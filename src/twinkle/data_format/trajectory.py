# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import sys
import uuid
from copy import deepcopy
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

from .message import Message, Tool
from .sampling import SamplingState

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class Trajectory(TypedDict, total=False):
    messages: List[Message]
    tools: List[Tool]
    # PyArrow-stable encoding: each entry is (key, json.dumps(value)). Use the helpers below.
    user_data: List[Tuple[str, str]]
    images: Optional[List[Union[str, Any]]]
    videos: Optional[List[Union[str, Any]]]
    audios: Optional[List[Union[str, Any]]]
    prompt: Optional[str]
    request_id: str
    session_id: Optional[str]
    request_namespace: str
    sampling_state: SamplingState
    episode_state: Dict[str, Any]


def request_identity(trajectory, *, control=False):
    """不转换、不裁剪身份；旧控制仅针对旧尝试。"""
    if not isinstance(trajectory, Mapping):
        raise TypeError('生成请求必须为 trajectory/encoded input 映射。')
    request_id = trajectory.get('request_id')
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('request_id 必须在提交前创建，且为非空字符串。')
    session_id = trajectory.get('session_id')
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        raise ValueError('session_id 必须为非空字符串或 None。')
    namespace = trajectory.get('request_namespace', '')
    if not isinstance(namespace, str):
        raise TypeError('request_namespace 必须为字符串。')
    state = trajectory.get('sampling_state')
    if state is None:
        state = SamplingState()
    if not isinstance(state, SamplingState):
        raise TypeError('sampling_state 必须通过 schema 还原为 SamplingState。')
    state.__post_init__()
    return namespace, request_id, state.attempt


def trajectory_sticky_key(read, *, state='sampling_state'):
    """生成尝试的组合键；read 仅投影身份字段。"""
    namespace = read('trajectory.request_namespace', '')
    request_id = read('trajectory.request_id')
    attempt = read(f'trajectory.{state}.attempt', 0)
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('request_id 必须在提交前创建，且为非空字符串。')
    if not isinstance(namespace, str) or type(attempt) is not int or attempt < 0:
        raise ValueError('request_namespace 或尝试序号无效。')
    return namespace, request_id, attempt


def trajectory_sticky_affinity(read, *, argument='trajectory'):
    """会话亲和不代替一次生成尝试的调用归属。"""
    namespace = read(f'{argument}.request_namespace', '')
    session_id = read(f'{argument}.session_id')
    if session_id is None:
        session_id = read(f'{argument}.request_id')
    if not isinstance(namespace, str) or not isinstance(session_id, str) or not session_id:
        raise ValueError('会话亲和需要有效的命名空间和 session_id/request_id。')
    return namespace, session_id


def trajectory_sticky_fallback(read, *, state='sampling_state'):
    """未提交的暂停续接许可仍控制签发它的上一尝试。"""
    namespace, request_id, attempt = trajectory_sticky_key(read, state=state)
    if read(f'trajectory.{state}.status', 'new') == 'paused' and attempt > 0:
        return namespace, request_id, attempt - 1
    return None


def prepare_sampling_call(signature, component, args, kwargs):
    """批量便利入口在发送前补身份，同时保护嵌套输入。"""
    bound = signature.bind(component, *args, **kwargs)
    batch_arg = 'inputs' if 'inputs' in bound.arguments else 'trajectories'
    inputs = bound.arguments[batch_arg]
    if isinstance(inputs, Mapping):
        if batch_arg == 'trajectories':
            raise TypeError('rollout 批入口需要 List[Trajectory]。')
        inputs = [inputs]
    if isinstance(inputs, (list, tuple)):
        inputs = deepcopy(list(inputs))
        for trajectory in inputs:
            if not isinstance(trajectory, Mapping):
                raise TypeError('每个采样输入必须为映射。')
            if 'request_id' not in trajectory:
                trajectory['request_id'] = uuid.uuid4().hex
            request_identity(trajectory)
        bound.arguments[batch_arg] = inputs
    return bound.args[1:], bound.kwargs


def pack_value(value: Any) -> str:
    """Encode a single user_data value to a JSON string."""
    return json.dumps(value, ensure_ascii=False, default=str)


def pack_user_data(values: Any) -> List[Tuple[str, str]]:
    """Build a canonical ``user_data`` payload from a plain mapping.

    ``user_data`` must be a list of ``(key, json_string)`` pairs: a dict cannot be written to
    PyArrow (its struct schema would differ from shard to shard), and readers go through
    :func:`user_data_get`, which only understands the packed form — a dict is silently ignored
    there, so e.g. ``key_rounds`` would be dropped without any error. Always pack before
    attaching to a trajectory.
    """
    if values is None:
        return []
    if isinstance(values, Mapping):
        return [(k, v if isinstance(v, str) else pack_value(v)) for k, v in values.items()]
    return [(k, v if isinstance(v, str) else pack_value(v)) for k, v in values]


def attach_user_data(trajectory: Trajectory, **values: Any) -> Trajectory:
    """Return ``trajectory`` with ``values`` merged into its packed ``user_data``.

    ``user_data`` is a list of ``(key, json_string)`` pairs rather than a dict,
    so it cannot be updated in place with ``update()``; going through
    :func:`pack_user_data` keeps it in the one shape readers understand.
    """
    merged: Dict[str, Any] = {}
    for entry in trajectory.get('user_data') or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            merged[entry[0]] = entry[1]
    merged.update(values)
    out = dict(trajectory)
    out['user_data'] = pack_user_data(merged)
    return out


def ground_truth(trajectory: Trajectory, key: str = 'ground_truth') -> Any:
    """A trajectory's gold label: the packed ``user_data`` entry first, the flat column second.

    Both spellings carry the same label -- a twinkle pipeline packs ``user_data``, a swift scoring
    row keeps dataset columns flat -- so a Reward reads one helper and works in either world.
    """
    value = user_data_get(trajectory.get('user_data'), key)
    if value is None:
        value = trajectory.get(key)
    return value


def msg_content_text(msg: Dict[str, Any]) -> str:
    """Extract plain text from a message's content (str | list | dict)."""
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(p.get('text', '') for p in c if isinstance(p, dict) and p.get('type') == 'text')
    if isinstance(c, dict) and c.get('type') == 'text':
        return c.get('text', '')
    return ''


def last_assistant_text(trajectory: Trajectory) -> str:
    """The newest assistant turn's text -- the completion a reward scores.

    Multimodal messages carry content as a list of parts, so text goes through
    :func:`msg_content_text` instead of being read as a plain string. The *last* assistant
    turn: a conversation that went through tools has several, and the model's answer is
    the turn it finished on.
    """
    for message in reversed(trajectory.get('messages') or []):
        if isinstance(message, dict) and message.get('role') == 'assistant':
            return msg_content_text(message)
    return ''


def user_data_get(items: Any, key: str, default: Any = None) -> Any:
    """Look up the first value matching ``key`` in packed user_data, decoded."""
    if isinstance(items, Mapping):
        raise TypeError('user_data must be a list of (key, json_string) pairs, got a mapping. '
                        'Wrap it with twinkle.data_format.pack_user_data().')
    if not isinstance(items, list):
        return default
    for entry in items:
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == key:
            v = entry[1]
            if not isinstance(v, str):
                return v
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return v
    return default
