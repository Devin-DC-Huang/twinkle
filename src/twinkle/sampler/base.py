# Copyright (c) ModelScope Contributors. All rights reserved.
"""采样业务：批生成、一次尝试的 Future，以及真实暂停和终止。"""
import asyncio
import hashlib
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import CancelledError, Future
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, Union

import twinkle
from peft import PeftConfig
from twinkle import remote_function
from twinkle.data_format import (InputFeature, PoolingParams, PoolingResponse, SampleResponse, SampledSequence,
                                 SamplingMask, SamplingParams, SamplingState, Trajectory, is_encoded)
from twinkle.data_format.trajectory import (request_identity, trajectory_sticky_affinity,
                                            trajectory_sticky_fallback, trajectory_sticky_key)
from twinkle.patch import Patch
from twinkle.template import Template
from twinkle.utils import construct_class


@dataclass
class _SamplingTask:
    key: tuple
    trajectory: dict
    params: SamplingParams
    options: dict
    future: Future = field(default_factory=Future)
    command: Optional[str] = None
    active_ids: set = field(default_factory=set)
    result: Optional[dict] = None
    finished: Optional[float] = None
    resumed: bool = False
    revoked: bool = False


class Sampler(ABC):
    def __init__(self):
        self.engine = None
        self.template = None
        self._sampling_lock = threading.RLock()
        self._sampling_tasks = OrderedDict()
        self._request_tasks = {}
        self._sampling_closed = False
        self._sampling_retention = 300.0
        self._sampling_capacity = 4096

    @abstractmethod
    def sample(self, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
               sampling_params: Optional[SamplingParams] = None, adapter_name: str = '', *,
               num_samples: int = 1) -> List[SampleResponse]:
        """批生成仍返回与输入等长的 SampleResponse 列表，不降为串行 submit。"""
        raise NotImplementedError

    def encode(self, inputs, pooling_params: Optional[PoolingParams] = None,
               adapter_name: str = '') -> List[PoolingResponse]:
        raise NotImplementedError(f'{type(self).__name__} 不支持 pooling/encode。')

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _not_encoded(inputs):
        if not isinstance(inputs, dict):
            raise TypeError('采样输入必须为映射。')
        return not is_encoded(inputs)

    def _is_trajectory(self, inputs):
        if isinstance(inputs, list):
            return bool(inputs) and self._is_trajectory(inputs[0])
        return isinstance(inputs, dict) and self._not_encoded(inputs)

    def _normalize_inputs(self, inputs):
        return [inputs] if isinstance(inputs, dict) else list(inputs)

    def encode_trajectory(self, trajectory: Trajectory, adapter_name='', add_generation_prompt=True):
        if self.template is None:
            raise ValueError(f"未设置 adapter '{adapter_name}' 的 template。")
        from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine
        return TransformersEngine.encode_trajectory(self.template, trajectory, add_generation_prompt)

    def decode_response(self, token_ids, adapter_name=''):
        if self.template is None:
            raise ValueError(f"未设置 adapter '{adapter_name}' 的 template。")
        return self.template.decode(token_ids)

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True, replay=True)
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        if 'model_id' not in kwargs and hasattr(self, 'model_id'):
            kwargs['model_id'] = self.model_id
        self.template = construct_class(template_cls, Template, twinkle.template, **kwargs)

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True)
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig) -> None:
        raise NotImplementedError

    @staticmethod
    def _params(value):
        if value is None:
            return SamplingParams()
        if isinstance(value, dict):
            return SamplingParams.from_dict(value)
        if not isinstance(value, SamplingParams):
            raise TypeError('sampling_params 必须为 SamplingParams、dict 或 None。')
        result = deepcopy(value)
        result.__post_init__()
        return result

    def _expire_tasks(self):
        now = time.monotonic()
        for key, task in tuple(self._sampling_tasks.items()):
            retained = task.result is not None and task.result['sampling_state'].status == 'paused'
            if task.finished is not None and (not retained or task.revoked or task.resumed):
                if now - task.finished > self._sampling_retention or len(self._sampling_tasks) >= self._sampling_capacity:
                    self._sampling_tasks.pop(key)

    @staticmethod
    def _engine_request_id(key, branch):
        namespace, request_id, attempt = key
        # 长度前缀避免 Unicode、分隔符或数字样字符串造成身份碰撞。
        raw = f'{len(namespace)}:{namespace}{len(request_id)}:{request_id}'.encode('utf-8')
        return f'{hashlib.sha256(raw).hexdigest()}:{attempt}:{branch}'

    @remote_function(dispatch='all', collect='first', lazy_collect=True, enable_continous_work=True,
                     sticky=True, sticky_key=trajectory_sticky_key, sticky_mode='acquire',
                     sticky_affinity=trajectory_sticky_affinity)
    def submit(self, trajectory: Trajectory, sampling_params=None, adapter_name=None, adapter_path=None,
               *, use_base_model=None, token_only=None, **options) -> Future:
        key = request_identity(trajectory)
        snapshot = deepcopy(trajectory)
        state = snapshot.get('sampling_state')
        params = self._params(sampling_params if sampling_params is not None else state.params if state else None)
        call_options = deepcopy(options)
        for name, value in (('adapter_name', adapter_name), ('adapter_path', adapter_path),
                            ('use_base_model', use_base_model), ('token_only', token_only)):
            if value is not None:
                call_options[name] = deepcopy(value)
        with self._sampling_lock:
            self._expire_tasks()
            if self._sampling_closed:
                raise RuntimeError('采样器已关闭。')
            if key in self._sampling_tasks:
                raise ValueError(f'重复提交同一尝试：{key!r}')
            if len(self._sampling_tasks) >= self._sampling_capacity:
                raise RuntimeError('生成与暂停保留容量已满；调用方必须施加 backpressure。')
            previous = None
            if key[-1] > 0:
                previous = self._sampling_tasks.get((*key[:-1], key[-1] - 1))
                if (state is None or state.status != 'paused' or previous is None or previous.result is None
                        or previous.revoked or previous.resumed
                        or state.resume_token != previous.result['sampling_state'].resume_token):
                    raise ValueError('未知、过期或已撤销的续接许可。')
                if state.response is None or params.num_samples != len(state.response.sequences):
                    raise ValueError('续接必须保留原分支，不能改变 num_samples。')
                if call_options:
                    raise ValueError('续接沿用原 adapter 和生成选项；只能显式更新 sampling_params。')
                call_options = deepcopy(previous.options)
                previous.resumed = True
            elif state is not None and state.status != 'new':
                raise ValueError('已完成或停止的轨迹不能作为初始尝试重新提交。')
            else:
                call_options = {'adapter_name': '', 'adapter_path': None,
                                'use_base_model': False, 'token_only': False, **call_options}
            task = _SamplingTask(key, snapshot, params, call_options)
            task.future.set_running_or_notify_cancel()
            self._sampling_tasks[key] = task
            for branch in range(params.num_samples):
                self._request_tasks[self._engine_request_id(key, branch)] = task
            coroutine = self._execute_submission(task)
            try:
                asyncio.run_coroutine_threadsafe(coroutine, self._async_loop)
            except BaseException:
                coroutine.close()
                self._sampling_tasks.pop(key)
                for branch in range(params.num_samples):
                    self._request_tasks.pop(self._engine_request_id(key, branch), None)
                if previous is not None:
                    previous.resumed = False
                raise
            return task.future

    def _control_task(self, trajectory, command):
        key = request_identity(trajectory, control=True)
        with self._sampling_lock:
            self._expire_tasks()
            task = self._sampling_tasks.get(key)
            if task is None:
                # 尚未提交的续接许可只能控制签发它的尝试，不能寻找其他在途请求。
                state = trajectory.get('sampling_state')
                previous = self._sampling_tasks.get((*key[:-1], key[-1] - 1))
                if (state is not None and state.status == 'paused' and previous is not None
                        and not previous.resumed and previous.result is not None
                        and state.resume_token == previous.result['sampling_state'].resume_token):
                    task = previous
            if task is None:
                raise KeyError(f'未知或过期的生成尝试：{key!r}')
            if task.finished is not None:
                if (command == 'stop' and not task.resumed and task.result is not None
                        and task.result['sampling_state'].status == 'paused'):
                    task.revoked = True
                return task
            if task.command is None:
                task.command = command
            return task

    async def _control_active(self, task):
        with self._sampling_lock:
            ids = tuple(task.active_ids)
        # 控制失败不取消 Future、不清记录；必须报告真实清理尚未完成。
        await asyncio.gather(*(self._abort_request(request_id) for request_id in ids))

    @remote_function(dispatch='all', collect='first', lazy_collect=False, control=True,
                     sticky=True, sticky_key=trajectory_sticky_key, sticky_mode='use',
                     sticky_affinity=trajectory_sticky_affinity, sticky_fallback=trajectory_sticky_fallback)
    def pause(self, trajectory: Trajectory) -> Trajectory:
        task = self._control_task(trajectory, 'pause')
        if not task.future.done():
            asyncio.run_coroutine_threadsafe(self._control_active(task), self._async_loop).result()
        result = task.future.result()
        with self._sampling_lock:
            if task.revoked:
                raise CancelledError('暂停状态已由 stop 撤销。')
        return deepcopy(result)

    @remote_function(dispatch='all', collect='first', lazy_collect=False, control=True,
                     sticky=True, sticky_key=trajectory_sticky_key, sticky_mode='use',
                     sticky_affinity=trajectory_sticky_affinity, sticky_fallback=trajectory_sticky_fallback)
    def stop(self, trajectory: Trajectory) -> None:
        task = self._control_task(trajectory, 'stop')
        if not task.future.done():
            asyncio.run_coroutine_threadsafe(self._control_active(task), self._async_loop).result()
        try:
            task.future.result()
        except CancelledError:
            pass
        with self._sampling_lock:
            if not task.resumed and task.result is not None and task.result['sampling_state'].status == 'paused':
                task.revoked = True
        return None

    async def _execute_submission(self, task):
        error = None
        result = None
        try:
            response = await self._generate_submission(task)
            partial = any(sequence.stop_reason == 'abort' for sequence in response.sequences)
            if partial and task.command is None:
                raise RuntimeError('引擎自行中止了生成；没有业务暂停许可。')
            result = deepcopy(task.trajectory)
            result['sampling_state'] = SamplingState(
                attempt=task.key[-1] + int(partial), status='paused' if partial else 'completed',
                response=response, params=deepcopy(task.params), resume_token=uuid.uuid4().hex if partial else None)
        except BaseException as exc:
            error = exc
        with self._sampling_lock:
            if task.command == 'stop' and error is None:
                error = CancelledError('生成已真实停止并清理。')
                result = None
            task.result = result
            task.finished = time.monotonic()
            for branch in range(task.params.num_samples):
                self._request_tasks.pop(self._engine_request_id(task.key, branch), None)
            if error is not None:
                task.future.set_exception(error)
            else:
                task.future.set_result(deepcopy(result))

    async def _generate_submission(self, task):
        state = task.trajectory.get('sampling_state')
        previous = state.response if state is not None else None
        sequences = [None] * task.params.num_samples
        groups = {}
        for branch in range(task.params.num_samples):
            old = previous.sequences[branch] if previous is not None else None
            if old is not None and old.stop_reason != 'abort':
                sequences[branch] = deepcopy(old)
                continue
            remaining = task.params.max_tokens
            if old is not None and remaining is not None:
                remaining = max(0, remaining - len(old.tokens))
                if remaining == 0:
                    sequences[branch] = deepcopy(old)
                    sequences[branch].stop_reason = 'length'
                    continue
            if old is None:
                feature = deepcopy(task.trajectory)
                feature.pop('sampling_state', None)
            elif old.new_input_feature is not None:
                feature = deepcopy(old.new_input_feature)
            else:
                # token_only 的恢复仍是精确 token-in，不经过 decoded 文本。
                feature = deepcopy(task.trajectory)
                feature['input_ids'] = list(previous.prompt_token_ids) + list(old.tokens)
                feature.pop('sampling_state', None)
                feature.pop('messages', None)
            groups.setdefault(remaining, []).append((branch, feature, old))
        first = previous
        for budget, entries in groups.items():
            params = deepcopy(task.params)
            params.num_samples = 1
            params.max_tokens = budget
            responses = await self._generate_inputs(
                [feature for _, feature, _ in entries], params,
                request_ids=[self._engine_request_id(task.key, branch) for branch, _, _ in entries], **task.options)
            if len(responses) != len(entries):
                raise ValueError('后端未按输入项返回等长结果。')
            for (branch, _, old), response in zip(entries, responses):
                if len(response.sequences) != 1:
                    raise ValueError('每个受控分支必须返回一条准确序列。')
                if first is None:
                    first = response
                new = response.sequences[0]
                sequences[branch] = self._merge_sequence(old, new) if old is not None else new
        return SampleResponse(sequences=sequences, prompt_token_ids=deepcopy(first.prompt_token_ids),
                              prompt_logprobs=deepcopy(first.prompt_logprobs),
                              topk_prompt_logprobs=deepcopy(first.topk_prompt_logprobs))

    def _merge_sequence(self, old, new):
        merged = deepcopy(new)
        merged.tokens = list(old.tokens) + list(new.tokens)
        if old.logprobs is None or new.logprobs is None:
            if old.tokens and new.tokens and (old.logprobs is None) != (new.logprobs is None):
                raise ValueError('续接不能丢弃已有或新增 token 的 logprobs。')
            merged.logprobs = deepcopy(old.logprobs if not new.tokens else new.logprobs)
        else:
            merged.logprobs = deepcopy(list(old.logprobs) + list(new.logprobs))
        if old.sampling_mask is not None or new.sampling_mask is not None:
            if old.tokens and old.sampling_mask is None or new.tokens and new.sampling_mask is None:
                raise ValueError('续接的采样 mask 缺失。')
            left = old.sampling_mask or SamplingMask([], [0])
            right = new.sampling_mask or SamplingMask([], [0])
            merged.sampling_mask = SamplingMask(list(left.token_ids) + list(right.token_ids),
                                                list(left.offsets) + [len(left.token_ids) + offset
                                                                      for offset in right.offsets[1:]])
        if old.routed_experts is not None or new.routed_experts is not None:
            if old.routed_experts is None or new.routed_experts is None:
                raise ValueError('续接的 routed_experts 缺失。')
            import numpy as np
            merged.routed_experts = np.concatenate([old.routed_experts, new.routed_experts], axis=0)
        merged.decoded = self.template.decode(merged.tokens) if self.template is not None else None
        return merged

    async def _generate_inputs(self, inputs, sampling_params, **kwargs):
        raise NotImplementedError(f'{type(self).__name__} 必须实现批生成内核。')

    async def _abort_request(self, request_id):
        await self.engine.abort_request(request_id)

    async def _sample_engine_request(self, request_id, coroutine, *, prompt_token_ids=None):
        with self._sampling_lock:
            owner = self._request_tasks.get(request_id)
            if owner is not None and owner.command is not None:
                coroutine.close()
                return SampleResponse([SampledSequence(stop_reason='abort', tokens=[])],
                                      prompt_token_ids=list(prompt_token_ids or []))
            task = asyncio.create_task(coroutine)
            if owner is not None:
                owner.active_ids.add(request_id)
        try:
            return await asyncio.shield(task)
        except BaseException:
            async def cleanup():
                try:
                    await self._abort_request(request_id)
                finally:
                    await asyncio.gather(task, return_exceptions=True)
            cleanup_task = asyncio.create_task(cleanup())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            cleanup_task.result()
            raise
        finally:
            with self._sampling_lock:
                if owner is not None:
                    owner.active_ids.discard(request_id)

    @staticmethod
    async def _gather_generation_tasks(coroutines):
        tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
        group = asyncio.gather(*tasks)
        try:
            return await asyncio.shield(group)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            cleanup = asyncio.gather(*tasks, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            if not group.cancelled():
                group.exception()
            raise

    def _close_generations(self):
        with self._sampling_lock:
            self._sampling_closed = True
            tasks = tuple(self._sampling_tasks.values())
            for task in tasks:
                if task.finished is None and task.command is None:
                    task.command = 'stop'
                if task.finished is not None and not task.resumed:
                    task.revoked = True
        errors = []
        for task in tasks:
            try:
                if not task.future.done():
                    asyncio.run_coroutine_threadsafe(self._control_active(task), self._async_loop).result()
                task.future.result()
            except CancelledError:
                pass
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('采样器存在清理未完成的生成。') from errors[0]
