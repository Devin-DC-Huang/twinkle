# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, List

from twinkle.data_format import Trajectory


def last_assistant_text(trajectory: Trajectory) -> str:
    """The newest assistant turn's text -- the completion every Reward scores."""
    for message in reversed(trajectory.get('messages') or []):
        if message.get('role') == 'assistant':
            return message.get('content') or ''
    return ''


def ground_truth(trajectory: Trajectory, key: str = 'ground_truth') -> Any:
    """A trajectory's gold label: the packed ``user_data`` entry first, the flat column second.

    Both spellings carry the same label -- a twinkle pipeline packs ``user_data``, a swift scoring
    row keeps dataset columns flat -- so a Reward reads one helper and works in either world.
    """
    from twinkle.data_format import user_data_get
    value = user_data_get(trajectory.get('user_data'), key)
    if value is None:
        value = trajectory.get(key)
    return value


class Reward:
    """Score completed trajectories: ``reward(trajectories, **kwargs) -> one float each."""

    def __call__(self, trajectories: List[Trajectory], **kwargs: Any) -> List[float]:
        ...
