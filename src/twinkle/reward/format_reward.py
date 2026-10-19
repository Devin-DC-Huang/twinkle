# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, List

from twinkle.data_format import Trajectory
from twinkle.reward.base import Reward, last_assistant_text


class FormatReward(Reward):

    @staticmethod
    def format_reward(completion: str) -> float:
        """Format reward: checks <think> and <answer> tags."""
        has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
        return 1.0 if (has_think and has_answer) else 0.0

    def __call__(self, trajectories: List[Trajectory], **kwargs: Any) -> List[float]:
        return [self.format_reward(last_assistant_text(trajectory)) for trajectory in trajectories]
