"""The generic scoring worker: the ONE remote boundary that orchestrates reward scoring.

Algorithm-agnostic scoring orchestration for the streaming RL driver: resolve rollout-sample
refs where the data lives, rebuild the scoring rows, run every ORM scorer, and return small
sample-major reward rows -- the driver never materializes token-scale data. Both scoring
CHANNELS (ORM outcome rows, PRM per-token process rewards) are protocol here and live in this
base; algorithm-specific halves (advantage fusion, teacher reweighting) stay in the caller's
subclass. The PRM step SEGMENTATION is an injected plugin callable (``prm_step_scorer``):
the caller's object, executed inside this actor.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from twinkle import remote_class, remote_function

#: The CPU DeviceGroup name the scoring worker actor lives in.
REF_ONLY_REMOTE_GROUP = 'rl_payload'


def _resolve(ref: Any) -> Any:
    """Resolve a LazyCollect / ObjectRef in THIS actor, never in the driver."""
    if callable(ref) and getattr(ref, '_is_lazy_collect', False):
        return ref()
    return ref


def _score_rows(samples: Sequence[Any]) -> List[dict]:
    """The trajectory row a scorer sees: the recorded conversation, else prompt + own completion.

    A row IS a trajectory dict -- ``messages``, the sample's flat extras, and its
    ``response_token_ids`` (the length-aware rules score on it) -- so every scorer kind reads one
    shape whether it is a rule plugin, a twinkle ``Reward``, or a model handle.
    """
    rows = []
    for sample in samples:
        row = {**(getattr(sample, 'extra', None) or {}),
               'response_token_ids': list(getattr(sample, 'response_token_ids', None) or [])}
        messages = getattr(sample, 'messages', None)
        if messages is None:
            prompt = list(getattr(sample, 'prompt', None) or [])
            if not prompt or prompt[-1].get('role') != 'assistant':
                prompt = prompt + [{'role': 'assistant', 'content': sample.decoded}]
            messages = prompt
        row['messages'] = list(messages)
        rows.append(row)
    return rows


@remote_class(execute='first')
class ScoreWorker:
    """Score rollout samples where the data lives: one remote boundary for the ORM channel.

    The driver (or the caller's group-completion hook) hands samples -- the payloads may be
    LazyCollect / ObjectRef refs under a remote rollout -- and this actor resolves them, rebuilds
    the scoring rows, runs every scorer (duck-typed: ``scorer(rows)`` for rules and twinkle
    ``Reward`` classes, ``scorer.score`` for model handles), and returns the ``[N][n_scorers]``
    sample-major matrix with ``None`` for
    an unparseable verdict or a NaN, so the caller's nan-aware sums skip them. Subclasses add
    the algorithm-specific halves (advantage fusion, teacher reweighting) on the same actor.
    """

    def __init__(self, *, orm_scorers: Sequence[Any], template: Any = None,
                 prm_scorers: Sequence[Any] = (), prm_step_scorer: Any = None,
                 prm_weights: Optional[Sequence[float]] = None):
        self.orm_scorers = list(orm_scorers)
        self.template = template
        self.prm_scorers = list(prm_scorers)
        self.prm_step_scorer = prm_step_scorer
        self._prm_weights = list(prm_weights) if prm_weights is not None else None
        self._global_step = 0

    def _one_reward(self, scorer: Any, rows: List[dict]) -> List[float]:
        """One scorer's batch over trajectory rows -- the single call shape every kind shares.

        A model handle answers ``scorer.score(rows)``; every other kind (rule plugin, judge, a
        twinkle ``Reward``) is itself callable on the SAME rows: the row IS a trajectory dict
        (``messages`` plus the sample's extras), so type differences live in construction and
        the runtime protocol stays one shape.
        """
        score = getattr(scorer, 'score', None)
        values = score(rows) if callable(score) else scorer(rows)
        return [None if value is None or value != value else float(value) for value in values]

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def score_group(self, samples: List[Any]) -> List[List[float]]:
        """Score one group with every ORM scorer -> the ``[N][n_scorers]`` sample-major matrix.

        Rows are rebuilt from the resolved samples (the recorded conversation for a multi-turn
        rollout, prompt + assistant completion otherwise), so a rule, a scalar RM and a judge
        all see the same view. The matrix is already transposed sample-major with ``None`` /
        NaN normalized to ``None`` -- the caller writes rows straight onto the samples.
        """
        resolved = [_resolve(sample) for sample in samples]
        rows = _score_rows(resolved)
        matrix = [self._one_reward(scorer, rows) for scorer in self.orm_scorers]
        return [[None if value is None or value != value else float(value) for value in column]
                for column in zip(*matrix)] if matrix else [[] for _ in samples]

    def _weighted_step_score(self, step_rows: List[Dict[str, Any]]) -> List[float]:
        """One weighted process score per PRM step row (nan-aware: a judge miss never poisons it)."""
        matrix = [self._one_reward(scorer, step_rows) for scorer in self.prm_scorers]
        if not matrix:
            raise RuntimeError('the PRM channel resolved no scorers.')
        weights = (list(self._prm_weights) if self._prm_weights is not None else [1.0] * len(matrix))
        if len(weights) != len(matrix):
            raise ValueError(f'prm_weights length {len(weights)} != number of PRM scorers {len(matrix)}.')
        # The nan-aware weighted sum: None / NaN steps score as zero (nansum semantics) so an
        # unparseable verdict degrades that step instead of poisoning the whole process reward.
        rows = []
        for per_scorer in zip(*matrix):
            total = 0.0
            for value, weight in zip(per_scorer, weights):
                total += 0.0 if value is None else float(value) * weight
            rows.append(total)
        return rows

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def score_prm(self, samples: List[Any]) -> List[Any]:
        """Per-sample per-token process rewards for one group (heavy: stays a ref until consumed).

        Each resolved sample is segmented by the ``prm_step_scorer`` plugin (its own step spans,
        multi-turn context included -- it owns both); every step is scored in one batch through
        the ``prm_scorers`` channel (same duck-typed view as ``score_group``) and broadcast onto
        the response tokens by the plugin. The plugin signature is
        ``prm_step_scorer(sample, decode, score_steps, columns) -> List[float]``.
        """
        if self.prm_step_scorer is None or not self.prm_scorers:
            raise RuntimeError('score_prm was called without a wired PRM channel.')
        process: List[Any] = []
        for sample in samples:
            resolved = _resolve(sample)
            row = _score_rows([resolved])[0]
            process.append(self.prm_step_scorer(
                resolved, self.template.decode, self._weighted_step_score,
                columns={key: value for key, value in row.items() if key != 'messages'}))
        return process

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def prm_ready(self) -> bool:
        """Whether this actor carries a wired PRM channel (probed once by the driver)."""
        return self.prm_step_scorer is not None and bool(self.prm_scorers)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def log_completions(self, samples: List[Any], rewards: List[List[float]], step: int,
                        path: str) -> None:
        """Append one group's completions + rewards to ``completions.jsonl`` (resolved here)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as stream:
            for sample, per_func in zip(samples, rewards):
                resolved = _resolve(sample)
                stream.write(json.dumps({
                    'step': step,
                    'prompt': getattr(resolved, 'prompt', None),
                    'completion': resolved.decoded,
                    'rewards': per_func,
                }, ensure_ascii=False) + '\n')

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def set_global_step(self, step: int) -> None:
        """Track the driver's optimizer step for step-indexed logs."""
        self._global_step = int(step)
