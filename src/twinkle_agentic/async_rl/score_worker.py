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
import math
import threading
from copy import deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function

#: The CPU DeviceGroup name the scoring worker actor lives in.
REF_ONLY_REMOTE_GROUP = 'rl_payload'


def _score_rows(samples: Sequence[Any]) -> List[dict]:
    """The trajectory row a scorer sees: the recorded conversation, else prompt + own completion.

    A row IS a trajectory dict -- ``messages``, the sample's flat extras, and its
    ``response_token_ids`` (the length-aware rules score on it) -- so every scorer kind reads one
    shape whether it is a rule plugin, a twinkle ``Reward``, or a model handle.
    """
    rows = []
    for sample in samples:
        tokens = getattr(sample, 'response_token_ids', None)
        row = {**(getattr(sample, 'extra', None) or {}),
               'response_token_ids': list(tokens) if tokens is not None else []}
        messages = getattr(sample, 'messages', None)
        if messages is None:
            prompt = list(getattr(sample, 'prompt', None) or [])
            if not prompt or prompt[-1].get('role') != 'assistant':
                prompt = prompt + [{'role': 'assistant', 'content': sample.decoded}]
            messages = prompt
        row['messages'] = list(messages)
        rows.append(row)
    return rows


def _run_grouped(tasks: Sequence[Tuple[Any, Callable[[], Any]]]) -> List[Any]:
    """Run scoring tasks grouped by placement: groups overlap as threads, within a group serial.

    A thread parked on its own actors' ray.get releases the GIL, so scorers on disjoint device
    groups (each RM on its own cards, a judge on its own engine) overlap wall-clock; one group
    stays strictly serial -- two forwards on the same cards would only race for memory. A single
    group runs inline (the all-rules shape starts no pool) and the first exception propagates
    unchanged.
    """
    groups: Dict[Any, List[int]] = {}
    for index, (key, _task) in enumerate(tasks):
        groups.setdefault(key, []).append(index)
    results: List[Any] = [None] * len(tasks)
    if len(groups) <= 1:
        for index, (_key, task) in enumerate(tasks):
            results[index] = task()
        return results

    def _run_group(indices: List[int]) -> None:
        for index in indices:
            results[index] = tasks[index][1]()

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(_run_group, indices) for indices in groups.values()]
        for future in futures:
            future.result()
    return results


@remote_class(execute='first')
class ScoreWorker:
    """Score rollout samples where the data lives: one remote boundary for the ORM channel.

    The driver (or the caller's group-completion hook) hands samples -- the payloads may be
    LazyCollect / ObjectRef refs under a remote rollout -- and this actor resolves them, rebuilds
    the scoring rows, runs every scorer (duck-typed: ``scorer(rows)`` for rules and twinkle
    ``Reward`` classes, ``scorer.score`` for model handles; scorers on disjoint device groups
    overlap as threads, one group strictly serial), and returns the ``[N][n_scorers]``
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
        self._submission_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='score-worker')
        self._group_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='score-group')
        self._score_lock = threading.RLock()
        if self._prm_weights is not None:
            if len(self._prm_weights) != len(self.prm_scorers):
                raise ValueError('prm_weights 必须与 PRM scorer 等长。')
            if any(not math.isfinite(float(value)) for value in self._prm_weights):
                raise ValueError('prm_weights 必须为有限数。')

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def prepare_sample(self, trajectory: dict, *, prompt_id: str, extra: dict, prompt: list,
                       group_key: str, policy_version: int, version_span: int) -> Any:
        """转换也由 worker 推进；本地调用不阻塞 driver。"""
        return self._submission_executor.submit(
            self._prepare_sample, trajectory, prompt_id, extra, prompt, group_key, policy_version, version_span)

    def _prepare_sample(self, trajectory, prompt_id, extra, prompt, group_key, policy_version, version_span):
        from twinkle_agentic.rollout.multi_turn import trajectory_to_rollout_sample
        from twinkle_agentic.rollout.ledger import sampled_token_logprobs
        from twinkle.data_format.sampling import SamplingState
        state = trajectory.get('sampling_state')
        if isinstance(state, SamplingState):
            if state.status != 'completed' or state.response is None or len(state.response.sequences) != 1:
                raise ValueError('单样本训练只接受一条已完成采样分支。')
            sequence = state.response.sequences[0]
            if sequence.new_input_feature is None:
                raise ValueError('训练采样必须保留 new_input_feature；不能用 decoded 重编码。')
            feature = deepcopy(sequence.new_input_feature)
            logprobs = sequence.logprobs
            if sequence.tokens and logprobs is None:
                raise ValueError('训练采样缺少真实旧 logprobs。')
            if logprobs is not None and len(logprobs) != len(sequence.tokens):
                raise ValueError('采样 logprobs 与 token 数量不一致。')
            trajectory = dict(feature, response_token_ids=list(sequence.tokens),
                              rollout_logprobs=sampled_token_logprobs(sequence.tokens, logprobs),
                              sampling_mask=sequence.sampling_mask,
                              truncated=sequence.stop_reason == 'length', stop_reason=sequence.stop_reason)
            trajectory['messages'] = deepcopy(prompt) + [{'role': 'assistant', 'content': sequence.decoded}]
        elif trajectory.get('episode_state', {}).get('status') != 'completed':
            raise ValueError('暂停轨迹不能转换成完整训练 episode。')
        sample = trajectory_to_rollout_sample(trajectory, prompt_id, deepcopy(extra))
        sample.prompt = deepcopy(prompt)
        sample.group_key = group_key
        sample.policy_version = policy_version
        sample.version_span = version_span
        return sample

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def submit(self, samples: List[Any], *, transform: Any = None, with_prm: bool = False) -> Future:
        """本地与远程均非阻塞；普通 callable 在评分 worker 执行。"""
        return self._submission_executor.submit(self._score_submission, samples, transform, with_prm)

    def _score_submission(self, samples, transform, with_prm):
        with self._score_lock:
            samples = deepcopy(samples)
            if transform is not None:
                for index, sample in enumerate(samples):
                    replacement = transform(sample)
                    if replacement is not None:
                        samples[index] = replacement
            if with_prm:
                rows, processes = self.score_group_prm(samples)
                if len(processes) != len(samples):
                    raise ValueError('PRM 必须按样本返回等长结果。')
                for sample, process in zip(samples, processes):
                    if process is not None and len(process) != len(sample.response_token_ids):
                        raise ValueError('PRM token 向量与 response_token_ids 长度不一致。')
                    sample.prm_process = process
            else:
                rows = self.score_group(samples)
            if len(rows) != len(samples):
                raise ValueError('ORM 必须按样本返回等长结果。')
            for sample, row in zip(samples, rows):
                sample.orm_reward = row
            return {'samples': samples, 'orm': rows}

    @remote_function(dispatch='all', collect='first', lazy_collect=True, control=True)
    def decide_group(self, group_key: str, samples: List[Any], *, regenerate=None,
                     baseline_fn=None, can_regenerate: bool = False) -> Future:
        """组钩子在评分结束后至多执行一次，不占用 driver 推进线程。"""
        return self._group_executor.submit(
            self._decide_group, group_key, samples, regenerate, baseline_fn, can_regenerate)

    def _decide_group(self, group_key, samples, regenerate, baseline_fn, can_regenerate):
        retry = bool(regenerate(group_key, samples)) if regenerate is not None else False
        if retry and can_regenerate:
            return {'regenerate': True, 'baseline': None}
        baseline = baseline_fn(group_key, samples) if baseline_fn is not None else None
        return {'regenerate': False, 'baseline': baseline}

    @remote_function(dispatch='all', collect='first', lazy_collect=False, control=True)
    def completion_state(self, trajectory: dict) -> str:
        """仅返回业务终态；不让控制面读取部分 token 载荷。"""
        if 'episode_state' in trajectory:
            return trajectory['episode_state']['status']
        return trajectory['sampling_state'].status

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True, shutdown=True)
    def close(self) -> None:
        """退出前等全部评分任务真正结束。"""
        self._group_executor.shutdown(wait=True)
        self._submission_executor.shutdown(wait=True)

    def _one_reward(self, scorer: Any, rows: List[dict]) -> List[float]:
        """One scorer's batch over trajectory rows -- the single call shape every kind shares.

        A model handle answers ``scorer.score(rows)``; every other kind (rule plugin, judge, a
        twinkle ``Reward``) is itself callable on the SAME rows: the row IS a trajectory dict
        (``messages`` plus the sample's extras), so type differences live in construction and
        the runtime protocol stays one shape.
        """
        score = getattr(scorer, 'score', None)
        values = score(rows) if callable(score) else scorer(rows)
        values = list(values)
        if len(values) != len(rows):
            raise ValueError(f'评分器返回 {len(values)} 行，预期 {len(rows)} 行。')
        return [None if value is None or value != value else float(value) for value in values]

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def score_group(self, samples: List[Any]) -> List[List[float]]:
        """Score one group with every ORM scorer -> the ``[N][n_scorers]`` sample-major matrix.

        Rows are rebuilt from the resolved samples (the recorded conversation for a multi-turn
        rollout, prompt + assistant completion otherwise), so a rule, a scalar RM and a judge
        all see the same view. The matrix is already transposed sample-major with ``None`` /
        NaN normalized to ``None`` -- the caller writes rows straight onto the samples.
        """
        with self._score_lock:
            rows = _score_rows(samples)
            matrix = _run_grouped([
                (getattr(scorer, 'placement_key', None),
                 lambda scorer=scorer: self._one_reward(scorer, rows))
                for scorer in self.orm_scorers])
            return [[None if value is None or value != value else float(value) for value in column]
                    for column in zip(*matrix)] if matrix else [[] for _ in samples]

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def score_group_prm(self, samples: List[Any]) -> Tuple[List[List[Optional[float]]], List[Any]]:
        """ORM 小矩阵与 PRM token 向量共享一个可投影结果。"""
        with self._score_lock:
            return self._score_group_prm(samples)

    def _score_group_prm(self, samples):
        if self.prm_step_scorer is None or not self.prm_scorers:
            raise RuntimeError('score_group_prm was called without a wired PRM channel.')
        resolved = samples
        rows = _score_rows(resolved)
        tasks = [(getattr(scorer, 'placement_key', None),
                  lambda scorer=scorer: self._one_reward(scorer, rows))
                 for scorer in self.orm_scorers]
        prm_key = next((key for key in (getattr(scorer, 'placement_key', None)
                                        for scorer in self.prm_scorers) if key is not None), None)
        for sample, row in zip(resolved, rows):
            columns = {key: value for key, value in row.items() if key != 'messages'}
            tasks.append((prm_key, lambda sample=sample, columns=columns: self.prm_step_scorer(
                sample, self.template.decode, self._weighted_step_score, columns=columns)))
        results = _run_grouped(tasks)
        matrix = results[:len(self.orm_scorers)]
        rewards = ([[None if value is None or value != value else float(value) for value in column]
                    for column in zip(*matrix)] if matrix else [[] for _ in samples])
        return rewards, results[len(self.orm_scorers):]

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
                resolved = sample
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
