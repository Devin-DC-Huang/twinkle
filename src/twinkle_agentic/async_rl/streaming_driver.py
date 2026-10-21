# Copyright (c) ModelScope Contributors. All rights reserved.
"""持续推进生成和评分；训练确认与权重发布不占用推进线程。"""
from __future__ import annotations

import copy
import math
import queue
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from twinkle.infra import call, project
from twinkle_agentic.rollout.multi_turn import RolloutSample
from .score_worker import ScoreWorker
from .types import PartitionAdmission, RolloutPolicy


class GroupDecision(Enum):
    ACCEPT = 'accept'
    AUGMENT = 'augment'
    REGENERATE = 'regenerate'


@dataclass(frozen=True)
class PromptRef:
    ordinal: int
    row: Any


@dataclass(eq=False)
class BufferRecord:
    sample: Any
    version: int
    group_key: str
    score_ref: Any = None


@dataclass
class DriverStats:
    admitted: int = 0
    collected: int = 0
    consumed: int = 0
    off_policy_consumed: int = 0
    dropped_stale: int = 0
    dropped_incomplete: int = 0
    train_pulls: int = 0
    optimizer_steps: int = 0
    publishes: int = 0
    final_version: int = 0
    train_active_time: float = 0.0
    idle_time: float = 0.0
    last_pull_idle_ratio: float = 0.0
    submit_time: float = 0.0
    poll_time: float = 0.0
    collect_time: float = 0.0
    score_time: float = 0.0
    publish_time: float = 0.0
    capacity_blocked_passes: int = 0
    staleness_blocked_passes: int = 0
    ready_high_watermark: int = 0
    inflight_high_watermark: int = 0
    total_time: float = 0.0


@dataclass(eq=False)
class _Inflight:
    trajectory: dict
    future: Any
    policy: RolloutPolicy
    group_key: str
    prompt: PromptRef
    control: Any = None
    paused: Any = None
    state_check: Any = None
    lease: Any = None
    released: bool = False


@dataclass
class _Group:
    prompt: PromptRef
    version: int
    regenerations: int = 0
    resolved: bool = False
    decision: Any = None


@dataclass
class _Delivery:
    batch_id: int
    records: list
    batch: Any
    delivered_at: float = field(default_factory=time.monotonic)


@dataclass
class _Publication:
    push: Callable
    after_batch: int
    partial: bool
    completion: Future = field(default_factory=Future)
    ready: threading.Event = field(default_factory=threading.Event)
    confirmation: Future = field(default_factory=Future)
    phase: str = 'requested'
    window: Any = None
    started: float = 0.0
    pushing: bool = False


_ITER_DONE = object()
_NULL_POLICY = RolloutPolicy('__null__', None, 0, None)


class StreamingDriver:
    """单一控制面；所有准入、评分、组装、pin 和版本写入归推进线程。"""

    def __init__(self, *, prompt_stream: Iterable[Any], max_staleness: int,
                 num_generations: int, groups_per_partition: int, buffer_depth: int,
                 num_train_epochs: float = 1.0, rollout: Any = None,
                 submit: Optional[Callable] = None, pause: Optional[Callable] = None,
                 stop: Optional[Callable] = None, resume: Optional[Callable] = None,
                 request_builder: Optional[Callable] = None,
                 sampling_params: Any = None, assembly_ready: Optional[Callable] = None,
                 pull_rule: Optional[Literal['group', 'prefix']] = None, need_rows: Optional[int] = None,
                 run_id: Optional[str] = None, consume: Optional[Callable] = None,
                 group_key: Optional[Callable] = None, score: Optional[Callable] = None,
                 assemble: Optional[Callable] = None, regenerate: Optional[Callable] = None,
                 baseline_fn: Optional[Callable] = None, max_group_regenerations: int = 0,
                 prune: Optional[Callable] = None, initial_adapter_path: Optional[Callable] = None,
                 snapshot_model: Any = None, snapshot_dir: Optional[str] = None,
                 adapter_name: Optional[str] = None, max_steps: Optional[int] = None,
                 reached_max: Optional[Callable] = None, score_worker: Any = None,
                 resource_adapter: Any = None, resource_pool: Any = None,
                 allow_partial_rollout: bool = False, sleep: Optional[Callable] = None):
        for name, value, minimum in (('max_staleness', max_staleness, 0),
                                     ('num_generations', num_generations, 1),
                                     ('groups_per_partition', groups_per_partition, 1),
                                     ('buffer_depth', buffer_depth, 1),
                                     ('max_group_regenerations', max_group_regenerations, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} 必须为不小于 {minimum} 的整数。')
        if not math.isfinite(num_train_epochs) or num_train_epochs <= 0:
            raise ValueError('num_train_epochs 必须为正的有限数。')
        if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
            raise ValueError('max_steps 必须为非负整数或 None。')
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError('run_id 必须为非空字符串或 None。')
        self._rollout = rollout
        self._submit = submit if submit is not None else self._default_submit
        self._pause = pause if pause is not None else getattr(rollout, 'pause', None)
        self._stop = stop if stop is not None else getattr(rollout, 'stop', None)
        self._resume = resume if resume is not None else getattr(rollout, 'submit', None)
        self._request_builder = request_builder or self._default_request
        if submit is None and not callable(getattr(rollout, 'submit', None)):
            raise TypeError('rollout 必须提供业务 submit，或显式配置 submit。')
        if pull_rule not in (None, 'group', 'prefix'):
            raise ValueError('pull_rule 只支持 group/prefix。')
        if need_rows is not None and (type(need_rows) is not int or need_rows < 1):
            raise ValueError('need_rows 必须为正整数。')
        if assembly_ready is None:
            if need_rows is None or pull_rule is None:
                raise ValueError('需要 assembly_ready 或 pull_rule 与 need_rows。')
            assembly_ready = (self._make_group_pull_rule(num_generations, need_rows)
                              if pull_rule == 'group' else self._make_prefix_pull_rule(need_rows))
        quantum = (math.lcm(num_generations, need_rows) if pull_rule == 'group' else need_rows)
        if buffer_depth < max(num_generations, quantum or num_generations):
            raise ValueError('buffer_depth 必须容纳一组生成与一个完整训练批次。')
        for name, fn in (('submit', self._submit), ('stop', self._stop),
                         ('request_builder', self._request_builder), ('assembly_ready', assembly_ready)):
            if not callable(fn):
                raise TypeError(f'{name} 必须为 callable。')
        for name, fn in (('consume', consume), ('score', score), ('assemble', assemble),
                         ('regenerate', regenerate), ('baseline_fn', baseline_fn),
                         ('prune', prune), ('initial_adapter_path', initial_adapter_path),
                         ('reached_max', reached_max), ('group_key', group_key)):
            if fn is not None and not callable(fn):
                raise TypeError(f'{name} 必须为 callable 或 None。')
        if allow_partial_rollout and (not callable(self._pause) or not callable(self._resume)):
            raise TypeError('显式续接需要业务 pause 与 resume submit。')
        self._prompt_stream = prompt_stream
        self._num_train_epochs = num_train_epochs
        self._max_staleness = max_staleness
        self._num_generations = num_generations
        self._groups_per_partition = groups_per_partition
        self._buffer_depth = buffer_depth
        self._sampling_params = copy.deepcopy(sampling_params)
        self._assembly_ready = assembly_ready
        self._pull_rule = pull_rule
        self._consume = consume
        self._score = score
        self._assemble = assemble
        self._group_key = group_key or (lambda ref: str(ref.ordinal))
        self._regenerate = regenerate
        self._baseline_fn = baseline_fn
        self._max_group_regenerations = max_group_regenerations
        self._run_id = run_id
        self._policy = _NULL_POLICY
        self._policy_pins = {}
        self._windows = []
        self._next_step = 0
        self._training_step = None
        self._snapshot_model = snapshot_model
        self._snapshot_dir = snapshot_dir
        self._snapshot_adapter_name = adapter_name
        self._snapshot_paths = set()
        self._prune = prune or (self._default_prune_snapshots if snapshot_model is not None else lambda: None)
        self._initial_adapter_path = initial_adapter_path or (
            self._save_snapshot_version0 if snapshot_model is not None else lambda: None)
        self._max_steps = max_steps
        self._budget_steps = 0
        self._reached_max = reached_max or self._default_reached_max
        self._worker = score_worker if score_worker is not None else ScoreWorker(orm_scorers=[])
        self._owns_worker = score_worker is None
        for name in ('prepare_sample', 'submit', 'decide_group', 'completion_state', 'prm_ready'):
            if not callable(getattr(self._worker, name, None)):
                raise TypeError(f'score_worker 必须提供 {name}。')
        self._prm_ready = bool(self._worker.prm_ready())
        self._resource_adapter = resource_adapter
        self._resource_pool = resource_pool
        self._allow_partial = allow_partial_rollout
        self._sleep = sleep or time.sleep
        self.stats = DriverStats()
        self._prompt_seq = 0
        self._group_seq = 0
        self._handles = OrderedDict()
        self._preparing = []
        self._ready = []
        self._groups = {}
        self._regenerations = []
        self._waiting = None
        self._batch_seq = 0
        self._acked_batch = 0
        self._consumer_batch = None
        self._commands = queue.Queue()
        self._batch_queue = queue.Queue(maxsize=1)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._finished = threading.Event()
        self._iter_thread = None
        self._run_error = None
        self._publication = None
        self._requested_publication = None
        self._controls = ThreadPoolExecutor(max_workers=2, thread_name_prefix='rollout-control')

    @staticmethod
    def _as_future(value):
        if callable(getattr(value, 'done', None)) and callable(getattr(value, 'result', None)):
            return value
        future = Future()
        future.set_result(value)
        return future

    @staticmethod
    def _value(value):
        return value.result() if callable(getattr(value, 'result', None)) else value

    @staticmethod
    def _sample_view(payload):
        """本地值直接复用；远程样本的重字段维持同一 Future 投影。"""
        small = {'truncated', 'baseline', 'orm_reward', 'policy_version', 'version_span',
                 'prompt_id', 'group_key'}
        values = {item.name: project(payload, item.name) for item in fields(RolloutSample)}
        for name in small:
            values[name] = StreamingDriver._value(values[name])
        return RolloutSample(**values)

    @staticmethod
    def _make_group_pull_rule(num_generations, need_rows):
        quantum = math.lcm(num_generations, need_rows) // num_generations
        def rule(buffer):
            groups = OrderedDict()
            for record in buffer:
                groups.setdefault(record.group_key, []).append(record)
            complete = [rows for rows in groups.values() if len(rows) == num_generations]
            count = len(complete) // quantum * quantum
            return [row for group in complete[:count] for row in group] or None
        return rule

    @staticmethod
    def _make_prefix_pull_rule(need_rows):
        def rule(buffer):
            count = len(buffer) // need_rows * need_rows
            return buffer[:count] or None
        return rule

    def _default_request(self, prompt, trajectory_idx):
        row = copy.deepcopy(prompt.row)
        row['request_id'] = uuid.uuid4().hex
        row['request_namespace'] = self._run_id or 'inference'
        row['session_id'] = row['request_id']
        row.pop('sampling_state', None)
        row.pop('episode_state', None)
        return row

    def _default_submit(self, trajectory, policy):
        kwargs = {}
        if self._sampling_params is not None:
            kwargs['sampling_params'] = copy.deepcopy(self._sampling_params)
        if policy.adapter_path is not None:
            kwargs['adapter_path'] = policy.adapter_path
        return self._rollout.submit(trajectory, **kwargs)

    def _save_snapshot(self, name):
        path = self._snapshot_model.save(name, output_dir=self._snapshot_dir,
                                        adapter_name=self._snapshot_adapter_name, save_optimizer=False)
        path = self._value(path)
        if not isinstance(path, str) or not path:
            raise ValueError('adapter snapshot 必须返回非空路径。')
        self._snapshot_paths.add(path)
        return path

    def _save_snapshot_version0(self):
        return self._save_snapshot('v0')

    def _default_publish_push(self, window):
        return self._save_snapshot(f'v{self._current_version() + 1}')

    def _default_prune_snapshots(self):
        keep = self._adapter_paths_to_keep()
        for path in tuple(self._snapshot_paths):
            if path not in keep:
                shutil.rmtree(path)
                self._snapshot_paths.remove(path)

    def _default_reached_max(self):
        return self._max_steps is not None and self._budget_steps >= self._max_steps

    def _register_initial_policy(self, path):
        if self._run_id is not None:
            self._policy = RolloutPolicy(self._run_id, self._snapshot_adapter_name, 0, path)

    def _acquire_policy(self):
        policy = self._policy
        if policy.adapter_path is not None:
            self._policy_pins[policy.adapter_path] = self._policy_pins.get(policy.adapter_path, 0) + 1
        return policy

    def _release_pin(self, entry):
        if entry.released:
            return
        path = entry.policy.adapter_path
        if path is not None:
            count = self._policy_pins.get(path, 0)
            if count < 1:
                raise RuntimeError(f'未取得 rollout pin：{path}')
            if count == 1:
                self._policy_pins.pop(path)
            else:
                self._policy_pins[path] = count - 1
        if entry.lease is not None:
            self._resource_pool.release_lease(entry.lease)
        entry.released = True

    def _adapter_paths_to_keep(self):
        return set(self._policy_pins) | ({self._policy.adapter_path} if self._policy.adapter_path else set())

    def _current_version(self):
        return self._policy.version

    def _live_windows(self):
        return self._windows

    def _open_window(self):
        if self._run_id is None:
            return True
        if self._windows and self._windows[-1].step == self._current_version():
            return True
        if len(self._windows) >= self._max_staleness + 1:
            return False
        window = PartitionAdmission(context_key=self._run_id,
                                    partition_id=f'{self._run_id}:{self._next_step}', step=self._next_step,
                                    target_groups=self._groups_per_partition, num_generations=self._num_generations)
        self._next_step += 1
        self._windows.append(window)
        return True

    def _begin_publish(self, window):
        if not self._windows or self._windows[0] is not window or self._training_step is not None:
            raise RuntimeError('发布只能关闭最早的存活窗口。')
        self._training_step = window.step

    def _finish_publish(self, window, path):
        self._policy = RolloutPolicy(self._run_id, self._snapshot_adapter_name, self._policy.version + 1, path)
        self._windows.remove(window)
        self._training_step = None

    def _clear_window(self, window):
        if window in self._windows:
            self._windows.remove(window)
        if self._training_step == window.step:
            self._training_step = None

    def _iter_rows(self):
        for epoch in range(math.ceil(self._num_train_epochs)):
            setter = getattr(self._prompt_stream, 'set_epoch', None)
            if callable(setter):
                setter(epoch)
            limit = self._num_train_epochs - epoch
            length = len(self._prompt_stream) if limit < 1 and hasattr(self._prompt_stream, '__len__') else None
            for index, batch in enumerate(self._prompt_stream):
                if length is not None and index >= math.ceil(length * limit):
                    break
                if isinstance(batch, (list, tuple)):
                    yield from batch
                else:
                    yield batch

    def _draw(self, rows):
        row = next(rows, None)
        if row is None:
            return None
        ref = PromptRef(self._prompt_seq, row)
        self._prompt_seq += 1
        return ref

    def _occupancy(self):
        return len(self._handles) + len(self._preparing) + len(self._ready) + (
            len(self._waiting.records) if self._waiting is not None else 0)

    def _admit(self, prompt, group_key, regenerations=0):
        group = _Group(prompt, self._current_version(), regenerations,
                       resolved=self._regenerate is None and self._baseline_fn is None)
        self._groups[group_key] = group
        for index in range(self._num_generations):
            trajectory = self._request_builder(prompt, index)
            from twinkle.data_format.trajectory import request_identity
            request_identity(trajectory)
            policy = self._acquire_policy()
            entry = _Inflight(trajectory, None, policy, group_key, prompt)
            try:
                started = time.perf_counter()
                entry.future = self._submit(trajectory, policy)
                if not callable(getattr(entry.future, 'done', None)) or not callable(getattr(entry.future, 'result', None)):
                    raise TypeError('业务 submit 必须返回统一 Future 协议对象。')
                self._handles[entry.future] = entry
                if self._resource_pool is not None:
                    entry.lease = self._resource_pool.pin_association('submit', (trajectory,), {})
                self.stats.submit_time += time.perf_counter() - started
                self.stats.admitted += 1
            except BaseException:
                if entry.future not in self._handles:
                    self._release_pin(entry)
                raise

    def _prepare(self, entry, trajectory):
        self._release_pin(entry)
        self.stats.collected += 1
        if self._current_version() - entry.policy.version > self._max_staleness:
            self.stats.dropped_stale += 1
            return
        row = entry.prompt.row
        metadata = dict(prompt_id=str(entry.prompt.ordinal),
                        extra={key: value for key, value in row.items() if key != 'messages'},
                        prompt=row['messages'], group_key=entry.group_key, policy_version=entry.policy.version,
                        version_span=(self._current_version() - entry.policy.version
                                      if entry.policy.adapter_path is None else 0))
        payload = self._as_future(self._worker.prepare_sample(trajectory, **metadata))
        self._preparing.append((entry, payload, metadata))

    def _collect_completed(self):
        progressed = False
        for future, entry in tuple(self._handles.items()):
            if entry.control is not None or entry.paused is not None or not future.done():
                continue
            error = future.exception()
            if error is not None:
                raise error
            self._prepare(entry, project(future))
            self._handles.pop(future)
            progressed = True
        fresh = []
        for entry, payload, metadata in tuple(self._preparing):
            if not payload.done():
                continue
            error = payload.exception()
            if error is not None:
                raise error
            sample = self._sample_view(payload)
            sample.prompt_id = metadata['prompt_id']
            sample.prompt = copy.deepcopy(metadata['prompt'])
            sample.extra = copy.deepcopy(metadata['extra'])
            sample.group_key = metadata['group_key']
            sample.policy_version = metadata['policy_version']
            sample.version_span = metadata['version_span']
            record = BufferRecord(sample, entry.policy.version, entry.group_key)
            fresh.append(record)
            self._ready.append(record)
            self._preparing.remove((entry, payload, metadata))
            progressed = True
        if fresh:
            started = time.perf_counter()
            outcome = self._worker.submit([record.sample for record in fresh],
                                          transform=self._score, with_prm=self._prm_ready)
            outcome = self._as_future(outcome)
            for index, record in enumerate(fresh):
                record.score_ref = (outcome, index)
            self.stats.score_time += time.perf_counter() - started
        return progressed

    def _reap_scores(self):
        progressed = False
        for record in self._ready:
            if record.score_ref is None:
                continue
            outcome, index = record.score_ref
            if not outcome.done():
                continue
            error = outcome.exception()
            if error is not None:
                raise error
            sample = self._sample_view(project(outcome, 'samples', index))
            previous = record.sample
            for name in ('prompt_id', 'prompt', 'extra', 'group_key', 'policy_version', 'version_span'):
                setattr(sample, name, getattr(previous, name))
            sample.orm_reward = self._value(project(outcome, 'orm', index))
            record.sample = sample
            record.score_ref = None
            progressed = True
        return progressed

    def _resolve_groups(self):
        for key, group in tuple(self._groups.items()):
            if group.resolved:
                continue
            records = [record for record in self._ready if record.group_key == key]
            if len(records) != self._num_generations or any(row.score_ref is not None for row in records):
                continue
            if group.decision is None:
                if self._regenerate is None and self._baseline_fn is None:
                    group.resolved = True
                    continue
                group.decision = self._as_future(self._worker.decide_group(
                    key, [row.sample for row in records], regenerate=self._regenerate,
                    baseline_fn=self._baseline_fn, can_regenerate=group.regenerations < self._max_group_regenerations))
            if not group.decision.done():
                continue
            verdict = group.decision.result()
            if verdict['regenerate']:
                for record in records:
                    self._ready.remove(record)
                self._regenerations.append((group.prompt, key, group.regenerations + 1))
                self._groups.pop(key)
            else:
                for record in records:
                    record.sample.baseline = verdict['baseline']
                group.resolved = True

    def _acknowledge(self, batch_id, steps):
        delivery = self._waiting
        if delivery is None or delivery.batch_id != batch_id:
            raise ValueError('未知、重复或乱序的训练确认。')
        self._waiting = None
        self._acked_batch = batch_id
        self._budget_steps += steps
        self.stats.optimizer_steps += steps
        self.stats.train_pulls += int(steps > 0)
        self.stats.consumed += len(delivery.records)
        self.stats.off_policy_consumed += sum(self._current_version() > row.version for row in delivery.records)
        self.stats.train_active_time += time.monotonic() - delivery.delivered_at
        if self._resource_adapter is not None:
            self._resource_adapter.observe(step=self.stats.optimizer_steps, driver_stats=self.stats,
                                           score_time=self.stats.score_time)

    def _receive_commands(self):
        while True:
            try:
                operation, value = self._commands.get_nowait()
            except queue.Empty:
                return
            if operation == 'ack':
                self._acknowledge(*value)
            elif operation == 'publish':
                if self._publication is not None:
                    raise RuntimeError('一次只允许一个未完成发布。')
                self._publication = value
            else:
                raise RuntimeError(f'未知 driver 命令：{operation}')

    def _progress_publication(self):
        publication = self._publication
        if publication is None or self._acked_batch < publication.after_batch:
            return
        if publication.phase == 'requested':
            if not self._windows:
                raise RuntimeError('没有可关闭的发布窗口。')
            publication.window = self._windows[0]
            publication.started = time.perf_counter()
            self._begin_publish(publication.window)
            if self._resource_adapter is not None:
                decision = self._resource_adapter.decide(step=self.stats.optimizer_steps, publish_boundary=True)
                self._resource_adapter.register_decision(decision, step=self.stats.optimizer_steps)
            if publication.partial:
                for entry in self._handles.values():
                    entry.control = call(self._pause, entry.trajectory, executor=self._controls)
            publication.phase = 'pausing' if publication.partial else 'draining'
        if publication.phase == 'pausing':
            for future, entry in tuple(self._handles.items()):
                if entry.paused is not None:
                    continue
                if entry.control is None or not entry.control.done() or not future.done():
                    return
                error = entry.control.exception() or future.exception()
                if error is not None:
                    raise error
                trajectory = project(entry.control)
                if entry.state_check is None:
                    entry.state_check = call(self._worker.completion_state, trajectory, executor=self._controls)
                if not entry.state_check.done():
                    return
                status = entry.state_check.result()
                if status == 'paused':
                    entry.paused = trajectory
                elif status == 'completed':
                    self._prepare(entry, trajectory)
                    self._handles.pop(future)
                else:
                    raise RuntimeError(f'pause 返回了非法业务终态：{status!r}')
            publication.phase = 'push_ready'
        elif publication.phase == 'draining' and not self._handles:
            publication.phase = 'push_ready'
        if publication.phase == 'push_ready':
            publication.ready.set()
            publication.phase = 'confirming'
        if publication.phase != 'confirming' or not publication.confirmation.done():
            return
        path = publication.confirmation.result()
        self._finish_publish(publication.window, path)
        self.stats.publishes += 1
        self.stats.publish_time += time.perf_counter() - publication.started
        self._drop_stale()
        for future, entry in tuple(self._handles.items()):
            if entry.paused is None:
                continue
            resume = entry.paused
            replacement = self._resume(resume)
            if not callable(getattr(replacement, 'done', None)) or not callable(getattr(replacement, 'result', None)):
                raise TypeError('续接 submit 必须返回统一 Future。')
            entry.trajectory = resume
            entry.future = replacement
            entry.paused = None
            entry.control = None
            entry.state_check = None
            self._handles.pop(future)
            self._handles[replacement] = entry
        self._prune()
        self._publication = None
        publication.completion.set_result(path)

    def _drop_stale(self):
        for record in tuple(self._ready):
            if self._current_version() - record.version > self._max_staleness and record.score_ref is None:
                self._ready.remove(record)
                self.stats.dropped_stale += 1

    def _deliver(self):
        if self._waiting is not None or self._publication is not None or self._reached_max():
            return False
        eligible = [row for row in self._ready if row.score_ref is None and
                    self._groups[row.group_key].resolved and
                    self._current_version() - row.version <= self._max_staleness]
        records = self._assembly_ready(eligible) if eligible else None
        if not records:
            return False
        if len({id(row) for row in records}) != len(records) or any(row not in eligible for row in records):
            raise ValueError('assembly_ready 必须返回已评分缓冲区的唯一子集。')
        for record in records:
            self._ready.remove(record)
        batch = self._assemble(records) if self._assemble is not None else [row.sample for row in records]
        self._batch_seq += 1
        self._waiting = _Delivery(self._batch_seq, records, batch)
        self._batch_queue.put_nowait(self._waiting)
        return True

    def _cleanup_groups(self):
        live = {entry.group_key for entry in self._handles.values()}
        live.update(entry.group_key for entry, _, _ in self._preparing)
        live.update(row.group_key for row in self._ready)
        live.update(key for _, key, _ in self._regenerations)
        if self._waiting is not None:
            live.update(row.group_key for row in self._waiting.records)
        for key in tuple(self._groups):
            if key not in live:
                self._groups.pop(key)

    def _pump(self):
        started = time.perf_counter()
        try:
            self._register_initial_policy(self._value(self._initial_adapter_path()))
            rows = self._iter_rows()
            pending = self._draw(rows)
            while not self._stop_event.is_set():
                self._receive_commands()
                self._progress_publication()
                progressed = self._collect_completed()
                progressed = self._reap_scores() or progressed
                self._resolve_groups()
                self._drop_stale()
                progressed = self._deliver() or progressed
                if self._reached_max() and self._publication is None:
                    break
                if self._publication is None and (self._max_staleness > 0 or self._waiting is None):
                    capacity = self._buffer_depth - self._occupancy()
                    if capacity < self._num_generations:
                        self.stats.capacity_blocked_passes += 1
                    while capacity >= self._num_generations and (pending is not None or self._regenerations):
                        if not self._open_window():
                            self.stats.staleness_blocked_passes += 1
                            break
                        if self._regenerations:
                            prompt, key, attempts = self._regenerations.pop(0)
                            self._admit(prompt, key, attempts)
                        else:
                            key = f'{self._group_key(pending)}#{self._group_seq}'
                            self._group_seq += 1
                            self._admit(pending, key)
                            pending = self._draw(rows)
                        capacity -= self._num_generations
                        progressed = True
                self._cleanup_groups()
                self.stats.ready_high_watermark = max(self.stats.ready_high_watermark, len(self._ready))
                self.stats.inflight_high_watermark = max(self.stats.inflight_high_watermark, len(self._handles))
                pending_scores = any(row.score_ref is not None for row in self._ready)
                pending_decisions = any(group.decision is not None and not group.resolved for group in self._groups.values())
                if (pending is None and not self._regenerations and not self._handles and not self._preparing
                        and not pending_scores and not pending_decisions and self._waiting is None
                        and self._publication is None):
                    if not self._deliver():
                        break
                if not progressed:
                    before = time.perf_counter()
                    self._sleep(0.01)
                    self.stats.idle_time += time.perf_counter() - before
        except BaseException as exc:
            self._run_error = exc
        finally:
            try:
                self._drain()
            except BaseException as exc:
                if self._run_error is None:
                    self._run_error = exc
                else:
                    self._run_error.add_note(f'退出清理失败：{exc}')
            self.stats.final_version = self._current_version()
            self.stats.total_time = time.perf_counter() - started
            self._finished.set()
            try:
                self._batch_queue.put_nowait(_ITER_DONE)
            except queue.Full:
                pass

    @staticmethod
    def _wait_finished(future):
        """异常状态即可证明实际结束；不为排干取得 token 载荷。"""
        error = future.exception()
        if error is not None:
            raise error

    def _drain(self):
        errors = []
        controls = [(entry, call(self._stop, entry.trajectory, executor=self._controls))
                    for entry in self._handles.values()]
        for entry, control in controls:
            try:
                control.result()
                if entry.control is not None:
                    self._wait_finished(entry.control)
                if entry.state_check is not None:
                    self._wait_finished(entry.state_check)
                try:
                    self._wait_finished(entry.future)
                except CancelledError:
                    pass
                self._release_pin(entry)
            except BaseException as exc:
                errors.append(exc)
        for _, future, _ in self._preparing:
            try:
                self._wait_finished(future)
            except BaseException as exc:
                errors.append(exc)
        outcomes = {row.score_ref[0] for row in self._ready if row.score_ref is not None}
        outcomes.update(group.decision for group in self._groups.values() if group.decision is not None)
        for outcome in outcomes:
            try:
                self._wait_finished(outcome)
            except BaseException as exc:
                errors.append(exc)
        publication = self._publication or self._requested_publication
        if publication is not None and not publication.completion.done():
            with self._lock:
                pushing = publication.pushing
                publication.pushing = True
            if pushing:
                try:
                    publication.confirmation.result()
                except BaseException as exc:
                    errors.append(exc)
            failure = self._run_error or CancelledError('driver 在发布完成前关闭，未确认权重不得恢复准入。')
            publication.completion.set_exception(failure)
            publication.ready.set()
        self._controls.shutdown(wait=True)
        if self._owns_worker:
            self._worker.close()
        if errors:
            raise RuntimeError('driver 的真实执行清理未完成，保留 pin 与不可用状态。') from errors[0]
        self._handles.clear()
        self._preparing.clear()
        self.stats.dropped_incomplete += len(self._ready)
        self._ready.clear()
        self._groups.clear()
        for window in tuple(self._windows):
            self._clear_window(window)

    def _start(self):
        with self._lock:
            if self._iter_thread is not None:
                raise RuntimeError('StreamingDriver 实例只运行一次。')
            self._iter_thread = threading.Thread(target=self._pump, name='streaming-driver', daemon=False)
            self._iter_thread.start()

    def run(self):
        """阻塞消费也在调用方线程执行；推进线程继续提交逐样本评分。"""
        if self._consume is None:
            raise RuntimeError('consume=None 时使用 iterator，不直接调用 run。')
        self._start()
        try:
            while True:
                delivery = self._next_delivery()
                if delivery is None:
                    break
                steps = self._consume(delivery.records)
                self.commit(1 if steps is None else steps)
        finally:
            self.close()

    def __iter__(self):
        if self._consume is not None:
            raise RuntimeError('有 consume 的 driver 使用 run，不使用 iterator。')
        self._start()
        return self

    def _next_delivery(self):
        with self._lock:
            if self._consumer_batch is not None:
                raise RuntimeError('读取下一批前必须确认上一批的实际 optimizer steps。')
        while True:
            if self._run_error is not None:
                raise self._run_error
            try:
                delivery = self._batch_queue.get(timeout=0.05)
            except queue.Empty:
                if self._finished.is_set():
                    return None
                continue
            if delivery is _ITER_DONE:
                if self._run_error is not None:
                    raise self._run_error
                return None
            with self._lock:
                self._consumer_batch = delivery.batch_id
            return delivery

    def __next__(self):
        delivery = self._next_delivery()
        if delivery is None:
            raise StopIteration
        return delivery.batch

    @property
    def batch_id(self):
        with self._lock:
            return self._consumer_batch

    def commit(self, optimizer_steps=1, *, batch_id=None):
        """确认唯一已交付批次；0 表示没有执行 optimizer step。"""
        if type(optimizer_steps) is not int or optimizer_steps < 0:
            raise ValueError('optimizer_steps 必须为非负整数。')
        with self._lock:
            expected = self._consumer_batch
            if expected is None or batch_id is not None and batch_id != expected:
                raise ValueError('未知、重复或乱序的训练确认。')
            self._consumer_batch = None
            self._commands.put(('ack', (expected, optimizer_steps)))
            publication = self._requested_publication
        if publication is not None and publication.after_batch == expected:
            self.wait_publish()

    def request_publish(self, push=None, *, partial=None):
        """登记发布请求；push 由调用方在 commit/wait_publish 中执行。"""
        if self._run_id is None:
            raise ValueError('推理 driver 不发布 RL 权重。')
        if push is None:
            if self._snapshot_model is None:
                raise ValueError('需要 push callable 或 snapshot_model。')
            push = self._default_publish_push
        if not callable(push):
            raise TypeError('push 必须为 callable。')
        with self._lock:
            if self._finished.is_set():
                raise RuntimeError('已结束的 driver 不接受发布。')
            if self._requested_publication is not None and not self._requested_publication.completion.done():
                raise RuntimeError('已有未完成的发布。')
            after_batch = self._consumer_batch or self._acked_batch
            publication = _Publication(push, after_batch, self._allow_partial if partial is None else bool(partial))
            if publication.partial and (not callable(self._pause) or not callable(self._resume)):
                raise ValueError('该组件没有真实 pause/resume，不能请求显式续接。')
            self._requested_publication = publication
            self._commands.put(('publish', publication))
            return publication.completion

    def wait_publish(self, timeout=None):
        """等待 pause/drain 后在本调用线程 push，再把成功或失败交回推进线程。"""
        publication = self._requested_publication
        if publication is None:
            raise RuntimeError('没有已登记发布。')
        if self._consumer_batch is not None and self._consumer_batch == publication.after_batch:
            raise RuntimeError('必须先 commit，不能在消费回调内等待自身训练确认。')
        deadline = None if timeout is None else time.monotonic() + timeout
        while not publication.ready.wait(0.05):
            if self._run_error is not None:
                raise self._run_error
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('等待发布排干超时；生成未被视为已经停止。')
        if publication.completion.done():
            return publication.completion.result()
        with self._lock:
            if self._stop_event.is_set():
                raise CancelledError('driver 已关闭，不再开始权重 push。')
            execute = not publication.pushing
            publication.pushing = True
        if execute:
            try:
                path = self._value(publication.push(publication.window))
            except BaseException as exc:
                publication.confirmation.set_exception(exc)
            else:
                publication.confirmation.set_result(path)
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        return publication.completion.result(timeout=remaining)

    def close(self, timeout=None):
        """退出先 stop/drain，再 join；超时只报告未完成，不丢弃后台任务。"""
        self._stop_event.set()
        thread = self._iter_thread
        if thread is None:
            self._controls.shutdown(wait=True)
            if self._owns_worker:
                self._worker.close()
            return
        if thread is threading.current_thread():
            raise RuntimeError('推进线程不能等待自身关闭。')
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError('driver 的 stop/drain 尚未完成。')
        if self._run_error is not None:
            raise self._run_error
