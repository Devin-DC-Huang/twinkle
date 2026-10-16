# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import threading
import uuid
from peft import PeftConfig
from typing import Any, List, Optional, Type, Union

import twinkle
from twinkle import remote_function
from twinkle.data_format import (InputFeature, PoolingParams, PoolingResponse, SampleResponse, SamplingParams,
                                 Trajectory)
from twinkle.infra import get_worker_id
from twinkle.patch import Patch
from twinkle.sampler import generation
from twinkle.template import Template
from twinkle.utils import construct_class, get_logger

logger = get_logger()


@dataclass
class SamplerWorker:
    """一个完整推理引擎的放置、准入状态及请求负载。"""

    worker_id: str
    actor: Any
    group: str
    ranks: tuple[int, ...]
    shared: bool = False
    state: str = 'WARMING'
    inflight: int = 0
    dispatching: int = 0
    service_time: float = 0.0


class Sampler(ABC):

    def _save_worker_factory(self, cls, args, kwargs, torch_kwargs):
        self._worker_factory = (cls, tuple(args), dict(kwargs), torch_kwargs)
        self._worker_lock = threading.RLock()
        self._worker_configurations = []

    def _configuration_call_lock(self, method):
        if method in ('set_template', 'add_adapter_to_sampler', 'apply_patch'):
            return getattr(self, '_worker_lock', None)
        if hasattr(self, '_sampler_workers') and method in ('pause_generation', 'resume_generation'):
            return self._worker_lock
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, '_actors'):
            # 远程 payload 处理只需要原生 handler 的可序列化元数据；准入和在途状态仍归原 driver。
            for key in ('_worker_lock', '_continous_work_state', '_sampler_workers', '_generation_owners',
                        '_suspended_workers', '_paused_generation_workers', '_creating_slots', '_worker_factory'):
                state.pop(key, None)
            if hasattr(self, '_sampler_workers'):
                state['_admission_owned_elsewhere'] = True
        return state

    def configure_workers(self, ranks, *, group=None, shared=False, state='WARMING'):
        """在 driver 上登记已有 actors；不修改训练 mesh 或引擎内部并行规格。"""
        if not hasattr(self, '_actors') or getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Sampler worker lifecycle requires its original Ray driver handler.')
        if hasattr(self, '_sampler_workers'):
            raise RuntimeError('Sampler workers have already been configured.')
        ranks = tuple(ranks)
        if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('Sampler physical ranks must be a non-empty sequence of unique non-negative integers.')
        from twinkle.infra._ray import RayHelper
        width = RayHelper.resource_manager.get_config(group or self.remote_group).gpus_per_worker
        if not self._actors or len(ranks) != len(self._actors) * width:
            raise ValueError('Physical ranks must match gpus_per_worker for every complete sampler actor slot.')
        if state not in ('WARMING', 'ACTIVE', 'DRAINING', 'PAUSED'):
            raise ValueError(f'Unknown sampler worker state: {state!r}.')
        if not hasattr(self, '_worker_lock'):
            self._worker_lock = threading.RLock()
        self._sampler_workers = {}
        self._generation_owners = {}
        self._suspended_workers = set()
        self._paused_generation_workers = set()
        self._creating_slots = set()
        self._round_robin = 0
        self._admission_closed = False
        self.service_time = 0.0
        for index, actor in enumerate(self._actors):
            worker_id = get_worker_id(actor)
            self._sampler_workers[worker_id] = SamplerWorker(
                worker_id, actor, group or self.remote_group,
                ranks[index * width:(index + 1) * width], shared=shared, state=state)

    def worker_ids(self, *, shared=None, states=None):
        with self._worker_lock:
            return tuple(worker_id for worker_id, worker in self._sampler_workers.items()
                         if (shared is None or worker.shared == shared)
                         and (states is None or worker.state in states))

    def worker_ranks(self, worker_ids):
        with self._worker_lock:
            return tuple(self._sampler_workers[worker_id].ranks for worker_id in worker_ids)

    def _sampling_worker_ids(self):
        if getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Adaptive sampling must be admitted by its original driver, not a serialized handler.')
        if not hasattr(self, '_sampler_workers'):
            return None
        with self._worker_lock:
            if self._admission_closed:
                raise RuntimeError('Sampler admission is closed.')
            return tuple(worker_id for worker_id, worker in self._sampler_workers.items()
                         if worker.state == 'ACTIVE' and worker_id not in self._suspended_workers)

    def _begin_worker_call(self, worker_ids, worker_selector):
        if not hasattr(self, '_sampler_workers'):
            return None
        from twinkle.infra import _select_workers

        # 选择、屏障和 RPC 发送使用同一临界区；普通调用发出后在锁外收集结果。
        self._worker_lock.acquire()
        try:
            if worker_ids is None and worker_selector is not None:
                worker_ids = getattr(self, worker_selector)()
            actors = _select_workers(self, worker_ids)
            worker_ids = tuple(get_worker_id(actor) for actor in actors)
            for worker_id in worker_ids:
                self._sampler_workers[worker_id].dispatching += 1
            return worker_ids
        except BaseException:
            self._worker_lock.release()
            raise

    def _end_worker_dispatch(self):
        self._worker_lock.release()

    def _end_worker_call(self, worker_ids):
        with self._worker_lock:
            for worker_id in worker_ids:
                worker = self._sampler_workers.get(worker_id)
                if worker is not None:
                    worker.dispatching -= 1

    def _worker_call_conflicts(self, current_method, pending_method, worker_id):
        if not hasattr(self, '_sampler_workers') or pending_method != 'sample':
            return True
        controls = ('pause_generation', 'resume_generation', 'abort_all_inflight', '_get_generation_status',
                    '_cancel_generation', '_cancel_all_generations', '_generation_count')
        return (current_method not in controls
                and worker_id not in getattr(self, '_paused_generation_workers', ()))

    def _worker_call_completed(self, method, worker_ids):
        with self._worker_lock:
            if method == 'pause_generation':
                self._paused_generation_workers.update(worker_ids)
            elif method == 'resume_generation':
                self._paused_generation_workers.difference_update(worker_ids)

    def _settle_worker_controls(self, method, submissions):
        """分派或收集失败后，逐个确认已发出的暂停/恢复 RPC。"""
        if method not in ('pause_generation', 'resume_generation'):
            return
        import ray

        for worker_id, ref in submissions:
            try:
                ray.get(ref)
            except BaseException:
                logger.warning('sampler actor %s 的 %s 未成功完成。', worker_id, method, exc_info=True)
            else:
                self._worker_call_completed(method, (worker_id,))

    @staticmethod
    def _normalize_sampling_args(args, kwargs):
        args, kwargs = list(args), dict(kwargs)
        inputs = args[0] if args else kwargs['inputs']
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        elif not isinstance(inputs, list) and not getattr(inputs, '_is_lazy_collect', False):
            inputs = [inputs]
        if args:
            args[0] = inputs
        else:
            kwargs['inputs'] = inputs
        return tuple(args), kwargs

    @staticmethod
    def _dispatch_sampling_inputs(worker_count, worker_index, args, kwargs, **_):
        """仅切分顶层 inputs，保留每条输入内部的 token、tensor 和多模态字段。"""
        args, kwargs = list(args), dict(kwargs)
        inputs = args[0] if args else kwargs['inputs']
        batch = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        size, remainder = divmod(len(batch), worker_count)
        start = worker_index * size + min(worker_index, remainder)
        stop = start + size + (worker_index < remainder)
        shard = batch[start:stop]
        for index, value in enumerate(args):
            if index > 0 and isinstance(value, list) and len(value) == len(batch):
                args[index] = value[start:stop]
        for key, value in kwargs.items():
            if key != 'inputs' and isinstance(value, list) and len(value) == len(batch):
                kwargs[key] = value[start:stop]
        if args:
            args[0] = shard
        else:
            kwargs['inputs'] = shard
        return tuple(args), kwargs

    def _record_worker_configuration(self, method, args, kwargs):
        if method not in ('set_template', 'add_adapter_to_sampler', 'apply_patch'):
            return
        with self._worker_lock:
            self._worker_configurations.append((method, tuple(args), dict(kwargs)))

    def add_workers(self, group, ranks, *, worker_slots=None, shared=False, engine_args=None):
        """从已预留的完整 slot 创建 actors；完成配置后仍不接收新请求。"""
        from twinkle.infra._ray import RayHelper
        from twinkle.utils import DeviceMesh

        if not hasattr(self, '_sampler_workers'):
            raise RuntimeError('Call configure_workers before adding sampler workers.')
        config = RayHelper.resource_manager.get_config(group)
        placements = RayHelper.resource_manager.get_group(group)
        width = getattr(config, 'gpus_per_worker', 1)
        ranks = tuple(ranks)
        if any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('Sampler physical ranks must be unique non-negative integers.')
        if len(ranks) != len(placements) * width:
            raise ValueError('Physical ranks must match the reserved sampler worker slots.')
        slots = tuple(range(len(placements))) if worker_slots is None else tuple(worker_slots)
        if any(type(slot) is not int or not 0 <= slot < len(placements) for slot in slots) or len(set(slots)) != len(slots):
            raise ValueError(f'Invalid sampler worker_slots: {slots!r}.')
        if not slots:
            return ()
        selected_ranks = tuple(ranks[slot * width:(slot + 1) * width] for slot in slots)
        reservations = {(group, slot_ranks) for slot_ranks in selected_ranks}
        with self._worker_lock:
            occupied = {(worker.group, worker.ranks) for worker in self._sampler_workers.values()}
            if self._admission_closed or reservations & (occupied | self._creating_slots):
                raise RuntimeError('Sampler is closing or a selected worker slot is occupied.')
            self._creating_slots.update(reservations)
        actors = []
        worker_ids = ()
        try:
            cls, args, kwargs, torch_kwargs = self._worker_factory
            kwargs = dict(kwargs, remote_group=group, device_mesh=DeviceMesh.from_sizes(dp_size=len(slots)))
            if engine_args is not None:
                kwargs['engine_args'] = dict(kwargs.get('engine_args', {}), **engine_args)
            actors = RayHelper.create_workers(
                cls, group, 'all', *args, instance_id=uuid.uuid4().hex,
                extra_torch_kwargs=torch_kwargs, max_concurrency=self._max_concurrency,
                worker_slots=list(slots), **kwargs)
            worker_ids = tuple(get_worker_id(actor) for actor in actors)
            with self._worker_lock:
                if self._admission_closed:
                    raise RuntimeError('Sampler closed while new workers were being constructed.')
                self._actors = tuple(self._actors) + tuple(actors)
                for actor, worker_id, physical_ranks in zip(actors, worker_ids, selected_ranks):
                    self._sampler_workers[worker_id] = SamplerWorker(
                        worker_id, actor, group, physical_ranks, shared=shared)
                self._worker_ready(_worker_ids=worker_ids)
                # 全量配置调用与新 actor 的安装/重放互斥，避免漏放或重复执行 patch。
                for method, args, kwargs in self._worker_configurations:
                    result = getattr(self, method)(*args, _worker_ids=worker_ids, **kwargs)
                    if callable(result):
                        result()
                return worker_ids
        except BaseException:
            try:
                self._discard_workers(actors)
            except Exception:
                logger.warning('构造失败后未能完整关闭 sampler actors。', exc_info=True)
            raise
        finally:
            with self._worker_lock:
                self._creating_slots.difference_update(reservations)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def _worker_ready(self):
        return True

    def activate_workers(self, worker_ids):
        with self._worker_lock:
            workers = [self._sampler_workers[worker_id] for worker_id in worker_ids]
            if any(worker.state == 'DRAINING' for worker in workers):
                raise RuntimeError('A draining sampler worker cannot accept new requests.')
            for worker in workers:
                worker.state = 'ACTIVE'

    def drain_workers(self, worker_ids):
        with self._worker_lock:
            for worker_id in worker_ids:
                self._sampler_workers[worker_id].state = 'DRAINING'

    def pause_workers(self, worker_ids):
        with self._worker_lock:
            for worker_id in worker_ids:
                worker = self._sampler_workers[worker_id]
                if worker.state in ('ACTIVE', 'WARMING'):
                    worker.state = 'PAUSED'

    def suspend_admission(self, worker_ids):
        with self._worker_lock:
            self._suspended_workers.update(worker_ids)

    def resume_admission(self, worker_ids):
        with self._worker_lock:
            self._suspended_workers.difference_update(worker_ids)

    def close_admission(self):
        with self._worker_lock:
            self._admission_closed = True

    def worker_inflight(self, worker_ids):
        self._reap_generation_owners()
        with self._worker_lock:
            return tuple(self._sampler_workers[worker_id].inflight for worker_id in worker_ids)

    @property
    def inflight_count(self):
        return sum(self.worker_inflight(self.worker_ids()))

    def workers_idle(self, worker_ids):
        """只检查控制状态，不通过读取生成输出判断 actor 是否可以退役。"""
        import ray
        worker_ids = tuple(worker_ids)
        self._reap_generation_owners()
        with self._worker_lock:
            if any(self._sampler_workers[worker_id].inflight or self._sampler_workers[worker_id].dispatching
                   for worker_id in worker_ids):
                return False
        state = getattr(self, '_continous_work_state', None)
        if state is not None:
            with state['lock']:
                refs = [ref for ref, (owner, _) in state['owners'].items() if owner in worker_ids]
            if refs:
                ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
                if len(ready) != len(refs):
                    return False
                from twinkle.infra import _cw_unregister
                _cw_unregister(self, '', refs)
        counts = self._generation_count(_worker_ids=worker_ids)
        counts = [counts] if isinstance(counts, int) else counts
        return all(count == 0 for count in counts)

    def remove_workers(self, worker_ids):
        worker_ids = tuple(worker_ids)
        with self._worker_lock:
            workers = [self._sampler_workers[worker_id] for worker_id in worker_ids]
            if any(worker.state == 'ACTIVE' for worker in workers):
                raise RuntimeError('Drain or pause sampler workers before removing them.')
            if not self.workers_idle(worker_ids):
                return False
            self._discard_workers(tuple(worker.actor for worker in workers))
            return True

    def _discard_workers(self, actors):
        """终止引擎与 actors；即使某个关闭失败，也继续清理其余成员。"""
        import ray
        from twinkle.infra import _cw_drop_locked

        actors = tuple(actors)
        worker_ids = {get_worker_id(actor) for actor in actors}
        error = None
        shutdown_refs = []
        for actor in actors:
            try:
                shutdown_refs.append(actor.shutdown.remote())
            except BaseException as exc:
                error = error or exc
        try:
            for ref in shutdown_refs:
                try:
                    ray.get(ref)
                except BaseException as exc:
                    error = error or exc
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except BaseException as exc:
                    error = error or exc
            lock = getattr(self, '_worker_lock', None)
            with (lock if lock is not None else nullcontext()):
                self._actors = tuple(actor for actor in self._actors if get_worker_id(actor) not in worker_ids)
                if hasattr(self, '_sampler_workers'):
                    for worker_id in worker_ids:
                        self._sampler_workers.pop(worker_id, None)
                    self._suspended_workers.difference_update(worker_ids)
                    self._paused_generation_workers.difference_update(worker_ids)
                    self._generation_owners = {
                        key: owner for key, owner in self._generation_owners.items() if owner.worker_id not in worker_ids
                    }
                state = getattr(self, '_continous_work_state', None)
                if state is not None:
                    with state['lock']:
                        refs = [ref for ref, (owner, _) in state['owners'].items() if owner in worker_ids]
                        _cw_drop_locked(state, refs)
                        for worker_id in worker_ids:
                            state['load'].pop(worker_id, None)
        if error is not None:
            raise error

    def close_workers(self):
        """仅在终止整个 sampler 时使用；关闭准入、取消请求并清理所有引擎。"""
        if getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Sampler lifecycle belongs to its original driver handler.')
        if not hasattr(self, '_actors'):
            return self.shutdown()
        if hasattr(self, '_sampler_workers'):
            self.close_admission()
        try:
            cancel = getattr(self, 'cancel_all_generations', None)
            if self._actors and cancel is not None:
                cancel()
        finally:
            self._discard_workers(tuple(self._actors))

    def __init__(self):
        self.engine = None
        self.template = None

    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
        *,
        num_samples: int = 1,
    ) -> List[SampleResponse]:
        """Sample responses for given inputs.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'/'videos'.
                - Trajectory: Must contain 'messages'. Requires template to be set.
            sampling_params: Sampling parameters.
            adapter_name: Optional LoRA adapter name.
            num_samples: Number of completions to generate per input prompt.
                        When > 1, returns num_samples sequences for each input.

        Returns:
            SampleResponse containing sampled sequences.
            Total sequences = len(inputs) * num_samples.
        """
        pass

    def encode(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        pooling_params: Optional[PoolingParams] = None,
        adapter_name: str = '',
    ) -> List[PoolingResponse]:
        """Run a pooling (non-generative) forward and return the pooled output per input.

        The pooling counterpart of :meth:`sample`: one forward per input, no decoding, producing an
        embedding vector / class scores / a relevance score. Only samplers whose backend can serve a
        pooling head (vLLM, SGLang) override this; the default raises so a backend without pooling
        support fails loudly rather than silently returning hidden states that look like embeddings.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s), as for :meth:`sample`.
            pooling_params: Which head to run and how to post-process it. Defaults to a plain embedding.
            adapter_name: Optional LoRA adapter name.

        Returns:
            One PoolingResponse per input, in input order.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support pooling/encode; use a vLLM or '
                                  'SGLang sampler, or a HF forward via build_model(task=...).')

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        ...

    @staticmethod
    def _not_encoded(inputs: Any) -> bool:
        """Check if inputs are not yet encoded (i.e., is Trajectory, not InputFeature).

        Shared with ``TransformersModel`` via :func:`twinkle.sampler.generation.not_encoded` so every
        caller agrees on where the encode boundary is.
        """
        return generation.not_encoded(inputs)

    def _is_trajectory(self, inputs: Any) -> bool:
        """Check if inputs are Trajectory type (not encoded)."""
        if isinstance(inputs, list):
            if not inputs:
                return False
            inputs = inputs[0]
        if isinstance(inputs, dict):
            return self._not_encoded(inputs)
        return False

    def _normalize_inputs(self, inputs) -> List:
        if isinstance(inputs, dict):
            return [inputs]
        return list(inputs)

    def encode_trajectory(self,
                          trajectory: Trajectory,
                          adapter_name: str = '',
                          add_generation_prompt: bool = True) -> InputFeature:
        if self.template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return generation.encode_trajectory(self.template, trajectory, add_generation_prompt)

    def decode_response(self, token_ids: List[int], adapter_name: str = '') -> str:
        """Decode token ids to text."""
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return template.decode(token_ids)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        if 'model_id' not in kwargs and hasattr(self, 'model_id'):
            kwargs['model_id'] = self.model_id
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        self.template = template

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig) -> None:
        raise NotImplementedError
