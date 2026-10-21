# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import copy
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from twinkle import DeviceGroup, Platform, find_free_port, find_node_ip, requires
from twinkle.utils.loader import run_worker_setup_hook
from .. import _torch_setup
from .resource_manager import ResourceManager

T = TypeVar('T')
_UNSET = object()


class LazyCollect:
    """可序列化的远程 Future；结果投影在数据所在进程完成。"""

    def __init__(self, futures, method, collect_func, device_mesh=None, timeout=None, *, path=(), root=None):
        self._futures = tuple(futures)
        self._method = method
        self._collect_func = collect_func
        self.device_mesh = device_mesh
        self._timeout = timeout
        self._path = tuple(path)
        self._root = root
        self._is_lazy_collect = True
        self._lock = threading.RLock()
        self._result = _UNSET
        self._error = _UNSET
        self._projection = None
        self._status_ref = None
        self._projections = {}

    def __getstate__(self):
        # 句柄只传计算描述，不把已收集载荷、锁或 driver 状态带给消费者。
        with self._lock:
            self.as_ref()
            return {key: value for key, value in self.__dict__.items()
                    if key not in ('_lock', '_result', '_error', '_projections')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._result = _UNSET
        self._error = _UNSET
        self._projections = {}

    def done(self):
        import ray
        with self._lock:
            if self._result is not _UNSET or self._error is not _UNSET:
                return True
            refs = (self.as_ref(),)
        if not refs:
            return True
        ready, _ = ray.wait(list(refs), num_returns=len(refs), timeout=0, fetch_local=False)
        return len(ready) == len(refs)

    ready = done

    def result(self, timeout=None):
        import ray
        limit = self._timeout if timeout is None else timeout
        if limit is not None and limit < 0:
            raise ValueError('timeout 必须为非负秒数或 None。')
        deadline = None if limit is None else time.monotonic() + limit
        acquired = self._lock.acquire() if limit is None else self._lock.acquire(timeout=limit)
        if not acquired:
            raise TimeoutError('等待远程结果收集锁超时。')
        try:
            if self._error is not _UNSET:
                raise self._error
            if self._result is not _UNSET:
                return self._result
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            try:
                value = ray.get(self.as_ref(), timeout=remaining)
                self._result = value
            except (ray.exceptions.GetTimeoutError, TimeoutError):
                # 等待超时不缓存为任务失败；下次仍能取得真正结果。
                raise TimeoutError('等待远程调用结果超时。') from None
            except Exception as exc:
                self._error = exc
                raise
            return self._result
        finally:
            self._lock.release()

    def exception(self, timeout=None):
        import ray
        limit = self._timeout if timeout is None else timeout
        if limit is not None and limit < 0:
            raise ValueError('timeout 必须为非负秒数或 None。')
        with self._lock:
            if self._error is not _UNSET:
                return self._error
            if self._result is not _UNSET:
                return None
            if self._status_ref is None:
                self._status_ref = RayHelper.status_ref(self.as_ref())
            ref = self._status_ref
        try:
            return ray.get(ref, timeout=limit)
        except ray.exceptions.GetTimeoutError:
            raise TimeoutError('等待远程异常结果超时。') from None

    def cancel(self):
        # RPC 已发送，无法据此证明业务执行尚未开始。
        return False

    def __call__(self):
        return self.result()

    def __await__(self):
        async def collect():
            return await asyncio.to_thread(self.result)
        return collect().__await__()

    def select(self, *path):
        if any(not isinstance(key, (str, int, slice)) or isinstance(key, bool) for key in path):
            raise TypeError('结果投影路径只接受字符串、整数或 slice。')
        full_path = self._path + tuple(path)
        # slice 不可哈希，路径缓存使用规范化索引。
        key = tuple((item.start, item.stop, item.step) if isinstance(item, slice) else item for item in full_path)
        with self._lock:
            if key not in self._projections:
                self._projections[key] = LazyCollect(
                    self._futures, self._method, self._collect_func, self.device_mesh,
                    self._timeout, path=full_path, root=self._root if self._root is not None else self)
            return self._projections[key]

    def __getitem__(self, key):
        return self.select(key)

    def as_ref(self):
        with self._lock:
            if self._projection is None:
                if self._root is None:
                    self._projection = RayHelper.collect_ref(
                        self._futures, self._method, self._collect_func, self.device_mesh)
                else:
                    self._projection = RayHelper.project_ref(self._root.as_ref(), self._path)
            return self._projection

    def __iter__(self):
        return iter(self.result())

    def __len__(self):
        return len(self.result())


class RayHelper:

    resource_manager: Optional[ResourceManager] = None

    _registry = None

    _remote_components: Dict[str, Any] = {}

    @staticmethod
    def _get_ray_custom_resources(device_groups: List[DeviceGroup]) -> Dict[str, float]:
        """Return custom accelerator resources needed for local Ray startup.

        Ray discovers CUDA GPUs itself.  Other accelerators, such as Ascend
        NPUs, need to be registered as Ray custom resources so that placement
        groups can request them.
        """
        device_types = {group.device_type.upper() for group in device_groups} - {'CPU'}

        # ResourceManager supports one accelerator type per run.  Only NPU
        # currently needs an explicit Ray custom-resource registration.
        if device_types != {'NPU'}:
            return {}

        try:
            import torch

            npu = getattr(torch, 'npu', None)
            npu_count = npu.device_count() if npu is not None and npu.is_available() else 0
        except (ImportError, AttributeError, RuntimeError):
            return {}

        return {'NPU': float(npu_count)} if npu_count > 0 else {}

    @staticmethod
    def init_registry():
        if RayHelper._registry is not None:
            return

        import ray

        @ray.remote
        class WorkerRegistry:
            """A config center to store global configs"""

            def __init__(self):
                self.config = {}

            def add_config(self, key: str, value: Any):
                self.config[key] = value

            def add_or_get(self, key: str, value: Any) -> Tuple[bool, Any]:
                """Add or get config, because ray is single threaded."""
                if key in self.config:
                    return self.config[key]
                self.config[key] = value
                return value

            def get_config(self, key: str):
                return self.config.get(key)

            def clear(self):
                self.config.clear()

        try:
            RayHelper._registry = ray.get_actor('config_registry')
        except ValueError:
            try:
                RayHelper._registry = WorkerRegistry.options(name='config_registry', lifetime='detached').remote()
            except ValueError:
                RayHelper._registry = ray.get_actor('config_registry')
        assert RayHelper._registry is not None

    @staticmethod
    def initialize(nproc_per_node: int, ncpu_proc_per_node: int, device_groups: List[DeviceGroup]):
        """Initialize RayHelper.

        Args:
            nproc_per_node: How many processes in one node.
            ncpu_proc_per_node: How many cpu processes in one node.
            device_groups: The device groups to initialize.

        Returns:
            None
        """
        requires('ray')
        import ray
        RayHelper.device_groups = device_groups
        if not RayHelper.ray_inited():
            resources = RayHelper._get_ray_custom_resources(device_groups)
            ray.init(ignore_reinit_error=True, resources=resources or None)

        if RayHelper.resource_manager is None:
            # Resource manager initializes only once in the pipeline process.
            RayHelper.resource_manager = ResourceManager(nproc_per_node, ncpu_proc_per_node, device_groups)
        RayHelper.init_registry()

    @staticmethod
    def teardown():
        """Teardown RayHelper."""
        if RayHelper.resource_manager is not None:
            RayHelper.resource_manager.destroy_placement_group()
            RayHelper.resource_manager = None

        if RayHelper._registry is not None:
            import ray
            try:
                ray.get(RayHelper._registry.clear.remote())
                ray.kill(RayHelper._registry)
            except:  # noqa
                pass
            RayHelper._registry = None

    @staticmethod
    def ray_inited():
        """Check if Ray is initialized."""
        try:
            import ray
        except ImportError:
            # not installed, not inited
            return False
        return ray.is_initialized()

    @staticmethod
    def is_worker():
        """Check if this process is the worker"""
        import ray
        return RayHelper.ray_inited() and ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE

    @staticmethod
    def execute_all_sync(method_name: str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]], *,
                         on_submit: Optional[Callable] = None, timeout=None):
        """Execute method and return results.

        timeout 为整个结果列表的等待上限，与 LazyCollect 相同；超时不终止业务执行。
        """
        import ray
        return ray.get(RayHelper.execute_all_async(method_name, workers_and_args, on_submit=on_submit), timeout=timeout)

    @staticmethod
    def execute_all_async(method_name: str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]], *,
                          on_submit: Optional[Callable] = None):
        """Execute method and return futures."""
        output = []
        for worker_and_args in workers_and_args:
            worker, args, kwargs = worker_and_args
            remote_call = getattr(worker, method_name)
            result = remote_call.remote(*args, **kwargs)
            output.append(result)
            # 逐个登记，后续 RPC 发送失败也不能丢失已发出的 refs。
            if on_submit is not None:
                on_submit(worker, result)
        return output

    @staticmethod
    def add_or_get_config(key: str, value: Any):
        import ray
        return ray.get(RayHelper._registry.add_or_get.remote(key, value))

    @staticmethod
    def add_config(key: str, value: Any):
        import ray
        ray.get(RayHelper._registry.add_config.remote(key, value))

    @staticmethod
    def get_config(key: str):
        import ray
        return ray.get(RayHelper._registry.get_config.remote(key))

    @staticmethod
    def _get_remote_component(component):
        """Avoid create remote component twice."""
        if component not in RayHelper._remote_components:
            import ray
            RayHelper._remote_components[component] = ray.remote(
                concurrency_groups={'ordinary': 1, 'control': 2, 'exclusive': 1})(component)
        return RayHelper._remote_components[component]

    @staticmethod
    def get_master_id_port(placement_group):
        import ray

        @ray.remote
        def get_node_address():
            return find_node_ip(), find_free_port()

        ip, port = ray.get(get_node_address.options(placement_group=placement_group, num_cpus=0.01).remote())
        return ip, port

    @staticmethod
    def do_get_and_collect_func(collect_func: Callable,
                                method: Union[str, Callable],
                                futures,
                                device_mesh,
                                timeout=None):
        """使用唯一的通用结果句柄。"""
        return LazyCollect(futures, method, collect_func, device_mesh, timeout=timeout)

    @staticmethod
    def do_get_and_collect(args, kwargs):
        args, kwargs = RayHelper.resolve((tuple(args), dict(kwargs)))
        return args, kwargs

    @staticmethod
    def resolve(value, *, _active=None, _memo=None):
        try:
            import ray
        except ImportError:
            ray = None
        active = set() if _active is None else _active
        memo = {} if _memo is None else _memo
        reference = isinstance(value, (LazyCollect, Future)) or ray is not None and isinstance(value, ray.ObjectRef)
        record = is_dataclass(value) and not isinstance(value, type)
        if not isinstance(value, (list, tuple, dict)) and not record and not reference:
            return value
        if len(active) >= 256:
            raise ValueError('远程调用参数的容器嵌套超过 256 层。')
        identity = id(value)
        if identity in active:
            raise ValueError('远程调用参数不支持自引用容器。')
        if identity in memo:
            return memo[identity]
        active.add(identity)
        try:
            if reference:
                payload = value.result() if isinstance(value, (LazyCollect, Future)) else ray.get(value)
                result = RayHelper.resolve(payload, _active=active, _memo=memo)
            elif record:
                result = copy.copy(value)
                for item in fields(value):
                    object.__setattr__(result, item.name,
                                       RayHelper.resolve(getattr(value, item.name), _active=active, _memo=memo))
            elif isinstance(value, dict):
                result = {key: RayHelper.resolve(item, _active=active, _memo=memo) for key, item in value.items()}
            else:
                items = [RayHelper.resolve(item, _active=active, _memo=memo) for item in value]
                if isinstance(value, tuple) and isinstance(getattr(type(value), '_fields', None), tuple):
                    result = type(value)(*items)
                else:
                    result = tuple(items) if isinstance(value, tuple) else items
            memo[identity] = result
            return result
        finally:
            active.remove(identity)

    @staticmethod
    def object_refs(value, _active=None):
        try:
            import ray
        except ImportError:
            ray = None
        active = set() if _active is None else _active
        if ray is not None and isinstance(value, ray.ObjectRef):
            return [value]
        if isinstance(value, LazyCollect):
            return [value.as_ref()]
        record = is_dataclass(value) and not isinstance(value, type)
        if not isinstance(value, (list, tuple, dict)) and not record:
            return []
        if len(active) >= 256:
            raise ValueError('远程调用参数的容器嵌套超过 256 层。')
        if id(value) in active:
            raise ValueError('远程调用参数不支持自引用容器。')
        active.add(id(value))
        try:
            items = ([getattr(value, item.name) for item in fields(value)] if record else
                     value.values() if isinstance(value, dict) else value)
            return [ref for item in items for ref in RayHelper.object_refs(item, active)]
        finally:
            active.remove(id(value))

    @staticmethod
    def has_ref(args, kwargs):
        return bool(RayHelper.object_refs((args, kwargs)))

    @staticmethod
    def _collect(recipe):
        futures, method, collector, device_mesh = recipe
        parts = RayHelper.resolve(list(futures))
        return collector(method, parts, device_mesh=device_mesh)

    @staticmethod
    def collect_ref(futures, method, collector, device_mesh):
        import ray
        return ray.remote(RayHelper._collect).options(num_cpus=0).remote(
            (futures, method, collector, device_mesh))

    @staticmethod
    def _project(refs, path):
        result = RayHelper.resolve(refs)[0]
        for key in path:
            result = result[key] if isinstance(result, (dict, list, tuple)) else getattr(result, key)
        return result

    @staticmethod
    def project_ref(ref, path):
        import ray
        return ray.remote(RayHelper._project).options(num_cpus=0).remote((ref,), path)

    @staticmethod
    def _status(refs):
        try:
            RayHelper.resolve(refs)
        except Exception as exc:
            return exc
        return None

    @staticmethod
    def status_ref(ref):
        import ray
        return ray.remote(RayHelper._status).options(num_cpus=0).remote((ref,))

    @staticmethod
    def create_workers(worker_cls: Type[T],
                       group: str,
                       execute: Literal['all', 'peer', 'first'],
                       *args,
                       instance_id,
                       extra_torch_kwargs: Optional[Dict[str, Any]] = None,
                       max_concurrency: Optional[int] = None,
                       worker_slots: Optional[List[int]] = None,
                       **kwargs) -> List[T]:
        # TODO when will remote create remote?
        # Should it peer create peer? or peer create all?
        # Whether the input data of each remote is independent, or they are a part of the whole device mesh?
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        workers = []
        device_config = RayHelper.resource_manager.get_config(group)
        placement_groups = RayHelper.resource_manager.get_group(group)
        worker_cls = RayHelper._get_remote_component(worker_cls)
        ranks = device_config.ranks
        if isinstance(ranks, int):
            ranks = list(range(ranks))
        assert len(placement_groups) == len(ranks)
        if worker_slots is not None:
            slots = tuple(worker_slots)
            if any(type(slot) is not int or not 0 <= slot < len(ranks) for slot in slots) or len(set(slots)) != len(slots):
                raise ValueError(f'worker_slots must be unique indices in [0, {len(ranks)}), got {slots!r}.')
            if not slots:
                return []
            placement_groups = [placement_groups[slot] for slot in slots]
            ranks = [ranks[slot] for slot in slots]
        key = f'{group}-{worker_cls.__class__.__name__}-{instance_id}'
        if execute == 'peer':
            # Create the peer worker
            # 0 1 2 3
            # | | | |
            # 0 1 2 3
            _slice = Platform.get_peer_index(len(ranks))
            placement_groups = placement_groups[_slice]
            ranks = ranks[_slice]
            ip, port = RayHelper.get_master_id_port(placement_groups[0]['placement_group'])
            ip = RayHelper.add_or_get_config(key + '-ip', ip)
            port = RayHelper.add_or_get_config(key + '-port', port)
        elif execute == 'first':
            placement_groups = placement_groups[:1]
            ranks = ranks[:1]
            ip, port = RayHelper.get_master_id_port(placement_groups[0]['placement_group'])
        else:
            ip, port = RayHelper.get_master_id_port(placement_groups[0]['placement_group'])

        device_type_upper = (device_config.device_type or '').upper()
        if device_type_upper != 'CPU':
            world_size = len(ranks)
            device_type = Platform.get_platform(device_type_upper).__name__
            for pg_idx, (deploy_pg, gpu) in enumerate(zip(placement_groups, ranks)):
                deploy_pg: Dict
                cluster_name = group
                worker_name = key + '-' + str(pg_idx)
                env_vars = os.environ.copy()
                env_vars.update({
                    'WORLD_SIZE':
                    str(world_size),
                    'RANK':
                    str(pg_idx),
                    'LOCAL_RANK':
                    str(0),
                    'CLUSTER_NAME':
                    cluster_name,
                    'WORKER_NAME':
                    worker_name,
                    Platform.get_platform(device_type_upper).visible_device_env():
                    ','.join([str(r) for r in deploy_pg['gpu_rank']]),
                    'TWINKLE_MODE':
                    'ray',
                    _torch_setup.EXTRA_TORCH_KWARGS_ENV:
                    _torch_setup.to_env_value(extra_torch_kwargs),
                })

                env_vars['MASTER_ADDR'] = ip
                env_vars['MASTER_PORT'] = str(port)

                # Prevent Ray from overriding CUDA_VISIBLE_DEVICES set in runtime_env
                # This is critical for multi-GPU workers (gpus_per_worker > 1)
                env_vars.update(ResourceManager.noset_env())

                if max_concurrency is not None:
                    # Read back in the worker to size the thread pool that runs
                    # blocking continuous-work methods off the actor's event loop.
                    env_vars['TWINKLE_ACTOR_MAX_CONCURRENCY'] = str(max_concurrency)

                # Ray serialises runtime_env to JSON; a callable setup hook is rejected by Ray 2.58
                # ("Object of type function is not JSON serializable"). Pass the import path instead.
                runtime_env = RuntimeEnv(
                    env_vars=env_vars,
                    worker_process_setup_hook='twinkle.utils.loader.run_worker_setup_hook')

                worker_options = {
                    'scheduling_strategy':
                    PlacementGroupSchedulingStrategy(placement_group=deploy_pg['placement_group']),
                    'name': worker_name,
                    'namespace': 'default',
                    'runtime_env': runtime_env,
                    'num_cpus': 0.01,
                    'max_restarts': 0,
                    'max_task_retries': 0,
                }

                # The placement group already reserves a GPU worker's full
                # allocation and ``CUDA_VISIBLE_DEVICES`` above pins that actor
                # to the ranks selected by DeviceGroup.  Do not additionally
                # request a fractional Ray GPU resource for CUDA workers.
                #
                # With a fractional request, Ray translates the placement-group
                # GPU id through the already narrowed visible-device list.  For
                # the second GPU that becomes index 1 into ``['1']``, killing
                # the worker in ``set_visible_accelerator_ids`` before its
                # constructor runs.  The CPU claim and placement-group strategy
                # are sufficient to place the actor in the GPU-owning bundle.
                if device_type != 'GPU':
                    # Use a custom resource key for non-GPU accelerators
                    # (for example, NPU).
                    worker_options['resources'] = {device_type: 0.01}

                if max_concurrency is not None:
                    worker_options['max_concurrency'] = max_concurrency
                worker_options['concurrency_groups'] = {
                    'ordinary': max_concurrency or 1, 'control': 2, 'exclusive': 1}

                try:
                    worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                except BaseException:
                    for created in workers:
                        ray.kill(created, no_restart=True)
                    raise
                workers.append(worker)
        else:
            world_size = len(ranks)
            workers = []
            # For CPU case, don't set visible device environment variables
            _visible_device_env = {}
            for rank, (deploy_pg, index) in enumerate(zip(placement_groups, list(range(world_size)))):
                deploy_pg: Dict
                cluster_name = group
                worker_name = key + '-' + str(rank)
                env_vars = os.environ.copy()
                env_vars.update({
                    'WORLD_SIZE': str(world_size),
                    'RANK': str(rank),
                    'LOCAL_RANK': str(0),
                    'CLUSTER_NAME': cluster_name,
                    'WORKER_NAME': worker_name,
                    'TWINKLE_MODE': 'ray',
                    _torch_setup.EXTRA_TORCH_KWARGS_ENV: _torch_setup.to_env_value(extra_torch_kwargs),
                    **_visible_device_env
                })
                if max_concurrency is not None:
                    # Read back in the worker to size the thread pool that runs
                    # blocking continuous-work methods off the actor's event loop.
                    env_vars['TWINKLE_ACTOR_MAX_CONCURRENCY'] = str(max_concurrency)

                # Ray serialises runtime_env to JSON; a callable setup hook is rejected by Ray 2.58
                # ("Object of type function is not JSON serializable"). Pass the import path instead.
                runtime_env = RuntimeEnv(
                    env_vars=env_vars,
                    worker_process_setup_hook='twinkle.utils.loader.run_worker_setup_hook')

                worker_options = {
                    'scheduling_strategy':
                    PlacementGroupSchedulingStrategy(placement_group=deploy_pg['placement_group']),
                    'name': worker_name,
                    'namespace': 'default',
                    'runtime_env': runtime_env,
                    'num_cpus': 0.01,
                    'max_restarts': 0,
                    'max_task_retries': 0,
                }

                if max_concurrency is not None:
                    worker_options['max_concurrency'] = max_concurrency
                worker_options['concurrency_groups'] = {
                    'ordinary': max_concurrency or 1, 'control': 2, 'exclusive': 1}

                try:
                    worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                except BaseException:
                    for created in workers:
                        ray.kill(created, no_restart=True)
                    raise
                workers.append(worker)
        return workers
