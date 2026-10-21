# Copyright (c) ModelScope Contributors. All rights reserved.
"""通用调用预留、关联路由与 worker 生命周期；不读取组件领域声明。"""
import copy
import functools
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .continuous_work import _WORKER_ASYNC_PREFIX, _WORKER_RUNTIME_METHOD, get_worker_id


@dataclass
class WorkerSlot:
    worker_id: str
    actor: Any
    group: str
    ranks: Tuple[int, ...]
    shared: bool = False
    state: str = 'WARMING'


@dataclass(frozen=True)
class WorkerSlotView:
    worker_id: str
    state: str
    shared: bool
    inflight: int
    dispatching: int
    load: int


class WorkerPoolView:
    def __init__(self, slots, suspended=(), bindings=None, active_sessions=()):
        self._slots = tuple(slots)
        self._suspended = frozenset(suspended)
        self._by_id = {slot.worker_id: slot for slot in slots}
        self.bindings = dict(bindings or {})
        self.active_sessions = frozenset(active_sessions)

    @property
    def active_ids(self):
        return tuple(slot.worker_id for slot in self._slots if slot.state == 'ACTIVE')

    @property
    def admissible_ids(self):
        return tuple(key for key in self.active_ids if key not in self._suspended)

    def slot(self, worker_id):
        return self._by_id.get(worker_id)

    def __iter__(self):
        return iter(self._slots)

    def __len__(self):
        return len(self._slots)


@dataclass(frozen=True)
class CallContext:
    mode: Optional[str] = None
    key: Any = None
    session: Any = None
    fallback: Any = None


class Router:
    def prepare_context(self, method, args, kwargs):
        return CallContext(*self.association(method, args, kwargs))

    def select_context(self, method, args, kwargs, view, context):
        return self.select(method, args, kwargs, view)

    def select(self, method, args, kwargs, view):
        return None

    def association(self, method, args, kwargs):
        return None, None


class LeastInflightRouter(Router):
    def __init__(self):
        self._rotation = 0

    def select(self, method, args, kwargs, view):
        ids = view.admissible_ids
        if not ids:
            return None
        offset = self._rotation % len(ids)
        rotated = ids[offset:] + ids[:offset]
        winner = min(rotated, key=lambda key: view.slot(key).load)
        self._rotation = ids.index(winner) + 1
        return (winner,)


class StickyRouter(LeastInflightRouter):
    """解析方法级通信声明；活跃调用归属与软会话亲和分别处理。"""

    def __init__(self, *, key: Optional[Callable] = None, affinity: Optional[Callable] = None, bindings=None,
                 capacity=4096, ttl=300.0, fallback_key: Optional[Callable] = None, methods=None):
        super().__init__()
        if type(capacity) is not int or capacity < 1 or ttl <= 0:
            raise ValueError('StickyRouter capacity/ttl 必须为正数。')
        if any(mode not in ('acquire', 'use') for mode in (bindings or {}).values()):
            raise ValueError('关联规则只支持 acquire/use。')
        self.key = key
        self.affinity = affinity
        self.bindings = dict(bindings or {})
        self.methods = dict(methods or {})
        self.fallback_key = fallback_key
        self.capacity = capacity
        self.ttl = float(ttl)
        self._sessions = OrderedDict()
        self._leases = {}
        self._lock = threading.RLock()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != '_lock'}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def association(self, method, args, kwargs):
        mode = self.bindings.get(method)
        key = self.key(method, args, kwargs) if mode is not None else None
        if mode is not None and key is None:
            raise ValueError(f'{method} 缺少关联键。')
        if key is not None:
            hash(key)
        return mode, key

    @staticmethod
    def _read_path(arguments, path, default=None):
        """仅解引用选定字段；缺省字段不需要拉取完整参数。"""
        from . import project
        from ._ray.ray_helper import RayHelper
        try:
            import ray
        except ImportError:
            ray = None
        parts = path.split('.')
        keys = [int(name) if name.isdecimal() else name for name in parts[1:]]
        try:
            value = arguments[parts[0]]
            for index, key in enumerate(keys):
                if ray is not None and isinstance(value, ray.ObjectRef):
                    value = RayHelper.project_ref(value, keys[index:])
                    break
                value = project(value, key)
            return RayHelper.resolve(value)
        except (KeyError, AttributeError, IndexError):
            return default

    def prepare_context(self, method, args, kwargs):
        declaration = self.methods.get(method)
        if declaration is None:
            mode, key = self.association(method, args, kwargs)
            session = self.affinity(method, args, kwargs) if self.affinity is not None else None
            fallback = self.fallback_key(method, args, kwargs) if mode == 'use' and self.fallback_key else None
            return CallContext(mode, key, session, fallback)
        bound = declaration['signature'].bind(None, *args, **kwargs)
        bound.apply_defaults()
        read = functools.partial(self._read_path, bound.arguments)
        def resolve(selector):
            return selector(read) if callable(selector) else read(selector) if selector is not None else None
        mode = declaration['mode']
        key = resolve(declaration['key'])
        if key is None:
            raise ValueError(f'{method} 缺少关联键。')
        session = key if declaration['affinity'] is None else resolve(declaration['affinity'])
        fallback = resolve(declaration['fallback']) if mode == 'use' else None
        for value in (key, session, fallback):
            if value is not None:
                hash(value)
        return CallContext(mode, key, session, fallback)

    def select(self, method, args, kwargs, view):
        return self.select_context(method, args, kwargs, view, self.prepare_context(method, args, kwargs))

    def select_context(self, method, args, kwargs, view, context):
        now = time.monotonic()
        mode, key = context.mode, context.key
        if mode == 'use':
            ids = view.bindings.get(key)
            if ids is None and context.fallback is not None:
                ids = view.bindings.get(context.fallback)
            if ids is None:
                raise KeyError(f'未知或过期的调用关联：{key!r}')
            if any(view.slot(worker_id) is None for worker_id in ids):
                raise RuntimeError('关联 worker 已退役；不能把控制请求发送到其他实例。')
            return ids
        if mode == 'acquire' and key in view.bindings:
            raise ValueError(f'调用关联已存在：{key!r}')
        session = context.session
        if session is None:
            return super().select(method, args, kwargs, view)
        hash(session)
        pinned = self._leases.get(session)
        previous = self._sessions.get(session)
        if pinned is not None:
            if any(view.slot(worker_id) is None for worker_id in pinned):
                raise RuntimeError('硬租约所属 worker 已失效。')
            ids = pinned
        elif previous is not None and (session in view.active_sessions or now - previous[1] <= self.ttl):
            ids = previous[0]
        else:
            ids = None
        if ids and all(worker_id in view.admissible_ids for worker_id in ids):
            self._sessions[session] = (ids, now)
            self._sessions.move_to_end(session)
            return ids
        if pinned is not None:
            raise RuntimeError('硬租约 worker 当前不可准入。')
        if session in view.active_sessions:
            raise RuntimeError('会话仍有实际在途调用，暂不能迁移。')
        ids = super().select(method, args, kwargs, view)
        if ids is not None:
            self._sessions[session] = (ids, now)
            self._sessions.move_to_end(session)
            for expired, (_, stamp) in tuple(self._sessions.items()):
                if expired not in self._leases and expired not in view.active_sessions:
                    if now - stamp > self.ttl or len(self._sessions) > self.capacity:
                        self._sessions.pop(expired)
        return ids

    def acquire_lease(self, key, worker_ids):
        ids = tuple(worker_ids)
        if not ids:
            raise ValueError('硬租约必须关联存活 worker。')
        with self._lock:
            if key in self._leases and self._leases[key] != ids:
                raise RuntimeError('不能更改尚未释放的路由硬租约。')
            self._leases[key] = ids

    def release_lease(self, key):
        with self._lock:
            self._leases.pop(key, None)


@dataclass
class _Call:
    call_id: str
    method: str
    worker_ids: Tuple[str, ...]
    group: str
    key: Any = None
    charge: int = 1
    refs: tuple = ()
    thread_id: int = field(default_factory=threading.get_ident)
    finished: Optional[float] = None
    session: Any = None
    started: float = field(default_factory=time.monotonic)


class WorkerPool:
    """一份调用记录承担选择、负载、完成和排干；等待不占用选择锁。"""

    @classmethod
    def create(cls, owner):
        """全部进程连接同一个池；序列化代理时不复制调用记录。"""
        import ray
        from ._ray import RayHelper
        declarations = {name: getattr(type(owner), name)._rpc for name in dir(type(owner))
                        if getattr(getattr(type(owner), name), '_rpc', None) is not None}
        sticky = {name: options for name, options in declarations.items() if options['sticky']}
        execution = {name: 'control' if options['control'] else 'exclusive'
                     for name, options in declarations.items()
                     if options['control'] or getattr(getattr(type(owner), name), '_exclusive', False)
                     or getattr(getattr(type(owner), name), '_sync', False)}
        shutdown = [name for name, options in declarations.items() if options['shutdown']]
        if len(shutdown) > 1:
            raise ValueError('一个组件只能声明一个 shutdown 方法。')
        snapshot = SimpleNamespace(
            _actors=tuple(owner._actors), remote_group=owner.remote_group,
            _max_concurrency=owner._max_concurrency, _worker_factory=owner._worker_factory,
            device_mesh=getattr(owner, 'device_mesh', None))
        # 完成通知不能与正在等待互斥锁的调用共用执行容量。
        coordinator = ray.remote(num_cpus=0, max_concurrency=8, max_restarts=0, max_task_retries=0,
                                 concurrency_groups={'ordinary': 4, 'control': 2, 'exclusive': 1,
                                                     'completion': 1})(cls).remote(
            snapshot, resources=RayHelper.resource_manager, registry=RayHelper._registry,
            router=StickyRouter(methods=sticky) if sticky else None, execution=execution,
            replay=tuple(name for name, options in declarations.items() if options['replay']),
            shutdown=shutdown[0] if shutdown else None)
        proxy = cls.__new__(cls)
        proxy.__dict__['_coordinator'] = coordinator
        ray.get(coordinator.invoke.options(concurrency_group='control').remote('initialize_slots', (), {}))
        return proxy

    def __getattribute__(self, name):
        if not name.startswith('_') and name not in ('create', 'invoke'):
            coordinator = object.__getattribute__(self, '__dict__').get('_coordinator')
            if coordinator is not None:
                if name in ('configured', 'inflight_count', 'service_time', 'execution', 'replay'):
                    return self._invoke('_read', (name,), {})
                if name == 'lock':
                    raise RuntimeError('跨进程调用池不暴露局部锁；选择与发送须由池原子执行。')
                if callable(getattr(type(self), name, None)):
                    return functools.partial(self._invoke, name)
        return object.__getattribute__(self, name)

    def _invoke(self, operation, *args, **kwargs):
        import ray
        if operation == '_read':
            args, kwargs = args
        lane = 'exclusive' if operation in ('add_workers', 'wait_idle', 'close_workers', 'remove_workers') else 'control'
        if operation == 'dispatch_call':
            method = args[0]
            lane = ray.get(self._coordinator.invoke.options(concurrency_group='control').remote(
                'execution_group', (method,), {'sync': kwargs.get('sync', False),
                                              'exclusive': kwargs.get('exclusive', False)}))
        return ray.get(self._coordinator.invoke.options(concurrency_group=lane).remote(operation, args, kwargs))

    def invoke(self, operation, args, kwargs):
        if operation == '_read':
            return getattr(self, args[0])
        return getattr(self, operation)(*args, **kwargs)

    def __getstate__(self):
        if '_coordinator' not in self.__dict__:
            raise TypeError('WorkerPool 的实际调用记录不能复制到其他进程。')
        return {'_coordinator': self._coordinator}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(self, owner, *, router=None, execution=None, replay=(), shutdown=None, ttl=300.0,
                 resources=None, registry=None):
        from ._ray import RayHelper
        if resources is not None:
            RayHelper.resource_manager = resources
            RayHelper._registry = registry
        self._owner = owner
        self._lock = threading.RLock()
        self._configuration_lock = threading.Lock()
        self.router = router
        self._default_router = LeastInflightRouter()
        self.execution = dict(execution or {})
        self.replay = frozenset(replay)
        self.shutdown = shutdown
        self.ttl = ttl
        self.workers = {}
        self._calls = {}
        self._bindings = OrderedDict()
        self.suspended_workers = set()
        self.creating_slots = set()
        self.admission_closed = False
        self.configurations = []
        self._leases = {}
        self._lease_counts = {}
        self._service_time = 0.0

    def initialize_slots(self):
        from ._ray import RayHelper
        placements = RayHelper.resource_manager.get_group(self._owner.remote_group)
        config = RayHelper.resource_manager.get_config(self._owner.remote_group)
        self.install_slots(self._owner._actors, [item['physical_ranks']
                           for item in placements[:len(self._owner._actors)]],
                           self._owner.remote_group, shared=config.shared)
        self._runtime('ready', self.worker_ids())
        self.activate_workers(self.worker_ids())

    def execution_group(self, method, *, sync=False, exclusive=False):
        if sync or exclusive or method in self.replay:
            return 'exclusive'
        return self.execution.get(method, 'ordinary')

    def dispatch_call(self, method, args, kwargs, **options):
        group = self.execution_group(method, sync=options.get('sync', False),
                                     exclusive=options.get('exclusive', False))
        if group == 'control':
            # control 用于停止或查询在途工作，不能排在互斥调用之后。
            return self._dispatch_call(method, args, kwargs, **options)
        exclusive = group == 'exclusive'
        # 普通调用仅在选择和发送期间持锁；互斥调用把锁交给完成通知，覆盖真实执行期。
        self._configuration_lock.acquire()
        transferred = False
        try:
            handle = self._dispatch_call(method, args, kwargs, **options)
            transferred = exclusive
            return handle
        finally:
            if not transferred:
                self._configuration_lock.release()

    def _dispatch_call(self, method, args, kwargs, *, dispatch='all', execute='all', collect='none',
                       sync=False, exclusive=False, timeout=None, worker_ids=None, independent=False,
                       batch_arg=None, aligned_args=(), signature=None):
        from ._ray.ray_helper import LazyCollect, RayHelper
        from .dispatch import _dispatch_args, _collect_func, _get_workers
        from .continuous_work import _cw_batch_len, _run_continous_work
        group = self.execution_group(method, sync=sync, exclusive=exclusive)
        if group == 'exclusive' and independent:
            raise ValueError('互斥调用不能使用独立请求路由。')
        if independent and not RayHelper.has_ref(args, kwargs):
            size = _cw_batch_len(args, kwargs)
            if batch_arg is not None:
                value = signature.bind(None, *args, **kwargs).arguments[batch_arg]
                size = len(value) if isinstance(value, list) else None
            if size is not None:
                return _run_continous_work(None, method, args, kwargs, size, timeout, self,
                                          batch_arg=batch_arg, aligned_args=aligned_args, signature=signature)
        context = self.prepare_context(method, args, kwargs)
        def send_parts(actors):
            if RayHelper.has_ref(args, kwargs):
                parts = [(actor, args, dict(kwargs, _dispatch_context=(len(actors), index, dispatch, execute)))
                         for index, actor in enumerate(actors)]
            else:
                parts = _dispatch_args(actors, dispatch, execute, self._owner.device_mesh, args, kwargs)
            refs = []
            record = self.reserve(method, tuple(get_worker_id(actor) for actor in actors),
                                  args, kwargs, group=group, context=context)
            try:
                for actor, part_args, part_kwargs in parts:
                    refs.extend(self.send(method, (get_worker_id(actor),), part_args, part_kwargs,
                                          group=group, record=record))
            finally:
                self.finish_reservations()
            return LazyCollect(refs, collect, _collect_func, self._owner.device_mesh,
                               None if group == 'exclusive' else timeout)

        with self.lock:
            ids = self.choose(method, args, kwargs, independent=independent, worker_ids=worker_ids,
                              group=group, context=context)
            actors = _get_workers([self.workers[key].actor for key in ids], execute)
            ids = tuple(get_worker_id(actor) for actor in actors)
            if group != 'exclusive':
                # 普通调用的选择、预留与发送在同一临界区完成。
                return send_parts(actors)
            prior = set(ids) & self.suspended_workers
            self.suspended_workers.update(ids)
        # 关闭范围后排干；任何失败均保留关闭状态。
        self.wait_idle(ids, timeout=timeout)
        with self.lock:
            handle = send_parts(actors)
        import ray
        coordinator = ray.get_runtime_context().current_actor
        completion = coordinator.invoke.options(concurrency_group='completion').remote(
            'complete_exclusive', (handle, method, args, kwargs, ids, tuple(prior)), {})
        return LazyCollect((completion,), 'first', _collect_func, self._owner.device_mesh, timeout)

    def complete_exclusive(self, handle, method, args, kwargs, ids, prior):
        """独占锁延续到实际执行结束；调用方等待超时不会重新开放范围。"""
        try:
            error = handle.exception()
            if error is not None:
                raise error
            self.record_configuration(method, args, kwargs)
            with self.lock:
                reopen = tuple(key for key in ids if key not in prior and self.workers[key].state == 'ACTIVE')
            self.resume_admission(reopen)
            return handle
        finally:
            self._configuration_lock.release()

    def prepare_context(self, method, args, kwargs):
        """身份提取可等待小结果投影，但不得占用调用池的选择锁。"""
        router = self.router
        return router.prepare_context(method, args, kwargs) if router is not None else CallContext()

    def pin_association(self, method, args, kwargs):
        """按集中路由规则将已发送关联的会话固定到同一 worker。"""
        context = self.prepare_context(method, args, kwargs)
        with self.lock:
            self.reap()
            if not isinstance(self.router, StickyRouter):
                raise ValueError('关联硬租约需要配置 StickyRouter。')
            binding = self._bindings.get(context.key)
            session = context.session
            if binding is None or session is None:
                raise KeyError('硬租约需要已有调用关联及会话键。')
            self.acquire_lease(session, binding.worker_ids)
            return session

    def acquire_lease(self, key, worker_ids):
        ids = tuple(worker_ids)
        with self.lock:
            if any(worker_id not in self.workers for worker_id in ids) or not ids:
                raise ValueError('硬租约必须关联存活 worker。')
            if key in self._leases and self._leases[key] != ids:
                raise RuntimeError('硬租约未释放，不能迁移。')
            self._leases[key] = ids
            self._lease_counts[key] = self._lease_counts.get(key, 0) + 1
            if isinstance(self.router, StickyRouter):
                self.router.acquire_lease(key, ids)

    def release_lease(self, key):
        with self.lock:
            count = self._lease_counts.get(key, 0)
            if count < 1:
                raise KeyError(f'未知或已释放的硬租约：{key!r}')
            if count > 1:
                self._lease_counts[key] = count - 1
                return
            self._lease_counts.pop(key)
            self._leases.pop(key)
            if isinstance(self.router, StickyRouter):
                self.router.release_lease(key)

    @property
    def lock(self):
        return self._lock

    @property
    def configured(self):
        return bool(self.workers)

    def configure(self, *, router=None, execution=None, replay=None, shutdown=None):
        with self._configuration_lock, self.lock:
            self.reap()
            if self._calls or self._leases:
                raise RuntimeError('修改调用配置前必须排干并释放硬租约。')
            if router is not None:
                self.router = router
            if execution is not None:
                if any(group not in ('ordinary', 'control', 'exclusive') for group in execution.values()):
                    raise ValueError('execution 只支持 ordinary/control/exclusive。')
                self.execution = dict(execution)
            if replay is not None:
                self.replay = frozenset(replay)
            if shutdown is not None:
                self.shutdown = shutdown
        return True

    def reap(self):
        import ray
        with self.lock:
            refs = tuple(dict.fromkeys(ref for call in self._calls.values() for ref in call.refs))
            ready = set(ray.wait(list(refs), num_returns=len(refs), timeout=0, fetch_local=False)[0]) if refs else set()
            now = time.monotonic()
            for call_id, call in tuple(self._calls.items()):
                if call.refs and all(ref in ready for ref in call.refs):
                    call.finished = now
                    if call.group == 'ordinary':
                        self._service_time += now - call.started
                    self._calls.pop(call_id)
            for key, call in tuple(self._bindings.items()):
                if call.finished is not None and call.session not in self._leases and now - call.finished > self.ttl:
                    self._bindings.pop(key)
            if len(self._bindings) > 4096:
                for key, call in tuple(self._bindings.items()):
                    if len(self._bindings) <= 4096:
                        break
                    if call.finished is not None and call.session not in self._leases:
                        self._bindings.pop(key)

    def view(self):
        with self.lock:
            self.reap()
            slots = []
            for key, slot in self.workers.items():
                calls = [call for call in self._calls.values() if key in call.worker_ids]
                load = sum(call.charge for call in calls if call.group == 'ordinary')
                slots.append(WorkerSlotView(key, slot.state, slot.shared, len(calls),
                                            sum(not call.refs for call in calls), load))
            return WorkerPoolView(slots, self.suspended_workers,
                                  {key: call.worker_ids for key, call in self._bindings.items()},
                                  {call.session for call in self._calls.values() if call.session is not None})

    def choose(self, method, args, kwargs, *, independent=False, worker_ids=None, group=None, context=None):
        context = self.prepare_context(method, args, kwargs) if context is None else context
        with self.lock:
            view = self.view()
            group = group or self.execution_group(method)
            if group == 'ordinary' and self.admission_closed:
                raise RuntimeError('WorkerPool 准入已关闭。')
            mode = context.mode
            if worker_ids is not None:
                ids = tuple(worker_ids)
            elif self.router is not None and (independent or mode is not None):
                ids = self.router.select_context(method, args, kwargs, view, context)
            elif independent:
                ids = self._default_router.select(method, args, kwargs, view)
            else:
                ids = tuple(self.workers) if group != 'ordinary' else view.admissible_ids
            if not ids:
                raise RuntimeError('没有可准入的 worker。')
            for key in ids:
                slot = view.slot(key)
                if slot is None:
                    raise ValueError(f'未知 worker：{key!r}')
                if slot.state == 'CLOSING' or group == 'ordinary' and key not in view.admissible_ids:
                    raise RuntimeError(f'worker {key} 当前不可准入。')
                limit = getattr(self._owner, '_max_concurrency', None)
                if independent and group == 'ordinary' and limit and slot.load >= limit:
                    raise RuntimeError(f'worker {key} 已满载；应通过调用方 backpressure 稍后重试。')
            return tuple(ids)

    def reserve(self, method, worker_ids, args, kwargs, *, charge=1, group=None, context=None):
        context = self.prepare_context(method, args, kwargs) if context is None else context
        with self.lock:
            group = group or self.execution.get(method, 'exclusive' if method in self.replay else 'ordinary')
            mode, key = context.mode, context.key
            if mode == 'acquire' and key in self._bindings:
                raise ValueError(f'重复调用关联：{key!r}')
            call = _Call(uuid.uuid4().hex, method, tuple(worker_ids), group, key if mode == 'acquire' else None, charge)
            call.session = context.session
            self._calls[call.call_id] = call
            if mode == 'acquire':
                self._bindings[key] = call
            return call

    def send(self, method, worker_ids, args, kwargs, *, charge=1, group=None, record=None):
        context = self.prepare_context(method, args, kwargs) if record is None else None
        pending = list(record) if isinstance(record, (list, tuple)) else [record] if record is not None else []
        with self.lock:
            group = group or self.execution_group(method)
            if any(call.call_id not in self._calls or not set(worker_ids).issubset(call.worker_ids)
                   for call in pending):
                raise ValueError('发送目标必须属于本次调用的有效预留。')
            if not pending:
                pending = [self.reserve(method, worker_ids, args, kwargs, charge=charge, group=group,
                                        context=context)]
            refs = []
            try:
                for key in worker_ids:
                    actor = self.workers[key].actor
                    remote = getattr(actor, _WORKER_ASYNC_PREFIX + method).options(concurrency_group=group)
                    refs.append(remote.remote(*args, **dict(kwargs, _infra_group=group)))
            finally:
                # 部分发送失败的已发调用仍在记录中，后续必须真实排干。
                for call in pending:
                    call.refs += tuple(refs)
            return refs

    def finish_reservations(self):
        with self.lock:
            for key, call in tuple(self._calls.items()):
                if call.thread_id == threading.get_ident() and not call.refs:
                    self._calls.pop(key)
                    if call.key is not None and self._bindings.get(call.key) is call:
                        self._bindings.pop(call.key)

    def record_configuration(self, method, args, kwargs):
        if method in self.replay:
            snapshot = (method, copy.deepcopy(args), copy.deepcopy(kwargs))
            with self.lock:
                self.configurations.append(snapshot)

    def worker_ids(self, *, shared=None, states=None):
        with self.lock:
            return tuple(key for key, slot in self.workers.items()
                         if (shared is None or slot.shared == shared) and (states is None or slot.state in states))

    def worker_ranks(self, worker_ids):
        with self.lock:
            return tuple(self.workers[key].ranks for key in worker_ids)

    def install_slots(self, actors, ranks_per_actor, group, *, shared=False, state='WARMING'):
        if state not in ('WARMING', 'ACTIVE', 'DRAINING', 'PAUSED'):
            raise ValueError(f'未知 worker 状态：{state!r}')
        actors, ranks_per_actor = tuple(actors), tuple(ranks_per_actor)
        if len(actors) != len(ranks_per_actor):
            raise ValueError('actors 与 ranks_per_actor 必须等长。')
        with self.lock:
            for actor, ranks in zip(actors, ranks_per_actor):
                key = get_worker_id(actor)
                self.workers[key] = WorkerSlot(key, actor, group, tuple(ranks), shared, state)

    def configure_workers(self, ranks, *, group=None, shared=False, state='WARMING'):
        from ._ray import RayHelper
        ranks = tuple(ranks)
        actors = tuple(self._owner._actors)
        if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('物理 ranks 必须是唯一的非负整数。')
        group = group or self._owner.remote_group
        width = RayHelper.resource_manager.get_config(group).gpus_per_worker
        if len(ranks) != len(actors) * width:
            raise ValueError('物理 ranks 必须覆盖每个完整 actor。')
        with self.lock:
            if self._calls:
                raise RuntimeError('重新登记物理放置前必须排干调用。')
            self.install_slots(actors, [ranks[i * width:(i + 1) * width] for i in range(len(actors))],
                               group, shared=shared, state=state)

    def _runtime(self, operation, worker_ids, *, timeout=None):
        import ray
        refs = [getattr(self.workers[key].actor, _WORKER_RUNTIME_METHOD)
                .options(concurrency_group='control').remote(operation) for key in worker_ids]
        return ray.get(refs, timeout=timeout)

    def activate_workers(self, worker_ids):
        ids = tuple(worker_ids)
        with self.lock:
            if self.admission_closed or any(self.workers[key].state not in ('WARMING', 'PAUSED') for key in ids):
                raise RuntimeError('只有初始化完成的 WARMING/PAUSED worker 可以激活。')
        self._runtime('ready', ids)
        with self.lock:
            for key in ids:
                self.workers[key].state = 'ACTIVE'
                self.suspended_workers.discard(key)

    def drain_workers(self, worker_ids):
        ids = tuple(worker_ids)
        with self.lock:
            for key in ids:
                self.workers[key].state = 'DRAINING'

    def pause_workers(self, worker_ids):
        self.suspend_admission(worker_ids)
        with self.lock:
            for key in worker_ids:
                self.workers[key].state = 'PAUSED'

    def suspend_admission(self, worker_ids):
        ids = tuple(worker_ids)
        with self.lock:
            self.suspended_workers.update(ids)

    def resume_admission(self, worker_ids):
        ids = tuple(worker_ids)
        with self.lock:
            if self.admission_closed or any(self.workers[key].state != 'ACTIVE' for key in ids):
                raise RuntimeError('关闭或非 ACTIVE 的 worker 不能重新开放准入。')
            self.suspended_workers.difference_update(ids)

    def close_admission(self):
        with self.lock:
            self.admission_closed = True
        self.suspend_admission(self.worker_ids())

    def worker_inflight(self, worker_ids):
        view = self.view()
        return tuple(view.slot(key).inflight for key in worker_ids)

    @property
    def inflight_count(self):
        with self.lock:
            self.reap()
            return len(self._calls)

    @property
    def service_time(self):
        """普通调用累计在途秒数；每份调用记录只在真实结束时累计一次。"""
        with self.lock:
            self.reap()
            return self._service_time

    def workers_idle(self, worker_ids):
        ids = tuple(worker_ids)
        if any(self.worker_inflight(ids)):
            return False
        if any(set(ids).intersection(owner) for owner in self._leases.values()):
            return False
        return all(self._runtime('idle', ids))

    def wait_idle(self, worker_ids, *, timeout=None):
        import ray
        deadline = None if timeout is None else time.monotonic() + timeout
        ids = tuple(worker_ids)
        while True:
            with self.lock:
                self.reap()
                calls = [call for call in self._calls.values() if set(ids).intersection(call.worker_ids)]
                pending = list(dict.fromkeys(ref for call in calls for ref in call.refs))
            if not calls:
                break
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            if remaining == 0:
                raise TimeoutError('排干 worker 超时；准入保持关闭。')
            if pending:
                ray.wait(pending, num_returns=len(pending), timeout=remaining, fetch_local=False)
            else:
                time.sleep(0.01)
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        self._runtime('drain', ids, timeout=remaining)

    def add_workers(self, group, ranks, *, worker_slots=None, shared=False, engine_args=None):
        from ._ray import RayHelper
        from twinkle.utils import DeviceMesh
        placements = RayHelper.resource_manager.get_group(group)
        width = RayHelper.resource_manager.get_config(group).gpus_per_worker
        ranks = tuple(ranks)
        slots = tuple(range(len(placements))) if worker_slots is None else tuple(worker_slots)
        if (len(ranks) != len(placements) * width or any(type(rank) is not int or rank < 0 for rank in ranks)
                or len(set(ranks)) != len(ranks)):
            raise ValueError('物理 ranks 必须覆盖所有预留放置且唯一。')
        if any(type(slot) is not int or not 0 <= slot < len(placements) for slot in slots) or len(set(slots)) != len(slots):
            raise ValueError('worker_slots 必须是唯一的有效放置索引。')
        if not slots:
            return ()
        selected = tuple(ranks[slot * width:(slot + 1) * width] for slot in slots)
        reservations = {(group, item) for item in selected}
        with self.lock:
            occupied = {(slot.group, slot.ranks) for slot in self.workers.values()}
            if self.admission_closed or reservations.intersection(occupied | self.creating_slots):
                raise RuntimeError('放置已占用或 WorkerPool 正在关闭。')
            self.creating_slots.update(reservations)
        actors = ()
        try:
            cls, args, kwargs, torch_kwargs = self._owner._worker_factory
            kwargs = dict(kwargs, remote_group=group, device_mesh=DeviceMesh.from_sizes(dp_size=len(slots)))
            if engine_args is not None:
                kwargs['engine_args'] = dict(kwargs.get('engine_args', {}), **engine_args)
            actors = RayHelper.create_workers(cls, group, 'all', *args, instance_id=uuid.uuid4().hex,
                                             extra_torch_kwargs=torch_kwargs,
                                             max_concurrency=self._owner._max_concurrency,
                                             worker_slots=list(slots), **kwargs)
            ids = tuple(get_worker_id(actor) for actor in actors)
            with self._configuration_lock:
                with self.lock:
                    if self.admission_closed:
                        raise RuntimeError('worker 创建期间 WorkerPool 已关闭。')
                    self.install_slots(actors, selected, group, shared=shared)
                    self._owner._actors = tuple(self._owner._actors) + tuple(actors)
                for method, replay_args, replay_kwargs in self.configurations:
                    from ._ray.ray_helper import LazyCollect
                    from .dispatch import _collect_func
                    refs = self.send(method, ids, replay_args, replay_kwargs, group='exclusive')
                    LazyCollect(refs, 'none', _collect_func).result()
                return ids
        except BaseException:
            if actors:
                import ray
                for actor in actors:
                    ray.kill(actor, no_restart=True)
                with self.lock:
                    for actor in actors:
                        self.workers.pop(get_worker_id(actor), None)
                    failed = {get_worker_id(actor) for actor in actors}
                    self._owner._actors = tuple(actor for actor in self._owner._actors if get_worker_id(actor) not in failed)
            raise
        finally:
            with self.lock:
                self.creating_slots.difference_update(reservations)
                self.finish_reservations()

    def remove_workers(self, worker_ids, *, timeout=None):
        import ray
        ids = tuple(worker_ids)
        with self._configuration_lock:
            with self.lock:
                if any(self.workers[key].state == 'ACTIVE' for key in ids):
                    raise RuntimeError('移除前必须先排干或暂停 worker。')
            if not self.workers_idle(ids):
                return False
            with self.lock:
                self.reap()
                if (any(set(ids).intersection(call.worker_ids) for call in self._calls.values())
                        or any(set(ids).intersection(owner) for owner in self._leases.values())):
                    return False
                for key in ids:
                    self.workers[key].state = 'CLOSING'
            try:
                if self.shutdown is not None:
                    refs = self.send(self.shutdown, ids, (), {}, group='exclusive')
                    ray.get(refs, timeout=timeout)
                for key in ids:
                    ray.kill(self.workers[key].actor, no_restart=True)
                with self.lock:
                    for key in ids:
                        self.workers.pop(key)
                        self.suspended_workers.discard(key)
                    self._owner._actors = tuple(actor for actor in self._owner._actors if get_worker_id(actor) not in ids)
                return True
            finally:
                self.finish_reservations()

    def close_workers(self, *, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        self.close_admission()
        ids = self.worker_ids()
        self.drain_workers(ids)
        self.wait_idle(ids, timeout=timeout)
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        if not self.remove_workers(ids, timeout=remaining):
            raise RuntimeError('存在未释放硬租约；不能关闭 worker。')
