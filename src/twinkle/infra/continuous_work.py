# Copyright (c) ModelScope Contributors. All rights reserved.
"""通用 worker 执行适配与独立请求的批分片，不维护第二份调用归属。"""
import asyncio
import functools
import inspect
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

_WORKER_ASYNC_PREFIX = '_twinkle_async_'
_WORKER_RUNTIME_METHOD = '_twinkle_runtime'
_CREATE_LOCK = threading.Lock()


def get_worker_id(worker):
    """actor ID 同时标识当前 incarnation。"""
    return worker._actor_id.hex()


def _worker_state(component):
    with _CREATE_LOCK:
        state = component.__dict__.get('_twinkle_execution')
        if state is None:
            state = {'lock': threading.RLock(), 'calls': {}, 'executors': {}}
            component.__dict__['_twinkle_execution'] = state
        return state


def _worker_executor(component, group):
    state = _worker_state(component)
    with state['lock']:
        if group not in state['executors']:
            sizes = {'ordinary': int(os.environ.get('TWINKLE_ACTOR_MAX_CONCURRENCY') or 1),
                     'control': 2, 'exclusive': 1}
            state['executors'][group] = ThreadPoolExecutor(
                max_workers=sizes[group], thread_name_prefix=f'twinkle-{group}')
        return state['executors'][group]


async def _await_result(value):
    """Future 始终在原 loop/线程完成；取消等待不取消正在运行的业务。"""
    while isinstance(value, Future) or inspect.isawaitable(value):
        if isinstance(value, Future):
            value = await asyncio.shield(asyncio.wrap_future(value))
        elif isinstance(value, asyncio.Future) and value.get_loop() is not asyncio.get_running_loop():
            async def wait_original(future):
                return await asyncio.shield(future)
            loop = value.get_loop()
            if not loop.is_running():
                raise RuntimeError('业务 Future 的事件循环未运行。')
            bridge = asyncio.run_coroutine_threadsafe(wait_original(value), loop)
            value = await asyncio.shield(asyncio.wrap_future(bridge))
        else:
            value = await asyncio.shield(value)
    return value


def _make_worker_async_companion(func, wrapper):
    @functools.wraps(func)
    async def companion(component, *args, **kwargs):
        # 准入只由调用池在发送前判定；worker 只登记实际执行。
        group = kwargs.pop('_infra_group', 'ordinary')
        if group not in ('ordinary', 'control', 'exclusive'):
            raise ValueError(f'未知执行组：{group!r}')
        state = _worker_state(component)
        call_id = uuid.uuid4().hex
        with state['lock']:
            state['calls'][call_id] = group
        async def execute():
            # 参数解引用和阻塞 body 不占用 Ray actor 的事件循环。
            from ._ray.ray_helper import RayHelper
            dispatch_context = kwargs.pop('_dispatch_context', None)
            call_args, call_kwargs = await asyncio.get_running_loop().run_in_executor(
                _worker_executor(component, group),
                functools.partial(RayHelper.do_get_and_collect, args, kwargs))
            if dispatch_context is not None:
                from .dispatch import _dispatch_args
                count, index, dispatch, execute_mode = dispatch_context
                mesh = getattr(component, 'device_mesh', None)
                parts = _dispatch_args([None] * count, dispatch, execute_mode, mesh, call_args, call_kwargs)
                _, call_args, call_kwargs = parts[index]
            if inspect.iscoroutinefunction(func):
                result = func(component, *call_args, **call_kwargs)
            else:
                result = await asyncio.get_running_loop().run_in_executor(
                    _worker_executor(component, group), functools.partial(func, component, *call_args, **call_kwargs))
            return await _await_result(result)
        task = asyncio.create_task(execute())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # 重复取消等待也不能提前归还执行容量。
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            if not task.cancelled():
                task.exception()
            raise
        except Exception as exc:
            from twinkle import infra
            from twinkle.notifier import notify_exception
            infra._maybe_load_worker_notifier()
            notify_exception(infra._notifier, f'{type(component).__name__}.{func.__name__}', exc, infra._name)
            raise
        finally:
            with state['lock']:
                state['calls'].pop(call_id, None)
    companion.__name__ = _WORKER_ASYNC_PREFIX + func.__name__
    return companion


async def _worker_runtime(component, operation):
    """仅通信生命周期；不调用任何领域方法。"""
    state = _worker_state(component)
    if operation not in ('ready', 'idle', 'drain'):
        raise ValueError(f'未知 worker 生命周期操作：{operation!r}')
    if operation == 'drain':
        while True:
            with state['lock']:
                if not state['calls']:
                    break
            await asyncio.sleep(0.01)
    with state['lock']:
        return not state['calls'] if operation == 'idle' else True


def _cw_batch_len(args, kwargs):
    return next((len(value) for value in (*args, *kwargs.values()) if isinstance(value, list)), None)


def _cw_sub_args(args, kwargs, indices, batch_len):
    def pick(value):
        return [value[index] for index in indices] if isinstance(value, list) and len(value) == batch_len else value
    return tuple(pick(value) for value in args), {key: pick(value) for key, value in kwargs.items()}


def _collect_ordered(plan, results, device_mesh=None):
    size, partitions = plan
    output = [None] * size
    if len(partitions) != len(results):
        raise ValueError('批调用的分片数量与结果数量不一致。')
    for indices, part in zip(partitions, results):
        if not isinstance(part, (list, tuple)) or len(part) != len(indices):
            raise TypeError('独立请求的批调用必须按输入项返回等长结果。')
        for index, value in zip(indices, part):
            output[index] = value
    return output


def _run_continous_work(component, func_name, args, kwargs, batch_len, timeout, pool,
                        *, batch_arg=None, aligned_args=(), signature=None):
    """在 pool 的同一临界区选择、预留、发送；只返回通用 Future。"""
    from ._ray.ray_helper import LazyCollect
    partitions = {}
    reservations = {}
    refs = []
    indices_by_ref = []
    prepared = []
    for index in range(batch_len):
        sub_args, sub_kwargs = _cw_sub_args(args, kwargs, [index], batch_len)
        if batch_arg is not None:
            bound = signature.bind(component, *args, **kwargs)
            for name in (batch_arg, *aligned_args):
                value = bound.arguments.get(name)
                if value is not None:
                    if not isinstance(value, list) or len(value) != batch_len:
                        raise ValueError(f'{name} 必须与 {batch_arg} 等长并为 list。')
                    bound.arguments[name] = [value[index]]
            sub_args, sub_kwargs = bound.args[1:], bound.kwargs
        prepared.append((sub_args, sub_kwargs, pool.prepare_context(func_name, sub_args, sub_kwargs)))
    with pool.lock:
        try:
            for index, (sub_args, sub_kwargs, context) in enumerate(prepared):
                ids = pool.choose(func_name, sub_args, sub_kwargs, independent=True, context=context)
                if len(ids) != 1:
                    raise ValueError('独立请求路由必须选择一个完整 worker。')
                partitions.setdefault(ids[0], []).append(index)
                record = pool.reserve(func_name, ids, sub_args, sub_kwargs, context=context)
                reservations.setdefault(ids[0], []).append(record)
            for worker_id, indices in partitions.items():
                sub_args, sub_kwargs = _cw_sub_args(args, kwargs, indices, batch_len)
                if batch_arg is not None:
                    bound = signature.bind(component, *args, **kwargs)
                    for name in (batch_arg, *aligned_args):
                        value = bound.arguments.get(name)
                        if value is not None:
                            bound.arguments[name] = [value[index] for index in indices]
                    sub_args, sub_kwargs = bound.args[1:], bound.kwargs
                ref = pool.send(func_name, (worker_id,), sub_args, sub_kwargs, charge=len(indices),
                                record=tuple(reservations[worker_id]))[0]
                refs.append(ref)
                indices_by_ref.append(tuple(indices))
        finally:
            pool.finish_reservations()
    return LazyCollect(refs, (batch_len, tuple(indices_by_ref)), _collect_ordered, timeout=timeout)
