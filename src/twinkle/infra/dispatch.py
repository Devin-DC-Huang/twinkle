# Copyright (c) ModelScope Contributors. All rights reserved.
"""The remote-dispatch mechanism: ``remote_class`` / ``remote_function`` and their helpers.

This module owns how a decorated call travels from driver to workers: worker
selection, argument dispatch (slice / slice_dp / all / callable), result
collection, and the driver-side orchestration around continuous work. Process-wide
state (mode, device groups, notifier) lives in ``twinkle.infra`` and is read
through the ``infra`` module reference at call time, so ``initialize`` keeps
single ownership of it.
"""
import functools
import inspect
import os
import sys
from typing import Any, Callable, List, Literal, Optional, TypeVar, Union

import numpy as np

from twinkle import infra
from twinkle.notifier import notify_exception
from twinkle.utils import DeviceMesh, Platform, check_unsafe, get_logger
from . import _torch_setup
from .continuous_work import (_WORKER_ASYNC_PREFIX, _WORKER_RUNTIME_METHOD, _make_worker_async_companion,
                              _worker_runtime)
from .worker_pool import WorkerPool

logger = get_logger()

T1 = TypeVar('T1', bound=object)


def _capture_caller() -> Optional[str]:
    """Return ``file:line`` of the first frame outside this module, or ``None``."""
    f = sys._getframe(1)
    while f and f.f_code.co_filename == __file__:
        f = f.f_back
    return f'{f.f_code.co_filename}:{f.f_lineno}' if f else None


def _tag_exc(exc: BaseException, caller: Optional[str]) -> None:
    """Stamp driver-caller location onto exc for both traceback and str(exc)."""
    if not caller:
        return
    try:
        marker = f'[twinkle] driver caller: {caller}'
        if marker not in (getattr(exc, '__notes__', None) or []):
            exc.add_note(marker)
        if not getattr(exc, '_twinkle_caller_augmented', False):
            prefix = f'[twinkle driver caller: {caller}] '
            exc.args = (prefix + str(exc.args[0]), *exc.args[1:]) if exc.args else (prefix.rstrip(), )
            exc._twinkle_caller_augmented = True
    except Exception:  # noqa
        pass


def _get_workers(workers, execute):
    if not workers:
        return []
    if execute == 'first':
        return [workers[0]]
    elif execute == 'all':
        return workers
    elif execute == 'peer':
        peer = workers[Platform.get_peer_index(len(workers))]
        return peer if isinstance(peer, (list, tuple)) else [peer]
    else:
        raise ValueError(f'Unsupported execute method: {execute}')


def get_device_placement(device_group=None) -> str:
    """Get the device placement graph, can be used to show the training topology.

    Args:
        device_group: The device group of the training, default will use the global `device_group`.

    Returns:
        A string containing the training topology.
    """
    if device_group is None:
        device_group = infra._device_group

    if device_group is None:
        return 'No device group provided.'

    WIDTH = 80

    def box_line(content='', align='left', prefix='│', suffix='│'):
        inner_width = WIDTH - 4
        if align == 'center':
            text = content.center(inner_width)
        else:
            text = content.ljust(inner_width)
        return f'{prefix} {text} {suffix}'

    def header_box(title):
        return [
            '╔' + '═' * (WIDTH - 2) + '╗',
            box_line(title, align='center', prefix='║', suffix='║'),
            '╚' + '═' * (WIDTH - 2) + '╝',
        ]

    def section_top(title=''):
        lines = ['┌' + '─' * (WIDTH - 2) + '┐']
        if title:
            lines.append(box_line(f'◈ {title}', prefix='│', suffix='│'))
            lines.append('├' + '─' * (WIDTH - 2) + '┤')
        return lines

    def section_bottom():
        return ['└' + '─' * (WIDTH - 2) + '┘']

    def format_ranks(ranks):
        if isinstance(ranks, list):
            if len(ranks) <= 16:
                return str(ranks)
            return f'{ranks[:6]} ... {ranks[-3:]} ({len(ranks)} total)'
        return str(ranks)

    def render_mesh_grid(mesh_array, dim_names):
        """Render a compact mesh visualization."""
        lines = []

        if mesh_array.ndim == 1:
            mesh_array = mesh_array.reshape(1, -1)

        if mesh_array.ndim > 2:
            lines.append(box_line(f'    ⊞ High-dim mesh: shape={mesh_array.shape}'))
            return lines

        rows, cols = mesh_array.shape
        max_rows, max_cols = 6, 10
        show_rows, show_cols = min(rows, max_rows), min(cols, max_cols)

        cell_w = max(4, len(str(mesh_array.max())) + 2)

        header = '      ' + ''.join(f'{i:^{cell_w}}' for i in range(show_cols))
        if cols > max_cols:
            header += ' ⋯'
        lines.append(box_line(f'    {header}'))

        # Top border
        border = '      ╭' + '─' * (cell_w * show_cols + show_cols - 1) + '╮'
        lines.append(box_line(f'    {border}'))

        # Data rows
        for r in range(show_rows):
            row_data = '│'.join(f'{mesh_array[r, c]:^{cell_w}}' for c in range(show_cols))
            row_str = f'   {r:>2} │{row_data}│'
            if cols > max_cols:
                row_str += ' ⋯'
            lines.append(box_line(f'    {row_str}'))

        if rows > max_rows:
            lines.append(box_line(f"         {'⋮':^{cell_w * show_cols}}"))

        # Bottom border
        border = '      ╰' + '─' * (cell_w * show_cols + show_cols - 1) + '╯'
        lines.append(box_line(f'    {border}'))

        return lines

    # Build output
    lines = header_box('DEVICE PLACEMENT TOPOLOGY')
    lines.append('')

    for group in device_group:
        lines.extend(section_top(f'DeviceGroup: {group.name}'))
        lines.append(box_line(f'  ├─ Device Type : {group.device_type}'))
        lines.append(box_line(f'  └─ Ranks       : {format_ranks(group.ranks)}'))

        if not group._device_mesh:
            lines.append(box_line(''))
            lines.append(box_line('  (No device meshes configured)', align='center'))
        else:
            for mesh_name, mesh in group._device_mesh.items():
                lines.append(box_line(''))
                lines.append(box_line(f'  ┌─ DeviceMesh: {mesh_name}'))

                # Dimensions
                if mesh.mesh_dim_names:
                    dim_info = ' × '.join(f'{name}={size}' for name, size in zip(mesh.mesh_dim_names, mesh.mesh.shape))
                    lines.append(box_line(f'  │  Dimensions : {dim_info}'))

                # Active parallelism
                parallelism = []
                for dim in ['pp', 'dp', 'tp', 'ep', 'sp', 'cp', 'fsdp']:
                    ws = mesh._get_world_size_for_dim(dim)
                    if ws is not None and ws > 1:
                        parallelism.append(f'{dim.upper()}={ws}')

                if parallelism:
                    lines.append(box_line(f"  │  Parallelism: {', '.join(parallelism)}"))

                # Mesh layout
                lines.append(box_line('  │'))
                lines.append(box_line('  └─ Mesh Layout:'))
                lines.extend(render_mesh_grid(mesh.mesh, mesh.mesh_dim_names or []))

        lines.append(box_line(''))
        lines.extend(section_bottom())
        lines.append('')

    return '\n' + '\n'.join(lines)


def _collect_func(method: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp'], Callable],
                  result: List[Any],
                  device_mesh: DeviceMesh = None):
    """Collect results

    Args:
        method:
            none: Return as is.
            flatten: Flat the nested results.
            mean: Average the results.
            sum: Sum the results.
            first: Only return the first result.
            last_pp: Only return the results of the last pp rank.
        result: The results returned by workers.
        device_mesh: The device_mesh, needed by `last_pp`
    Returns:
        The collected results.
    """
    if not result:
        return result

    if method == 'first':
        return result[0]
    if method == 'none' and len(result) == 1:
        return result[0]
    if isinstance(result[0], tuple):
        if any(type(item) is not type(result[0]) or len(item) != len(result[0]) for item in result):
            raise ValueError('各 worker 的 tuple 结果必须具有相同类型和长度。')
        output = []
        # if each result of a worker is a tuple
        for i in range(len(result[0])):
            # handle each element in a tuple
            _single_result = [r[i] for r in result]
            output.append(_collect_func(method, _single_result, device_mesh=device_mesh))
        if isinstance(getattr(type(result[0]), '_fields', None), tuple):
            return type(result[0])(*output)
        return tuple(output)
    if method == 'none':
        if isinstance(result, list) and len(result) == 1:
            # unwrap the result
            return result[0]
        else:
            return result
    elif method == 'flatten':
        # flatten
        flatten = [item for sublist in result for item in sublist]
        if isinstance(result[0], np.ndarray):
            return np.array(flatten)
        return type(result[0])(flatten)
    elif method in ('avg', 'mean'):
        if isinstance(result[0], dict):
            output = {}
            for key in result[0]:
                vals = [r[key] for r in result if key in r]
                try:
                    output[key] = np.mean(vals)
                except (TypeError, ValueError):
                    output[key] = vals
            return output
        return np.mean(result)
    elif method == 'sum':
        return np.sum(result)
    elif method == 'first':
        return result[0]
    elif method == 'last_pp':
        assert device_mesh is not None
        return [r for i, r in enumerate(result) if i in device_mesh.get_pp_last_ranks()]
    elif method == 'last_pp_first':
        # Return the first result from the last PP stage workers.
        # Falls back to result[0] when PP = 1 (all workers are the last stage).
        assert device_mesh is not None
        last_pp = [r for i, r in enumerate(result) if i in device_mesh.get_pp_last_ranks()]
        return last_pp[0] if last_pp else result[0]
    elif isinstance(method, Callable):
        # Callable
        return method(result, device_mesh=device_mesh)
    else:
        raise ValueError(f'Unsupported collect method: {method}')


def _dispatch_args(workers, dispatch, execute, device_mesh: Optional[DeviceMesh], args, kwargs):
    if execute == 'first':
        return [(workers[0], args, kwargs)]
    elif dispatch == 'all':
        return [(worker, args, kwargs) for worker in workers]
    elif dispatch == 'slice':
        # split arg to workers evenly
        result = []
        length = len(workers)

        def dispatch_func(arg, n):
            if isinstance(arg, list):
                # only list
                _args = []
                k, m = divmod(len(arg), n)
                for i in range(n):
                    _args.append(arg[i * k + min(i, m):(i + 1) * k + min(i + 1, m)])
                return _args
            else:
                return [arg] * n

        args = [dispatch_func(arg, length) for arg in args]
        kwargs = {k: dispatch_func(v, length) for k, v in kwargs.items()}
        for i in range(length):
            sliced_args = tuple(arg[i] for arg in args)
            sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
            result.append((workers[i], sliced_args, sliced_kwargs))

        return result
    elif dispatch == 'slice_dp':
        assert device_mesh is not None
        # split by dp. each worker in one ep will receive the same argument
        result = []
        # if device_mesh is not None:
        # TODO this may occurs error when remote calls remote
        # Comment this because remote_class supports `first``
        # assert device_mesh.world_size == len(workers)
        length = len(workers)
        # Map actor index to global_rank: with gpus_per_worker>1, consecutive
        # global ranks belong to the same actor (TP peers).
        _mesh_world = device_mesh.world_size if device_mesh is not None else length
        _rank_stride = max(1, _mesh_world // length)

        def dispatch_func(arg, n):
            import torch
            if isinstance(arg, list) or isinstance(arg, torch.Tensor):
                _args = []
                for i in range(n):
                    _args.append(arg[device_mesh.get_slice(
                        len(arg), device_mesh.get_data_rank_from_global_rank(i * _rank_stride))])
                return _args
            elif isinstance(arg, dict):
                _args = [{} for _ in range(n)]
                for key in arg.keys():
                    value = arg[key]
                    for i, v in enumerate(dispatch_func(value, n)):
                        _args[i][key] = v
                return _args
            else:
                return [arg] * n

        args = [dispatch_func(arg, length) for arg in args]
        kwargs = {k: dispatch_func(v, length) for k, v in kwargs.items()}

        for i in range(length):
            sliced_args = tuple(arg[i] for arg in args)
            sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
            result.append((workers[i], sliced_args, sliced_kwargs))

        # Raise early if some ranks got data and others didn't (causes hangs).
        def _check_uniform(slices):
            lens = [len(s) if s is not None and isinstance(s, (list, tuple)) else 0 for s in slices]
            return not lens or all(length > 0 for length in lens) or all(length == 0 for length in lens)

        for arg in args:
            if not _check_uniform(arg):
                raise ValueError(f'Batch too small for {length} workers, some ranks have no data. '
                                 f'Please increase batch size to at least {length}.')
        for v in kwargs.values():
            if not _check_uniform(v):
                raise ValueError(f'Batch too small for {length} workers, some ranks have no data. '
                                 f'Please increase batch size to at least {length}.')

        return result
    elif isinstance(dispatch, Callable):
        length = len(workers)
        result = []
        for i in range(length):
            sliced_args, sliced_kwargs = dispatch(length, i, args, kwargs, device_mesh=device_mesh)
            result.append((workers[i], sliced_args, sliced_kwargs))
        return result
    else:
        raise ValueError(f'Unsupported dispatch method: {dispatch}')


def _get_device_mesh_param_name(init_method) -> str:
    """Try to get the device_mesh param name"""
    sig = inspect.signature(init_method)
    for param in sig.parameters.values():
        ann = param.annotation
        if ann != inspect.Parameter.empty:
            if hasattr(ann, '__name__') and ann.__name__ == 'DeviceMesh':
                return param.name
            if 'DeviceMesh' in str(ann):
                return param.name
    return ''


def _get_device_mesh_param(args, kwargs):
    """Try to get the device_mesh param instance"""
    for arg in (list(args) + list(kwargs.values())):
        if isinstance(arg, DeviceMesh):
            return arg
    return None


def remote_class(execute: Literal['first', 'peer', 'all'] = 'all', max_concurrency: Optional[int] = None):
    """Patch each class used in remote clusters with this decorator.

    Use this decorator to wrap your class to enable it to execute in a remote cluster.

    Args:
        execute: which workers the class runs on.
        max_concurrency: Ray actor concurrency, i.e. how many of this class's
            methods one worker may run at once. ``None`` leaves Ray's default of
            1, under which concurrent calls to the same worker queue and run one
            after another. Only set it for a class whose methods tolerate running
            side by side: a class holding NCCL collectives does not, because two
            collectives interleaving on one rank deadlock. It is what
            ``enable_continous_work`` needs to reach the worker's engine
            concurrently instead of stopping at the actor boundary.
    """

    def decorator(cls):
        # 所有领域方法采用同一个执行适配，不识别方法名或组件种类。
        setattr(cls, _WORKER_RUNTIME_METHOD, _worker_runtime)
        for _name in dir(cls):
            _attr = getattr(cls, _name, None)
            _companion = getattr(_attr, '_worker_async_companion', None)
            if _companion is not None:
                setattr(cls, _WORKER_ASYNC_PREFIX + _name, _companion)
        # Get device mesh parameter name
        device_mesh_name = _get_device_mesh_param_name(cls.__init__)
        init_method = cls.__init__

        @functools.wraps(init_method)
        def new_init(self, *args, **kwargs):
            _caller = _capture_caller()
            _ctx = f'{cls.__name__}.__init__'
            if _caller:
                _ctx = f'{_ctx} <- {_caller}'
            try:
                infra._maybe_load_worker_notifier()
                _new_init_body(self, _caller, *args, **kwargs)
            except Exception as _e:
                _tag_exc(_e, _caller)
                notify_exception(infra._notifier, _ctx, _e, infra._name)
                raise

        def _new_init_body(self, _caller, *args, **kwargs):
            if infra._mode == 'local':
                # Get the actual device_mesh
                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name and infra._device_group is not None:
                    if device_mesh is None:
                        # Local mode can safely assign the default device mesh
                        device_mesh = infra._device_mesh
                        kwargs[device_mesh_name] = infra._device_mesh
                    assert len(infra._device_group) == 1  # only one device group is allowed
                    infra._device_group[0]._device_mesh[self.__class__.__name__] = device_mesh
                    if self.__class__.__name__ == 'DataLoader' and 'min_batch_size' not in kwargs:
                        # TODO An ugly special setting for dataloader to set the min batch size
                        kwargs['min_batch_size'] = device_mesh.data_world_size
                    init_method(self, *args, **kwargs)
                else:
                    if device_mesh is not None:
                        logger.warning(f'{cls.__name__} was given a device_mesh but it is being DROPPED: twinkle '
                                       'holds no device group, so call twinkle.initialize(...) before '
                                       'constructing it. Training will otherwise run with device_mesh=None '
                                       '(single-rank loss normalisation and no metric aggregation).')
                    args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                    kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                    init_method(self, *args, **kwargs)
            elif infra._mode == 'ray':
                from ._ray import RayHelper

                # In case the same class created twice in the same device group
                # Try to get the caller's line (resolved in ``new_init`` so it points
                # at user code, not at the wrapper itself).
                _cf, _, _cl = (_caller or f'{__file__}:0').rpartition(':')
                caller_file = _cf.replace(os.sep, '_').replace('.', '_')
                caller_line = _cl
                # Pass an instance_id is recommended
                instance_id = kwargs.pop('instance_id', '') + f'{caller_file}_{caller_line}'
                remote_group = kwargs.get('remote_group')
                if os.environ.get('WORKER_NAME') is None and remote_group is None:
                    logger.info(f'⚠️ Using local initialization of class: {cls}, please make sure the class '
                                'does not need remote execution.')
                # If cannot trust_remote_code, no callable and type can be used.
                check_unsafe(*args, **kwargs)

                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name:
                    if execute == 'first':
                        # Manually create a device_mesh because there is only one worker
                        device_mesh = DeviceMesh.from_sizes(dp_size=1)
                        kwargs[device_mesh_name] = device_mesh

                    if self.__class__.__name__ == 'DataLoader' and 'min_batch_size' not in kwargs:
                        # TODO An ugly special setting for dataloader to set the min batch size
                        kwargs['min_batch_size'] = kwargs['batch_size']

                    if remote_group:
                        if device_mesh is None:
                            if infra._device_mesh is not None:
                                device_mesh = infra._device_mesh
                                kwargs[device_mesh_name] = device_mesh
                            else:
                                raise ValueError('Set device_mesh=DeviceMesh(...) to enable ray.')

                    if infra._device_group and remote_group:
                        # usually this happens in driver because worker does not has a valid _device_group
                        # this is used to print the device_group info, so pass the worker is ok
                        device_group = [dg for dg in infra._device_group if dg.name == remote_group][0]
                        device_group._device_mesh[self.__class__.__name__] = device_mesh

                # This will solve the iterator cannot be passed through ray.
                def __iter__(_self):
                    if os.environ.get('WORKER_NAME'):
                        # This is a worker, iter keeps in the class, pass nothing to driver
                        _iter = _self.__iter_origin__()
                        assert _iter is not _self
                        _self._iter = _iter
                    else:
                        # This is executed in driver
                        return _self.__iter_origin__()

                def __next__(_self):
                    # Use _self._iter to get the next data
                    # Only one driver can use this at one time
                    try:
                        # Return a tuple, get the second output in the driver to stop the for loop
                        return next(_self._iter), False
                    except StopIteration:
                        return [], True

                if (not remote_group) or os.environ.get('CLUSTER_NAME') == remote_group:
                    # not remote_group: Ray mode with local component
                    # os.environ.get('CLUSTER_NAME') == remote_group: a normal worker's init
                    # All per-process torch setup (seed / determinism / tf32 / the worker CUDA-env pins)
                    # is isolated in _torch_setup, so none of it leaks into the Ray dispatch logic here.
                    _torch_setup.apply_in_worker(infra._extra_torch_kwargs)
                    if not device_mesh_name:
                        # pop the device_mesh
                        args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                        kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                        # if any handler is passed to other component, lazy collect should be false
                        # for example, dataset pass to the dataloader
                    kwargs.pop('remote_group', None)  # component does not need this
                    init_method(self, *args, **kwargs)
                else:
                    if hasattr(cls, '__iter__'):
                        _dispatch = self.__iter__._dispatch
                        _execute = self.__iter__._execute
                        _collect = self.__iter__._collect

                    if hasattr(cls, '__iter__'):
                        import ray
                        cls.__iter_origin__ = cls.__iter__
                        cls.__iter__ = __iter__
                        cls.__next__ = __next__
                        setattr(cls, _WORKER_ASYNC_PREFIX + '__iter__', _make_worker_async_companion(__iter__, None))
                        setattr(cls, _WORKER_ASYNC_PREFIX + '__next__', _make_worker_async_companion(__next__, None))

                    # Create remote workers
                    # Remove potential duplicate keys from kwargs before passing
                    kwargs_for_workers = kwargs.copy()
                    kwargs_for_workers.pop('instance_id', None)
                    # create_workers takes extra_torch_kwargs explicitly, so strip it from the forwarded
                    # component kwargs to avoid a duplicate-keyword collision.
                    kwargs_for_workers.pop('extra_torch_kwargs', None)

                    _actors = RayHelper.create_workers(
                        cls,
                        remote_group,
                        execute,
                        instance_id=instance_id,
                        extra_torch_kwargs=infra._extra_torch_kwargs,
                        max_concurrency=max_concurrency,
                        *args,
                        **kwargs_for_workers)
                    self._actors = _actors
                    self._worker_factory = (cls, tuple(args), dict(kwargs_for_workers), infra._extra_torch_kwargs)
                    # Remembered so remote_function knows this class's workers run
                    # methods side by side, and that it must therefore track what is
                    # in flight. Without concurrency Ray orders calls per actor and
                    # the tracking would be dead weight.
                    self._max_concurrency = max_concurrency
                    if hasattr(cls, '__iter__'):
                        # wraps again, because ray uses cls method to call remote
                        cls.__iter__ = remote_function(dispatch=_dispatch, execute=_execute, collect='none')(__iter__)
                        cls.__next__ = remote_function(dispatch=_dispatch, execute=_execute, collect=_collect)(__next__)
                    for arg in (list(args) + list(kwargs.values())):
                        # keeps the device_mesh in the handler
                        if isinstance(arg, DeviceMesh):
                            self.device_mesh = arg
                            break

                self.remote_group = remote_group
                self._instance_id = instance_id
                if '_actors' in self.__dict__ and '_worker_pool' not in self.__dict__:
                    self._worker_pool = WorkerPool.create(self)
            else:
                raise ValueError(f'Unsupported mode: {infra._mode}')

        def perf_on(self, *, profiler_spec=None):
            from twinkle.perf import Perf
            strategy = getattr(self, 'strategy', None)
            model = getattr(self, 'model', None)
            target = strategy.unwrap_model(model) if (strategy is not None and model is not None) else None
            self._profiler = Perf.enable_process(model=target, profiler_spec=profiler_spec)

        def perf_step(self):
            from twinkle.perf import Perf
            Perf.step_process(getattr(self, '_profiler', None))

        def perf_dump(self):
            from twinkle.perf import Perf
            return Perf.dump_process(getattr(self, '_profiler', None))

        def perf_off(self):
            from twinkle.perf import Perf
            Perf.disable_process(getattr(self, '_profiler', None))
            self._profiler = None

        if not hasattr(cls, 'perf_on'):
            cls.perf_on = remote_function(dispatch='all', execute=execute, collect='none',
                                          lazy_collect=False)(perf_on)
        if not hasattr(cls, 'perf_step'):
            cls.perf_step = remote_function(dispatch='all', execute=execute, collect='none')(perf_step)
        if not hasattr(cls, 'perf_dump'):
            cls.perf_dump = remote_function(dispatch='all', execute=execute, collect='none',
                                            lazy_collect=False)(perf_dump)
        if not hasattr(cls, 'perf_off'):
            cls.perf_off = remote_function(dispatch='all', execute=execute, collect='none',
                                           lazy_collect=False)(perf_off)

        # 后添加的通用 perf 方法也走同一个适配。
        for name in dir(cls):
            companion = getattr(getattr(cls, name, None), '_worker_async_companion', None)
            if companion is not None:
                setattr(cls, _WORKER_ASYNC_PREFIX + name, companion)
        original_getstate = getattr(cls, '__getstate__', None)
        original_setstate = getattr(cls, '__setstate__', None)

        def getstate(self):
            if '_actors' in self.__dict__:
                return {'_worker_pool': self._worker_pool, '_actors': self._actors,
                        'device_mesh': getattr(self, 'device_mesh', None),
                        'remote_group': self.remote_group, '_instance_id': self._instance_id,
                        '_max_concurrency': self._max_concurrency}
            return original_getstate(self) if original_getstate is not None else dict(self.__dict__)

        def setstate(self, state):
            if '_worker_pool' in state or original_setstate is None:
                self.__dict__.update(state)
            else:
                original_setstate(self, state)

        cls.__getstate__ = getstate
        cls.__setstate__ = setstate
        cls.__init__ = new_init
        return cls

    return decorator


def remote_function(dispatch: Union[Literal['slice', 'all', 'slice_dp', 'last_pp_first'], Callable] = 'slice',
                    execute: Literal['first', 'peer', 'all'] = 'all',
                    collect: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp'], Callable] = 'none',
                    sync: bool = False,
                    lazy_collect: Optional[bool] = None,
                    timeout: Optional[float] = None,
                    enable_continous_work: bool = False,
                    worker_selector: Optional[str] = None,
                    *, exclusive: bool = False, batch_arg: Optional[str] = None, aligned_args=(), prepare=None,
                    sticky: bool = False, sticky_key: Optional[Union[str, Callable]] = None,
                    sticky_mode: Literal['affinity', 'acquire', 'use'] = 'affinity',
                    sticky_affinity: Optional[Union[str, Callable]] = None,
                    sticky_fallback: Optional[Union[str, Callable]] = None,
                    control: bool = False, replay: bool = False, shutdown: bool = False):
    """Patch each method called from remote(which class should be decorated with `remote_class`) with this decorator.

    Args:
        dispatch: How to dispatch the arguments.
            'slice': load balance
            'all': all processes do the same thing
            'slice_dp': Slice the input by data ranks in device_mesh
            Callable: A callable that handles the dispatching
        execute: How to execute
            'first': Only first worker
            'peer': Only peer workers
            'all': All processes
        collect: How to collect the results.
            'none': Return as-is
            'flatten': Return a flattened list
            'mean': Return the mean value of all processes
            'sum': Return the sum value of all processes
            'first': Return the first worker's result but executed in each process, usually works for scenarios of all-gather.
            'mean'/'sum': Avg or sum the results.
            'first': Return the first worker's result, for example, get length
            'last_pp': Return the last pp's result.
            Callable: A callable that handles the collection
        sync: If True, use synchronous execution (execute_all_sync) instead of async.
            Required for methods with NCCL collective operations (e.g., Megatron forward_backward).
        lazy_collect: Do lazy collect, this boolean value decides whether this function needs lazy collect. If setting to None, it will follow the global setting.
        timeout: Timeout in seconds for ray.get() when collecting results. The decorator's
            explicitly declared value takes priority; the instance attribute ``_ray_get_timeout``
            is the fallback for methods that declare none (``timeout if timeout is not None
            else instance``).
        enable_continous_work: 独立输入按通用 Router 分配，合并 worker 批次并恢复原顺序；
            不适用于跨 rank collective。单项调用选择一个完整 worker。
        worker_selector: 旧的组件路由入口，迁移后不再接受。
        batch_arg/aligned_args: 显式声明批参数及其等长对齐参数。
        prepare: 可选的调用前参数规范化函数；本地与远程使用同一输入合同。
        exclusive: 等待在途执行结束，实际完成前不发送普通调用；infra 不识别业务方法名。
        sticky/sticky_key: 按同一键路由；键为字段路径（如 trajectory.request_id、inputs.0.session_id）或 callable。
            callable 接收 read(path, default=None)，只读取指定字段，不解析表达式或取得完整远程载荷。
        sticky_mode: affinity 为会话亲和；acquire 建立调用归属；use 只查询已有归属，不重新分配。
        sticky_affinity/sticky_fallback: 可选会话键及 use 的回退键，解析方式同 sticky_key。
        control: 独立控制执行组，可在普通调用或互斥排干期间执行。
        replay/shutdown: 新 worker 重放的初始化方法，以及退役时调用的无参关闭方法。
    """ # noqa

    def decorator(func: Callable[..., T1]) -> Callable[..., T1]:

        signature = inspect.signature(func)
        if any(type(value) is not bool for value in (exclusive, sticky, control, replay, shutdown)):
            raise TypeError('exclusive/sticky/control/replay/shutdown 必须为 bool。')
        if control and (sync or exclusive or replay):
            raise ValueError('control 不能与互斥、同步或重放声明组合。')
        if sticky_mode not in ('affinity', 'acquire', 'use'):
            raise ValueError('sticky_mode 只支持 affinity/acquire/use。')
        selectors = (sticky_key, sticky_affinity, sticky_fallback)
        if sticky and sticky_key is None:
            raise ValueError('sticky=True 必须声明 sticky_key。')
        if not sticky and (any(value is not None for value in selectors) or sticky_mode != 'affinity'):
            raise ValueError('粘滞路由参数需要 sticky=True。')
        for selector in selectors:
            if isinstance(selector, str):
                parts = selector.split('.')
                if (parts[0] not in signature.parameters
                        or any(not part.isidentifier() and not part.isdecimal() for part in parts)):
                    raise ValueError(f'无效的粘滞路由字段路径：{selector!r}')
            elif selector is not None and not callable(selector):
                raise TypeError('粘滞路由键必须为字段路径字符串、callable 或 None。')
        if prepare is not None and not callable(prepare):
            raise TypeError('prepare 必须为 callable 或 None。')
        if (sync or exclusive or replay) and enable_continous_work:
            raise ValueError('互斥或 collective 同步执行不能采用独立请求路由。')

        def invoke(self, args, kwargs, *, wait=True):
            proxy = '_actors' in self.__dict__
            caller = _capture_caller() if proxy else None
            context = f'{type(self).__name__}.{func.__name__}'
            try:
                if prepare is not None:
                    args, kwargs = prepare(signature, self, args, kwargs)
                if not proxy:
                    # 本地保留业务返回类型；参数引用仍由通信适配统一解析。
                    from ._ray.ray_helper import RayHelper
                    args, kwargs = RayHelper.do_get_and_collect(args, kwargs)
                    return func(self, *args, **kwargs)
                if worker_selector is not None:
                    raise ValueError('worker_selector 已移除；请通过 remote_function 声明 sticky 路由。')
                check_unsafe(*args, **kwargs)
                worker_ids = kwargs.pop('_worker_ids', None)
                limit = timeout if timeout is not None else getattr(self, '_ray_get_timeout', None)
                pool = self._worker_pool
                result = pool.dispatch_call(func.__name__, args, kwargs,
                                            dispatch=dispatch, execute=execute, collect=collect,
                                            sync=sync, exclusive=exclusive, timeout=limit, worker_ids=worker_ids,
                                            independent=enable_continous_work, batch_arg=batch_arg,
                                            aligned_args=aligned_args, signature=signature)
                if func.__name__ == '__iter__':
                    result.result()
                    return self
                if func.__name__ == '__len__':
                    return result.result()
                if func.__name__ == '__next__':
                    pair = result.result()
                    flags = pair[1]
                    if any(flags) if isinstance(flags, (list, tuple)) else flags:
                        raise StopIteration
                    return pair[0]
                effective_lazy = infra._lazy_collect if lazy_collect is None else lazy_collect
                return result if not wait or effective_lazy and not sync else result.result()
            except StopIteration:
                raise
            except Exception as exc:
                _tag_exc(exc, caller)
                infra._maybe_load_worker_notifier()
                notify_exception(infra._notifier, context, exc, infra._name)
                raise

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T1:
            return invoke(self, args, kwargs)

        wrapper._invoke = invoke
        wrapper._execute = execute
        wrapper._collect = collect
        wrapper._dispatch = dispatch
        wrapper._lazy_collect = lazy_collect
        wrapper._sync = sync
        wrapper._exclusive = exclusive
        wrapper._enable_continous_work = enable_continous_work
        wrapper._rpc = {
            'sticky': sticky, 'key': sticky_key, 'mode': sticky_mode,
            'affinity': sticky_affinity, 'fallback': sticky_fallback, 'signature': signature,
            'control': control, 'replay': replay, 'shutdown': shutdown,
        }
        wrapper._worker_async_companion = _make_worker_async_companion(func, wrapper)
        return wrapper

    return decorator
