# Copyright (c) ModelScope Contributors. All rights reserved.
"""twinkle.perf -- a general-purpose performance profiling component.

Two independent, low-intrusion parts:

* :class:`TimeRecorder` / :func:`timed` / :func:`span` -- a synchronous- and
  async-agnostic wall-clock span recorder that nests into an overall timing tree
  and summarizes to a printable report or a metric-ready ``dict[str, float]``.
  Pure Python (no torch), so it can time ANY callable.
* :class:`ModuleProfiler` / :class:`ActivationStatsProfiler` -- a forward-hook
  profiler that gates by step, selects target modules, and leaves *what to
  measure* open via the :meth:`ModuleProfiler.collect` subclass hook. Attaches to
  any ``nn.Module`` subclass.

Both are disabled/no-op by default, so importing perf never changes existing
behaviour. Neither part imports twinkle/swift business modules or Ray: the
driver-side orchestration (:class:`Perf`) duck-types the per-rank handles, and the
process-local primitives (``Perf.enable_process`` / ``step_process`` / ``dump_process``
/ ``disable_process``) run in whatever process calls them and own the whole
recorder+profiler protocol. The only Ray crossing is :func:`twinkle.infra.remote_class`
wrapping those primitives so they execute on every rank -- perf itself stays
standalone and reusable without it.
"""

from .control import Perf
from .module_profiler import (ActivationStatsProfiler, ModuleProfiler, ParamStatsProfiler, build_module_profiler)
from .recorder import TimeRecorder, default_recorder, disable_perf, enable_perf, span, timed
from .stats import describe_shape, primary_tensor, tensor_moments, tensor_numel

__all__ = [
    'Perf',
    'TimeRecorder',
    'default_recorder',
    'timed',
    'span',
    'enable_perf',
    'disable_perf',
    'ModuleProfiler',
    'ActivationStatsProfiler',
    'ParamStatsProfiler',
    'build_module_profiler',
    'primary_tensor',
    'tensor_moments',
    'tensor_numel',
    'describe_shape',
]
