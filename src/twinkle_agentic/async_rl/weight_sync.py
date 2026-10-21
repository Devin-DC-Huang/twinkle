# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight-publication strategies for the async-RL trainer.

When a training partition finishes, the trainer must make the just-trained
policy visible to the sampler before the next rollout.  How that happens is
model- and deployment-specific, so ``TrainerWorker`` receives it as an injected
``save_adapter`` callable instead of hard-coding one mechanism.  The strategies
here are the concrete, named forms of that callable:

* :class:`AdapterSnapshotSync` -- save a LoRA adapter to disk and let the sampler
  load that snapshot.  Every version is a distinct path, so several can stay
  resident at once and ``max_staleness`` may exceed 1.
* :class:`InPlaceWeightSync` -- overwrite the sampler's single live copy of the
  weights (twinkle's ``CheckpointEngineManager``).  The transport is a deployment
  choice, not a separate strategy: ``mode='colocate'`` pushes over CUDA IPC on a
  shared GPU, ``mode='standalone'`` pushes over NCCL to a disaggregated sampler
  (verl's checkpoint-engine approach).  How much staleness this allows follows
  from the deployment, not from there being one copy: a colocated sampler must
  sit idle while it is overwritten, so at most one batch can be in flight
  (``max_staleness <= 1``); a disaggregated sampler keeps generating on its own
  GPUs between sparse syncs, so queued samples may be several versions old
  (``max_staleness > 1``), exactly verl's fully-async regime.

A strategy is callable with the same ``(admission) -> handle`` signature
``TrainerWorker`` already expects of ``save_adapter``, so it drops into that seam
unchanged.  The handle is the version-pinning token: a disk path for
:class:`AdapterSnapshotSync`, ``None`` for :class:`InPlaceWeightSync` (its single
live copy needs no pinning).  Both are plain objects holding one callback, so
they cross the Ray actor boundary exactly like the ``partial`` callables the
pipeline injects today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

# 发布窗口由调用方控制面提供，策略只负责权重 push。


class WeightSyncStrategy(ABC):
    """Publishes newly-trained weights and returns their version-pinning handle.

    Owned by the dev-side controller (``swift.dev.recipe.rl.streaming.StreamingPublisher``), NOT by the
    ``StreamingDriver``: the driver was purified into a rollout pump that holds no weight/model control, so the
    controller keeps this strategy and hands its bound :meth:`publish` to ``StreamingDriver.request_publish``
    once per publication. The driver then calls it as ``push(window)`` inside its own version/window bracket
    (close the oldest window before, bump the version + drop stale + prune after) -- the strategy never lives
    on the driver and the driver never decides the cadence.
    """

    @abstractmethod
    def publish(self, admission: Any) -> str | None:
        """Make the trained policy visible to the sampler and return its handle.

        The handle is a disk path when each version is a separate snapshot, or
        ``None`` when the sampler's single live copy is overwritten in place.
        """

    def __call__(self, admission: Any) -> str | None:
        return self.publish(admission)


class AdapterSnapshotSync(WeightSyncStrategy):
    """Save each trained version as a LoRA adapter snapshot on disk.

    ``save_fn`` is the existing disk-save seam (``pipeline._save_adapter``): it
    writes the adapter and returns its non-empty checkpoint path, which the
    control plane then pins by reference count.  Because each version lives at
    its own path, several can be resident at once, so this strategy does not
    bound ``max_staleness``.
    """

    def __init__(self, save_fn: Callable[[Any], str]):
        if not callable(save_fn):
            raise TypeError(f'AdapterSnapshotSync needs a callable save_fn, got {type(save_fn).__name__}')
        self._save_fn = save_fn

    def publish(self, admission: Any) -> str:
        return self._save_fn(admission)


class InPlaceWeightSync(WeightSyncStrategy):
    """同步 sampler 的当前权重；pause/drain 与显式续接由 driver 管理。

    sync_fn 完成后才确认发布成功；策略不控制生成请求，也不自动恢复失败发布。
    """

    def __init__(self, sync_fn: Callable[[], None]):
        if not callable(sync_fn):
            raise TypeError(f'InPlaceWeightSync needs a callable sync_fn, got {type(sync_fn).__name__}')
        self._sync_fn = sync_fn

    def publish(self, admission: Any) -> None:
        result = self._sync_fn()
        if callable(getattr(result, 'result', None)):
            result.result()
        return None


def build_weight_sync_strategy(
    name: str,
    *,
    save_fn: Callable[[Any], str] | None = None,
    sync_fn: Callable[[], None] | None = None,
) -> WeightSyncStrategy:
    """Build the weight-sync strategy selected by ``name``.

    ``save_fn`` is required for ``adapter_snapshot`` and ``sync_fn`` for
    ``in_place``.  driver 在调用策略前完成真实 pause/drain。There is no separate
    ``nccl`` strategy: NCCL is the transport ``in_place`` uses when the sampler is
    disaggregated, chosen by the deployment (``CheckpointEngineManager`` mode), not
    by this name.
    """
    if name == 'adapter_snapshot':
        if save_fn is None:
            raise ValueError("weight_sync_strategy='adapter_snapshot' requires a save_fn")
        return AdapterSnapshotSync(save_fn)
    if name == 'in_place':
        if sync_fn is None:
            raise ValueError("weight_sync_strategy='in_place' requires a sync_fn")
        return InPlaceWeightSync(sync_fn)
    if name == 'nccl':
        raise ValueError(
            "weight_sync_strategy='nccl' is not a separate strategy: NCCL is how 'in_place' transports "
            "weights to a disaggregated sampler (CheckpointEngineManager mode='standalone'). Use 'in_place' "
            "and select colocate (IPC) vs standalone (NCCL) via the deployment setting.")
    raise ValueError(f'unknown weight_sync_strategy: {name!r}')
