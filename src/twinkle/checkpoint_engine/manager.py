# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
import threading
from typing import List, Literal, Optional

from twinkle.infra import get_worker_id

from twinkle import Platform, get_logger
from .base import CheckpointEngine
from .mixin import CheckpointEngineMixin
from .zmq_auth import generate_auth_key

logger = get_logger()

CheckpointEngineMode = Literal['auto', 'naive', 'colocate', 'standalone']
_VALID_MODES = {'auto', 'naive', 'colocate', 'standalone'}


class CheckpointEngineManager:
    """Weight synchronization manager for local and Ray deployments.

    ``mode`` selects one of three synchronization paths:

    * ``naive`` streams a local model's weight generator directly into a local sampler.
    * ``colocate`` connects Ray model and sampler actors sharing GPUs through CUDA IPC.
    * ``standalone`` connects disaggregated Ray actors through NCCL/HCCL.

    ``auto`` resolves local objects to ``naive`` and Ray actor handlers to ``standalone``. It never
    guesses ``colocate`` because actor placement cannot be inferred reliably from the driver.

    Architecture (following verl's CheckpointEngineManager):

        Trainer GPU(s)                          Rollout GPU(s)
        ┌──────────────────┐                    ┌──────────────────┐
        │ TransformersModel│                    │   vLLMSampler    │
        │  (Ray actors)    │                    │  (Ray actors)    │
        │        │         │                    │        │         │
        │        ▼         │                    │        ▼         │
        │ CheckpointEngine │ NCCL/HCCL/CUDA IPC │ CheckpointEngine │
        │  send_weights()  │ ─────────────────► │ receive_weights()│
        │                  │                    │        │         │
        │                  │                    │        ▼         │
        │                  │                    │ VLLMEngine       │
        │                  │                    │  update_weights()│
        │                  │                    │   (CUDA IPC)     │
        │                  │                    │        │         │
        │                  │                    │        ▼         │
        │                  │                    │ vLLM subprocess  │
        │                  │                    │  load_weights()  │
        └──────────────────┘                    └──────────────────┘

    Usage:
        >>> manager = CheckpointEngineManager(model=model, sampler=sampler)
        >>> manager.sync_weights()  # Call after each training step

    With colocated Ray actors, the caller also owns the memory schedule, because only it knows where
    in the loop the device is free. The sampler must have its weights resident to be written into --
    ``sleep(1)`` puts them on the host -- and the trainer has to step aside before a rollout:

        >>> manager = CheckpointEngineManager(model=model, sampler=sampler, mode='colocate')
        >>> sampler.wake_up(tags=['weights'])   # able to receive, still without a KV cache
        >>> manager.sync_weights()
        >>> model.offload_to_cpu()              # the trainer's turn is over
        >>> sampler.wake_up()                   # KV cache, then generate
        >>> ...
        >>> sampler.sleep()                     # and back the other way
        >>> model.reload_to_gpu()

    This is deliberately not done for the caller: waking a sampler that is already awake is not a
    no-op in vLLM, so guessing the current state here would trade one foot-gun for another.
    """

    def __init__(
        self,
        model: 'CheckpointEngineMixin',
        sampler: 'CheckpointEngineMixin',
        platform: str = 'GPU',
        mode: CheckpointEngineMode = 'auto',
        channel: str = 'default',
        colocate_trainer_ranks: Optional[List[int]] = None,
        receiver_worker_ids: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.requested_mode = mode
        self.mode = self._resolve_mode(mode, model, sampler)
        self.backend_cls = self.decide_backend_engine(platform, self.mode)
        self._sync_lock = threading.RLock()
        self.receiver_worker_ids = self._validate_receiver_ids(receiver_worker_ids)
        self._receiver_snapshot = self.receiver_worker_ids
        self._channel_closed = False
        if not isinstance(channel, str) or not channel:
            raise ValueError(f'checkpoint channel must be a non-empty string, got {channel!r}.')
        self.channel = channel
        if colocate_trainer_ranks is not None:
            if self.mode != 'colocate':
                raise ValueError('colocate_trainer_ranks is valid only for mode="colocate".')
            colocate_trainer_ranks = self._validate_colocate_ranks(colocate_trainer_ranks, self.receiver_worker_ids)
        elif self.mode == 'colocate' and self.receiver_worker_ids is not None:
            raise ValueError(
                'mode="colocate" with an explicit receiver_worker_ids subset also requires '
                'colocate_trainer_ranks: an arbitrary subset cannot be paired with trainer ranks by '
                'the uniform default topology, and a mismatched pair would hang on the first bucket.')
        self.colocate_trainer_ranks = (None if colocate_trainer_ranks is None
                                       else tuple(colocate_trainer_ranks))
        self.backend = None if self.mode == 'naive' else self.backend_cls.backend_name
        # Signs the engine's control frames. Generated here and delivered to both sides through the
        # Ray control path, so a peer that guessed the endpoint path still cannot inject frames.
        self._auth_key = generate_auth_key()

        if self.mode != 'naive':
            try:
                self._activate_checkpoint_channel()
            except BaseException:
                try:
                    self.close()
                except Exception:
                    logger.warning('Failed to fully close checkpoint channel %s after initialization failure.',
                                   channel, exc_info=True)
                raise

        # LoRA sync state: tracks whether the first full sync has been done.
        # After the first sync, only LoRA adapter weights are transferred.
        self.base_sync_done: bool = False
        # Cached peft_config dict for LoRA-only sync.
        # Fetched lazily from the model on first LoRA sync.
        self._peft_config: dict | None = None
        self._model_keys: Optional[List[str]] = None

    @staticmethod
    def _resolve_mode(
        mode: CheckpointEngineMode,
        model: 'CheckpointEngineMixin',
        sampler: 'CheckpointEngineMixin',
    ) -> Literal['naive', 'colocate', 'standalone']:
        if mode not in _VALID_MODES:
            valid = ', '.join(sorted(_VALID_MODES))
            raise ValueError(f'Unknown checkpoint engine mode {mode!r}; expected one of: {valid}.')

        model_has_actors = bool(getattr(model, '_actors', None))
        sampler_has_actors = bool(getattr(sampler, '_actors', None))
        if model_has_actors != sampler_has_actors:
            raise ValueError(
                'CheckpointEngineManager requires model and sampler to use the same deployment shape: '
                'both must be local objects or both must be Ray actor handlers.')

        if mode == 'auto':
            return 'standalone' if model_has_actors else 'naive'
        if mode == 'naive' and model_has_actors:
            raise ValueError("mode='naive' requires local model and sampler objects without Ray actors.")
        if mode in ('colocate', 'standalone') and not model_has_actors:
            raise ValueError(f"mode={mode!r} requires model and sampler to be Ray actor handlers.")
        return mode

    @staticmethod
    def decide_backend_engine(
        platform: Optional[str] = None,
        mode: Literal['naive', 'colocate', 'standalone'] = 'standalone',
    ) -> Optional['CheckpointEngine']:
        if mode == 'naive':
            return None

        platform_name = Platform.get_platform(platform).__name__
        if mode == 'colocate':
            from twinkle.checkpoint_engine import IPCCheckpointEngine
            return IPCCheckpointEngine
        if mode != 'standalone':
            raise ValueError(f'Cannot select a backend for unresolved mode {mode!r}.')
        if platform_name == 'GPU':
            from twinkle.checkpoint_engine import NCCLCheckpointEngine
            return NCCLCheckpointEngine
        elif platform_name == 'NPU':
            from twinkle.checkpoint_engine import HCCLCheckpointEngine
            return HCCLCheckpointEngine
        else:
            raise NotImplementedError

    def _validate_receiver_ids(self, worker_ids):
        if worker_ids is None:
            return None
        if self.mode == 'naive':
            raise ValueError('receiver_worker_ids requires Ray checkpoint receivers.')
        worker_ids = tuple(worker_ids)
        if any(not isinstance(worker_id, str) for worker_id in worker_ids):
            raise ValueError('receiver_worker_ids must contain stable actor ID strings.')
        known = {get_worker_id(actor) for actor in self.sampler._actors}
        if len(set(worker_ids)) != len(worker_ids) or set(worker_ids) - known:
            raise ValueError(f'receiver_worker_ids contains duplicate or unknown actor IDs: {worker_ids!r}.')
        return worker_ids

    def _validate_colocate_ranks(self, ranks, worker_ids):
        ranks = tuple(ranks)
        count = self.sampler.device_mesh.data_world_size if worker_ids is None else len(worker_ids)
        if any(type(rank) is not int or not 0 <= rank < self.model.device_mesh.world_size for rank in ranks):
            raise ValueError('Each IPC receiver requires a physical trainer sender rank within the trainer world.')
        if len(ranks) != count or len(set(ranks)) != len(ranks):
            raise ValueError('Each IPC receiver requires one unique physical trainer sender rank.')
        return ranks

    def set_receiver_workers(self, worker_ids, *, colocate_trainer_ranks=None):
        """Register the receiver set for the next sync; communication changes are applied
        lazily, only inside sync_weights."""
        with self._sync_lock:
            worker_ids = self._validate_receiver_ids(worker_ids)
            if colocate_trainer_ranks is not None:
                if self.mode != 'colocate':
                    raise ValueError('colocate_trainer_ranks requires mode="colocate".')
                self.colocate_trainer_ranks = self._validate_colocate_ranks(colocate_trainer_ranks, worker_ids)
            elif self.mode == 'colocate':
                if worker_ids is not None and worker_ids != self.receiver_worker_ids:
                    # A changed receiver subset with stale ranks pairs receivers with trainers that
                    # share no GPU; the mismatch would only surface -- as a hang -- inside sync.
                    raise ValueError(
                        'Changing receiver_worker_ids in mode="colocate" requires colocate_trainer_ranks '
                        'for the new set; the previous ranks no longer describe the pairing.')
                if worker_ids is None:
                    # Back to the full receiver set: the uniform default topology applies again.
                    self.colocate_trainer_ranks = None
            self.receiver_worker_ids = worker_ids

    def _sampler_call(self, method, *args, **kwargs):
        if self._receiver_snapshot is not None:
            kwargs['_worker_ids'] = self._receiver_snapshot
        return getattr(self.sampler, method)(*args, **kwargs)

    def _activate_checkpoint_channel(self) -> None:
        if self._channel_closed:
            self._receiver_snapshot = self.receiver_worker_ids
        self._channel_closed = False
        self.model.activate_checkpoint_engine(self.backend, self.channel, auth_key=self._auth_key)
        self._sampler_call('activate_checkpoint_engine', self.backend, self.channel, auth_key=self._auth_key)

    def _close_checkpoint_channel(self):
        if self.mode != 'naive':
            try:
                self.model.reset_checkpoint_engine(self.channel, self.backend)
            finally:
                self._sampler_call('reset_checkpoint_engine', self.channel, self.backend)

    def close(self) -> None:
        """Tear down this channel, cleaning only the receiver snapshot that the last sync used."""
        with self._sync_lock:
            if self._channel_closed:
                return
            try:
                self._close_checkpoint_channel()
            finally:
                self._receiver_snapshot = ()
                self._channel_closed = True
                self.base_sync_done = False

    def sync_weights(self, merge_and_sync=True):
        with self._sync_lock:
            if self._channel_closed or self.receiver_worker_ids != self._receiver_snapshot:
                self.close()
                self._receiver_snapshot = self.receiver_worker_ids
                self.base_sync_done = False
            if self._receiver_snapshot == ():
                return
            try:
                self._sync_weights(merge_and_sync)
            except BaseException:
                # A failed sync can leave the transport half-torn -- a NCCL group in error state, a
                # dead socket -- and the next sync only rebuilds on membership change. Close here so
                # the next attempt rebuilds the channel instead of reusing the broken one.
                self.close()
                raise

    def _sync_weights(self, merge_and_sync=True):
        """
        Synchronize the weights between the model and the sampler.

        This method ensures that the sampler's weights are consistent with the model's
        current state. It supports two synchronization modes: full merge-and-sync or
        separate base-and-LoRA sync.

        Args:
            merge_and_sync (bool, optional): Whether to merge and sync the weights.
                - If True: LoRA weights are merged into the base model, then the
                combined weights are synchronized to the sampler on every call.
                - If False: On the first call, base model weights are synced to the
                sampler. On subsequent calls, only the LoRA adapter weights are
                synced incrementally.
                Defaults to True.

        Returns:
            None
        """
        if self.mode == 'naive':
            self._sync_weights_naive(merge_and_sync)
            return

        self._activate_checkpoint_channel()
        model_metadata = self.model.prepare_checkpoint_engine([True]
                                                              + [False] * (self.model.device_mesh.world_size - 1))
        self._sampler_call('prepare_checkpoint_engine', False)
        model_world_size = self.model.device_mesh.world_size
        sampler_world_size = (self.sampler.device_mesh.data_world_size if self._receiver_snapshot is None
                              else len(self._receiver_snapshot))
        topology_kwargs = {}
        if self.colocate_trainer_ranks is not None:
            topology_kwargs['sender_ranks'] = self.colocate_trainer_ranks
        model_kwargs, sampler_kwargs = self.backend_cls.build_topology(
            model_world_size,
            sampler_world_size,
            [model_metadata],
            **topology_kwargs,
        )
        # Launch both init calls concurrently — TCPStore server (model rank 0)
        # blocks until all clients (sampler ranks) connect, so these MUST NOT
        # be serialised.  lazy_collect=True makes them return futures.
        model_init = self.model.init_checkpoint_process_group(**model_kwargs)
        sampler_init = self._sampler_call('init_checkpoint_process_group', **sampler_kwargs)
        model_init()  # wait for model init to complete
        sampler_init()  # wait for sampler init to complete

        peft_config = None
        if self.base_sync_done and not merge_and_sync:
            if self._peft_config is None:
                self._peft_config = self.model.get_peft_config_dict()
            peft_config = self._peft_config

        self._ensure_model_keys()

        model_result = self.model.send_weights(
            base_sync_done=self.base_sync_done, merge_and_sync=merge_and_sync, model_keys=self._model_keys)
        sampler_result = self._sampler_call('receive_weights', base_sync_done=self.base_sync_done,
                                           peft_config=peft_config)
        model_result()
        sampler_result()

        self.model.finalize_checkpoint_engine()
        self._sampler_call('finalize_checkpoint_engine')

        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')

    def _ensure_model_keys(self):
        if self._model_keys is not None:
            return

        if hasattr(self.sampler, 'get_state_keys'):
            self._model_keys = self._sampler_call('get_state_keys')

        if self._model_keys is None:
            self._model_keys = []

        # vLLM may have grouped params - use word boundaries to avoid substring matches
        import re
        _STACKED_MAPPINGS = [
            (re.compile(r'\bqkv_proj\b'), ('q_proj', 'k_proj', 'v_proj', 'q', 'k', 'v')),
            (re.compile(r'\bgate_up_proj\b'), ('gate_proj', 'up_proj')),
            (re.compile(r'\bin_proj_qkvz\b'), ('in_proj_qkv', 'in_proj_z')),
            (re.compile(r'\bin_proj_ba\b'), ('in_proj_b', 'in_proj_a')),
            (re.compile(r'\blanguage_model\.model\b'), ('model.language_model', )),
            (re.compile(r'^visual\.'), ('model.visual.', )),
        ]

        def _expand_keys(keys):
            result = set(keys)
            for key in keys:
                for pattern, individuals in _STACKED_MAPPINGS:
                    if pattern.search(key):
                        for ind in individuals:
                            result.add(pattern.sub(ind, key))
            return result

        # Two passes for chain expansion (e.g., language_model.model + qkv_proj)
        expanded = _expand_keys(self._model_keys)
        expanded = _expand_keys(expanded)
        self._model_keys = list(expanded)

    def _sync_weights_naive(self, merge_and_sync):
        """Stream model weights directly into a local sampler."""
        peft_config = None
        if self.base_sync_done and not merge_and_sync:
            if self._peft_config is None:
                self._peft_config = self.model.get_peft_config_dict()
            peft_config = self._peft_config

        self._ensure_model_keys()
        weights = self.model._get_weight_generator(
            base_sync_done=self.base_sync_done,
            merge_and_sync=merge_and_sync,
            model_keys=self._model_keys,
        )
        self.sampler.receive_weights(
            weights=weights,
            base_sync_done=self.base_sync_done,
            peft_config=peft_config,
        )

        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')

    # ── Paired EMA sync (policy → reference) ────────────────────────────────

    def sync_ema(self, alpha: float):
        """Synchronize reference weights as an EMA of policy weights.

        Each policy rank k and reference rank k form an independent 2-rank NCCL group:
        policy sends its LOCAL shard parameters, reference receives and applies
        ``param.mul_(alpha).add_(received, alpha=1-alpha)``.  Peak memory per rank is two engine
        buckets -- the engine allocates its staging buffers before the send/receive roles split,
        so the unused direction's buffer is paid for too (configurable, default 2x3 GB),
        independent of model size.

        Requires policy and reference to have the SAME parallel_spec (identical sharding),
        so local parameter names and shapes match one-to-one across the pair.

        The receiver side is always the full reference world: this method routes every remote
        call through :meth:`_sampler_call`, so a manager constructed with ``receiver_worker_ids``
        would filter them -- construct EMA managers without a receiver subset.

        Args:
            alpha: EMA decay factor.  ``ref = alpha * ref + (1 - alpha) * policy``.
                   Closer to 1 means slower tracking.
        """
        with self._sync_lock:
            if self.mode == 'naive':
                self._sync_ema_naive(alpha)
                return

            self._activate_checkpoint_channel()
            world_size = self.model.device_mesh.world_size
            ref_world_size = self.sampler.device_mesh.world_size
            if world_size != ref_world_size:
                raise ValueError(
                    f'Paired EMA requires policy and reference to have the same world_size '
                    f'(policy={world_size}, reference={ref_world_size}); use the same parallel_spec.')

            # Phase 1: all policy ranks prepare as master (each gets a unique TCPStore port).
            all_metadata = self.model.prepare_ema_sender()
            if not isinstance(all_metadata, list):
                all_metadata = [all_metadata]
            # Reference ranks prepare as non-master (allocate receive buffers).
            # Scalar False: dispatch='slice' broadcasts it identically to every rank.
            self._sampler_call('prepare_checkpoint_engine', False)

            # Phase 2: init paired 2-rank groups (policy rank k = NCCL rank 0, ref rank k = rank 1).
            model_init = self.model.init_checkpoint_process_group(
                rank=[0] * world_size,
                world_size=[2] * world_size,
                master_metadata=all_metadata,
            )
            sampler_init = self._sampler_call(
                'init_checkpoint_process_group',
                rank=[1] * world_size,
                world_size=[2] * world_size,
                master_metadata=all_metadata,
            )
            model_init()
            sampler_init()

            # Phase 3: concurrent send (local shard) / receive+EMA.
            send_future = self.model.send_weights(local=True)
            recv_future = self._sampler_call('receive_weights_ema', alpha)
            send_future()
            recv_future()

            # Phase 4: finalize.
            self.model.finalize_checkpoint_engine()
            self._sampler_call('finalize_checkpoint_engine')

    def _sync_ema_naive(self, alpha: float):
        """In-process EMA: iterate policy's local params and blend into reference's."""
        import torch
        policy_state = dict(self.model._get_local_state_dict())
        ref_state = dict(self.sampler._get_local_state_dict())
        with torch.no_grad():
            for name, policy_param in policy_state.items():
                ref_param = ref_state[name]
                ref_param.data.mul_(alpha).add_(policy_param.data, alpha=1.0 - alpha)
