# Copyright (c) ModelScope Contributors. All rights reserved.

from twinkle import Platform, remote_function
from twinkle.checkpoint_engine.base import CheckpointEngine


class CheckpointEngineMixin:

    _checkpoint_engine: CheckpointEngine = None
    _checkpoint_engines = None
    _checkpoint_engine_key = None
    _bucket_size: int = 3072 << 20  # 2 GB
    # Set by CheckpointEngineManager when the roles share a GPU; see set_checkpoint_engine_backend.
    _checkpoint_engine_backend: str = None
    # Per-channel HMAC keys, delivered by the manager over the (trusted) Ray control path.
    _checkpoint_engine_auth_keys = None

    def _select_checkpoint_engine(self, backend: str, channel: str, auth_key: str = '') -> None:
        if not isinstance(channel, str) or not channel:
            raise ValueError(f'checkpoint channel must be a non-empty string, got {channel!r}.')
        if backend not in {'ipc', 'nccl', 'hccl'}:
            raise ValueError(f'checkpoint backend must be ipc/nccl/hccl, got {backend!r}.')
        self._checkpoint_engine_backend = backend
        self._checkpoint_engine_key = (backend, channel)
        if auth_key:
            if self._checkpoint_engine_auth_keys is None:
                self._checkpoint_engine_auth_keys = {}
            self._checkpoint_engine_auth_keys[(backend, channel)] = auth_key
        engines = getattr(self, '_checkpoint_engines', None)
        self._checkpoint_engine = engines.get(self._checkpoint_engine_key) if engines else None

    @remote_function(dispatch='all', lazy_collect=False)
    def activate_checkpoint_engine(self, backend: str, channel: str = 'default', auth_key: str = ''):
        """Select one persistent transport channel without discarding the other channels.

        ``auth_key`` signs the engine's control frames (IPC) over the unauthenticated socket path;
        it travels through the Ray control path, which the deployment already trusts.
        """
        self._select_checkpoint_engine(backend, channel, auth_key)

    @remote_function(dispatch='all', lazy_collect=False)
    def set_checkpoint_engine_backend(self, backend: str):
        """Choose the default transport channel for backward-compatible callers."""
        self._select_checkpoint_engine(backend, 'default')

    def _get_or_create_checkpoint_engine(self) -> 'CheckpointEngine':
        """Get or create the engine for the currently active transport channel."""
        backend = self._checkpoint_engine_backend
        if backend is None:
            backend = 'nccl' if Platform.get_platform().__name__ == 'GPU' else 'hccl'
        key = self._checkpoint_engine_key or (backend, 'default')
        engines = getattr(self, '_checkpoint_engines', None)
        if engines is None:
            engines = {}
            self._checkpoint_engines = engines
        engine = engines.get(key)
        if engine is None:
            if backend == 'ipc':
                # Colocated: the sampler is on this GPU, so weights are mapped, not broadcast. NCCL
                # is not an option here -- it refuses two ranks on one device. The channel reaches
                # the engine so the endpoint separates colocate managers sharing a GPU pair; the
                # bucket only passes when the class default was overridden, because the IPC default
                # (512 MB) is deliberately smaller than this mixin's NCCL-tuned 3 GB.
                from twinkle.checkpoint_engine import IPCCheckpointEngine
                bucket = None if type(self)._bucket_size == CheckpointEngineMixin._bucket_size else self._bucket_size
                auth_keys = self._checkpoint_engine_auth_keys or {}
                engine = IPCCheckpointEngine(bucket, channel=key[1], auth_key=auth_keys.get(key, ''))
            elif backend == 'nccl':
                from twinkle.checkpoint_engine import NCCLCheckpointEngine
                engine = NCCLCheckpointEngine(self._bucket_size)
            elif backend == 'hccl':
                from twinkle.checkpoint_engine import HCCLCheckpointEngine

                # Reusing HCCL communicator across sync steps avoids frequent
                # stream/channel allocation and reduces resource exhaustion risk.
                engine = HCCLCheckpointEngine(
                    self._bucket_size,
                    rebuild_group=False,
                )
            engines[key] = engine
        self._checkpoint_engine_key = key
        self._checkpoint_engine = engine
        return engine

    @remote_function(collect='first', lazy_collect=False)
    def prepare_checkpoint_engine(self, is_master):
        engine = self._get_or_create_checkpoint_engine()
        engine.is_master = is_master
        return engine.prepare()

    @remote_function(dispatch='slice', lazy_collect=True)
    def init_checkpoint_process_group(self, rank: int, world_size: int, master_metadata):
        """Initialize process group for weight synchronization."""
        if isinstance(rank, list):
            assert len(rank) == 1
            rank = rank[0]
        if isinstance(world_size, list):
            assert len(world_size) == 1
            world_size = world_size[0]
        if isinstance(master_metadata, list):
            assert len(master_metadata) == 1
            master_metadata = master_metadata[0]
        engine = self._get_or_create_checkpoint_engine()
        engine.init_process_group(
            rank=rank,
            world_size=world_size,
            master_metadata=master_metadata,
        )

    @remote_function(dispatch='all', lazy_collect=False)
    def finalize_checkpoint_engine(self):
        """Finalize the active engine while retaining a reusable communication group."""
        engine = self._get_or_create_checkpoint_engine()
        engine.finalize()

    @remote_function(dispatch='all', lazy_collect=False)
    def reset_checkpoint_engine(self, channel: str = 'default', backend: str = None):
        """Destroy selected persistent channels when their remote peer is retired."""
        engines = getattr(self, '_checkpoint_engines', None) or {}
        keys = [key for key in engines if key[1] == channel and (backend is None or key[0] == backend)]
        for key in keys:
            engines.pop(key).close()
        if self._checkpoint_engine_key in keys:
            self._checkpoint_engine = None
            self._checkpoint_engine_key = None
