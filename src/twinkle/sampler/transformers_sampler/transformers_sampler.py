# Copyright (c) ModelScope Contributors. All rights reserved.
"""A :class:`~twinkle.sampler.Sampler` on plain transformers ``generate()``.

Drop-in for :class:`~twinkle.sampler.vLLMSampler`: same constructor shape
(``model_id, engine_args, device_mesh``), same ``sample`` / ``sample_stream`` / ``shutdown`` surface,
same synchronous-blocking convention. Swapping backends should not require touching call sites.

Two capabilities exist here that the vLLM path does not have:

- ``strict=False`` degrades **per input** instead of failing the batch. Offline sampling over a large
  corpus routinely hits a handful of rows that cannot be encoded (over-length, broken media refs); the
  legacy sampler set ``engine.strict = False`` for exactly this, and losing it would mean one bad row
  discards a whole run's worth of work.
- batching is explicit. transformers has no continuous batching, so inputs are grouped and padded up
  front by the engine's ``batch_sample``.

Not provided, deliberately: colocated weight sync. ``vLLMSampler`` mixes in ``CheckpointEngineMixin``
to receive NCCL-broadcast weights mid-training; this backend is for inference, evaluation and offline
sampling, and a half-working ``receive_weights`` would be worse than an absent one. ``update_weights``
on the engine covers the simple in-process refresh.

And not needed, in the one case where it would otherwise be: passing ``model=<TransformersModel>``
instead of ``model_id`` makes this a facade over a model that already exists, generating inside its
workers. Nothing is loaded and nothing is synced, because there is only ever one set of weights.
"""
from __future__ import annotations

import asyncio
import atexit
import threading
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Type, Union

from twinkle import DeviceMesh, get_logger, remote_class, remote_function, requires
from twinkle.data_format import (InputFeature, SampleResponse, SamplingParams, Trajectory,
                                 prepare_sampling_call, request_identity)
from twinkle.data_format.trajectory import trajectory_sticky_affinity
from twinkle.hub import HubOperation
from twinkle.patch import Patch, apply_patch
from twinkle.perf import timed
from twinkle.sampler.base import Sampler

logger = get_logger()


@remote_class()
class TransformersSampler(Sampler):
    """Sample with ``AutoModelForCausalLM.generate()``."""

    def __init__(self,
                 model_id: str = None,
                 engine_args: Dict[str, Any] = None,
                 device_mesh: DeviceMesh = None,
                 *,
                 model: Any = None,
                 **kwargs):
        """
        Args:
            model_id: HuggingFace model id or local path. Mutually exclusive with ``model``.
            engine_args: Forwarded to :class:`~twinkle.sampler.TransformersEngine` -- ``dtype``,
                ``device_map``, ``max_model_len``, ``attn_implementation``, ``trust_remote_code``, plus
                anything ``from_pretrained`` accepts. Also accepts ``max_batch_size`` (see below).
                Note that its ``model`` entry is a different thing from the ``model`` argument here: it
                is a bare ``nn.Module`` for the engine to generate with in this process, whereas ``model``
                below is a ``TransformersModel`` whose workers do the generating. Both avoid a reload;
                the first is what ``TransformersModel.generate`` uses internally.
            device_mesh: Data-parallel layout. Slicing is done by ``dispatch='slice_dp'`` on
                :meth:`sample`, so each rank only ever sees its own shard.
            model: A ``TransformersModel`` to generate from instead of loading one. Sampling is then
                done by its workers, on the weights they already hold -- no second copy of the model,
                and no weight sync, which is what makes this worth having during training. Construct
                this facade on the driver without a ``remote_group``: it holds no weights itself, so it
                needs no workers of its own, and the model's ``generate`` does the data-rank slicing.
                ``engine_args`` and ``device_mesh`` belong to the model in this mode and are ignored.
            **kwargs: Accepted and ignored, for signature parity with the other samplers.
        """
        super().__init__()
        requires('transformers')

        if (model_id is None) == (model is None):
            raise ValueError('Pass either model_id, to load a model to sample from, or model, to sample '
                             'from one that already exists -- not both and not neither.')
        self.model = model
        self.model_id = model_id
        self.device_mesh = device_mesh
        engine_kwargs = dict(engine_args or {})
        # How many prompts go into one padded generate(). Larger is faster until the padding waste
        # from mixed prompt lengths outweighs it, and until activations stop fitting.
        self.max_batch_size = engine_kwargs.pop('max_batch_size', 8)

        # Mirror the other samplers: a private loop on a background thread, so the public methods stay
        # synchronous and blocking even when the caller already runs inside an event loop (Ray workers
        # run uvloop, which forbids asyncio.run).
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name='TransformersSampler-EventLoop')
        self._async_thread.start()

        if model is not None:
            # Nothing to build: generation happens in the model's workers.
            self.engine = None
        else:
            from .transformers_engine import TransformersEngine
            self.engine: TransformersEngine = TransformersEngine(model_id, **engine_kwargs)

        self._shutdown_called = False
        atexit.register(self.shutdown)

    def _run_event_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _run_in_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._async_loop).result()

    def _iter_in_loop(self, async_gen):
        """Drain an async generator running on the background loop into a plain iterator."""
        import queue as stdlib_queue
        q: stdlib_queue.Queue = stdlib_queue.Queue()
        sentinel = object()

        async def _drain():
            try:
                async for item in async_gen:
                    q.put(item)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(sentinel)

        asyncio.run_coroutine_threadsafe(_drain(), self._async_loop)
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True)
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        apply_patch(self, patch_cls, **kwargs)

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True, replay=True)
    def set_template(self, template_cls, **kwargs):
        """Set the template that turns messages into token ids.

        Handed to the model in facade mode, because that is where encoding happens. It would not work
        here in any case: the base implementation defaults ``model_id`` from this object, and a facade
        has none.
        """
        if self.model is not None:
            return self.model.set_template(template_cls, **kwargs)
        return super().set_template(template_cls, **kwargs)

    # ------------------------------------------------------------------ sampling

    @remote_function(dispatch='slice_dp', collect='flatten', lazy_collect=False, enable_continous_work=True,
                     batch_arg='inputs', aligned_args=('adapter_paths',), prepare=prepare_sampling_call,
                     sticky=True, sticky_key=partial(trajectory_sticky_affinity, argument='inputs.0'))
    @timed
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        return_encoded: bool = False,
        use_base_model: bool = False,
        adapter_paths: Optional[List[Optional[str]]] = None,
        strict: bool = True,
        token_only: bool = False,
        **generation_kwargs,
    ) -> List[SampleResponse]:
        """Sample completions for ``inputs``.

        Args:
            inputs: ``InputFeature``(s) carrying ``input_ids``, or ``Trajectory``(s) carrying
                ``messages`` (which then requires ``set_template`` to have been called).
            sampling_params: ``max_tokens == 0`` means "score, do not generate", matching the vLLM
                backend: one token is generated and discarded so the prompt gets its logprobs.
            adapter_name: Name for the adapter loaded from ``adapter_path``.
            adapter_path: peft adapter to apply to every input in this call.
            adapter_paths: Per-input adapters, one entry per input. Accepted for parity with
                ``vLLMSampler``, but served differently: peft activates one adapter at a time, so
                inputs are grouped by adapter and each group gets its own batch. Correct, and slower
                than vLLM in proportion to the number of distinct adapters -- there is no way to mix
                adapters inside a single transformers forward. Mutually exclusive with ``adapter_path``.
            use_base_model: Ignore any loaded adapter for this call.
            strict: When False, an input that fails to encode or generate yields an empty
                ``SampleResponse`` with ``stop_reason='error'`` instead of aborting the batch, so one
                bad row cannot discard a long offline run. Positions are preserved: the result list
                always lines up with ``inputs``.

        Returns:
            One ``SampleResponse`` per input, each holding ``num_samples`` sequences.
        """
        return self._run_in_loop(self._generate_inputs(
            inputs, sampling_params, adapter_name, adapter_path, return_encoded=return_encoded,
            use_base_model=use_base_model, adapter_paths=adapter_paths, strict=strict,
            token_only=token_only, **generation_kwargs))

    async def _generate_inputs(self, inputs, sampling_params=None, adapter_name='', adapter_path=None, *,
                               return_encoded=False, use_base_model=False, adapter_paths=None, strict=True,
                               token_only=False, request_ids=None, **generation_kwargs):
        inputs = deepcopy(self._normalize_inputs(inputs))
        params = self._params(sampling_params)
        if not inputs:
            return []
        if request_ids is None:
            request_ids = [self._engine_request_id(request_identity(item), 0) for item in inputs]
        if len(request_ids) != len(inputs):
            raise ValueError('request_ids 与 inputs 必须等长。')
        with self._sampling_lock:
            owners = [self._request_tasks.get(rid) for rid in request_ids]
            stopped = [rid for rid, owner in zip(request_ids, owners) if owner is not None and owner.command is not None]
            if self.engine is not None:
                self.engine.register_requests(request_ids, stopped=stopped)
            for rid, owner in zip(request_ids, owners):
                if owner is not None:
                    owner.active_ids.add(rid)
        try:
            if self.model is not None:
                generation = asyncio.create_task(asyncio.to_thread(
                    self.model.generate, inputs, sampling_params=params, adapter_name=adapter_name,
                    adapter_path=adapter_path, return_encoded=return_encoded, use_base_model=use_base_model,
                    adapter_paths=adapter_paths, strict=strict, token_only=token_only, request_ids=request_ids,
                    stopped_mask=[rid in stopped for rid in request_ids], **generation_kwargs))
                try:
                    return await asyncio.shield(generation)
                except BaseException:
                    async def cleanup():
                        if not generation.done():
                            await asyncio.to_thread(self.model.stop_generation, request_ids)
                        await asyncio.gather(generation, return_exceptions=True)
                    cleanup_task = asyncio.create_task(cleanup())
                    while not cleanup_task.done():
                        try:
                            await asyncio.shield(cleanup_task)
                        except asyncio.CancelledError:
                            continue
                    cleanup_task.result()
                    raise
            return await self.engine.sample_inputs(
                inputs, params, adapter_name, adapter_path, template=self.template,
                return_encoded=return_encoded, use_base_model=use_base_model, adapter_paths=adapter_paths,
                strict=strict, token_only=token_only, request_ids=request_ids, registered=True,
                max_batch_size=self.max_batch_size, **generation_kwargs)
        finally:
            with self._sampling_lock:
                for rid, owner in zip(request_ids, owners):
                    if owner is not None:
                        owner.active_ids.discard(rid)

    async def _abort_request(self, request_id):
        if self.model is not None:
            await asyncio.to_thread(self.model.pause_generation, [request_id])
        else:
            await self.engine.abort_request(request_id)

    def sample_stream(
        self,
        inputs: Union[InputFeature, Trajectory, Dict[str, Any]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
    ):
        """Yield ``(delta_text, finish_reason)`` tuples for a single input."""
        if self.model is not None:
            yield from self.model.generate_stream(
                inputs, sampling_params=sampling_params, adapter_name=adapter_name, adapter_path=adapter_path)
            return
        params = self._coerce_params(sampling_params)
        feat = inputs if not self._not_encoded(inputs) else self.encode_trajectory(inputs, adapter_name)
        adapter_uri = self._resolve_adapter(adapter_path, adapter_name, use_base_model=False)
        prompt = self.template.get_vllm_input_ids(feat['input_ids']) if self.template else feat['input_ids']
        yield from self._iter_in_loop(
            self.engine.generate_stream(prompt=prompt, sampling_params=params, lora_request=adapter_uri))

    # ------------------------------------------------------------------ engine passthrough

    def _own_engine(self):
        """The engine this sampler loaded, or a clear failure when it is a facade over a model.

        A facade has no memory of its own to control, and quietly doing nothing would be worse: a caller
        putting a sampler to sleep to free the device would believe it had.
        """
        if self.engine is None:
            raise RuntimeError('This sampler generates through the model it was given, which owns those '
                               'weights and that memory; there is no engine here to control. Call the '
                               'corresponding method on the model instead.')
        return self.engine

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True)
    def get_state_keys(self):
        return self._run_in_loop(self._own_engine().get_state_keys())

    @remote_function(dispatch='all', collect='first', exclusive=True)
    @timed
    def sleep(self, level: int = 1) -> None:
        self._run_in_loop(self._own_engine().sleep(level=level))

    @remote_function(dispatch='all', collect='first', exclusive=True)
    @timed
    def wake_up(self, tags: List[str] = None) -> None:
        self._run_in_loop(self._own_engine().wake_up(tags=tags))

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True)
    def add_adapter_to_sampler(self, adapter_name: str, config: Any) -> None:
        raise NotImplementedError('TransformersSampler loads adapters from disk; pass adapter_path to sample().')

    @remote_function(dispatch='all', collect='first', lazy_collect=False, control=True, shutdown=True)
    def shutdown(self):
        """Release the model and stop the background loop. Idempotent; also runs via ``atexit``."""
        if self._shutdown_called:
            return
        self._close_generations()
        if self.engine is not None:
            self._run_in_loop(self.engine.shutdown())
        self._shutdown_called = True
        if self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_thread.is_alive():
            self._async_thread.join(timeout=5)
        if self._async_thread.is_alive():
            self._shutdown_called = False
            raise RuntimeError('TransformersSampler 事件循环尚未退出。')

    # ------------------------------------------------------------------ internals

    def _coerce_params(self, sampling_params) -> SamplingParams:
        return self._params(sampling_params)

    def _resolve_adapter(self, adapter_path: Optional[str], adapter_name: str, use_base_model: bool) -> Optional[str]:
        if use_base_model or adapter_path is None:
            return None
        local_path = HubOperation.download_model(model_id_or_path=adapter_path)
        return self.engine.load_lora(local_path, adapter_name or None)


