# Copyright (c) ModelScope Contributors. All rights reserved.
"""A :class:`~twinkle.sampler.BaseSamplerEngine` backed by plain transformers ``generate()``.

Why this exists alongside vLLM and sglang: those two are the throughput backends, but they cannot
load every architecture (custom / brand-new / ``trust_remote_code`` models often land in transformers
first), they cost a separate weight copy in the KV-cache format, and they are heavyweight optional
dependencies. This engine trades throughput for reach -- it runs anything ``AutoModelForCausalLM``
loads, with no extra install.

Three things differ from the vLLM engine and callers do feel them:

- **no continuous batching.** transformers generates one padded batch at a time, so a batch runs at
  the pace of its longest sequence and concurrent ``sample()`` calls serialize behind
  :attr:`_generate_lock`. Batching is therefore done up front (see :meth:`batch_sample`) rather than
  by the engine scheduler.
- **left padding is mandatory.** Decoder-only generation appends to the end of the sequence, so pads
  must sit on the left or the model continues from padding. The tokenizer's own ``padding_side`` is
  overridden per call rather than mutated globally, because the same tokenizer object may be shared
  with a training path that needs right padding.
- **logprobs come from ``scores``, not from a KV-cache side channel.** ``output_scores=True`` returns
  raw logits per step; they are log-softmaxed here. Prompt logprobs need a separate forward pass
  (:meth:`_prompt_logprobs`) because ``generate()`` never scores the prompt.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle import get_logger, requires
from twinkle.data_format import InputFeature, SampledSequence, SampleResponse, SamplingParams
from twinkle.sampler.base_engine import BaseSamplerEngine
from twinkle.utils import torch_util

logger = get_logger()


def _resolve_model_cls(model_cls: Any):
    """The transformers class to ``from_pretrained`` with.

    Defaults to ``AutoModelForCausalLM`` (plain text generation). A vision-language checkpoint's
    config is not a causal-LM config, so ``AutoModelForCausalLM`` rejects it -- the caller supplies
    the family's real ``...ForConditionalGeneration`` class instead, either as the class object or as
    a string: ``'module:ClassName'`` (the form dev's model loaders declare, and safe to ship through
    Ray ``engine_args``) or a bare name resolved against ``transformers`` (twinkle's
    ``TransformersModel.model_cls`` convention).
    """
    from transformers import AutoModelForCausalLM
    if model_cls is None:
        return AutoModelForCausalLM
    if isinstance(model_cls, str):
        if ':' in model_cls:
            import importlib
            module_name, _, cls_name = model_cls.partition(':')
            return getattr(importlib.import_module(module_name), cls_name)
        import transformers
        return getattr(transformers, model_cls)
    return model_cls


@dataclass
class _GenerationRequest:
    stopped: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    finished: Optional[float] = None


class _RequestStoppingCriteria:
    """在 token 边界记录准确长度，原生停止条件仍由 generate 执行。"""

    def __init__(self, controls, prompt_width, eos_ids):
        self.controls = controls
        self.prompt_width = prompt_width
        self.eos_ids = set(eos_ids)
        self.lengths = {}
        self.aborted = set()
        self.abort_lengths = {}

    def __call__(self, input_ids, scores, **kwargs):
        rows = input_ids.shape[0]
        if rows % len(self.controls):
            raise RuntimeError('Transformers 的生成分支数与请求数不匹配。')
        branches = rows // len(self.controls)
        length = input_ids.shape[-1] - self.prompt_width
        done = torch.zeros(rows, dtype=torch.bool, device=input_ids.device)
        for row in range(rows):
            if row not in self.lengths:
                if length > 0 and int(input_ids[row, -1]) in self.eos_ids:
                    self.lengths[row] = length
                elif self.controls[row // branches].stopped.is_set():
                    self.lengths[row] = length
                    self.aborted.add(row)
                    self.abort_lengths.setdefault(row // branches, length)
            done[row] = row in self.aborted
        return done


class _RequestLogitsProcessor:
    """beam 路径用 EOS 结束被停止请求，不改变其他请求的 logits。"""

    def __init__(self, tracker, eos_id):
        self.tracker = tracker
        self.eos_id = eos_id

    def __call__(self, input_ids, scores):
        branches = input_ids.shape[0] // len(self.tracker.controls)
        for index, control in enumerate(self.tracker.controls):
            if control.stopped.is_set():
                self.tracker.abort_lengths.setdefault(index, input_ids.shape[-1] - self.tracker.prompt_width)
                start = index * branches
                scores[start:start + branches] = -float('inf')
                scores[start:start + branches, self.eos_id] = 0
        return scores


class TransformersEngine(BaseSamplerEngine):
    """Wrap a HF causal LM (or a vision-language model, via ``model_cls``) so it satisfies the
    sampler-engine contract."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: Any = None,
        device_map: Any = None,
        max_model_len: Optional[int] = None,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
        model_cls: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        **model_kwargs,
    ):
        """
        Args:
            model_id: HuggingFace model id or local path.
            dtype: Torch dtype (or its string name) for the weights. Defaults to ``'auto'``.
            device_map: Passed to ``from_pretrained``. Defaults to the current accelerator, i.e. a
                single-device load -- ``'auto'`` would shard the model across every visible GPU,
                which is wrong under data parallelism where each rank owns one device.
            max_model_len: Prompt+completion budget. Prompts longer than this are rejected rather
                than silently truncated, because a truncated prompt produces a plausible-looking but
                meaningless completion.
            attn_implementation: e.g. ``'flash_attention_2'``, ``'sdpa'``.
            trust_remote_code: Allow custom modelling code from the checkpoint.
            model_cls: The transformers class to load with (class object, ``'module:ClassName'`` or a
                bare ``transformers`` name). Defaults to ``AutoModelForCausalLM``; a vision-language
                checkpoint needs its own ``...ForConditionalGeneration`` class (see
                :func:`_resolve_model_cls`).
            model: A pre-built model, for callers that already loaded one. Skips ``from_pretrained``.
            tokenizer: A pre-built tokenizer/processor, likewise.
            **model_kwargs: Forwarded to ``from_pretrained``.
        """
        requires('transformers')
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.max_model_len = max_model_len
        # Serializes generate() calls: a single nn.Module cannot run two generate loops at once, and
        # the lock is what turns concurrent callers into a queue instead of a corrupted forward.
        self._generate_lock = asyncio.Lock()
        self._model_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._requests = OrderedDict()
        self._closed = False
        self._adapters: Dict[str, str] = {}
        self._active_adapter: Optional[str] = None

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, 'pad_token_id', None) is None:
            # Padding is unavoidable for batched generation; eos is the conventional stand-in and is
            # masked out anyway, so it never contributes to the logits that matter.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model is None:
            model_cls_resolved = _resolve_model_cls(model_cls)
            load_kwargs: Dict[str, Any] = {
                'dtype': dtype if dtype is not None else 'auto',
                'trust_remote_code': trust_remote_code,
                **model_kwargs,
            }
            if attn_implementation is not None:
                load_kwargs['attn_implementation'] = attn_implementation
            load_kwargs['device_map'] = device_map if device_map is not None else torch_util.get_device(None)
            logger.info(f'TransformersEngine loading {model_id} with {model_cls_resolved.__name__} and {load_kwargs}')
            model = model_cls_resolved.from_pretrained(model_id, **load_kwargs)
        self.model = model
        self.model.eval()

    def register_requests(self, request_ids, *, stopped=()):
        if any(not isinstance(rid, str) or not rid for rid in request_ids) or len(set(request_ids)) != len(request_ids):
            raise ValueError('引擎请求号必须为逐输入唯一的非空字符串。')
        with self._request_lock:
            now = time.monotonic()
            for rid, control in tuple(self._requests.items()):
                if control.finished is not None and (now - control.finished > 300
                                                     or len(self._requests) + len(request_ids) > 4096):
                    self._requests.pop(rid)
            if self._closed:
                raise RuntimeError('TransformersEngine 已关闭。')
            if any(rid in self._requests for rid in request_ids):
                raise ValueError('重复的 Transformers 引擎请求号。')
            if len(self._requests) + len(request_ids) > 4096:
                raise RuntimeError('Transformers 请求保留容量已满。')
            for rid in request_ids:
                control = _GenerationRequest()
                if rid in stopped:
                    control.stopped.set()
                self._requests[rid] = control

    def request_stop(self, request_ids):
        with self._request_lock:
            missing = [rid for rid in request_ids if rid not in self._requests]
            if missing:
                raise KeyError(f'未知或过期的 Transformers 引擎请求号：{missing!r}')
            for rid in request_ids:
                if self._requests[rid].finished is None:
                    self._requests[rid].stopped.set()

    async def abort_request(self, request_id):
        self.request_stop([request_id])

    def finish_requests(self, request_ids):
        with self._request_lock:
            for rid in request_ids:
                control = self._requests[rid]
                if control.finished is None:
                    control.finished = time.monotonic()
                    control.done.set()

    def wait_requests(self, request_ids):
        with self._request_lock:
            controls = [self._requests[rid] for rid in request_ids]
        for control in controls:
            control.done.wait()

    def stop_requests(self, request_ids):
        with self._request_lock:
            local_ids = [rid for rid in request_ids if rid in self._requests]
            self.request_stop(local_ids)
        self.wait_requests(local_ids)

    @staticmethod
    def encode_trajectory(template, trajectory, add_generation_prompt=True):
        if template is None:
            raise ValueError('轨迹编码需要 template。')
        encoded = template.encode(deepcopy(trajectory), add_generation_prompt=add_generation_prompt)
        if encoded.get('input_ids') is None:
            raise ValueError('Template.encode 必须返回 input_ids。')
        return deepcopy(encoded)

    @staticmethod
    def error_response(feat, num_samples=1):
        ids = feat.get('input_ids')
        return SampleResponse([SampledSequence(stop_reason='error', tokens=[], decoded='')
                               for _ in range(num_samples)],
                              prompt_token_ids=[] if ids is None else list(ids))

    @classmethod
    def encode_all(cls, template, inputs_list, *, logprobs_only, strict):
        encoded, failures = {}, {}
        for index, item in enumerate(inputs_list):
            try:
                if not isinstance(item, dict):
                    raise TypeError('每个生成输入必须为映射。')
                encoded[index] = (deepcopy(item) if 'input_ids' in item else
                                  cls.encode_trajectory(template, item, not logprobs_only))
            except Exception as exc:
                if strict:
                    raise
                logger.warning(f'输入 {index} 编码失败：{exc}')
                encoded[index] = {}
                failures[index] = cls.error_response({})
        return encoded, failures

    @staticmethod
    def prompt_ids(template, feat, *, token_only=False):
        ids = feat['input_ids']
        if template is not None and not token_only:
            ids = template.get_vllm_input_ids(ids)
        return ids.tolist() if hasattr(ids, 'tolist') else list(ids)

    @staticmethod
    def collate_extra_model_inputs(chunk, encoded):
        collected = {}
        for index in chunk:
            for key, value in encoded[index].items():
                if key not in ('input_ids', 'attention_mask', 'labels', 'completion_mask', 'routed_experts'):
                    if isinstance(value, torch.Tensor):
                        collected.setdefault(key, []).append(value)
        return {key: torch.cat(values, dim=0) for key, values in collected.items()}

    @staticmethod
    def attach_features(template, results, encoded):
        for index, response in enumerate(results):
            feat = encoded.get(index)
            if response is None or not feat:
                continue
            for seq in response.sequences:
                if template is None:
                    seq.new_input_feature = deepcopy(feat)
                    seq.new_input_feature['input_ids'] = list(feat['input_ids']) + list(seq.tokens)
                else:
                    prompt = deepcopy(feat)
                    if prompt.get('labels') is None or len(prompt['labels']) != len(prompt['input_ids']):
                        prompt['labels'] = [-100] * len(prompt['input_ids'])
                    seq.new_input_feature = template.concat_input_feature(prompt, seq.tokens)

    @staticmethod
    def group_by_adapter(pending, adapter_paths, adapter_path):
        groups = {}
        for index in pending:
            path = adapter_path if adapter_paths is None else adapter_paths[index]
            groups.setdefault(path, []).append(index)
        return groups

    @staticmethod
    def chunks(items, size):
        if type(size) is not int or size < 1:
            raise ValueError('max_batch_size 必须为正整数。')
        for start in range(0, len(items), size):
            yield items[start:start + size]

    def generate_inputs(self, inputs, sampling_params=None, adapter_name='', adapter_path=None, *,
                        template=None, adapter_paths=None, use_base_model=False, strict=True,
                        token_only=False, return_encoded=False, request_ids=None, max_batch_size=8,
                        registered=False, **kwargs):
        if request_ids is not None and not registered:
            self.register_requests(request_ids)
        try:
            return self._prepare_generate_inputs(
                inputs, sampling_params, adapter_name, adapter_path, template=template,
                adapter_paths=adapter_paths, use_base_model=use_base_model, strict=strict,
                token_only=token_only, request_ids=request_ids, max_batch_size=max_batch_size, **kwargs)
        finally:
            if request_ids is not None:
                self.finish_requests(request_ids)

    def _prepare_generate_inputs(self, inputs, sampling_params, adapter_name, adapter_path, *,
                                 template, adapter_paths, use_base_model, strict, token_only,
                                 request_ids, max_batch_size, **kwargs):
        inputs = deepcopy(list(inputs))
        params = (SamplingParams.from_dict(sampling_params) if isinstance(sampling_params, dict) else
                  deepcopy(sampling_params) if sampling_params is not None else SamplingParams())
        if not isinstance(params, SamplingParams):
            raise TypeError('sampling_params 类型错误。')
        params.__post_init__()
        if not inputs:
            return []
        if request_ids is not None and len(request_ids) != len(inputs):
            raise ValueError('request_ids 与输入必须等长。')
        if adapter_paths is not None and (adapter_path is not None or len(adapter_paths) != len(inputs)):
            raise ValueError('adapter_path 与 adapter_paths 互斥，且 adapter_paths 必须逐输入等长。')
        if token_only and any('input_ids' not in item for item in inputs):
            raise ValueError('token_only 需要已编码的 input_ids。')
        with self._model_lock:
            return self._generate_inputs_locked(inputs, params, template, adapter_name, adapter_path,
                                                adapter_paths, use_base_model, strict, token_only,
                                                request_ids, max_batch_size, kwargs)

    def _generate_inputs_locked(self, inputs, params, template, adapter_name, adapter_path, adapter_paths,
                                use_base_model, strict, token_only, request_ids, max_batch_size, kwargs):
        self.model.eval()
        logprobs_only = params.max_tokens == 0
        if logprobs_only:
            params.max_tokens = 1
        encoded, failures = self.encode_all(template, inputs, logprobs_only=logprobs_only, strict=strict)
        results = [self.error_response(encoded[index], params.num_samples) if index in failures else None
                   for index in range(len(inputs))]
        pending = [index for index, response in enumerate(results) if response is None]
        from peft import PeftModel
        context = self.model.disable_adapter() if use_base_model and isinstance(self.model, PeftModel) else contextlib.nullcontext()
        with context:
            for path, indices in self.group_by_adapter(pending, adapter_paths, adapter_path).items():
                if path is not None and not use_base_model:
                    from twinkle.hub import HubOperation
                    path = HubOperation.download_model(model_id_or_path=path)
                    adapter_uri = self.load_lora(path, adapter_name or None)
                else:
                    adapter_uri = None
                for chunk in self.chunks(indices, max_batch_size):
                    def call_batch(selected):
                        with self._request_lock:
                            stopped = {i for i in selected if request_ids is not None
                                       and self._requests[request_ids[i]].stopped.is_set()}
                            active = [i for i in selected if i not in stopped]
                            controls = None if request_ids is None else [self._requests[request_ids[i]] for i in active]
                        responses = {i: SampleResponse(
                            [SampledSequence(stop_reason='abort', tokens=[]) for _ in range(params.num_samples)],
                            prompt_token_ids=self.prompt_ids(template, encoded[i], token_only=token_only))
                            for i in stopped}
                        if active:
                            generated = self.generate_batch(
                                [self.prompt_ids(template, encoded[i], token_only=token_only) for i in active],
                                params, num_samples=params.num_samples, logprobs=params.logprobs is not None,
                                include_prompt_logprobs=params.prompt_logprobs is not None,
                                topk_prompt_logprobs=params.prompt_logprobs or 0, adapter_uri=adapter_uri,
                                extra_model_inputs=self.collate_extra_model_inputs(active, encoded),
                                _controls=controls, **kwargs)
                            if len(generated) != len(active):
                                raise ValueError('Transformers 引擎结果与活动输入必须等长。')
                            responses.update(zip(active, generated))
                        return [responses[i] for i in selected]
                    try:
                        responses = call_batch(chunk)
                    except Exception as exc:
                        if strict:
                            raise
                        logger.warning(f'批生成失败，逐输入隔离：{exc}')
                        responses = []
                        for index in chunk:
                            try:
                                responses.append(call_batch([index])[0])
                            except Exception as inner:
                                logger.warning(f'输入 {index} 生成失败：{inner}')
                                responses.append(self.error_response(encoded[index], params.num_samples))
                    if len(responses) != len(chunk):
                        raise ValueError('Transformers 引擎结果与输入必须等长。')
                    for index, response in zip(chunk, responses):
                        if logprobs_only:
                            for sequence in response.sequences:
                                sequence.tokens = []
                                sequence.logprobs = None
                        results[index] = response
        if not token_only:
            self.attach_features(template, results, encoded)
        return results

    async def sample_inputs(self, *args, **kwargs):
        # 已登记而尚在排队的请求也必须进入最终清理。
        ids = kwargs.get('request_ids')
        registered = kwargs.get('registered', False)
        try:
            async with self._generate_lock:
                task = asyncio.create_task(asyncio.to_thread(self.generate_inputs, *args, **kwargs))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if ids is not None:
                        self.request_stop(ids)
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            continue
                    task.result()
                    raise
        finally:
            if ids is not None and registered:
                self.finish_requests(ids)

    # ------------------------------------------------------------------ engine contract

    async def get_tokenizer(self):
        return self.tokenizer

    async def sample(
        self,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
        *,
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
        **kwargs,
    ) -> SampleResponse:
        """Single-prompt sampling. Prefer :meth:`batch_sample` -- see the class docstring on why."""
        responses = await self.batch_sample(
            [prompt_token_ids],
            sampling_params,
            num_samples=num_samples,
            logprobs=logprobs,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            adapter_uri=adapter_uri,
            request_ids=None if request_id is None else [request_id],
            **kwargs,
        )
        return responses[0]

    def generate_batch(
        self,
        prompt_token_ids: Sequence[List[int]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        extra_model_inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SampleResponse]:
        """Generate for several prompts in one padded forward, synchronously.

        The blocking core behind :meth:`batch_sample`. A caller with no event loop to keep responsive --
        a training model's worker generating inline (see ``TransformersModel.generate``) -- drives this
        directly rather than paying for a loop and a thread hop it has no use for.

        ``extra_model_inputs`` carries the already-processed multimodal tensors (``pixel_values`` and
        friends). Unlike vLLM, this engine cannot accept raw images -- the template's processor has to
        run first -- which is the contract note in ``BaseSamplerEngine.sample``.
        """
        if not prompt_token_ids:
            return []
        params = deepcopy(sampling_params) if sampling_params is not None else SamplingParams()
        ids = kwargs.pop('request_ids', None)
        controls = kwargs.pop('_controls', None)
        if ids is not None:
            if len(ids) != len(prompt_token_ids):
                raise ValueError('request_ids 与 prompt_token_ids 必须等长。')
            self.register_requests(ids)
            controls = [self._requests[rid] for rid in ids]
        try:
            with self._model_lock:
                self._check_lengths(prompt_token_ids, params)
                self._activate_adapter(adapter_uri)
                return self._generate_batch(list(prompt_token_ids), params, num_samples, logprobs,
                                            include_prompt_logprobs, topk_prompt_logprobs,
                                            extra_model_inputs or {}, kwargs, controls)
        finally:
            if ids is not None:
                self.finish_requests(ids)

    async def batch_sample(
        self,
        prompt_token_ids: Sequence[List[int]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        extra_model_inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SampleResponse]:
        """Async :meth:`generate_batch`: serialized behind a lock and run off the caller's event loop.

        Not part of :class:`BaseSamplerEngine`, but the reason this engine is usable at all: routing
        every prompt through :meth:`sample` would serialize them behind :attr:`_generate_lock` and
        give up the only batching transformers has. The lock turns concurrent callers into a queue
        instead of a corrupted forward (a single nn.Module cannot run two generate loops at once), and
        off-loading the blocking generate keeps the caller's loop responsive (the deploy recipe streams
        other requests on it).
        """
        ids = kwargs.pop('request_ids', None)
        if ids is not None:
            self.register_requests(ids)
        try:
            async with self._generate_lock:
                task = asyncio.create_task(asyncio.to_thread(
                    self.generate_batch, prompt_token_ids, sampling_params, num_samples=num_samples,
                    logprobs=logprobs, include_prompt_logprobs=include_prompt_logprobs,
                    topk_prompt_logprobs=topk_prompt_logprobs, adapter_uri=adapter_uri,
                    extra_model_inputs=extra_model_inputs,
                    _controls=None if ids is None else [self._requests[rid] for rid in ids], **kwargs))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if ids is not None:
                        self.request_stop(ids)
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            continue
                    task.result()
                    raise
        finally:
            if ids is not None:
                self.finish_requests(ids)

    async def generate_stream(
        self,
        prompt: List[int],
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[str] = None,
        **kwargs,
    ):
        """Yield ``(delta_text, finish_reason)`` as tokens are produced.

        ``TextIteratorStreamer`` only speaks to a blocking ``generate()`` running in another thread,
        so the bridge here is a thread + a thread-safe queue drained from the event loop. finish_reason
        is only known once generation ends, hence the final ``('', reason)`` tuple.
        """
        from transformers import TextIteratorStreamer

        params = sampling_params or SamplingParams()
        self._check_lengths([prompt], params)

        async with self._generate_lock:
            self._activate_adapter(lora_request)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = self._build_gen_kwargs(params, num_samples=1, logprobs=False)
            gen_kwargs['streamer'] = streamer
            inputs = self._to_model_inputs([prompt])
            produced: List[int] = []

            def _run():
                try:
                    out = self.model.generate(**inputs, **gen_kwargs)
                    produced.append(out.shape[-1] - inputs['input_ids'].shape[-1])
                except Exception as exc:  # surfaced to the consumer via the queue below
                    streamer.on_finalized_text('', stream_end=True)
                    raise exc

            thread = threading.Thread(target=_run, daemon=True, name='TransformersEngine-stream')
            thread.start()

            loop = asyncio.get_running_loop()
            sentinel = object()
            while True:
                delta = await loop.run_in_executor(None, lambda: next(streamer, sentinel))
                if delta is sentinel:
                    break
                if delta:
                    yield delta, None
            thread.join()
            new_tokens = produced[0] if produced else 0
            yield '', ('length' if params.max_tokens and new_tokens >= params.max_tokens else 'stop')

    # ------------------------------------------------------------------ weights / memory

    async def update_weights(self,
                             weights: Dict[str, torch.Tensor],
                             adapter_name: Optional[str] = None,
                             **kwargs) -> None:
        """In-place weight refresh for colocated training. Loads non-strictly: callers stream shards."""
        if adapter_name is not None:
            raise NotImplementedError('TransformersEngine cannot hot-swap adapter weights; pass adapter_uri '
                                      'to sample() with an on-disk adapter instead.')
        async with self._generate_lock:
            self.model.load_state_dict(weights, strict=False, assign=False)

    async def get_state_keys(self) -> List[str]:
        return list(self.model.state_dict().keys())

    async def sleep(self, **kwargs) -> None:
        """Park weights on CPU so a colocated trainer can have the GPU back."""
        async with self._generate_lock:
            self._device_before_sleep = next(self.model.parameters()).device
            self.model.to('cpu')
            torch_util.empty_cache()

    async def wake_up(self, **kwargs) -> None:
        async with self._generate_lock:
            self.model.to(getattr(self, '_device_before_sleep', torch_util.get_device(None)))

    async def shutdown(self) -> None:
        with self._request_lock:
            self._closed = True
            request_ids = [rid for rid, control in self._requests.items() if control.finished is None]
            self.request_stop(request_ids)
        await asyncio.to_thread(self.wait_requests, request_ids)
        async with self._generate_lock:
            with self._model_lock:
                self.model = None
                torch_util.empty_cache()

    # ------------------------------------------------------------------ LoRA

    def load_lora(self, adapter_path: str, adapter_name: Optional[str] = None) -> str:
        """Load a peft adapter and return the name to pass as ``adapter_uri``.

        Cached by path: peft raises on a duplicate name, and reloading the same weights is pure waste.
        Note that peft can hold many adapters but activates one at a time, so mixing adapters within a
        single batch is impossible here -- callers must group requests by adapter.
        """
        if adapter_path in self._adapters:
            return self._adapters[adapter_path]
        requires('peft')
        from peft import PeftModel

        name = adapter_name or f'adapter_{len(self._adapters)}'
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(adapter_path, adapter_name=name)
        else:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=name)
        self.model.eval()
        self._adapters[adapter_path] = name
        logger.info(f'TransformersEngine loaded LoRA {adapter_path} as {name!r}')
        return name

    def _activate_adapter(self, adapter_uri: Optional[str]) -> None:
        """Switch the active peft adapter, or disable adapters entirely when None."""
        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            if adapter_uri:
                raise ValueError(f'adapter {adapter_uri!r} requested but no LoRA is loaded; call load_lora() first.')
            return
        if adapter_uri is None:
            # disable_adapter() is a context manager, so base-model sampling is expressed by the
            # caller passing use_base_model, handled in the sampler. Here None means "keep whatever
            # single adapter is loaded" only if exactly one exists.
            return
        if adapter_uri != self._active_adapter:
            self.model.set_adapter(adapter_uri)
            self._active_adapter = adapter_uri

    # ------------------------------------------------------------------ internals

    def _check_lengths(self, prompts: Sequence[List[int]], params: SamplingParams) -> None:
        if self.max_model_len is None:
            return
        budget = self.max_model_len - (params.max_tokens or 0)
        longest = max(len(p) for p in prompts)
        if longest > budget:
            raise ValueError(f'Prompt of {longest} tokens leaves no room for {params.max_tokens} new tokens '
                             f'within max_model_len={self.max_model_len}. Shorten the prompt, lower '
                             'max_tokens, or raise max_model_len -- this engine will not truncate, because a '
                             'truncated prompt yields a fluent but meaningless completion.')

    def _build_gen_kwargs(self, params: SamplingParams, *, num_samples: int, logprobs: bool) -> Dict[str, Any]:
        gen_kwargs = params.to_transformers(self.tokenizer)
        gen_kwargs['num_return_sequences'] = num_samples
        if logprobs:
            gen_kwargs['output_scores'] = True
            gen_kwargs['return_dict_in_generate'] = True
        return gen_kwargs

    def _to_model_inputs(self, prompts: Sequence[List[int]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Left-pad the prompts into one batch on the model's device."""
        device = next(self.model.parameters()).device
        width = max(len(p) for p in prompts)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            attention_mask[row, width - len(prompt):] = 1
        inputs = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}
        for key, value in (extra or {}).items():
            inputs[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return inputs

    def _generate_batch(
        self,
        prompts: List[List[int]],
        params: SamplingParams,
        num_samples: int,
        want_logprobs: bool,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
        extra_model_inputs: Dict[str, Any],
        passthrough: Dict[str, Any],
        controls=None,
    ) -> List[SampleResponse]:
        inputs = self._to_model_inputs(prompts, extra_model_inputs)
        gen_kwargs = self._build_gen_kwargs(params, num_samples=num_samples, logprobs=want_logprobs)
        gen_kwargs.update(passthrough)
        prompt_width = inputs['input_ids'].shape[-1]
        tracker = None
        if controls is not None:
            from transformers import LogitsProcessorList, StoppingCriteriaList
            config = gen_kwargs.get('generation_config', self.model.generation_config)
            eos = gen_kwargs.get('eos_token_id', getattr(config, 'eos_token_id', None))
            eos_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
            tracker = _RequestStoppingCriteria(controls, prompt_width, eos_ids)
            criteria = list(gen_kwargs.get('stopping_criteria') or [])
            gen_kwargs['stopping_criteria'] = StoppingCriteriaList([*criteria, tracker])
            beams = gen_kwargs.get('num_beams', getattr(config, 'num_beams', 1))
            if beams > 1 and not eos_ids:
                raise ValueError('受控 beam 生成必须配置 eos_token_id。')
            if beams > 1:
                processors = list(gen_kwargs.get('logits_processor') or [])
                gen_kwargs['logits_processor'] = LogitsProcessorList(
                    [*processors, _RequestLogitsProcessor(tracker, eos_ids[0])])

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        # generate() returns prompt+completion; only the tail is new. With num_return_sequences=k the
        # batch dim is len(prompts)*k, ordered prompt-major.
        completions = sequences[:, prompt_width:]
        step_logprobs = _step_logprobs(output, params, prompt_width) if want_logprobs and hasattr(output, 'scores') else None

        prompt_logprobs = None
        if include_prompt_logprobs:
            prompt_logprobs = self._prompt_logprobs(inputs, topk_prompt_logprobs)

        max_new = gen_kwargs.get('max_new_tokens')
        responses: List[SampleResponse] = []
        for index, prompt in enumerate(prompts):
            seqs = []
            for offset in range(num_samples):
                row = index * num_samples + offset
                if tracker is not None and beams == 1:
                    length = tracker.lengths.get(row, completions.shape[-1])
                    tokens = completions[row, :length].tolist()
                    aborted = row in tracker.aborted
                else:
                    tokens = completions[row].tolist()
                    eos = gen_kwargs.get('eos_token_id', self.tokenizer.eos_token_id)
                    eos_ids = set([] if eos is None else [eos] if isinstance(eos, int) else eos)
                    for position, token in enumerate(tokens):
                        if token in eos_ids:
                            tokens = tokens[:position + 1]
                            break
                    aborted = tracker is not None and index in tracker.abort_lengths
                    if aborted:
                        tokens = tokens[:tracker.abort_lengths[index]]
                seqs.append(
                    SampledSequence(
                        stop_reason='abort' if aborted else 'length' if max_new and len(tokens) >= max_new else 'stop',
                        tokens=tokens,
                        logprobs=None if step_logprobs is None else step_logprobs[row][:len(tokens)],
                        decoded=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    ))
            responses.append(
                SampleResponse(
                    sequences=seqs,
                    prompt_token_ids=list(prompt),
                    prompt_logprobs=None if prompt_logprobs is None else prompt_logprobs[index],
                ))
        return responses

    def _prompt_logprobs(self, inputs: Dict[str, Any], topk: int) -> List[List[Optional[float]]]:
        """Score the prompt itself -- ``generate()`` never does, so this is a second forward pass."""
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = inputs['input_ids'][:, 1:]
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out: List[List[Optional[float]]] = []
        for row in range(picked.shape[0]):
            mask = (inputs['attention_mask'][row, :-1].bool()
                    & inputs['attention_mask'][row, 1:].bool())
            # The first real token has no predecessor to be predicted from, hence the leading None,
            # matching vLLM's prompt_logprobs convention.
            out.append([None] + picked[row][mask].tolist())
        return out


def _step_logprobs(output: Any, params: SamplingParams, prompt_width: int) -> List[List[List[Tuple[int, float]]]]:
    """``generate()`` scores -> per-row, per-step ``[(token_id, logprob), ...]``.

    ``params.logprobs`` follows vLLM's meaning: 0 (or None) asks only for the token actually sampled,
    k > 0 additionally asks for the k most likely alternatives at that step.
    """
    topk = params.logprobs or 0
    rows = output.sequences.shape[0]
    result: List[List[List[Tuple[int, float]]]] = [[] for _ in range(rows)]
    beam_indices = getattr(output, 'beam_indices', None)
    for step, scores in enumerate(output.scores):
        if prompt_width + step >= output.sequences.shape[-1]:
            break
        logprobs = torch.log_softmax(scores.float(), dim=-1)
        chosen = output.sequences[:, prompt_width + step]
        for row in range(rows):
            source = row if beam_indices is None else int(beam_indices[row, step])
            if source < 0:
                continue
            token = int(chosen[row])
            entries = [(token, float(logprobs[source, token]))]
            if topk > 0:
                values, indices = torch.topk(logprobs[source], min(topk, logprobs.shape[-1]))
                entries.extend((int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist()) if int(i) != token)
            result[row].append(entries)
    return result


def _strip_padding(tokens: torch.Tensor, pad_id: Optional[int]) -> List[int]:
    """Drop the trailing pads a short sequence in a batch accumulates once it hits eos."""
    ids = tokens.tolist()
    if pad_id is None:
        return ids
    while ids and ids[-1] == pad_id:
        ids.pop()
    return ids
