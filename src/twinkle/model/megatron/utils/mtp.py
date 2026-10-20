# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-token-prediction (MTP) training helpers for the megatron backend.

The MegatronModel keeps thin same-named methods that delegate here, so subclass
overrides and duck-typed callers (``getattr(model, 'pop_mtp_metrics')``) keep
working; the logic lives in this module, next to nothing else.
"""
from __future__ import annotations

from typing import Optional


def mtp_pass_enabled(enable_intent: bool, *, forward_only: bool, task: str, disable_lora: bool) -> bool:
    """Whether this pass should hand next-token targets to the MTP heads.

    The config declares the *intent* (``enable_mtp_training``); only the call site knows
    whether a pass will be backward'ed. That split is necessary, not redundant: megatron's
    ``forward_only`` schedule skips the backward but neither switches the module to eval
    nor disables autograd, so ``self.training`` and ``torch.is_grad_enabled()`` are both
    True on a log-prob pass. Running the MTP block there would cost an extra transformer
    layer plus a full-vocabulary projection per unroll step, and throw every bit of it
    away.
    """
    if not enable_intent:
        return False
    if forward_only:
        # old/reference log-probs. No backward, so the MTP forward would be pure waste.
        return False
    if task != 'causal_lm':
        # A pooling head has no next-token target to give them.
        return False
    if disable_lora:
        # The base-policy pass of a LoRA run; its gradients are discarded by construction.
        return False
    return True


def iter_mtp_modules(unwrapped_models):
    """The MTP blocks across model chunks, skipping ranks and chunks that hold none.

    ``mtp`` sits on the language model, which for a multimodal model is one level down. PP
    ranks other than the last have no MTP block at all, and a PeftModel forwards the
    lookup to its base.
    """
    for _model in unwrapped_models:
        mtp = getattr(_model, 'mtp', None)
        if mtp is None:
            language_model = getattr(_model, 'language_model', None)
            mtp = getattr(language_model, 'mtp', None) if language_model is not None else None
        if mtp is not None:
            yield mtp


def warn_if_mtp_not_trainable(mtp_modules) -> None:
    """Warn when MTP training is asked for but nothing can receive the gradient.

    LoRA is the way to get here that config validation cannot catch: it freezes every
    base parameter, and the MTP layers are base parameters unless named in
    ``target_modules``. The result is a run that pays for the MTP forward and reports an
    mtp_loss that never moves, which is far harder to notice than an error.
    """
    from twinkle.utils import get_logger

    if not mtp_modules:
        # Expected on every PP rank but the last; those ranks have no MTP block to check.
        return
    if any(p.requires_grad for mtp in mtp_modules for p in mtp.parameters()):
        return
    get_logger().warning('enable_mtp_training is set, but no MTP parameter has requires_grad=True, so the '
                         'MTP loss will be computed and then discarded. With LoRA this means the MTP '
                         'layers are not covered by target_modules.')


def pop_mtp_metrics(last_num_microbatches: int, loss_scale: Optional[float] = None) -> dict[str, float]:
    """Drain megatron's MTP loss tracker and return one entry per MTP depth, plus a total.

    Draining rather than reading: the tracker accumulates across microbatches and megatron
    never clears it, so a caller that only read it would report a monotonically growing
    loss.

    Args:
        last_num_microbatches: Microbatch count of the last ``forward_backward`` -- the
            tracker holds a per-microbatch sum while the reported loss is a mean over
            them. It is passed in from the model (not counted by the caller) because
            ``forward_backward`` picks its own ``micro_batch_size`` and re-splits the
            inputs, so a caller counting the batches it handed over would divide by the
            wrong number.
        loss_scale: Multiplier for the accumulated values; defaults to
            ``1 / last_num_microbatches``.

    Returns:
        ``{'mtp_1_loss': ..., ..., 'mtp_loss': <sum>}``, plus ``mtp_{i}_accept`` per depth
        when the megatron version reports acceptance counts. Empty when MTP is off or
        nothing ran.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    tracker = MTPLossLoggingHelper.tracker
    # mcore >= 0.19 renamed 'values' to 'loss_values' and added acceptance counts.
    loss_key = 'loss_values' if 'loss_values' in tracker else 'values'
    if loss_key not in tracker:
        return {}
    if loss_scale is None:
        loss_scale = 1.0 / max(last_num_microbatches, 1)

    if hasattr(MTPLossLoggingHelper, 'reduce_metrics_in_tracker'):
        MTPLossLoggingHelper.reduce_metrics_in_tracker()
    else:
        reduce_mtp_tracker_legacy(tracker, loss_key)

    losses = (tracker[loss_key] * loss_scale).tolist()
    metrics = {f'mtp_{i + 1}_loss': float(v) for i, v in enumerate(losses)}
    metrics['mtp_loss'] = float(sum(losses))

    correct, total = tracker.get('correct_values'), tracker.get('total_values')
    if correct is not None and total is not None:
        for i, (c, t) in enumerate(zip(correct.tolist(), total.tolist(), strict=True)):
            # The acceptance rate is what decides whether speculative decoding pays off at
            # rollout time, so it is worth surfacing next to the loss rather than only the loss.
            metrics[f'mtp_{i + 1}_accept'] = float(c / t) if t else 0.0

    if hasattr(MTPLossLoggingHelper, 'clean_metrics_in_tracker'):
        MTPLossLoggingHelper.clean_metrics_in_tracker()
    else:
        MTPLossLoggingHelper.clean_loss_in_tracker()
    return metrics


def reduce_mtp_tracker_legacy(tracker, loss_key: str) -> None:
    """The pre-0.19 reduction, which had no ``reduce_metrics_in_tracker`` to call.

    Order matters: the sum across the reduce group first, then the average across the
    group the loss is replicated over. Swapping them averages partial sums.
    """
    import torch.distributed as dist

    values = tracker[loss_key]
    if tracker.get('reduce_group') is not None:
        dist.all_reduce(values, group=tracker['reduce_group'])
    if tracker.get('avg_group') is not None:
        dist.all_reduce(values, group=tracker['avg_group'], op=dist.ReduceOp.AVG)
