# GRPO-family variant losses (plan T10a-04..T10a-11, T10a-14, T10a-15):
# GPG / DRO / GeoMean / KLCov / ClipCov / DPPO-TV / DPPO-KL / REAL / compute_sdar_loss / MOPDLoss.
import math

import pytest
import torch

from twinkle.loss import (ClipCovLoss, DPPOKLLoss, DPPOTVLoss, DROLoss, GPGLoss, GeoMeanLoss, KLCovLoss,
                          MOPDLoss, OPSDLoss, REALLoss)
from twinkle.loss.sdar import compute_sdar_loss


def _rl_batch(batch_size=2, seq_len=4, gap=0.0, adv=1.0):
    """A fully-valid [B, T] batch with per-token-constant logps and old_logps.

    ``gap`` sets old_logps = logps - gap (a constant per-token log-ratio of ``gap``),
    ``adv`` sets a constant per-token advantage. All masks are ones so every loss family
    shares the same aggregation denominators.
    """
    labels = torch.ones(batch_size, seq_len, dtype=torch.long)
    logps = torch.full((batch_size, seq_len), 0.3)
    old_logps = logps - gap
    advantages = torch.full((batch_size, seq_len), adv)
    return {'labels': labels}, {'logps': logps}, old_logps, advantages


def _loss(fn, batch, ref_logps=None, **extra):
    inputs, outputs, old_logps, advantages = batch
    return fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages,
              **extra)['loss'].item()


class TestGPGLoss:
    """T10a-04 (GPG half): pure REINFORCE -- the ratio is ignored."""

    def test_equals_neg_mean_logps_times_adv(self):
        # per_token = -logps * adv; with the default token-mean aggregation this is exact.
        fn = GPGLoss()
        loss = _loss(fn, _rl_batch(adv=1.0))
        assert loss == pytest.approx(-0.3, abs=1e-6)
        # Doubling the advantage doubles the loss (the ratio never enters).
        loss2 = _loss(fn, _rl_batch(adv=2.0))
        assert loss2 == pytest.approx(2 * loss, abs=1e-6)

    def test_ratio_ignored(self):
        # A nonzero old_logps gap must NOT change GPG's loss (no importance ratio).
        a = _loss(GPGLoss(), _rl_batch(gap=0.0))
        b = _loss(GPGLoss(), _rl_batch(gap=1.0))
        assert a == pytest.approx(b, abs=1e-9)


class TestDROLoss:
    """T10a-04 (DRO half): quadratic log-ratio penalty on top of REINFORCE."""

    def test_penalty_gap_hand_computed(self):
        # DRO - GPG on the same batch = mean(0.5 * dro_beta * gap^2) over valid tokens.
        gap, dro_beta = 0.4, 0.1
        batch = _rl_batch(gap=gap)
        dro = _loss(DROLoss(dro_beta=dro_beta), batch)
        gpg = _loss(GPGLoss(), batch)
        expected_penalty = 0.5 * dro_beta * gap * gap
        assert dro - gpg == pytest.approx(expected_penalty, abs=1e-6)

    def test_zero_gap_equals_gpg(self):
        batch = _rl_batch(gap=0.0)
        assert _loss(DROLoss(dro_beta=0.5), batch) == pytest.approx(_loss(GPGLoss(), batch), abs=1e-9)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match='dro_beta'):
            DROLoss(dro_beta=0.0)

    def test_kl_penalty_adds(self):
        batch = _rl_batch(gap=0.5)
        ref = torch.full((2, 4), 0.1)
        without = _loss(DROLoss(dro_beta=0.1, beta=0.0), batch)
        with_kl = _loss(DROLoss(dro_beta=0.1, beta=0.01), batch, ref_logps=ref)
        assert with_kl > without


class TestGeoMeanLoss:
    """T10a-05 (GeoMean): sequence-level geometric-mean ratio, degenerate hand-check."""

    def test_zero_log_ratio_hand_computed(self):
        # log_ratio = 0 everywhere -> seq_ratio = 1, seq_adv = adv -> loss = -mean(adv).
        loss = _loss(GeoMeanLoss(), _rl_batch(adv=1.0, gap=0.0))
        assert loss == pytest.approx(-1.0, abs=1e-6)
        loss_neg = _loss(GeoMeanLoss(), _rl_batch(adv=-2.0, gap=0.0))
        assert loss_neg == pytest.approx(2.0, abs=1e-6)

    def test_epsilon_bounds_geometric_ratio(self):
        # A gap beyond epsilon clips the log-ratio, so loss stays finite and bounded.
        loss = _loss(GeoMeanLoss(), _rl_batch(adv=1.0, gap=10.0))
        assert math.isfinite(loss)


class TestKLCovLoss:
    """T10a-07 (KLCov): covariance-selected tokens get the extra |log-ratio| KL."""

    def test_all_selected_penalty_hand_computed(self):
        # kl_cov_ratio=1.0 selects every token; the whole formula re-computed independently:
        # per_token = -adv * exp(gap) + ppo_kl_coef * |gap|, token-mean over identical tokens.
        gap, coef = 0.4, 2.0
        batch = _rl_batch(gap=gap)
        klcov = _loss(KLCovLoss(kl_cov_ratio=1.0, ppo_kl_coef=coef), batch)
        expected = -1.0 * math.exp(gap) + coef * abs(gap)
        assert klcov == pytest.approx(expected, abs=1e-6)

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError, match='kl_cov_ratio'):
            KLCovLoss(kl_cov_ratio=0.0)

    def test_finite_with_default_ratio(self):
        assert math.isfinite(_loss(KLCovLoss(), _rl_batch(gap=0.3)))


class TestClipCovLoss:
    """T10a-08 (ClipCov): PPO clipping with covariance zeroing -- contract anchors."""

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError, match='clip_cov_ratio'):
            ClipCovLoss(clip_cov_ratio=0.0)

    def test_finite_and_kl_adds(self):
        batch = _rl_batch(gap=0.3)
        ref = torch.full((2, 4), 0.0)
        assert math.isfinite(_loss(ClipCovLoss(), batch))
        base = _loss(ClipCovLoss(beta=0.0), batch)
        with_kl = _loss(ClipCovLoss(beta=0.01), batch, ref_logps=ref)
        assert with_kl > base


class TestDPPOLosses:
    """T10a-10: DPPO-TV and DPPO-KL both degenerate to the REINFORCE term at log_ratio 0."""

    def test_tv_and_kl_equal_gpg_at_zero_gap(self):
        batch = _rl_batch(gap=0.0, adv=1.0)
        gpg = _loss(GPGLoss(), batch)
        tv = _loss(DPPOTVLoss(), batch)
        kl = _loss(DPPOKLLoss(), batch)
        assert tv == pytest.approx(gpg, abs=1e-6)
        assert kl == pytest.approx(gpg, abs=1e-6)

    def test_finite_with_nonzero_gap(self):
        batch = _rl_batch(gap=0.5)
        assert math.isfinite(_loss(DPPOTVLoss(), batch))
        assert math.isfinite(_loss(DPPOKLLoss(), batch))


class TestREALLoss:
    """T10a-04 (REAL): group soft-constraint, hand-computed against the closed form."""

    def _real_batch(self, advs, token_logp=0.3, old=0.1):
        n = len(advs)
        labels = torch.ones(n, 2, dtype=torch.long)
        logps = torch.full((n, 2), token_logp, requires_grad=True)
        old_logps = torch.full((n, 2), old)
        advantages = torch.tensor([[a, a] for a in advs], dtype=torch.float32)
        return {'labels': labels}, {'logps': logps}, old_logps, advantages

    def test_mixed_group_hand_computed(self):
        # One group of 4: scores all 0.2 (tau=0.5 -> scaled 0.4), advantages [1,-1,1,-1].
        # neg = log(1 + 2*exp(0.4)); pos = log(1 + 2*exp(-0.4)); loss = neg + pos (one group).
        fn = REALLoss(real_tau=0.5, num_generations=4)
        batch = self._real_batch([1.0, -1.0, 1.0, -1.0])
        inputs, outputs, old_logps, advantages = batch
        out = fn(inputs, outputs, old_logps=old_logps, ref_logps=None, advantages=advantages)
        expected = math.log(1 + 2 * math.exp(0.4)) + math.log(1 + 2 * math.exp(-0.4))
        assert out['loss'].item() == pytest.approx(expected, rel=1e-5)

    def test_degenerate_group_zero_but_connected(self):
        # An all-positive group carries no contrastive signal: the loss is exactly zero
        # while staying connected to autograd (x * 0.0, not a detached constant).
        fn = REALLoss(real_tau=0.5, num_generations=2)
        inputs, outputs, old_logps, advantages = self._real_batch([1.0, 2.0])
        out = fn(inputs, outputs, old_logps=old_logps, ref_logps=None, advantages=advantages)
        assert out['loss'].item() == pytest.approx(0.0, abs=1e-9)
        assert out['loss'].requires_grad

    def test_indivisible_batch_rejected(self):
        fn = REALLoss(real_tau=0.5, num_generations=3)
        inputs, outputs, old_logps, advantages = self._real_batch([1.0, -1.0, 1.0, -1.0])
        with pytest.raises(AssertionError):
            fn(inputs, outputs, old_logps=old_logps, ref_logps=None, advantages=advantages)


class TestSdarLoss:
    """T10a-11: the confidence-gated self-distillation auxiliary, hand-computed."""

    def _batch(self):
        student = torch.tensor([[0.2, 0.4]], requires_grad=True)
        teacher = torch.tensor([[0.4, 0.2]])
        mask = torch.ones(1, 2)
        return student, teacher, mask

    def test_hand_computed(self):
        student, teacher, mask = self._batch()
        loss, metrics = compute_sdar_loss(student, teacher, mask, gate_beta=5.0)
        # delta = [0.2, -0.2]; gate = [sigmoid(1), sigmoid(-1)] = [0.73106, 0.26894]
        # kl = [0.2, -0.2]; gated = [0.14621, -0.05379]; token-mean = 0.04621.
        gate = [1 / (1 + math.exp(-1.0)), 1 / (1 + math.exp(1.0))]
        expected = (gate[0] * 0.2 + gate[1] * -0.2) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)
        assert 0.0 <= metrics['sdar/gate_mean'] <= 1.0

    def test_gradient_only_through_student(self):
        student, teacher, mask = self._batch()
        teacher = teacher.clone().requires_grad_(True)
        loss, _ = compute_sdar_loss(student, teacher, mask)
        loss.backward()
        assert student.grad is not None and torch.isfinite(student.grad).all()
        assert teacher.grad is None

    def test_aggregation_modes(self):
        student, teacher, mask = self._batch()
        token_mean, _ = compute_sdar_loss(student, teacher, mask, loss_agg_mode='token-mean')
        seq_sum, _ = compute_sdar_loss(student, teacher, mask, loss_agg_mode='seq-mean-token-sum')
        seq_mean, _ = compute_sdar_loss(student, teacher, mask, loss_agg_mode='seq-mean-token-mean')
        # One sequence of two valid tokens: seq-mean-token-sum = 2 * token-mean; seq-mean-token-mean == token-mean.
        assert seq_sum.item() == pytest.approx(2 * token_mean.item(), abs=1e-6)
        assert seq_mean.item() == pytest.approx(token_mean.item(), abs=1e-6)

    def test_unknown_agg_mode_raises(self):
        student, teacher, mask = self._batch()
        with pytest.raises(ValueError, match='loss_agg_mode'):
            compute_sdar_loss(student, teacher, mask, loss_agg_mode='nope')

    def test_high_beta_saturates_gate(self):
        # A huge gate_beta makes the gate a hard 0/1 step on the sign of delta.
        student = torch.tensor([[0.0, 0.0]])
        teacher = torch.tensor([[1.0, -1.0]])
        mask = torch.ones(1, 2)
        loss, metrics = compute_sdar_loss(student, teacher, mask, gate_beta=100.0)
        # gate ~ [1, 0]; kl = [1, -1]; gated = [1, 0]; token-mean = 0.5.
        assert loss.item() == pytest.approx(0.5, abs=1e-3)


def _opsd_batch(gap=0.3):
    torch.manual_seed(42)
    labels = torch.ones(1, 2, dtype=torch.long)
    logps = torch.full((1, 2), 0.5)
    return {'labels': labels}, {'logps': logps}, logps.detach() + gap


class TestMOPDLoss:
    """T10a-06: the multi-teacher mixture fusion."""

    def test_single_teacher_weight_one_equals_opsd(self):
        # K=1, w=[1.0]: logsumexp of one unit-weighted channel is that channel.
        inputs, outputs, teacher = _opsd_batch(gap=0.3)
        mopd = MOPDLoss()([t for t in [teacher]], teacher_weights=[1.0])if False else MOPDLoss()
        a = mopd(inputs, outputs, teacher_logps=[teacher], teacher_weights=[1.0])['loss'].item()
        b = OPSDLoss()(inputs, outputs, teacher_logps=teacher)['loss'].item()
        assert a == pytest.approx(b, abs=1e-6)

    def test_bare_tensor_delegates_to_opsd(self):
        inputs, outputs, teacher = _opsd_batch(gap=0.3)
        a = MOPDLoss()(inputs, outputs, teacher_logps=teacher)['loss'].item()
        b = OPSDLoss()(inputs, outputs, teacher_logps=teacher)['loss'].item()
        assert a == pytest.approx(b, abs=1e-9)

    def test_two_teacher_mixture_hand_computed(self):
        # student logp 0.5; teachers [0.0, -1.0]; equal weights ->
        # mix = log(0.5*1 + 0.5*e^-1); r = mix - 0.5; per-token = exp(r) - r - 1; token-mean over 2.
        inputs, outputs, _ = _opsd_batch()
        t1, t2 = torch.zeros(1, 2), torch.full((1, 2), -1.0)
        out = MOPDLoss()(inputs, outputs, teacher_logps=[t1, t2], teacher_weights=[0.5, 0.5])
        mix = math.log(0.5 + 0.5 * math.exp(-1.0))
        r = mix - 0.5
        expected = math.exp(r) - r - 1.0
        assert out['loss'].item() == pytest.approx(expected, abs=1e-6)

    def test_uniform_weights_default(self):
        # teacher_weights=None means 1/K each -- same as passing them explicitly.
        inputs, outputs, _ = _opsd_batch()
        t1, t2 = torch.zeros(1, 2), torch.full((1, 2), -1.0)
        a = MOPDLoss()(inputs, outputs, teacher_logps=[t1, t2])['loss'].item()
        b = MOPDLoss()(inputs, outputs, teacher_logps=[t1, t2], teacher_weights=[0.5, 0.5])['loss'].item()
        assert a == pytest.approx(b, abs=1e-9)

    @pytest.mark.parametrize('weights', [[0.5], [-0.5, 1.5], [0.0, 0.0]])
    def test_invalid_weights_raise(self, weights):
        inputs, outputs, _ = _opsd_batch()
        t1, t2 = torch.zeros(1, 2), torch.full((1, 2), -1.0)
        with pytest.raises(ValueError):
            MOPDLoss()(inputs, outputs, teacher_logps=[t1, t2], teacher_weights=weights)

    def test_backward_produces_finite_grads(self):
        # T10a-14 (MOPD instance): gradients flow through the student logps.
        logps = torch.full((1, 2), 0.5, requires_grad=True)
        inputs = {'labels': torch.ones(1, 2, dtype=torch.long)}
        outputs = {'logps': logps}
        t1, t2 = torch.zeros(1, 2), torch.full((1, 2), -1.0)
        out = MOPDLoss()(inputs, outputs, teacher_logps=[t1, t2])
        out['loss'].backward()
        assert logps.grad is not None and torch.isfinite(logps.grad).all()
