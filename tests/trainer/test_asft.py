"""CPU unit tests for the vendored ASFT loss helpers (``afterlife/trainer/asft.py``).

These cover the parts that do not require the Triton CE kernel (no GPU): the
KL-gating resolution (local review fix #1), label shifting, logit transforms,
DFT weighting via the CE-losses path, and KL divergence. End-to-end loss and
trainer behaviour require a GPU smoke test (design doc step 7).
"""

import unittest

import torch

from afterlife.trainer.asft import (
    _compute_dft_weights,
    _compute_kl_divergence,
    _compute_kl_seq_chunked,
    build_shift_labels,
    effective_logits,
    resolve_effective_mode,
)


class TestResolveEffectiveMode(unittest.TestCase):
    """Review fix #1: kl_weight == 0.0 must skip the KL/reference path."""

    def test_asft_with_zero_kl_collapses_to_dft(self):
        self.assertEqual(resolve_effective_mode("asft", 0.0), ("dft", False))

    def test_sft_plus_kl_with_zero_kl_collapses_to_sft(self):
        self.assertEqual(resolve_effective_mode("sft+kl", 0.0), ("sft", False))

    def test_asft_with_nonzero_kl_stays_active(self):
        self.assertEqual(resolve_effective_mode("asft", 0.5), ("asft", True))

    def test_sft_plus_kl_with_nonzero_kl_stays_active(self):
        self.assertEqual(resolve_effective_mode("sft+kl", 0.25), ("sft+kl", True))

    def test_dft_never_activates_kl(self):
        self.assertEqual(resolve_effective_mode("dft", 0.9), ("dft", False))

    def test_sft_never_activates_kl(self):
        self.assertEqual(resolve_effective_mode("sft", 0.9), ("sft", False))


class TestAsftLossHelpers(unittest.TestCase):
    def test_build_shift_labels_shifts_and_caps_last(self):
        labels = torch.tensor([[10, 11, 12, 13]])
        shifted = build_shift_labels(labels)
        self.assertTrue(torch.equal(shifted, torch.tensor([[11, 12, 13, -100]])))

    def test_effective_logits_identity_without_scale_or_cap(self):
        x = torch.tensor([[1.0, -2.0, 3.0]])
        self.assertTrue(torch.allclose(effective_logits(x), x))

    def test_effective_logits_applies_scaling(self):
        x = torch.tensor([[1.0, -2.0, 3.0]])
        self.assertTrue(torch.allclose(effective_logits(x, logit_scaling=2.0), x * 2.0))

    def test_effective_logits_applies_softcapping(self):
        x = torch.tensor([[1.0, -2.0, 3.0]])
        t = 5.0
        self.assertTrue(
            torch.allclose(
                effective_logits(x, logit_softcapping=t), t * torch.tanh(x / t)
            )
        )

    def test_dft_weights_from_ce_losses_are_exp_neg_ce_masked(self):
        # DFT weight w_t = p_theta(y_t) = exp(-ce_t), zeroed on ignored tokens.
        ce = torch.tensor([[0.0, 1.0, 2.0]])
        valid = torch.tensor([[True, True, False]])
        weights = _compute_dft_weights(None, None, ce_losses=ce, valid_mask=valid)
        expected = torch.exp(-ce) * valid
        self.assertTrue(torch.allclose(weights, expected))

    def test_kl_is_zero_for_identical_distributions(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 8)
        kl = _compute_kl_divergence(logits, logits.clone())
        self.assertTrue(torch.allclose(kl, torch.zeros_like(kl), atol=1e-5))

    def test_kl_is_nonnegative_and_direction_asymmetric(self):
        torch.manual_seed(1)
        cur = torch.randn(1, 3, 6)
        ref = torch.randn(1, 3, 6)
        kl_fwd = _compute_kl_divergence(cur, ref, kl_direction="forward")
        kl_rev = _compute_kl_divergence(cur, ref, kl_direction="reverse")
        self.assertTrue(bool((kl_fwd >= -1e-6).all()))
        self.assertTrue(bool((kl_rev >= -1e-6).all()))
        self.assertFalse(torch.allclose(kl_fwd, kl_rev))


class TestAsftTrainerWiring(unittest.TestCase):
    """Prove the local trainer/config are real (not just mocked in qlora tests)."""

    def test_asft_trainer_subclasses_unsloth_trainer(self):
        from unsloth.trainer import UnslothTrainer

        from afterlife.trainer.asft import ASFTStreamingConfig as LossStreamingConfig
        from afterlife.trainer.asft_trainer import ASFTStreamingConfig, ASFTTrainer

        self.assertTrue(issubclass(ASFTTrainer, UnslothTrainer))
        # ASFTStreamingConfig is re-exported from the vendored loss module.
        self.assertIs(ASFTStreamingConfig, LossStreamingConfig)


class TestKlSeqChunkedCheckpoint(unittest.TestCase):
    """The chunked + gradient-checkpointed KL must equal the plain full KL in value and
    gradient. This keeps full-vocabulary fp32 KL within a constrained GPU budget:
    per-token KL is independent across positions (chunking is exact) and the checkpoint
    recompute is deterministic (no dropout/RNG)."""

    def _inputs(self):
        torch.manual_seed(7)
        cur = torch.randn(1, 9, 16)
        ref = torch.randn(1, 9, 16)
        return cur, ref

    def test_chunked_equals_full_value(self):
        cur, ref = self._inputs()
        full = _compute_kl_divergence(cur, ref).reshape(1, 9)
        chunked = _compute_kl_seq_chunked(cur, ref, seq_chunk_size=4, checkpoint=False)
        self.assertEqual(tuple(chunked.shape), (1, 9))
        self.assertTrue(torch.allclose(full, chunked, atol=1e-5))

    def test_checkpoint_equals_noncheckpoint_value_and_grad(self):
        cur, ref = self._inputs()
        a_in = cur.clone().requires_grad_(True)
        a = _compute_kl_seq_chunked(a_in, ref, seq_chunk_size=4, checkpoint=False)
        (ga,) = torch.autograd.grad(a.sum(), a_in)
        b_in = cur.clone().requires_grad_(True)
        b = _compute_kl_seq_chunked(b_in, ref, seq_chunk_size=4, checkpoint=True)
        (gb,) = torch.autograd.grad(b.sum(), b_in)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.assertTrue(torch.allclose(ga, gb, atol=1e-5))

    def test_checkpoint_requested_but_no_grad_is_still_correct(self):
        # checkpoint=True but inputs don't require grad -> no checkpoint taken, value still exact.
        cur, ref = self._inputs()
        out = _compute_kl_seq_chunked(cur, ref, seq_chunk_size=4, checkpoint=True)
        full = _compute_kl_divergence(cur, ref).reshape(1, 9)
        self.assertTrue(torch.allclose(out, full, atol=1e-5))

    def test_checkpoint_with_inference_mode_reference(self):
        # The real reference forward runs under torch.inference_mode(); checkpoint must be able
        # to save the reference slice for backward (we clone it). Regression for
        # "Inference tensors cannot be saved for backward".
        cur, _ = self._inputs()
        with torch.inference_mode():
            ref = torch.randn(1, 9, 16)
        self.assertTrue(ref.is_inference())
        cur_in = cur.clone().requires_grad_(True)
        out = _compute_kl_seq_chunked(cur_in, ref, seq_chunk_size=4, checkpoint=True)
        # backward must not raise
        (grad,) = torch.autograd.grad(out.sum(), cur_in)
        self.assertEqual(tuple(grad.shape), tuple(cur_in.shape))


if __name__ == "__main__":
    unittest.main()
