import sys
import types
import unittest
from unittest.mock import patch


try:
    import torch
except Exception as exc:  # pragma: no cover - environment-dependent import guard
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

# Import cut_cross_entropy ONCE at module load. Importing it mid-process (e.g. lazily inside
# install_fused_dft -> _cce_available during one test) re-triggers torch._inductor's TORCH_LIBRARY
# registration and poisons later tests in the same process. In the real run Unsloth imports it
# before our code, so this mirrors production. Guarded: it's CUDA/Triton-only.
try:  # pragma: no cover - environment-dependent
    import cut_cross_entropy  # noqa: F401
except Exception:
    pass


IGNORE_INDEX = -100


def _shift_labels(labels):
    shifted = torch.empty_like(labels)
    shifted[..., :-1] = labels[..., 1:]
    shifted[..., -1] = IGNORE_INDEX
    return shifted


def _softcap(logits, softcap):
    if softcap is None or softcap == 0:
        return logits
    return torch.tanh(logits / softcap) * softcap


def _reference_dft_loss(hidden, weight, bias, labels, *, n_items=None, softcap=30.0):
    logits = torch.nn.functional.linear(hidden.to(weight.dtype), weight, bias)
    logits = _softcap(logits, softcap)
    shifted = _shift_labels(labels)
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, weight.shape[0]).float().contiguous(),
        shifted.view(-1).to(logits.device).contiguous(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    denom = n_items
    if denom is None:
        denom = (shifted.view(-1) != IGNORE_INDEX).sum().clamp_min(1)
    return ((-ce).exp().detach() * ce).sum() / denom


def _reference_nll_loss(hidden, weight, bias, labels, *, n_items=None, softcap=30.0):
    logits = torch.nn.functional.linear(hidden.to(weight.dtype), weight, bias)
    logits = _softcap(logits, softcap)
    shifted = _shift_labels(labels)
    reduction = "sum" if n_items is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, weight.shape[0]).float().contiguous(),
        shifted.view(-1).to(logits.device).contiguous(),
        reduction=reduction,
        ignore_index=IGNORE_INDEX,
    )
    return loss / n_items if n_items is not None else loss


class TestFusedDftInstall(unittest.TestCase):
    def test_install_fused_dft_patches_both_unsloth_symbols(self):
        from afterlife.trainer.dft_fused import (
            compute_fused_dft_loss,
            install_fused_dft,
            unsloth_fused_dft_loss,
        )

        ce_mod = types.ModuleType("unsloth_zoo.fused_losses.cross_entropy_loss")

        def original_compute(*_args, **_kwargs):
            raise AssertionError("original compute loss should be patched")

        def original_fused(*_args, **_kwargs):
            raise AssertionError("original fused loss should be patched")

        ce_mod.compute_fused_ce_loss = original_compute
        ce_mod.unsloth_fused_ce_loss = original_fused
        fused_losses_mod = types.ModuleType("unsloth_zoo.fused_losses")
        zoo_mod = types.ModuleType("unsloth_zoo")

        with patch.dict(
            sys.modules,
            {
                "unsloth_zoo": zoo_mod,
                "unsloth_zoo.fused_losses": fused_losses_mod,
                "unsloth_zoo.fused_losses.cross_entropy_loss": ce_mod,
            },
        ):
            install_fused_dft()
            # Fast path (CCE) replaces unsloth_fused_ce_loss; dense fallback replaces the inner fn.
            self.assertIs(ce_mod.unsloth_fused_ce_loss, unsloth_fused_dft_loss)
            self.assertIs(ce_mod.compute_fused_ce_loss, compute_fused_dft_loss)
            install_fused_dft()  # idempotent
            self.assertIs(ce_mod.unsloth_fused_ce_loss, unsloth_fused_dft_loss)
            self.assertIs(ce_mod.compute_fused_ce_loss, compute_fused_dft_loss)


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_IMPORT_ERROR}")
class TestFusedDftLoss(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 8, 5, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(13, 5, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(13, dtype=torch.float32, requires_grad=True)
        labels = torch.randint(0, 13, (2, 8), dtype=torch.long)
        labels[0, 3] = IGNORE_INDEX
        labels[1, 6] = IGNORE_INDEX
        shifted = _shift_labels(labels)
        n_items = (shifted.view(-1) != IGNORE_INDEX).sum().to(torch.float32)
        return hidden, weight, bias, labels, n_items

    def _clone_inputs(self, hidden, weight, bias):
        return (
            hidden.detach().clone().requires_grad_(True),
            weight.detach().clone().requires_grad_(True),
            bias.detach().clone().requires_grad_(True),
        )

    def test_fused_dft_gradient_matches_full_logit_reference(self):
        from afterlife.trainer.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, n_items = self._inputs()
        candidate_loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        candidate_loss.backward()
        candidate_grads = (hidden.grad.clone(), weight.grad.clone(), bias.grad.clone())

        ref_hidden, ref_weight, ref_bias = self._clone_inputs(hidden, weight, bias)
        ref_loss = _reference_dft_loss(
            ref_hidden,
            ref_weight,
            ref_bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        ref_loss.backward()
        ref_grads = (ref_hidden.grad, ref_weight.grad, ref_bias.grad)

        for candidate_grad, ref_grad in zip(candidate_grads, ref_grads):
            self.assertTrue(
                torch.allclose(candidate_grad, ref_grad, rtol=1e-5, atol=1e-6),
                (candidate_grad - ref_grad).abs().max().item(),
            )

        nll_hidden, nll_weight, nll_bias = self._clone_inputs(hidden, weight, bias)
        nll_loss = _reference_nll_loss(
            nll_hidden,
            nll_weight,
            nll_bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        nll_loss.backward()
        self.assertFalse(
            torch.allclose(candidate_grads[0], nll_hidden.grad, rtol=1e-5, atol=1e-6)
        )

    def test_fused_dft_loss_value_and_scale(self):
        from afterlife.trainer.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, n_items = self._inputs()
        candidate_loss, (detached_loss,) = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        ref_loss = _reference_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        nll_loss = _reference_nll_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )

        self.assertTrue(torch.allclose(candidate_loss, ref_loss, rtol=1e-6, atol=1e-7))
        self.assertTrue(
            torch.allclose(detached_loss, ref_loss.detach(), rtol=1e-6, atol=1e-7)
        )
        self.assertLess(candidate_loss.item(), nll_loss.item())

    def test_softcap_is_applied(self):
        from afterlife.trainer.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, n_items = self._inputs()
        hidden = (hidden.detach() * 40).requires_grad_(True)
        weight = (weight.detach() * 8).requires_grad_(True)

        candidate_loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        softcapped_ref = _reference_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        uncapped_ref = _reference_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=None,
        )

        self.assertTrue(
            torch.allclose(candidate_loss, softcapped_ref, rtol=1e-6, atol=1e-7)
        )
        self.assertFalse(
            torch.allclose(candidate_loss, uncapped_ref, rtol=1e-4, atol=1e-5)
        )

    def test_mask_and_normalizer(self):
        from afterlife.trainer.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, _n_items = self._inputs()
        labels = labels.clone()
        labels[0, 2] = IGNORE_INDEX
        shifted = _shift_labels(labels)
        n_items = (shifted.view(-1) != IGNORE_INDEX).sum().to(torch.float32)

        loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()

        self.assertTrue(
            torch.allclose(hidden.grad[0, 1], torch.zeros_like(hidden.grad[0, 1]))
        )

        changed_hidden = hidden.detach().clone()
        changed_hidden[0, 1] = changed_hidden[0, 1] + 1000
        changed_loss, _ = compute_fused_dft_loss(
            changed_hidden.requires_grad_(True),
            weight.detach().clone().requires_grad_(True),
            bias.detach().clone().requires_grad_(True),
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        self.assertTrue(
            torch.allclose(loss.detach(), changed_loss.detach(), rtol=1e-6, atol=1e-7)
        )


def _cce_cuda_available() -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        from triton.runtime import driver

        driver.active.get_current_device()
    except Exception:
        return False
    return True


@unittest.skipIf(
    not _cce_cuda_available(),
    "CUDA + cut_cross_entropy required for the fused CCE path",
)
class TestUnslothFusedDftLossCCE(unittest.TestCase):
    """The fast path: unsloth_fused_dft_loss runs CCE under NORMAL autograd (not functorch).

    cut_cross_entropy is CUDA/Triton-only, so this is gated. It asserts the CCE-DFT gradient
    equals the dense full-logit DFT reference AND differs from NLL — the same guard as the
    dense path, on the path we actually run for speed.
    """

    def test_cce_fast_path_matches_dense_dft_and_differs_from_nll(self):
        from afterlife.trainer.dft_fused import unsloth_fused_dft_loss

        dev = "cuda"
        torch.manual_seed(11)
        N, V, D = 256, 2048, 64
        hidden = torch.randn(1, N, D, device=dev, dtype=torch.bfloat16)
        weight = torch.randn(V, D, device=dev, dtype=torch.bfloat16) * 0.1
        labels = torch.randint(0, V, (1, N), device=dev, dtype=torch.long)
        labels[0, 5] = IGNORE_INDEX
        labels[0, 100] = IGNORE_INDEX
        shifted = _shift_labels(labels)
        n_items = (shifted.view(-1) != IGNORE_INDEX).sum().to(torch.float32)

        h = hidden.clone().requires_grad_(True)
        loss = unsloth_fused_dft_loss(
            None,
            h,
            weight,
            None,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()
        g_cce = h.grad.float().clone()

        hr = hidden.float().clone().requires_grad_(True)
        logits = torch.tanh((hr.view(-1, D) @ weight.float().t()) / 30.0) * 30.0
        ce = torch.nn.functional.cross_entropy(
            logits, shifted.view(-1), reduction="none", ignore_index=IGNORE_INDEX
        )
        loss_ref = ((-ce).exp().detach() * ce).sum() / n_items
        loss_ref.backward()
        g_ref = hr.grad.float().view_as(g_cce)

        hn = hidden.float().clone().requires_grad_(True)
        ln = torch.tanh((hn.view(-1, D) @ weight.float().t()) / 30.0) * 30.0
        nll = (
            torch.nn.functional.cross_entropy(
                ln, shifted.view(-1), reduction="sum", ignore_index=IGNORE_INDEX
            )
            / n_items
        )
        nll.backward()
        g_nll = hn.grad.float().view_as(g_cce)

        scale = g_ref.abs().max().clamp_min(1e-12).item()
        self.assertTrue(
            torch.allclose(loss.float(), loss_ref.float(), rtol=2e-2, atol=1e-4)
        )
        self.assertLess((g_cce - g_ref).abs().max().item(), max(3e-2 * scale, 1e-4))
        self.assertGreater((g_cce - g_nll).abs().max().item(), 1e-2 * scale)


if __name__ == "__main__":
    unittest.main()
