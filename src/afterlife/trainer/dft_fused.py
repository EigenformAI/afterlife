from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from loguru import logger


_DEFAULT_IGNORE_INDEX = -100
_LOGGED_MESSAGES: set[str] = set()
_ORIGINAL_COMPUTE_FUSED_CE_LOSS: Any | None = None
_ORIGINAL_UNSLOTH_FUSED_CE_LOSS: Any | None = None
_CCE_AVAILABLE: bool | None = None


def _log_once(message: str) -> None:
    if message in _LOGGED_MESSAGES:
        return
    logger.info(message)
    _LOGGED_MESSAGES.add(message)


def _cce_available() -> bool:
    global _CCE_AVAILABLE
    if _CCE_AVAILABLE is None:
        try:
            importlib.import_module("cut_cross_entropy")
            _CCE_AVAILABLE = True
        except Exception as exc:  # pragma: no cover - import-environment dependent
            logger.warning(
                "cut_cross_entropy unavailable ({}); DFT will use the dense path", exc
            )
            _CCE_AVAILABLE = False
    return _CCE_AVAILABLE


# --------------------------------------------------------------------------------------
# Dense path: the inner loss-fn for Unsloth's chunked UnslothFusedLoss (runs UNDER
# torch.func.grad_and_value). This is the robust-but-slower fallback. CCE cannot be used
# here: cut_cross_entropy's autograd.Function lacks setup_context and raises under functorch.
# --------------------------------------------------------------------------------------
def compute_fused_dft_loss(
    hidden_states: Any,
    lm_head_weight: Any,
    lm_head_bias: Any,
    labels: Any,
    n_items: Any | None = None,
    scaling: Any | None = None,
    shift_labels: bool = True,
    **kwargs: Any,
) -> tuple[Any, tuple[Any]]:
    """Dense per-chunk DFT loss. Differentiated by Unsloth via torch.func.grad_and_value, so
    reweighting per-token CE by the detached target probability yields the DFT gradient."""

    import torch

    ignore_index = int(kwargs.get("ignore_index", _DEFAULT_IGNORE_INDEX))
    label_smoothing = float(kwargs.get("label_smoothing", 0.0))
    device = lm_head_weight.device

    if shift_labels:
        shifted_labels = torch.empty_like(labels, device=device)
        shifted_labels[..., :-1] = labels[..., 1:]
        shifted_labels[..., -1] = ignore_index
        labels = shifted_labels

    logits = torch.nn.functional.linear(
        hidden_states.to(dtype=lm_head_weight.dtype, device=device),
        lm_head_weight,
        lm_head_bias,
    )
    vocab_size = lm_head_weight.shape[0]

    logit_scale_multiply = kwargs.get("logit_scale_multiply", None)
    logit_scale_divide = kwargs.get("logit_scale_divide", None)
    logit_softcapping = kwargs.get("logit_softcapping", None)
    if logit_scale_multiply != 0 and logit_scale_multiply is not None:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 0 and logit_scale_divide is not None:
        logits = logits / logit_scale_divide
    if logit_softcapping != 0 and logit_softcapping is not None:
        logits = logits / logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * logit_softcapping

    flat_labels = labels.view(-1).to(device).contiguous()
    ce = torch.nn.functional.cross_entropy(
        input=logits.view(-1, vocab_size).float().contiguous(),
        target=flat_labels,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    loss = ((-ce).exp().detach() * ce).sum()

    divisor = n_items
    if divisor is None:
        divisor = (flat_labels != ignore_index).sum().clamp_min(1)
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype=torch.float32, device=device)
    if divisor.numel() != 1:
        divisor = divisor.ravel()[0]
    loss = loss / divisor.to(dtype=loss.dtype, device=device)

    scaled_loss = loss * scaling if scaling is not None else loss
    _log_once(
        "DFT-fused active: dense chunked -p.detach()*log p (fallback), softcap={}".format(
            logit_softcapping
        )
    )
    return scaled_loss, (loss.detach(),)


# --------------------------------------------------------------------------------------
# Fast path: a drop-in replacement for unsloth_fused_ce_loss (called by the model forward
# under NORMAL autograd, not functorch). Computes DFT via cut_cross_entropy's fused kernel
# (reduction="none" + detached target-prob reweight) — no full logits, ~2.8x faster than the
# dense path. Falls back to the dense chunked path (never NLL) for cases CCE can't handle.
# --------------------------------------------------------------------------------------
def unsloth_fused_dft_loss(
    trainer: Any,
    hidden_states: Any,
    lm_head_weight: Any,
    lm_head_bias: Any,
    labels: Any,
    mask: Any | None = None,
    n_items: Any | None = None,
    scaling: Any | None = None,
    target_gb: Any | None = None,
    torch_compile: bool | None = True,
    overwrite: bool | None = False,
    shift_labels: bool = True,
    **kwargs: Any,
) -> Any:
    import torch

    # Mixed-precision (fp16 GradScaler) scaling; bf16 has no scaler -> None.
    scaler = (
        getattr(getattr(trainer, "accelerator", None), "scaler", None)
        if trainer is not None
        else None
    )
    if scaler is not None:
        scaling = scaler.get_scale()
    if hasattr(scaling, "get_scale"):
        scaling = scaling.get_scale()

    logit_scale_multiply = kwargs.get("logit_scale_multiply", None)
    logit_scale_divide = kwargs.get("logit_scale_divide", None)
    can_use_cce = (
        _cce_available()
        and lm_head_bias is None
        and not logit_scale_multiply
        and not logit_scale_divide
        and scaling is None
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
    )

    if not can_use_cce:
        # Edge cases (bias / logit-scale / fp16 loss-scaling / CCE unavailable): use the original
        # chunked machinery, which now runs DENSE DFT because compute_fused_ce_loss is also patched.
        _log_once(
            "DFT-fused: dense chunked fallback (bias/scale/fp16/CCE-unavailable; cce={}, scaling={})".format(
                _cce_available(), scaling
            )
        )
        return _ORIGINAL_UNSLOTH_FUSED_CE_LOSS(
            trainer,
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            mask=mask,
            n_items=n_items,
            scaling=scaling,
            target_gb=target_gb,
            torch_compile=torch_compile,
            overwrite=overwrite,
            shift_labels=shift_labels,
            **kwargs,
        )

    from cut_cross_entropy import linear_cross_entropy

    ignore_index = int(kwargs.get("ignore_index", _DEFAULT_IGNORE_INDEX))
    softcap = kwargs.get("logit_softcapping", None)
    if softcap == 0:
        softcap = None
    device = lm_head_weight.device
    hidden_states = hidden_states.to(device=device)

    # Shift + mask exactly as the original fused path does.
    if shift_labels:
        shifted = torch.empty_like(labels, device=device)
        shifted[..., :-1] = labels[..., 1:]
        if mask is not None:
            mask = mask.to(device=device)
            shifted[..., :-1][mask[..., 1:] == 0] = ignore_index
        shifted[..., -1] = ignore_index
        labels = shifted
    labels_flat = labels.reshape(-1).to(device=device).contiguous()
    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

    divisor = n_items
    if divisor is None:
        divisor = (labels_flat != ignore_index).sum().clamp_min(1)
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype=torch.float32, device=device)
    if divisor.numel() != 1:
        divisor = divisor.ravel()[0]

    # ce_t = -log p_target_t (no full logits materialized). filter_eps="auto" sparsifies the
    # backward (drops sub-threshold per-token CE gradients) exactly as Unsloth's NLL path does
    # (loss_utils.fused_linear_cross_entropy passes accuracy_threshold="auto"); the dropped grads
    # are numerically negligible and this recovers most of NLL's loss-path speed.
    ce = linear_cross_entropy(
        hidden_flat,
        lm_head_weight,
        labels_flat,
        reduction="none",
        softcap=softcap,
        shift=False,
        ignore_index=ignore_index,
        filter_eps="auto",
    )
    # DFT: sum_t p_t.detach() * ce_t. Backprop through CCE's reduction="none" feeds p as the
    # per-token grad_out -> de = p*(softmax-onehot) = the DFT gradient (verified).
    loss = ((-ce).exp().detach() * ce).sum() / divisor.to(
        dtype=torch.float32, device=device
    )
    _log_once(
        "DFT-fused active: CCE per-token reweight (fast path), softcap={}".format(
            softcap
        )
    )
    return loss


def _repatch_generated_modules() -> int:
    """Backstop: rebind unsloth_fused_ce_loss in any already-imported Unsloth compiled module.

    The generated per-model module imports the loss fns by value
    (`from ...cross_entropy_loss import unsloth_fused_ce_loss`). Patching the source symbol
    before that import is enough, but if the module was imported first we rebind it here.
    """
    count = 0
    for name, module in list(sys.modules.items()):
        if "compiled_module" not in name or module is None:
            continue
        if getattr(module, "unsloth_fused_ce_loss", None) not in (
            None,
            unsloth_fused_dft_loss,
        ):
            module.unsloth_fused_ce_loss = unsloth_fused_dft_loss
            count += 1
        if getattr(module, "compute_fused_ce_loss", None) not in (
            None,
            compute_fused_dft_loss,
        ):
            module.compute_fused_ce_loss = compute_fused_dft_loss
            count += 1
    if count:
        logger.info("Re-bound fused DFT loss into {} compiled-module symbol(s)", count)
    return count


def install_fused_dft(ignore_index: int = -100) -> None:
    """Patch Unsloth's loss seams to compute fused DFT.

    Primary (fast): replace unsloth_fused_ce_loss with a CCE-based DFT (normal autograd).
    Fallback (robust): also patch compute_fused_ce_loss to dense DFT for the chunked path /
    edge cases. Both are DFT — never silent NLL. Intentionally version-fragile: an import or
    attribute error should fail the run rather than silently train NLL.
    """

    global \
        _DEFAULT_IGNORE_INDEX, \
        _ORIGINAL_COMPUTE_FUSED_CE_LOSS, \
        _ORIGINAL_UNSLOTH_FUSED_CE_LOSS
    _DEFAULT_IGNORE_INDEX = int(ignore_index)
    # Route CausalLM models through unsloth_fused_ce_loss too (ConditionalGeneration always does);
    # pin chunks for the dense fallback (read at import; harmless for the CCE path which never chunks).
    os.environ["UNSLOTH_ENABLE_CCE"] = "0"
    os.environ.setdefault("UNSLOTH_CE_LOSS_N_CHUNKS", "16")

    ce_mod = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")

    if _ORIGINAL_UNSLOTH_FUSED_CE_LOSS is None:
        _ORIGINAL_UNSLOTH_FUSED_CE_LOSS = getattr(ce_mod, "unsloth_fused_ce_loss")
    if _ORIGINAL_COMPUTE_FUSED_CE_LOSS is None:
        _ORIGINAL_COMPUTE_FUSED_CE_LOSS = getattr(ce_mod, "compute_fused_ce_loss")

    setattr(ce_mod, "compute_fused_ce_loss", compute_fused_dft_loss)
    setattr(ce_mod, "unsloth_fused_ce_loss", unsloth_fused_dft_loss)
    _repatch_generated_modules()
    logger.info(
        "Installed fused DFT loss patch (fast=CCE:{}, fallback=dense) into unsloth_zoo",
        _cce_available(),
    )


__all__ = [
    "compute_fused_dft_loss",
    "unsloth_fused_dft_loss",
    "install_fused_dft",
    "_repatch_generated_modules",
]
