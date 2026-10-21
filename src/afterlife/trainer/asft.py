"""Anchored Supervised Fine-Tuning (ASFT) Loss Module for Unsloth.

This module implements ASFT as an optional loss-path for fine-tuning.
All logic is gated behind `asft_enabled` and does not modify existing
fast_cross_entropy_loss behavior.

Unsloth also includes additional performance and VRAM optimizations around
the same ASFT objective. In this repository we refer to that practical,
optimized implementation as **ASFT+**.

ASFT Modes:
- SFT: Standard cross-entropy loss
- DFT: CE weighted by model's confidence (detached)
- SFT+KL: CE + KL divergence from reference model
- ASFT: DFT + KL divergence (full ASFT loss)
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss
from unsloth.utils.packing import mask_packed_sequence_boundaries

import logging

# Vendored from unslothai/unsloth PR #4239 (branch dh/recover-3918-asft) at commit
# 5dc8ee0b3597c5c78de66db132499b431a0aa8c2, re-homed under afterlife/trainer/ for the local
# Local modifications vs. that commit:
#   1. kl_weight == 0.0 short-circuits the KL/reference path (resolve_effective_mode).
#   2. KL/streaming tensor reshapes use .reshape(...) for non-contiguous chunks.
#   3. Sequence-KV streaming fallbacks are logged instead of failing silently.
ASFT_SOURCE_COMMIT = "5dc8ee0b3597c5c78de66db132499b431a0aa8c2"

logger = logging.getLogger(__name__)

__all__ = [
    "ASFTStreamingConfig",
    "ASFT_SOURCE_COMMIT",
    "effective_logits",
    "fast_cross_entropy_loss_per_token",
    "build_shift_labels",
    "get_reference_forward_callable",
    "compute_asft_loss",
    "resolve_effective_mode",
]


# Default chunk sizes for streaming strategies
_DEFAULT_SEQ_CHUNK_SIZE = 256
_DEFAULT_REF_MICROBATCH_DIVISOR = 2  # batch_size // this value


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ASFTStreamingConfig:
    """Configuration for ASFT streaming strategies to reduce VRAM peak.

    Afterlife: only ``seq_chunk_size`` and ``force_fp32_kl`` affect the active
    loss path. The reference forward always runs in full; ``mode``,
    ``enabled``, ``ref_strategy`` and ``ref_microbatch_size`` are accepted for
    NSL2 compatibility but have no effect (train_sft warns when they are set).

    Attributes:
        mode: High-level streaming mode.
            - "off": Disable streaming.
            - "auto": Try seq_kv_cache with automatic batch-micro fallback.
            - "batch": Use batch microbatching only.
            - "seq": Use seq_kv_cache (with fallback).
            - "hybrid": Combine batch micro + seq_kv_cache.
        enabled: Whether streaming is enabled.
        ref_strategy: Strategy for reference model forward pass.
            - "none": Full reference forward (no streaming).
            - "batch_micro": Microbatch reference forward by batch dimension.
            - "seq_kv_cache": Sequence chunking via KV cache.
        ref_microbatch_size: Microbatch size for batch_micro or seq_kv_cache.
        seq_chunk_size: Chunk size for seq_kv_cache strategy (e.g., 128-512).
        kl_token_chunk_size: Optional extra chunking of valid tokens for KL.
        force_fp32_kl: Whether to force FP32 for KL computation.
    """

    mode: Optional[Literal["off", "auto", "batch", "seq", "hybrid"]] = None
    enabled: bool = False
    ref_strategy: Literal["none", "batch_micro", "seq_kv_cache"] = "none"
    ref_microbatch_size: Optional[int] = None
    seq_chunk_size: Optional[int] = None
    kl_token_chunk_size: Optional[int] = None
    force_fp32_kl: bool = True


# -----------------------------------------------------------------------------
# A1) Helper: effective_logits - Apply logit scaling and softcapping
# -----------------------------------------------------------------------------


def _resolve_logit_params(
    model: Optional[nn.Module],
    logit_softcapping: Optional[float],
    logit_scaling: Optional[float],
) -> Tuple[float, float]:
    if model is not None:
        config = getattr(model, "config", None)
        if config is not None:
            if logit_softcapping is None:
                logit_softcapping = getattr(config, "final_logit_softcapping", 0)
                if logit_softcapping is None:
                    logit_softcapping = 0
            if logit_scaling is None:
                logit_scaling = getattr(config, "logit_scale", 0)
                if logit_scaling is None:
                    logit_scaling = 0
                if logit_scaling == 0:
                    logit_scaling = getattr(config, "logit_scaling", 0)
                    if logit_scaling is None:
                        logit_scaling = 0
                model_type = getattr(config, "model_type", None)
                if model_type == "granite":
                    logits_scaling = getattr(config, "logits_scaling", 1)
                    if logits_scaling is None:
                        logits_scaling = 1
                    logit_scaling = 1 / logits_scaling
                elif model_type == "falcon_h1":
                    logit_scaling = getattr(config, "lm_head_multiplier", 0)
                    if logit_scaling is None:
                        logit_scaling = 0

    if logit_softcapping is None:
        logit_softcapping = 0
    if logit_scaling is None:
        logit_scaling = 0

    return logit_softcapping, logit_scaling


def effective_logits(
    logits: torch.Tensor,
    model: Optional[nn.Module] = None,
    logit_softcapping: Optional[float] = None,
    logit_scaling: Optional[float] = None,
) -> torch.Tensor:
    """Apply logit scaling and softcapping consistent with Unsloth's Triton CE kernel.

    This ensures DFT weights and KL are computed on the same distribution as CE.

    Args:
        logits: Input logits tensor.
        model: Model to extract config from (optional if softcapping/scaling provided).
        logit_softcapping: Softcapping value (e.g., for Gemma 2). If None, read from model.
        logit_scaling: Logit scaling value (e.g., for Cohere). If None, read from model.

    Returns:
        Transformed logits with scaling and softcapping applied.
    """
    logit_softcapping, logit_scaling = _resolve_logit_params(
        model,
        logit_softcapping,
        logit_scaling,
    )

    # Convert to float32 for stability
    x = logits.float()

    # Apply scaling: t * x
    if logit_scaling != 0:
        x = logit_scaling * x

    # Apply softcapping: t * tanh(x / t)
    if logit_softcapping != 0:
        x = logit_softcapping * torch.tanh(x / logit_softcapping)

    return x


# -----------------------------------------------------------------------------
# A2) Helper: fast_cross_entropy_loss_per_token
# -----------------------------------------------------------------------------


def fast_cross_entropy_loss_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token cross-entropy loss using Unsloth's Triton kernel.

    This is a wrapper around Fast_CrossEntropyLoss that returns token-level
    losses without reduction, suitable for ASFT weighting.

    Args:
        logits: Logits tensor of shape (B, T, V) or (B*T, V).
        labels: Labels tensor of shape (B, T) or (B*T,).
        logit_softcapping: Softcapping value for the kernel.
        logit_scaling: Scaling value for the kernel.
        ignore_index: Index to ignore in loss computation.

    Returns:
        Tuple of:
            - losses: Per-token losses of shape (B*T,) where ignored = 0.
            - valid_mask: Boolean mask of shape (B*T,) indicating valid tokens.
    """
    # Flatten if needed
    if logits.dim() == 3:
        batch, seq_len, vocab_size = logits.shape
        logits = logits.view(batch * seq_len, vocab_size)
        labels = labels.view(batch * seq_len)
    else:
        vocab_size = logits.shape[-1]

    # Create valid mask before computing loss
    valid_mask = labels != ignore_index

    labels_for_kernel = labels
    if ignore_index != -100:
        labels_for_kernel = labels.clone()
        labels_for_kernel[labels_for_kernel == ignore_index] = -100

    # Compute per-token CE using Unsloth's Triton kernel
    # The kernel already handles ignore_index (-100) internally and returns 0 for those
    losses = Fast_CrossEntropyLoss.apply(
        logits,
        labels_for_kernel,
        logit_softcapping,
        logit_scaling,
    )

    return losses, valid_mask


# -----------------------------------------------------------------------------
# A3) Helper: build_shift_labels - Unsloth-style label shifting
# -----------------------------------------------------------------------------


def build_shift_labels(
    labels: torch.Tensor,
    packed_seq_lengths: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build shifted labels in Unsloth style.

    Unsloth CE path "shifts labels, not logits" - the last token becomes -100.

    shift_labels[..., :-1] = labels[..., 1:]
    shift_labels[..., -1] = -100

    Args:
        labels: Original labels tensor of shape (B, T).
        packed_seq_lengths: Optional packed sequence lengths for boundary masking.
        ignore_index: Index to use for ignored positions.

    Returns:
        Shifted labels tensor of same shape as input.
    """
    shift_labels = torch.empty_like(labels)
    shift_labels[..., :-1] = labels[..., 1:]
    shift_labels[..., -1] = ignore_index

    # Apply packing boundary masking if needed
    if packed_seq_lengths is not None:
        mask_packed_sequence_boundaries(
            shift_labels,
            packed_seq_lengths,
            ignore_index = ignore_index,
        )

    return shift_labels


# -----------------------------------------------------------------------------
# A4) Helper: get_reference_forward_callable
# -----------------------------------------------------------------------------


@contextmanager
def _inference_eval_context(model: nn.Module):
    """Context manager for inference mode with eval() state preserved."""
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        if was_training:
            model.train()


def get_reference_forward_callable(
    model: nn.Module,
    reference_policy: Literal["disable_adapter", "frozen_copy"] = "disable_adapter",
    original_model: Optional[nn.Module] = None,
    return_outputs: bool = False,
) -> Callable[..., torch.Tensor]:
    """Get a callable for reference model forward pass.

    Args:
        model: The main model (may have LoRA adapters).
        reference_policy: How to get reference distribution:
            - "disable_adapter": Use model with adapters disabled (requires PEFT).
            - "frozen_copy": Use a frozen deepcopy of the model.
        original_model: Optional pre-created frozen model for "frozen_copy" policy.

    Returns:
        Callable that takes forward inputs (without labels) and returns logits.
    """
    # Check for PEFT/LoRA adapters
    has_adapters = hasattr(model, "disable_adapter")

    if reference_policy == "disable_adapter" and has_adapters:
        # Use adapter-disabled model
        def ref_forward(**forward_inputs) -> torch.Tensor:
            with _inference_eval_context(model):
                disable_adapter = model.disable_adapter
                if hasattr(disable_adapter, "__enter__") and hasattr(
                    disable_adapter, "__exit__"
                ):
                    context_manager = disable_adapter
                else:
                    context_manager = disable_adapter()
                with context_manager:
                    outputs = model(**forward_inputs)
                    return outputs if return_outputs else outputs.logits

        return ref_forward

    elif reference_policy == "frozen_copy" or not has_adapters:
        # Use frozen copy
        if original_model is None:
            # Create frozen copy
            original_model = deepcopy(model)
            original_model.eval()
            original_model.requires_grad_(False)

        def ref_forward(**forward_inputs) -> torch.Tensor:
            with _inference_eval_context(original_model):
                outputs = original_model(**forward_inputs)
                return outputs if return_outputs else outputs.logits

        return ref_forward

    else:
        raise ValueError(f"Unknown reference_policy: {reference_policy}")


# -----------------------------------------------------------------------------
# Internal: KL divergence computation
# -----------------------------------------------------------------------------


def _compute_kl_divergence(
    cur_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    model: Optional[nn.Module] = None,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    force_fp32: bool = True,
    kl_direction: Literal["forward", "reverse"] = "forward",
) -> torch.Tensor:
    """Compute per-token KL divergence (forward KL by default).

    KL naming is frequently confused due to PyTorch's kl_div signature
    (target is the weighting distribution). Definitions here follow standard
    math/RLHF convention:
      - Forward KL: KL(p_ref || p_cur), expectation over p_ref (mass-covering).
      - Reverse KL: KL(p_cur || p_ref), expectation over p_cur (mode-seeking).
    The original ASFT repo's paper text says "reverse KL" but its code uses
    F.kl_div(log(cur), ref), which is forward KL; we match the code behavior.
    Reverse KL is available via kl_direction="reverse".

    Args:
        cur_logits: Current model logits (B*T, V) or (B, T, V).
        ref_logits: Reference model logits (same shape as cur_logits).
        model: Model for extracting config (optional).
        logit_softcapping: Softcapping value.
        logit_scaling: Scaling value.
        force_fp32: Whether to compute in FP32 for stability.
        kl_direction: "forward" for KL(p_ref || p_cur), "reverse" for KL(p_cur || p_ref).

    Returns:
        Per-token KL divergence of shape (B*T,) or (B, T).
    """
    # Flatten if 3D
    if cur_logits.dim() == 3:
        batch, seq_len, vocab_size = cur_logits.shape
        cur_logits = cur_logits.reshape(batch * seq_len, vocab_size)
        ref_logits = ref_logits.reshape(batch * seq_len, vocab_size)

    # Apply effective logits transformation
    cur_eff = effective_logits(cur_logits, model, logit_softcapping, logit_scaling)
    ref_eff = effective_logits(ref_logits, model, logit_softcapping, logit_scaling)

    if force_fp32:
        cur_eff = cur_eff.float()
        ref_eff = ref_eff.float()

    if kl_direction == "forward":
        # Forward KL: KL(p_ref || p_cur)
        cur_logp = F.log_softmax(cur_eff, dim = -1)
        ref_p = F.softmax(ref_eff, dim = -1)
        # Using F.kl_div: kl_div(input=log_cur, target=ref) computes KL(ref || cur)
        kl = F.kl_div(cur_logp, ref_p, reduction = "none").sum(dim = -1)
    elif kl_direction == "reverse":
        # Reverse KL: KL(p_cur || p_ref)
        ref_logp = F.log_softmax(ref_eff, dim = -1)
        cur_p = F.softmax(cur_eff, dim = -1)
        kl = F.kl_div(ref_logp, cur_p, reduction = "none").sum(dim = -1)
    else:
        raise ValueError(f"Unknown kl_direction: {kl_direction}")

    return kl


# -----------------------------------------------------------------------------
# Internal: DFT weight computation
# -----------------------------------------------------------------------------


def _compute_dft_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model: Optional[nn.Module] = None,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    ignore_index: int = -100,
    ce_losses: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute DFT weights: probability of target token under current model.

    w = p(label) where p = softmax(effective_logits), detached.
    If ce_losses is provided, compute weights via exp(-ce_losses) to avoid
    a full softmax over the vocab.

    Args:
        logits: Model logits (B*T, V) or (B, T, V).
        labels: Labels tensor (B*T,) or (B, T).
        model: Model for extracting config.
        logit_softcapping: Softcapping value.
        logit_scaling: Scaling value.
        ignore_index: Index to ignore.
        ce_losses: Optional per-token CE losses aligned with labels.
        valid_mask: Optional mask of valid tokens (same shape as labels).

    Returns:
        DFT weights of same shape as labels, detached.
    """
    if ce_losses is not None:
        weights = torch.exp(-ce_losses.detach())
        if valid_mask is not None:
            weights = weights * valid_mask
        return weights

    # Flatten if 3D
    if logits.dim() == 3:
        batch, seq_len, vocab_size = logits.shape
        logits = logits.view(batch * seq_len, vocab_size)
        labels = labels.view(batch * seq_len)
    else:
        vocab_size = logits.shape[-1]

    # Apply effective logits transformation
    logits_eff = effective_logits(logits, model, logit_softcapping, logit_scaling)

    # Compute softmax probabilities
    p = F.softmax(logits_eff.float(), dim = -1)

    # Safe labels for gather (clamp -100 to 0)
    safe_labels = labels.clamp(min = 0, max = vocab_size - 1)

    # Gather probabilities at target positions
    weights = p.gather(dim = -1, index = safe_labels.unsqueeze(-1)).squeeze(-1)

    # Detach - weights should not receive gradients
    return weights.detach()


# -----------------------------------------------------------------------------
# Streaming helpers
# -----------------------------------------------------------------------------


def _unwrap_reference_outputs(
    ref_outputs: Any,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """Extract logits and past_key_values from reference outputs."""
    if hasattr(ref_outputs, "logits"):
        return ref_outputs.logits, getattr(ref_outputs, "past_key_values", None)
    if isinstance(ref_outputs, (tuple, list)) and len(ref_outputs) > 0:
        past_key_values = ref_outputs[1] if len(ref_outputs) > 1 else None
        return ref_outputs[0], past_key_values
    return ref_outputs, None


def _slice_batch_inputs(
    forward_inputs: Dict[str, Any],
    batch_size: int,
    b_start: int,
    b_end: int,
) -> Dict[str, Any]:
    """Slice batch-first tensors for microbatch processing."""
    mb_inputs = {}
    for key, value in forward_inputs.items():
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            mb_inputs[key] = value[b_start:b_end]
        else:
            mb_inputs[key] = value
    return mb_inputs


def _compute_kl_batch_micro(
    model: nn.Module,
    cur_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    ref_forward: Callable,
    forward_inputs: Dict[str, Any],
    microbatch_size: int,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    force_fp32: bool = True,
    kl_direction: Literal["forward", "reverse"] = "forward",
) -> torch.Tensor:
    """Compute KL using batch microbatching strategy.

    Processes reference forward in microbatches to reduce peak VRAM.

    Args:
        model: Current model.
        cur_logits: Current model logits (B, T, V).
        shift_labels: Shifted labels (B, T).
        valid_mask: Valid token mask (B, T).
        ref_forward: Reference forward callable.
        forward_inputs: Forward inputs (without labels).
        microbatch_size: Size of each microbatch.
        logit_softcapping: Softcapping value.
        logit_scaling: Scaling value.
        force_fp32: Whether to use FP32 for KL.
        kl_direction: "forward" for KL(p_ref || p_cur), "reverse" for KL(p_cur || p_ref).

    Returns:
        KL tensor of shape (B, T).
    """
    batch_size = cur_logits.shape[0]
    kl = torch.zeros_like(shift_labels, dtype = torch.float32)

    for b_start in range(0, batch_size, microbatch_size):
        b_end = min(b_start + microbatch_size, batch_size)

        # Slice inputs for microbatch
        mb_inputs = _slice_batch_inputs(forward_inputs, batch_size, b_start, b_end)

        # Get reference logits for microbatch
        ref_outputs_mb = ref_forward(**mb_inputs)
        ref_logits_mb, _ = _unwrap_reference_outputs(ref_outputs_mb)
        cur_logits_mb = cur_logits[b_start:b_end]

        # Compute KL for this microbatch
        kl_mb = _compute_kl_divergence(
            cur_logits_mb,
            ref_logits_mb,
            model,
            logit_softcapping,
            logit_scaling,
            force_fp32,
            kl_direction,
        )

        # Reshape if needed
        if kl_mb.dim() == 1:
            mb_batch = b_end - b_start
            kl_mb = kl_mb.reshape(mb_batch, -1)

        kl[b_start:b_end] = kl_mb

        # Free memory
        del ref_logits_mb

    return kl


def _compute_kl_seq_kv_cache(
    model: nn.Module,
    cur_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    ref_forward: Callable,
    forward_inputs: Dict[str, Any],
    seq_chunk_size: int,
    microbatch_size: Optional[int] = None,
    allow_auto_microbatch_fallback: bool = True,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    force_fp32: bool = True,
    kl_direction: Literal["forward", "reverse"] = "forward",
) -> torch.Tensor:
    """Compute KL using sequence chunking with KV cache strategy.

    Processes reference forward in sequence chunks to reduce peak VRAM.
    Falls back to full forward if model doesn't support caching.

    Args:
        model: Current model.
        cur_logits: Current model logits (B, T, V).
        shift_labels: Shifted labels (B, T).
        valid_mask: Valid token mask (B, T).
        ref_forward: Reference forward callable.
        forward_inputs: Forward inputs (without labels).
        seq_chunk_size: Size of each sequence chunk.
        microbatch_size: Optional microbatch size for batch dimension.
        allow_auto_microbatch_fallback: Allow automatic microbatch fallback on errors.
        logit_softcapping: Softcapping value.
        logit_scaling: Scaling value.
        force_fp32: Whether to use FP32 for KL.
        kl_direction: "forward" for KL(p_ref || p_cur), "reverse" for KL(p_cur || p_ref).

    Returns:
        KL tensor of shape (B, T).
    """
    batch_size, seq_len, vocab_size = cur_logits.shape
    device = cur_logits.device

    packed_seq_lengths = forward_inputs.get("packed_seq_lengths", None)
    if packed_seq_lengths is not None:
        # Avoid seq_kv_cache with packed sequences; fall back to batch/full reference.
        fallback_microbatch = None
        if microbatch_size is not None and microbatch_size < batch_size:
            fallback_microbatch = microbatch_size
        elif allow_auto_microbatch_fallback:
            fallback_microbatch = max(1, batch_size // _DEFAULT_REF_MICROBATCH_DIVISOR)
            if fallback_microbatch >= batch_size:
                fallback_microbatch = None
        if fallback_microbatch is not None:
            return _compute_kl_batch_micro(
                model,
                cur_logits,
                shift_labels,
                valid_mask,
                ref_forward,
                forward_inputs,
                fallback_microbatch,
                logit_softcapping,
                logit_scaling,
                force_fp32,
                kl_direction,
            )
        ref_outputs = ref_forward(**forward_inputs)
        ref_logits, _ = _unwrap_reference_outputs(ref_outputs)
        kl_full = _compute_kl_divergence(
            cur_logits,
            ref_logits,
            model,
            logit_softcapping,
            logit_scaling,
            force_fp32,
            kl_direction,
        )
        if kl_full.dim() == 1:
            kl_full = kl_full.reshape(batch_size, seq_len)
        return kl_full

    if microbatch_size is not None:
        microbatch_size = max(1, microbatch_size)
    if microbatch_size is not None and microbatch_size < batch_size:
        kl = torch.zeros(batch_size, seq_len, dtype = torch.float32, device = device)
        for b_start in range(0, batch_size, microbatch_size):
            b_end = min(b_start + microbatch_size, batch_size)
            mb_inputs = _slice_batch_inputs(forward_inputs, batch_size, b_start, b_end)
            kl_mb = _compute_kl_seq_kv_cache(
                model,
                cur_logits[b_start:b_end],
                shift_labels[b_start:b_end],
                valid_mask[b_start:b_end],
                ref_forward,
                mb_inputs,
                seq_chunk_size,
                microbatch_size = None,
                allow_auto_microbatch_fallback = False,
                logit_softcapping = logit_softcapping,
                logit_scaling = logit_scaling,
                force_fp32 = force_fp32,
                kl_direction = kl_direction,
            )
            if kl_mb.dim() == 1:
                mb_batch = b_end - b_start
                kl_mb = kl_mb.reshape(mb_batch, -1)
            kl[b_start:b_end] = kl_mb
        return kl

    kl = torch.zeros(batch_size, seq_len, dtype = torch.float32, device = device)

    # Process in chunks with KV cache
    past_key_values = None

    for s_start in range(0, seq_len, seq_chunk_size):
        s_end = min(s_start + seq_chunk_size, seq_len)

        # Build chunk inputs
        chunk_inputs = {}
        for key, value in forward_inputs.items():
            if key == "input_ids":
                chunk_inputs[key] = value[:, s_start:s_end]
            elif key == "attention_mask":
                # For chunked processing, need attention mask up to s_end
                chunk_inputs[key] = value[:, :s_end]
            elif key == "position_ids":
                chunk_inputs[key] = value[:, s_start:s_end]
            elif (
                torch.is_tensor(value)
                and value.dim() >= 2
                and value.shape[1] == seq_len
            ):
                chunk_inputs[key] = value[:, s_start:s_end]
            else:
                chunk_inputs[key] = value

        # Add past_key_values if available
        if past_key_values is not None:
            chunk_inputs["past_key_values"] = past_key_values

        chunk_inputs["use_cache"] = True

        try:
            # Get reference logits for chunk
            # Note: ref_forward may not support all these kwargs
            ref_outputs = ref_forward(**chunk_inputs)
            ref_logits_chunk, ref_past_key_values = _unwrap_reference_outputs(
                ref_outputs
            )
            if ref_past_key_values is None and s_end < seq_len:
                # Can't continue without cache; fall back to batch micro if allowed
                logger.warning(
                    "ASFT seq-KV streaming: reference returned no KV cache; "
                    "falling back to batch-micro/full reference forward."
                )
                fallback_microbatch = None
                if allow_auto_microbatch_fallback:
                    fallback_microbatch = (
                        microbatch_size
                        if microbatch_size is not None
                        else max(1, batch_size // _DEFAULT_REF_MICROBATCH_DIVISOR)
                    )
                if fallback_microbatch is not None and fallback_microbatch < batch_size:
                    return _compute_kl_batch_micro(
                        model,
                        cur_logits,
                        shift_labels,
                        valid_mask,
                        ref_forward,
                        forward_inputs,
                        fallback_microbatch,
                        logit_softcapping,
                        logit_scaling,
                        force_fp32,
                        kl_direction,
                    )
                ref_outputs = ref_forward(**forward_inputs)
                ref_logits, _ = _unwrap_reference_outputs(ref_outputs)
                kl_full = _compute_kl_divergence(
                    cur_logits,
                    ref_logits,
                    model,
                    logit_softcapping,
                    logit_scaling,
                    force_fp32,
                    kl_direction,
                )
                if kl_full.dim() == 1:
                    kl_full = kl_full.reshape(batch_size, seq_len)
                return kl_full
            past_key_values = ref_past_key_values

            cur_logits_chunk = cur_logits[:, s_start:s_end]

            # Compute KL for this chunk
            kl_chunk = _compute_kl_divergence(
                cur_logits_chunk,
                ref_logits_chunk,
                model,
                logit_softcapping,
                logit_scaling,
                force_fp32,
                kl_direction,
            )

            if kl_chunk.dim() == 1:
                chunk_len = s_end - s_start
                kl_chunk = kl_chunk.reshape(batch_size, chunk_len)

            kl[:, s_start:s_end] = kl_chunk

            del ref_logits_chunk

        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            # Fallback to batch micro or full forward on KV cache errors
            # These exceptions typically indicate the model doesn't support
            # the chunked KV cache approach (e.g., missing past_key_values support)
            logger.warning(
                "ASFT seq-KV streaming failed (%s: %s); falling back to "
                "batch-micro/full reference forward.",
                type(e).__name__,
                e,
            )
            fallback_microbatch = None
            if allow_auto_microbatch_fallback:
                fallback_microbatch = (
                    microbatch_size
                    if microbatch_size is not None
                    else max(1, batch_size // _DEFAULT_REF_MICROBATCH_DIVISOR)
                )
            if fallback_microbatch is not None and fallback_microbatch < batch_size:
                return _compute_kl_batch_micro(
                    model,
                    cur_logits,
                    shift_labels,
                    valid_mask,
                    ref_forward,
                    forward_inputs,
                    fallback_microbatch,
                    logit_softcapping,
                    logit_scaling,
                    force_fp32,
                    kl_direction,
                )
            ref_outputs = ref_forward(**forward_inputs)
            ref_logits, _ = _unwrap_reference_outputs(ref_outputs)
            kl_full = _compute_kl_divergence(
                cur_logits,
                ref_logits,
                model,
                logit_softcapping,
                logit_scaling,
                force_fp32,
                kl_direction,
            )
            if kl_full.dim() == 1:
                kl_full = kl_full.reshape(batch_size, seq_len)
            return kl_full

    return kl


# -----------------------------------------------------------------------------
# A5) Helper: resolve effective mode under KL gating (local fix #1)
# -----------------------------------------------------------------------------


def _compute_kl_seq_chunked(
    cur_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    *,
    model: Optional[nn.Module] = None,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    force_fp32: bool = True,
    kl_direction: Literal["forward", "reverse"] = "forward",
    seq_chunk_size: int = _DEFAULT_SEQ_CHUNK_SIZE,
    checkpoint: bool = True,
) -> torch.Tensor:
    """Per-token KL over the sequence, computed in chunks of ``seq_chunk_size``.

    ``ref_logits`` is a no-grad constant (a single full reference forward); only the
    current-model side carries gradient. When ``checkpoint`` is set and the current logits
    require grad, each chunk's KL is wrapped in ``torch.utils.checkpoint`` so its full-vocab
    fp32 buffers (``effective_logits``/``log_softmax`` over the 262k vocab) are recomputed in
    backward instead of retained — bounding both the transient (one chunk) and the retained
    (~0) fp32 footprint, so full-vocab fp32 KL fits a 24GB GPU.

    Mathematically identical to a single full ``_compute_kl_divergence`` (per-token KL is
    independent across positions; checkpoint recompute is deterministic — no dropout/RNG).
    Returns ``(B, T)``.
    """
    batch_size, seq_len, _ = cur_logits.shape
    out = cur_logits.new_zeros((batch_size, seq_len), dtype = torch.float32)
    use_ckpt = checkpoint and cur_logits.requires_grad
    for start in range(0, seq_len, max(1, seq_chunk_size)):
        end = min(start + seq_chunk_size, seq_len)
        cur_chunk = cur_logits[:, start:end]
        ref_chunk = ref_logits[:, start:end]
        if use_ckpt:
            # ref_logits comes from the reference forward run under torch.inference_mode();
            # checkpoint SAVES its inputs for backward and inference-mode tensors cannot be
            # saved, so clone the (chunk-sized, cheap) reference slice to a normal tensor. The
            # current-model slice is a view of the grad-tracked logits and saves fine.
            ref_chunk = ref_chunk.clone()

            # Closure captures the non-tensor args so checkpoint only sees tensors (no
            # "non-tensor input" warnings); recompute is exact (no RNG in the KL math).
            def _kl_fn(cur_c, ref_c):
                return _compute_kl_divergence(
                    cur_c, ref_c, model, logit_softcapping, logit_scaling,
                    force_fp32, kl_direction,
                )

            kl_chunk = _grad_checkpoint(_kl_fn, cur_chunk, ref_chunk, use_reentrant = False)
        else:
            kl_chunk = _compute_kl_divergence(
                cur_chunk, ref_chunk, model, logit_softcapping, logit_scaling,
                force_fp32, kl_direction,
            )
        out[:, start:end] = kl_chunk.reshape(batch_size, end - start)
    return out


def resolve_effective_mode(
    asft_mode: Literal["sft", "dft", "sft+kl", "asft"],
    kl_weight: float,
) -> Tuple[Literal["sft", "dft", "sft+kl", "asft"], bool]:
    """Resolve the effective loss mode and whether the KL term is active.

    The KL/reference path is only meaningful when the mode requests KL *and*
    kl_weight is non-zero. When kl_weight == 0.0 we collapse:
        - "asft"   -> "dft"  (DFT-equivalent ablation, no reference forward)
        - "sft+kl" -> "sft"  (plain CE, no reference forward)
    so the caller can skip the reference forward and any frozen-copy allocation.

    Returns (effective_mode, kl_active).
    """
    if asft_mode in ("sft+kl", "asft") and kl_weight != 0.0:
        return asft_mode, True
    if asft_mode == "asft":
        return "dft", False
    if asft_mode == "sft+kl":
        return "sft", False
    return asft_mode, False


# -----------------------------------------------------------------------------
# A6) Core: compute_asft_loss
# -----------------------------------------------------------------------------


def compute_asft_loss(
    model: nn.Module,
    inputs: Dict[str, Any],
    *,
    asft_mode: Literal["sft", "dft", "sft+kl", "asft"] = "asft",
    kl_weight: float = 0.0,
    kl_direction: Literal["forward", "reverse"] = "forward",
    reference_policy: Literal["disable_adapter", "frozen_copy"] = "disable_adapter",
    streaming_config: Optional[ASFTStreamingConfig] = None,
    original_model: Optional[nn.Module] = None,
    normalize_by: Literal["tokens", "weights"] = "tokens",
    checkpoint_kl: bool = True,
    return_outputs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
    """Compute ASFT loss.

    This is the main entry point for ASFT loss computation.

    Args:
        model: The model to train.
        inputs: Input dictionary containing input_ids, labels, etc.
        asft_mode: Loss mode:
            - "sft": Standard CE loss
            - "dft": CE weighted by model confidence
            - "sft+kl": CE + KL divergence from reference
            - "asft": DFT + KL divergence (full ASFT)
        kl_weight: Weight for KL term (only for sft+kl and asft modes).
        kl_direction: "forward" for KL(p_ref || p_cur), "reverse" for KL(p_cur || p_ref).
        reference_policy: How to get reference distribution.
        streaming_config: Configuration for streaming strategies.
        original_model: Optional pre-created frozen reference model.
        normalize_by: "tokens" (default, matches reference) or "weights" for DFT/ASFT.
        return_outputs: Whether to return model outputs alongside loss.

    Returns:
        Loss tensor, or (loss, outputs) tuple if return_outputs=True.
    """
    if streaming_config is None:
        streaming_config = ASFTStreamingConfig()

    # Resolve streaming mode (new API) vs legacy enabled/ref_strategy
    mode = streaming_config.mode
    if mode is None:
        _streaming_enabled = streaming_config.enabled
        _ref_strategy = streaming_config.ref_strategy
    else:
        if mode == "off":
            _streaming_enabled = False
            _ref_strategy = "none"
        elif mode == "batch":
            _streaming_enabled = True
            _ref_strategy = "batch_micro"
        elif mode in ("seq", "auto", "hybrid"):
            _streaming_enabled = True
            _ref_strategy = "seq_kv_cache"
        else:
            raise ValueError(f"Unknown streaming mode: {mode}")

    # Get model config for softcapping/scaling
    logit_softcapping, logit_scaling = _resolve_logit_params(
        model,
        None,
        None,
    )

    # Build forward inputs (without labels/num_items to force logits materialization)
    forward_inputs = {
        k: v for k, v in inputs.items() if k not in {"labels", "num_items_in_batch"}
    }

    # Main forward pass - ASFT always needs logits
    outputs = model(**forward_inputs)
    logits = outputs.logits  # (B, T, V)

    # Get labels and build shift_labels Unsloth-style
    labels = inputs["labels"]
    packed_seq_lengths = inputs.get("packed_seq_lengths", None)
    shift_labels = build_shift_labels(labels, packed_seq_lengths)

    # Valid mask and normalization
    valid_mask = shift_labels != -100
    n_items_tokens = inputs.get("num_items_in_batch", None)
    if n_items_tokens is None:
        n_items_tokens = valid_mask.sum()
    n_items_tokens = max(n_items_tokens, 1)  # Avoid division by zero

    # Handle edge case: no valid tokens
    if valid_mask.sum() == 0:
        zero_loss = logits.sum() * 0.0
        if return_outputs:
            return zero_loss, outputs
        return zero_loss

    # Compute per-token CE loss
    ce_losses, _ = fast_cross_entropy_loss_per_token(
        logits, shift_labels, logit_softcapping, logit_scaling
    )
    # Reshape to (B, T)
    batch_size, seq_len = shift_labels.shape
    ce_losses = ce_losses.view(batch_size, seq_len)

    # Initialize token losses. Resolve KL gating first (local fix #1): when
    # kl_weight == 0.0 the KL/reference path is skipped entirely and "asft"/
    # "sft+kl" collapse to "dft"/"sft".
    effective_mode, kl_active = resolve_effective_mode(asft_mode, kl_weight)
    dft_weights = None
    if effective_mode == "sft":
        # Standard SFT: just CE
        token_loss = ce_losses

    elif effective_mode == "dft":
        # DFT: CE weighted by model confidence
        dft_weights = _compute_dft_weights(
            logits,
            shift_labels,
            model,
            logit_softcapping,
            logit_scaling,
            ce_losses = ce_losses,
            valid_mask = valid_mask,
        )
        dft_weights = dft_weights.view(batch_size, seq_len)
        token_loss = ce_losses * dft_weights

    elif effective_mode in ("sft+kl", "asft"):
        # KL divergence to the reference policy. One full reference forward (no-grad), then the
        # per-token KL is computed in sequence chunks, each gradient-checkpointed when
        # ``checkpoint_kl`` is set so the full-vocab fp32 logit buffers are recomputed in
        # backward rather than retained — this keeps gemma-4's 262k-vocab fp32 KL within a 24GB
        # GPU. The legacy batch_micro / seq_kv_cache reference-streaming paths are subsumed by
        # this (left defined above for reference); chunk size honours
        # ``streaming_config.seq_chunk_size``.
        ref_forward = get_reference_forward_callable(
            model,
            reference_policy,
            original_model,
            return_outputs = False,
        )
        ref_outputs = ref_forward(**forward_inputs)
        ref_logits, _ = _unwrap_reference_outputs(ref_outputs)
        kl = _compute_kl_seq_chunked(
            logits,
            ref_logits,
            model = model,
            logit_softcapping = logit_softcapping,
            logit_scaling = logit_scaling,
            force_fp32 = streaming_config.force_fp32_kl,
            kl_direction = kl_direction,
            seq_chunk_size = streaming_config.seq_chunk_size or _DEFAULT_SEQ_CHUNK_SIZE,
            checkpoint = checkpoint_kl,
        )
        kl = kl.reshape(batch_size, seq_len)
        del ref_logits

        if effective_mode == "sft+kl":
            # SFT + KL
            token_loss = ce_losses + kl_weight * kl
        else:
            # Full ASFT: DFT + KL
            dft_weights = _compute_dft_weights(
                logits,
                shift_labels,
                model,
                logit_softcapping,
                logit_scaling,
                ce_losses = ce_losses,
                valid_mask = valid_mask,
            )
            dft_weights = dft_weights.view(batch_size, seq_len)
            dft_loss = ce_losses * dft_weights
            token_loss = dft_loss + kl_weight * kl

    else:
        raise ValueError(f"Unknown asft_mode: {asft_mode}")

    # Final reduction: sum over valid tokens, divide by chosen normalizer.
    normalizer = n_items_tokens
    if normalize_by == "weights" and dft_weights is not None:
        weight_sum = dft_weights[valid_mask].sum()
        normalizer = weight_sum.clamp_min(1e-8)
    elif normalize_by != "tokens":
        raise ValueError(f"Unknown normalize_by: {normalize_by}")
    loss = token_loss[valid_mask].sum() / normalizer

    if return_outputs:
        return loss, outputs
    return loss
