"""Local ASFT trainer.

A thin subclass of unsloth's ``UnslothTrainer`` that routes ``compute_loss``
through the vendored ASFT objective in :mod:`afterlife.trainer.asft` without requiring
an Unsloth fork.

It mirrors ``ASFTTrainer`` from unslothai/unsloth PR #4239, adapted for the
current TRL/Transformers stack and the vendored loss module. Local changes vs.
the PR:

* ``kl_weight == 0.0`` no longer allocates a frozen reference copy — the KL term
  is inactive, so there is nothing to be a reference *for* (review fix #2).
* When ASFT is enabled we force ``UNSLOTH_RETURN_LOGITS=1`` so the model
  materialises full per-position logits. Setting this before ``super().__init__``
  also makes unsloth's ``SFTTrainer`` init disable padding-free / sample-packing
  (see ``unsloth.trainer._patch_sft_trainer_auto_packing``), which would
  otherwise flatten the batch and break the ``(B, T)`` assumptions in
  ``compute_asft_loss``.

With ``asft_enabled=False`` (the default) this class is behaviourally identical
to ``UnslothTrainer``: no environment change, and ``compute_loss`` delegates to
the parent.
"""

from __future__ import annotations

import os
import warnings
from copy import deepcopy
from typing import Literal, Optional

from unsloth.trainer import UnslothTrainer

from afterlife.trainer.asft import (
    ASFTStreamingConfig,
    compute_asft_loss,
    resolve_effective_mode,
)

__all__ = ["ASFTTrainer", "ASFTStreamingConfig"]

# The vendored loss accepts the full 4-mode set (incl. the "sft" debug alias);
# qlora's config exposes only the {"dft", "sft+kl", "asft"} subset.
_AsftMode = Literal["sft", "dft", "sft+kl", "asft"]
_KlDirection = Literal["forward", "reverse"]
_ReferencePolicy = Literal["disable_adapter", "frozen_copy"]
_NormalizeBy = Literal["tokens", "weights"]


class ASFTTrainer(UnslothTrainer):
    """``UnslothTrainer`` + an optional Anchored-SFT loss path."""

    def __init__(
        self,
        *args,
        asft_enabled: bool = False,
        asft_mode: _AsftMode = "asft",
        kl_weight: float = 0.0,
        kl_direction: _KlDirection = "forward",
        reference_policy: _ReferencePolicy = "disable_adapter",
        asft_streaming: Optional[ASFTStreamingConfig] = None,
        normalize_by: _NormalizeBy = "tokens",
        **kwargs,
    ):
        if asft_enabled:
            # ASFT reads outputs.logits from a label-free forward; force unsloth to
            # return full logits (and, as a side effect, disable padding-free /
            # packing at SFTTrainer init time).
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        super().__init__(*args, **kwargs)

        self.asft_enabled: bool = asft_enabled
        self.asft_mode: _AsftMode = asft_mode
        self.kl_weight: float = kl_weight
        self.kl_direction: _KlDirection = kl_direction
        self.reference_policy: _ReferencePolicy = reference_policy
        self.asft_streaming: ASFTStreamingConfig = (
            asft_streaming or ASFTStreamingConfig()
        )
        self.normalize_by: _NormalizeBy = normalize_by
        # Lazily created only if a frozen reference copy is actually required.
        self._asft_original_model = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ASFT disabled -> behave exactly like the parent trainer.
        if not self.asft_enabled:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        num_items_in_batch = kwargs.get("num_items_in_batch")
        if num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch

        _, kl_active = resolve_effective_mode(self.asft_mode, self.kl_weight)

        # Allocate a frozen reference copy only when KL is active (review fix #2)
        # and a frozen copy is the only way to obtain a reference distribution.
        if kl_active:
            needs_frozen_copy = self.reference_policy == "frozen_copy" or (
                self.reference_policy == "disable_adapter"
                and not hasattr(model, "disable_adapter")
            )
            if needs_frozen_copy and self._asft_original_model is None:
                if self.reference_policy == "frozen_copy":
                    warnings.warn(
                        "ASFT: creating a frozen copy of the model for the KL "
                        "reference. This doubles VRAM; prefer 'disable_adapter' "
                        "for LoRA.",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "ASFT: 'disable_adapter' unavailable on this model; "
                        "falling back to a frozen copy for the KL reference. "
                        "This doubles VRAM.",
                        stacklevel=2,
                    )
                self._asft_original_model = deepcopy(model)
                self._asft_original_model.eval()
                self._asft_original_model.requires_grad_(False)

        return compute_asft_loss(
            model=model,
            inputs=inputs,
            asft_mode=self.asft_mode,
            kl_weight=self.kl_weight,
            kl_direction=self.kl_direction,
            reference_policy=self.reference_policy,
            streaming_config=self.asft_streaming,
            original_model=self._asft_original_model,
            normalize_by=self.normalize_by,
            return_outputs=return_outputs,
        )
