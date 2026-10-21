"""Training methods the agent can choose between, and the knobs it may turn.

The agent picks a method and, optionally, a few hyperparameters. The operator
config supplies per-method defaults and hard caps; anything outside the bounds
below is rejected rather than silently clamped, so the agent learns the limits.
"""

from __future__ import annotations

import math
from typing import Any

from afterlife.config import Config

METHODS: dict[str, dict[str, str]] = {
    "sft": {
        "summary": "Standard supervised fine-tuning (cross-entropy on the final assistant turn).",
        "notes": "The only one of the three that learned genuinely new facts in our tests, but hard to dose: "
        "a little too little and nothing sticks, a little too much and the new material leaks into "
        "unrelated answers. Sensitive to learning rate and step count.",
    },
    "dft": {
        "summary": "Dynamic fine-tuning: cross-entropy weighted by the model's own token probability.",
        "notes": "Down-weights tokens the model finds unlikely. Good at reinforcing things you already "
        "half-do; poor at new facts (in our tests it learned the shape of answers but not their content, "
        "and degraded at high learning rate x steps).",
    },
    "asft": {
        "summary": "Anchored SFT: DFT plus a KL penalty that keeps the adapter close to the base model.",
        "notes": "Tolerated a much wider range of hyperparameters than SFT and stayed stable in every "
        "setting we tried. Best for refining existing behaviour; in our tests it did not pick up new "
        "facts within 150 steps. Costs an extra forward pass per step. asft_kl_weight=0 reduces it to DFT.",
    },
}

# Each optimizer step sees BATCH_SIZE x GRAD_ACCUM rows; passed to the trainer
# explicitly so passes-per-example below stays true.
BATCH_SIZE, GRAD_ACCUM = 2, 4

# name -> (type, lo, hi, methods it applies to or None for all)
PARAM_BOUNDS: dict[str, tuple[type, float, float, tuple[str, ...] | None]] = {
    "learning_rate": (float, 1e-6, 1e-3, None),
    "max_steps": (int, 1, 10_000, None),  # further capped by [operator].max_steps
    "epochs": (float, 0.1, 100, None),  # passes over the training rows; sets max_steps
    "lora_rank": (int, 4, 64, None),  # 64 matches the vLLM --max-lora-rank we document
    "lora_alpha": (int, 1, 256, None),
    "warmup_steps": (int, 0, 1000, None),
    "seed": (int, 0, 2**31, None),
    "asft_kl_weight": (float, 0.0, 10.0, ("asft",)),
}


class RecipeError(Exception):
    """A recipe that can't run. ``refused`` marks the operator's limits, as opposed to bad input."""

    def __init__(self, message: str, *, refused: bool = False) -> None:
        super().__init__(message)
        self.refused = refused


def steps_per_pass(train_rows: int) -> int:
    """Optimizer steps for one pass over train_rows (the last step of a pass may be partial)."""
    return max(1, math.ceil(math.ceil(train_rows / BATCH_SIZE) / GRAD_ACCUM))


def rows_per_pass(config: Config, train_rows: int) -> int:
    """Rows in one pass: the selection's target turns plus any rehearsal rows."""
    op = config.operator
    return train_rows + (op.rehearsal_rows_per_epoch if op.rehearsal_dataset else 0)


def passes_per_example(max_steps: int, train_rows: int) -> float:
    return round(max_steps / steps_per_pass(train_rows), 2) if train_rows else 0.0


def describe_methods(config: Config) -> dict[str, Any]:
    allowed = config.operator.allowed_methods
    return {
        "methods": {
            name: {**info, "allowed": name in allowed, "defaults": config.methods.get(name, {})}
            for name, info in METHODS.items()
        },
        "tunable_params": {
            name: {"type": t.__name__, "min": lo, "max": hi, "methods": list(m) if m else "all"}
            for name, (t, lo, hi, m) in PARAM_BOUNDS.items()
        },
        "operator_caps": {
            "max_steps": config.operator.max_steps,
            "max_examples_per_run": config.operator.max_examples_per_run,
        },
        "steps_and_passes": f"Each step trains on {BATCH_SIZE * GRAD_ACCUM} rows (one row per target turn). "
        "With a fixed max_steps, a small selection is seen many times and a large one only a few: "
        "`preview` and `train --request` report passes_per_example. Set epochs instead of max_steps "
        "to fix the number of passes; the steps are then worked out from the selection.",
    }


def parse_param(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise RecipeError(f"expected key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if key not in PARAM_BOUNDS:
        raise RecipeError(f"{key!r} is not tunable; tunable: {', '.join(PARAM_BOUNDS)}")
    kind = PARAM_BOUNDS[key][0]
    try:
        value = kind(float(raw)) if kind is int and "e" in raw.lower() else kind(raw)
    except ValueError as exc:
        raise RecipeError(f"{key} must be {kind.__name__}") from exc
    return key, value


def resolve_recipe(
    config: Config, method: str, overrides: dict[str, Any], train_rows: int | None = None
) -> dict[str, Any]:
    """Merge operator defaults with agent overrides; validate against bounds and caps.

    ``epochs`` is turned into max_steps using ``train_rows``; without it (no
    selection known yet) only the bounds are checked.
    """
    if method not in METHODS:
        raise RecipeError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    if method not in config.operator.allowed_methods:
        raise RecipeError(f"method {method!r} is disabled by the operator", refused=True)

    params: dict[str, Any] = dict(config.methods.get(method, {}))
    for key, value in overrides.items():
        if key not in PARAM_BOUNDS:
            raise RecipeError(f"{key!r} is not tunable")
        kind, lo, hi, methods = PARAM_BOUNDS[key]
        if methods and method not in methods:
            raise RecipeError(f"{key} only applies to {', '.join(methods)}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RecipeError(f"{key} must be a number")
        if kind is int and value != int(value):
            raise RecipeError(f"{key} must be an integer")
        if not lo <= value <= hi:
            raise RecipeError(f"{key}={value} outside [{lo}, {hi}]")
        params[key] = kind(value)

    if "epochs" in overrides and "max_steps" in overrides:
        raise RecipeError("set epochs or max_steps, not both")
    cap = config.operator.max_steps
    if "epochs" in params and train_rows:
        if "max_steps" not in overrides:
            params["max_steps"] = max(1, math.ceil(params["epochs"] * steps_per_pass(train_rows)))
        if params["max_steps"] > cap:
            most = cap / steps_per_pass(train_rows)
            raise RecipeError(
                f"epochs={params['epochs']:g} over {train_rows} rows needs {params['max_steps']} steps; "
                f"operator cap is {cap} steps (at most {most:.2f} epochs for this selection)",
                refused=True,
            )
    params.setdefault("max_steps", 100)
    if params["max_steps"] > cap:
        raise RecipeError(f"max_steps={params['max_steps']} exceeds operator cap {cap}", refused=True)
    params.setdefault("lora_alpha", params.get("lora_rank", 32))
    recipe = {"method": method, "params": params, "overrides": sorted(overrides)}
    if train_rows:
        recipe["train_rows"] = train_rows
        recipe["passes_per_example"] = passes_per_example(params["max_steps"], train_rows)
    return recipe


def train_sft_kwargs(config: Config, recipe: dict[str, Any]) -> dict[str, Any]:
    """Translate a resolved recipe into keyword arguments for trainer.qlora.train_sft."""
    p = recipe["params"]
    kwargs: dict[str, Any] = {
        "inner_loss": recipe["method"],
        "max_seq_length": config.max_seq_length,
        "max_steps": p["max_steps"],
        "learning_rate": p.get("learning_rate", 2e-4),
        "num_train_epochs": p.get("num_train_epochs", 1),  # operator config only (rehearsal variety)
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "lora_rank": p.get("lora_rank", 32),
        "lora_alpha": p["lora_alpha"],
        "warmup_steps": p.get("warmup_steps", 5),
        "seed": p.get("seed", 42),
        "export_format": "lora",
    }
    if recipe["method"] == "asft":
        kwargs["asft_mode"] = "asft"
        kwargs["asft_kl_weight"] = p.get("asft_kl_weight", 0.1)
        if kwargs["asft_kl_weight"] == 0:
            # ASFT without its KL anchor is DFT; the fused DFT path gets there
            # without ASFT's full-vocabulary logits.
            kwargs.update(inner_loss="dft", dft_impl="fused")
            del kwargs["asft_mode"], kwargs["asft_kl_weight"]
    if config.operator.rehearsal_dataset and config.operator.rehearsal_rows_per_epoch > 0:
        kwargs["rehearsal_dataset"] = config.operator.rehearsal_dataset
        kwargs["rehearsal_rows_per_epoch"] = config.operator.rehearsal_rows_per_epoch
    return kwargs
