"""Training methods the agent can choose between, and the knobs it may turn.

The agent picks a method and, optionally, a few hyperparameters. The operator
config supplies per-method defaults and hard caps; anything outside the bounds
below is rejected rather than silently clamped, so the agent learns the limits.
"""

from __future__ import annotations

from typing import Any

from afterlife.config import Config

METHODS: dict[str, dict[str, str]] = {
    "sft": {
        "summary": "Standard supervised fine-tuning (cross-entropy on the final assistant turn).",
        "notes": "Fast and simple. In NSL experiments it was sensitive to learning rate and step count: "
        "too much and it overwrites unrelated behaviour, too little and nothing sticks.",
    },
    "dft": {
        "summary": "Dynamic fine-tuning: cross-entropy weighted by the model's own token probability.",
        "notes": "Down-weights tokens the model finds very unlikely, which tends to make updates gentler than SFT.",
    },
    "asft": {
        "summary": "Anchored SFT: DFT plus a KL penalty that keeps the adapter close to the base model.",
        "notes": "In NSL experiments it tolerated a much wider range of hyperparameters than SFT. "
        "Costs extra compute for the reference forward pass. asft_kl_weight=0 reduces it to DFT.",
    },
}

# name -> (type, lo, hi, methods it applies to or None for all)
PARAM_BOUNDS: dict[str, tuple[type, float, float, tuple[str, ...] | None]] = {
    "learning_rate": (float, 1e-6, 1e-3, None),
    "max_steps": (int, 1, 10_000, None),  # further capped by [operator].max_steps
    "num_train_epochs": (int, 1, 10, None),
    "lora_rank": (int, 4, 64, None),  # 64 matches the vLLM --max-lora-rank we document
    "lora_alpha": (int, 1, 256, None),
    "warmup_steps": (int, 0, 1000, None),
    "seed": (int, 0, 2**31, None),
    "asft_kl_weight": (float, 0.0, 10.0, ("asft",)),
}


class RecipeError(Exception):
    pass


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


def resolve_recipe(config: Config, method: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge operator defaults with agent overrides; validate against bounds and caps."""
    if method not in METHODS:
        raise RecipeError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    if method not in config.operator.allowed_methods:
        raise RecipeError(f"method {method!r} is disabled by the operator")

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

    params.setdefault("max_steps", 100)
    if params["max_steps"] > config.operator.max_steps:
        raise RecipeError(
            f"max_steps={params['max_steps']} exceeds operator cap {config.operator.max_steps}"
        )
    params.setdefault("lora_alpha", params.get("lora_rank", 32))
    return {"method": method, "params": params, "overrides": sorted(overrides)}


def train_sft_kwargs(config: Config, recipe: dict[str, Any]) -> dict[str, Any]:
    """Translate a resolved recipe into keyword arguments for trainer.qlora.train_sft."""
    p = recipe["params"]
    kwargs: dict[str, Any] = {
        "inner_loss": recipe["method"],
        "max_seq_length": config.max_seq_length,
        "max_steps": p["max_steps"],
        "learning_rate": p.get("learning_rate", 2e-4),
        "num_train_epochs": p.get("num_train_epochs", 1),
        "lora_rank": p.get("lora_rank", 32),
        "lora_alpha": p["lora_alpha"],
        "warmup_steps": p.get("warmup_steps", 5),
        "seed": p.get("seed", 42),
        "export_format": "lora",
    }
    if recipe["method"] == "asft":
        kwargs["asft_mode"] = "asft"
        kwargs["asft_kl_weight"] = p.get("asft_kl_weight", 0.1)
    if config.operator.rehearsal_dataset and config.operator.rehearsal_rows_per_epoch > 0:
        kwargs["rehearsal_dataset"] = config.operator.rehearsal_dataset
        kwargs["rehearsal_rows_per_epoch"] = config.operator.rehearsal_rows_per_epoch
    return kwargs
