"""Operator configuration (`.afterlife/config.toml`).

The agent can read this file but the CLI never writes to it after `init`.
Everything that bounds what the agent may do lives here.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

DEFAULT_CONFIG_TOML = """\
# Afterlife operator config. The agent can read this; only you should edit it.

[model]
# Training checkpoint (Transformers / Unsloth BnB; not GGUF/AWQ/GPTQ).
base_model = "{base_model}"
max_seq_length = 2048

[operator]
# Master switch for the training worker.
training_enabled = true
# May the agent activate / roll back adapters itself? If false, only
# `afterlife activate --operator` works.
agent_may_activate = true
# Training methods the agent may choose from.
allowed_methods = ["sft", "dft", "asft"]
# Hard caps applied to every run, whatever the agent asks for.
max_examples_per_run = 500
max_steps = 300
min_minutes_between_runs = 0
# Drop candidates whose verification.status is not "verified".
require_verified = false
# Fraction of sessions held out for evaluation (split by session, not row).
eval_fraction = 0.15
# Max characters per training example (all messages together).
max_example_chars = 12000
# Optional general-text dataset mixed into training to limit forgetting,
# e.g. "wikitext" style HF dataset name. Empty disables rehearsal.
rehearsal_dataset = ""
rehearsal_rows_per_epoch = 0

# Per-method defaults. The agent may override the knobs listed in
# afterlife.recipes.PARAM_BOUNDS within those bounds.
[methods.sft]
learning_rate = 2e-4
max_steps = 100
lora_rank = 32

[methods.dft]
learning_rate = 2e-4
max_steps = 100
lora_rank = 32

[methods.asft]
learning_rate = 2e-4
max_steps = 100
lora_rank = 32
asft_kl_weight = 0.05
"""


class ConfigError(Exception):
    pass


@dataclass
class OperatorConfig:
    training_enabled: bool = True
    agent_may_activate: bool = True
    allowed_methods: list[str] = field(default_factory=lambda: ["sft", "dft", "asft"])
    max_examples_per_run: int = 500
    max_steps: int = 300
    min_minutes_between_runs: float = 0
    require_verified: bool = False
    eval_fraction: float = 0.15
    max_example_chars: int = 12000
    rehearsal_dataset: str = ""
    rehearsal_rows_per_epoch: int = 0


@dataclass
class Config:
    base_model: str
    max_seq_length: int
    operator: OperatorConfig
    methods: dict[str, dict[str, Any]]
    raw: dict[str, Any]


def render_default_config(base_model: str = DEFAULT_BASE_MODEL) -> str:
    return DEFAULT_CONFIG_TOML.format(base_model=base_model)


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"no config at {path}; run `afterlife init` first") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    model = raw.get("model", {})
    base_model = model.get("base_model")
    if not isinstance(base_model, str) or not base_model:
        raise ConfigError("[model].base_model must be set")

    known = OperatorConfig.__dataclass_fields__
    op_raw = raw.get("operator", {})
    unknown = sorted(set(op_raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown [operator] keys: {', '.join(unknown)}")
    operator = OperatorConfig(**op_raw)
    if not 0 <= operator.eval_fraction < 1:
        raise ConfigError("[operator].eval_fraction must be in [0, 1)")

    return Config(
        base_model=base_model,
        max_seq_length=int(model.get("max_seq_length", 2048)),
        operator=operator,
        methods={k: dict(v) for k, v in raw.get("methods", {}).items()},
        raw=raw,
    )
