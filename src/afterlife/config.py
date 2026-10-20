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
# Known training checkpoint -> full-precision model to serve with vLLM.
KNOWN_SERVE_MODELS = {
    "unsloth/Qwen3-8B-unsloth-bnb-4bit": "Qwen/Qwen3-8B",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-7B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

DEFAULT_CONFIG_TOML = """\
# Afterlife operator config. The agent can read this; only you should edit it.

[model]
# Training checkpoint (Transformers / Unsloth BnB; not GGUF/AWQ/GPTQ).
base_model = "{base_model}"
# What vLLM serves with the adapter on top. Current vLLM cannot load
# bitsandbytes checkpoints, so this is normally the full-precision original;
# a LoRA trained on the 4-bit checkpoint applies to it as usual (QLoRA).
serve_model = "{serve_model}"
# Longer is better for agent transcripts; lower it if training runs out of memory.
max_seq_length = 4096

[eval]
# Per-probe generation budget. Thinking models need room to finish thinking.
probe_max_new_tokens = 2048
# Held-out / train rows scored for loss (each is one forward pass per variant).
max_loss_rows = 200
# Training prompts re-asked after training, answers shown next to the target
# (one turn per experience, first ones first).
recall_rows = 8

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
# Fraction of sessions held out for evaluation, applied only to experiences
# picked by the agent's mode or policy. Explicit nominations are never held
# out; eval_only experiences always are. Re-split for every request.
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
asft_kl_weight = 0.1
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
class EvalConfig:
    probe_max_new_tokens: int = 2048
    max_loss_rows: int = 200
    recall_rows: int = 8


@dataclass
class Config:
    base_model: str
    serve_model: str
    max_seq_length: int
    operator: OperatorConfig
    eval: EvalConfig
    methods: dict[str, dict[str, Any]]
    raw: dict[str, Any]


def render_default_config(base_model: str = DEFAULT_BASE_MODEL, serve_model: str | None = None) -> str:
    serve_model = serve_model or KNOWN_SERVE_MODELS.get(base_model, base_model)
    return DEFAULT_CONFIG_TOML.format(base_model=base_model, serve_model=serve_model)


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

    eval_raw = raw.get("eval", {})
    unknown = sorted(set(eval_raw) - set(EvalConfig.__dataclass_fields__))
    if unknown:
        raise ConfigError(f"unknown [eval] keys: {', '.join(unknown)}")

    return Config(
        base_model=base_model,
        serve_model=model.get("serve_model") or base_model,
        max_seq_length=int(model.get("max_seq_length", 4096)),
        operator=operator,
        eval=EvalConfig(**eval_raw),
        methods={k: dict(v) for k, v in raw.get("methods", {}).items()},
        raw=raw,
    )
