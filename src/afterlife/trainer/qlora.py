from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence

from dotenv import dotenv_values, find_dotenv
from loguru import logger
from afterlife.trainer.artifacts import (
    ConfiguredTrainingExportFormat,
    TrainingExportFormat,
    resolve_training_export_format as _resolve_training_export_format,
)


DEFAULT_MAX_SEQ_LENGTH = 2048
IGNORE_INDEX = -100
InnerLoss = Literal["sft", "dft", "asft"]
DftImpl = Literal["fused", "trl"]
AsftMode = Literal["dft", "sft+kl", "asft"]
AsftKlDirection = Literal["forward", "reverse"]
AsftReferencePolicy = Literal["disable_adapter", "frozen_copy"]
AsftStreamingMode = Literal["off", "auto", "batch", "seq", "hybrid"]
AsftNormalizeBy = Literal["tokens", "weights"]
SFTConfig: Any | None = None
SFTTrainer: Any | None = None
ASFTTrainer: Any | None = None
ASFTStreamingConfig: Any | None = None
load_dataset: Any | None = None
SEQUENCE_FEATURE_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "mm_token_type_ids",
)
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Fully-qualified regex that scopes LoRA to the language model only.
# VLMs like Gemma 4 (Gemma4ForConditionalGeneration) expose vision_tower.* modules
# with the same projection suffixes; matching by suffix attaches LoRA to those too,
# and vLLM rejects such adapters (vision tower keys are not in its expected set).
LORA_TARGET_MODULES_LM_ONLY_REGEX = (
    r"^(?!.*vision_tower).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES = (
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
)


class _FastLanguageModelProxy:
    """Import Unsloth only when training actually loads or patches the model."""

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        from unsloth import FastLanguageModel as _FastLanguageModel

        return _FastLanguageModel.from_pretrained(*args, **kwargs)

    def get_peft_model(self, *args: Any, **kwargs: Any) -> Any:
        from unsloth import FastLanguageModel as _FastLanguageModel

        return _FastLanguageModel.get_peft_model(*args, **kwargs)


FastLanguageModel = _FastLanguageModelProxy()


class _SimpleDataset(list[dict[str, Any]]):
    @classmethod
    def from_list(cls, rows: list[dict[str, Any]]) -> "_SimpleDataset":
        return cls(rows)

    @property
    def column_names(self) -> list[str]:
        names: list[str] = []
        for row in self:
            for key in row:
                if key not in names:
                    names.append(key)
        return names


class _SimpleSFTConfig:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "lr_scheduler_type" and isinstance(value, str):
                value = SimpleNamespace(value=value)
            setattr(self, key, value)


def _dataset_from_list(rows: list[dict[str, Any]]) -> Any:
    try:
        from datasets import Dataset
    except ImportError:
        return _SimpleDataset.from_list(rows)
    return Dataset.from_list(rows)


def _load_sft_classes() -> tuple[Any, Any]:
    global SFTConfig, SFTTrainer
    if SFTTrainer is not None and SFTConfig is None:
        return _SimpleSFTConfig, SFTTrainer
    if SFTConfig is None:
        from trl.trainer.sft_config import SFTConfig as _SFTConfig

        SFTConfig = _SFTConfig
    if SFTTrainer is None:
        from trl.trainer.sft_trainer import SFTTrainer as _SFTTrainer

        SFTTrainer = _SFTTrainer
    return SFTConfig, SFTTrainer


def _load_asft_classes() -> tuple[Any, Any, Any]:
    global SFTConfig, ASFTTrainer, ASFTStreamingConfig
    try:
        if SFTConfig is None:
            from trl.trainer.sft_config import SFTConfig as _SFTConfig

            SFTConfig = _SFTConfig
        if ASFTTrainer is None or ASFTStreamingConfig is None:
            from afterlife.trainer.asft_trainer import (
                ASFTStreamingConfig as _ASFTStreamingConfig,
                ASFTTrainer as _ASFTTrainer,
            )

            ASFTTrainer = _ASFTTrainer
            ASFTStreamingConfig = _ASFTStreamingConfig
    except ImportError as exc:
        raise RuntimeError(
            "inner_loss='asft' requires the vendored ASFT modules "
            "(afterlife/trainer/asft.py + afterlife/trainer/asft_trainer.py); import failed."
        ) from exc
    return SFTConfig, ASFTTrainer, ASFTStreamingConfig


def _install_fused_dft() -> None:
    from afterlife.trainer.dft_fused import install_fused_dft

    install_fused_dft(ignore_index=IGNORE_INDEX)


def _load_dataset(name: str, *, split: str) -> Any:
    global load_dataset
    if load_dataset is None:
        from datasets import load_dataset as _hf_load_dataset

        load_dataset = _hf_load_dataset
    return load_dataset(name, split=split)


def _export_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(distribution_name: str) -> str | None:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _package_vcs_commit(distribution_name: str) -> str | None:
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    vcs_info = payload.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    for key in ("commit_id", "requested_revision"):
        value = vcs_info.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _training_package_versions() -> dict[str, str | None]:
    return {
        "unsloth_version": _package_version("unsloth"),
        "unsloth_zoo_version": _package_version("unsloth-zoo"),
        "trl_version": _package_version("trl"),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "torch_version": _package_version("torch"),
        "bitsandbytes_version": _package_version("bitsandbytes"),
        "triton_version": _package_version("triton"),
    }


def _unsloth_asft_fork_commit() -> str | None:
    # Provenance of the vendored ASFT source (Approach D): prefer an explicit env
    # override, else the commit the loss module was vendored from.
    override = os.environ.get("UNSLOTH_ASFT_FORK_COMMIT")
    if override:
        return override
    try:
        from afterlife.trainer.asft import ASFT_SOURCE_COMMIT

        return ASFT_SOURCE_COMMIT
    except Exception:
        return _package_vcs_commit("unsloth")


def _qualified_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if isinstance(module, str) and isinstance(name, str):
        return f"{module}.{name}"
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def _validate_training_base_model(base_model: str) -> None:
    lower_model = base_model.lower()
    invalid_format = None
    if lower_model.endswith(".gguf") or "gguf" in lower_model:
        invalid_format = "GGUF"
    elif "awq" in lower_model:
        invalid_format = "AWQ"
    elif "gptq" in lower_model:
        invalid_format = "GPTQ"

    if invalid_format is None:
        return

    examples = ", ".join(
        repr(example) for example in SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES
    )
    raise ValueError(
        f"Unsupported training base model '{base_model}': {invalid_format} is an inference/export format, not a Transformers training checkpoint. "
        f"Use a Transformers or Unsloth BnB model instead, for example {examples}."
    )


def _load_wandb_api_key_from_dotenv() -> str | None:
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return None

    value = dotenv_values(dotenv_path).get("WANDB_API_KEY")
    if not isinstance(value, str) or not value:
        return None
    return value


def _cleanup_torch_cuda_state() -> None:
    gc.collect()

    import torch

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()


def _mixed_precision_kwargs() -> dict[str, bool]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {"bf16": False, "fp16": False}

    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return {"bf16": bf16, "fp16": not bf16}


def _parse_max_memory_env() -> dict[int | str, str] | None:
    """Optional per-device weight-placement cap, e.g. NSL_TRAIN_MAX_MEMORY="0=5GiB,1=22GiB".

    Returned dict is passed straight to ``from_pretrained(max_memory=...)``, which
    skews accelerate's ``device_map="balanced"`` placement: a LOW cap on a device
    keeps fewer weight layers there, reserving runtime headroom. This is the lever
    for the ASFT/KL path, whose full-vocab logits + their backward gradient pile
    onto whichever GPU holds the ``lm_head`` — capping that GPU low pushes
    transformer blocks onto the other GPU so the logits peak fits. Default (unset)
    keeps the plain balanced split, so the SFT/DFT paths are unaffected.
    """
    raw = os.environ.get("NSL_TRAIN_MAX_MEMORY")
    if not raw:
        return None
    out: dict[int | str, str] = {}
    for pair in raw.split(","):
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        out[int(key) if key.isdigit() else key] = value
    return out or None


def _load_base_model(
    base_model: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
):
    _validate_training_base_model(base_model)
    _cleanup_torch_cuda_state()

    load_kwargs: dict[str, Any] = dict(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="balanced",
    )
    max_memory = _parse_max_memory_env()
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
        logger.info("Skewed weight placement via NSL_TRAIN_MAX_MEMORY={}", max_memory)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except RuntimeError as exc:
        if "No config file found" in str(exc):
            examples = ", ".join(
                repr(example) for example in SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES
            )
            raise RuntimeError(
                f"Failed to load training base model '{base_model}'. "
                f"Use a Transformers-compatible training checkpoint or an Unsloth BnB checkpoint, for example {examples}."
            ) from exc
        raise
    device_map_resolved = getattr(model, "hf_device_map", None)
    if device_map_resolved:
        per_device: dict[Any, int] = {}
        for dev in device_map_resolved.values():
            per_device[dev] = per_device.get(dev, 0) + 1
        logger.info("Resolved hf_device_map module counts per device: {}", per_device)
    return model, tokenizer


def _resolve_target_modules(model: Any, configured: Any) -> Any:
    """Resolve regex targets to leaf names, or paths when leaves are ambiguous."""
    if not isinstance(configured, str):
        return configured
    pattern = re.compile(configured)
    named = [name for name, _ in model.named_modules() if name]
    matched = [name for name in named if pattern.fullmatch(name)]
    if not matched:
        raise ValueError(
            f"LoRA target_modules regex {configured!r} matched no modules in the "
            "base model. Check the pattern against the model's module names."
        )
    matched_set = set(matched)
    leaf_names = {name.rsplit(".", 1)[-1] for name in matched}
    overmatch = any(
        name not in matched_set and name.rsplit(".", 1)[-1] in leaf_names
        for name in named
    )
    return sorted(matched) if overmatch else sorted(leaf_names)


def _attach_lora_adapter(model: Any, config: dict[str, Any]) -> Any:
    target_modules = _resolve_target_modules(
        model, config.get("target_modules", LORA_TARGET_MODULES_LM_ONLY_REGEX)
    )
    return FastLanguageModel.get_peft_model(
        model,
        r=config.get("rank", 32),
        target_modules=target_modules,
        lora_alpha=config.get("alpha", 32),
        lora_dropout=config.get("dropout", 0),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_training_export_format(
    model_name: str,
    configured_format: ConfiguredTrainingExportFormat = "auto",
) -> TrainingExportFormat:
    resolved = _resolve_training_export_format(model_name, configured_format)
    if model_name.strip().split(":", 1)[0] == "vllm" and resolved == "merged_16bit":
        logger.warning(
            "vLLM merged_16bit override selected. This will usually be slower than "
            "serving the AWQ base model with a LoRA adapter."
        )

    return resolved


def _save_training_artifact(
    model: Any,
    tokenizer: Any,
    output_path: Path,
    *,
    export_format: TrainingExportFormat,
    gguf_quantize: str = "f16",
) -> Path:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "gguf":
        model.save_pretrained_gguf(
            output_path, tokenizer, quantization_method=gguf_quantize
        )
        return output_path

    assert export_format in ["lora", "merged_16bit", "merged_4bit"], (
        f"Unexpected export format {export_format} for finetuned model"
    )

    if export_format == "lora":
        # NOTE: unsloth/FastLanguageModel's save_pretrained_merged is bugged.
        # Use PEFT's save_pretrained directly — save_pretrained_merged always
        # merges LoRA into the base weights (downloading full safetensors),
        # even when save_method="lora".
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    else:
        model.save_pretrained_merged(output_path, tokenizer, save_method=export_format)
    return output_path


def _training_metadata_path(artifact_path: Path) -> Path:
    if artifact_path.suffix == ".gguf":
        return artifact_path.with_suffix(".training_info.json")
    return artifact_path / "training_info.json"


def _as_1d_int_list(value: Any, *, key: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"Encoded field '{key}' is not a sequence")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Encoded field '{key}' is batched with {len(value)} rows")
        value = value[0]
    return [int(item) for item in value]


def _encoded_sequence(encoded: Any, key: str) -> list[int] | None:
    if isinstance(encoded, dict):
        value = encoded.get(key)
    else:
        value = getattr(encoded, key, None)
    if value is None:
        return None
    return _as_1d_int_list(value, key=key)


def _is_gemma_vlm_processor(tokenizer: Any) -> bool:
    class_name = tokenizer.__class__.__name__.lower()
    return (
        "gemma" in class_name
        and hasattr(tokenizer, "tokenizer")
        and hasattr(tokenizer, "image_processor")
    )


def _encode_text(tokenizer: Any, text: str) -> dict[str, list[int]]:
    try:
        encoded = tokenizer(text=text, truncation=False, padding=False)
    except Exception:  # noqa: BLE001 - VLM processors and text tokenizers differ here.
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if text_tokenizer is tokenizer:
            raise
        encoded = text_tokenizer(text, truncation=False, padding=False)

    row: dict[str, list[int]] = {}
    for key in SEQUENCE_FEATURE_KEYS:
        sequence = _encoded_sequence(encoded, key)
        if sequence is not None:
            row[key] = sequence

    input_ids = row.get("input_ids")
    if input_ids is None:
        raise ValueError("Tokenizer did not return input_ids for SFT row")

    for key, sequence in row.items():
        if len(sequence) != len(input_ids):
            raise ValueError(
                f"Encoded field '{key}' has length {len(sequence)} but input_ids has length {len(input_ids)}"
            )

    # Gemma 4 is a text+multimodal architecture whose training forward may require
    # text-only token type tensors. Add the semantically correct all-zero values
    # when the processor did not emit them, while avoiding these extra kwargs for
    # ordinary text-only models.
    if _is_gemma_vlm_processor(tokenizer):
        zeros = [0] * len(input_ids)
        row.setdefault("token_type_ids", zeros)
        row.setdefault("mm_token_type_ids", zeros)

    return row


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    return _encode_text(tokenizer, text)["input_ids"]


def _context_messages(prompt: str | None, context: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    # Afterlife: a row is either NSL2's single prompt or a full conversation
    # (system/user/assistant turns) leading up to the response.
    if context is not None:
        keep = ("role", "content", "tool_calls", "tool_call_id", "name")
        return [{k: m[k] for k in keep if k in m} for m in context]
    if prompt is None:
        raise ValueError("SFT row needs a prompt or context messages")
    return [{"role": "user", "content": prompt}]


def _build_query_response_sft_row(
    tokenizer: Any,
    *,
    prompt: str | None = None,
    raw_response: str,
    context: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Afterlife: render the conversation with the model's own chat template, so
    # multi-turn context looks exactly as it will at inference. Put the loss on
    # the response content and the end-of-turn marker
    # only. Everything the template renders before the response (the assistant
    # turn marker, an empty Qwen3 <think></think> block, Gemma's model-turn
    # scaffold) is treated as prompt, and whitespace after the end-of-turn token
    # is dropped: the model never generates either at inference. This keeps
    # thinking models from learning "skip thinking" from answer-only examples.
    # Falls back to the NSL2 boundary below when the response text cannot be
    # located on a token boundary in the rendered conversation.
    messages = _context_messages(prompt, context)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": raw_response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    start = full_text.rfind(raw_response) if raw_response else -1
    if start > 0:
        prefix_ids = _tokenize_text(tokenizer, full_text[:start])
        row = _encode_text(tokenizer, full_text)
        input_ids = row["input_ids"]
        if 0 < len(prefix_ids) < len(input_ids) and input_ids[: len(prefix_ids)] == prefix_ids:
            end = len(input_ids)
            decode = getattr(getattr(tokenizer, "tokenizer", tokenizer), "decode", None)
            if decode is not None:
                while end > len(prefix_ids) + 1 and not decode(input_ids[end - 1 : end]).strip():
                    end -= 1
            row["completion_mask"] = (
                [0] * len(prefix_ids)
                + [1] * (end - len(prefix_ids))
                + [0] * (len(input_ids) - end)
            )
            row["length"] = len(input_ids)
            return row
    return _build_query_response_sft_row_nsl(
        tokenizer, prompt=prompt, raw_response=raw_response, context=context
    )


def _build_query_response_sft_row_nsl(
    tokenizer: Any,
    *,
    prompt: str | None = None,
    raw_response: str,
    context: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Render the masked-prompt boundary WITHOUT the generation prompt. Gemma 4's
    # chat template, with add_generation_prompt=True, appends a "thinking channel"
    # scaffold ("<start_of_turn>model\n<|channel>thought\n<channel|>") that the
    # assistant-message rendering below does NOT emit (it places the response
    # directly after "<start_of_turn>model\n"). Using the generation prompt here
    # therefore makes prompt_text diverge from full_text right after the model turn
    # marker, so it is not a token-prefix and the mask cannot be built. The
    # assistant-style rendering (add_generation_prompt=False) is the user turn only,
    # which IS a clean prefix of the user+assistant conversation; the completion
    # then covers the model turn marker plus the response.
    messages = _context_messages(prompt, context)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": raw_response}],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = _tokenize_text(tokenizer, prompt_text)
    row = _encode_text(tokenizer, full_text)
    input_ids = row["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Cannot build completion-only SFT mask: tokenized query template is not "
            "a prefix of the tokenized query+response template. Check the chat template."
        )

    completion_len = len(input_ids) - len(prompt_ids)
    if completion_len <= 0:
        raise ValueError("Cannot build completion-only SFT row with no response tokens")

    row["completion_mask"] = [0] * len(prompt_ids) + [1] * completion_len
    # Expose the token length so a length-grouped sampler (group_by_length) can
    # batch similar-length rows together; this is TrainingArguments' default
    # length_column_name and is ignored by the collator.
    row["length"] = len(input_ids)
    return row


def _completion_survives_truncation(
    row: dict[str, Any],
    *,
    max_seq_length: int | None,
) -> bool:
    if max_seq_length is None:
        return True
    completion_mask = _as_1d_int_list(row["completion_mask"], key="completion_mask")
    return 1 in completion_mask[:max_seq_length]


def _build_completion_only_labels(
    *,
    input_ids: Sequence[int],
    completion_mask: Sequence[int],
    attention_mask: Sequence[int] | None = None,
) -> list[int]:
    if len(input_ids) != len(completion_mask):
        raise ValueError("input_ids and completion_mask must have the same length")
    if attention_mask is not None and len(input_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask must have the same length")

    labels: list[int] = []
    for index, token_id in enumerate(input_ids):
        is_completion = bool(completion_mask[index])
        is_attended = attention_mask is None or bool(attention_mask[index])
        labels.append(int(token_id) if is_completion and is_attended else IGNORE_INDEX)
    return labels


def _pad_token_id(tokenizer: Any) -> int:
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(text_tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer must define pad_token_id or eos_token_id")


class _CompletionOnlyDataCollator:
    def __init__(self, *, pad_token_id: int, max_seq_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    @staticmethod
    def _pad(values: Sequence[int], *, length: int, value: int) -> list[int]:
        return list(values) + [value] * (length - len(values))

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not examples:
            raise ValueError("Cannot collate an empty SFT batch")

        rows: list[dict[str, list[int]]] = []
        for example in examples:
            input_ids = _as_1d_int_list(example["input_ids"], key="input_ids")
            completion_mask = _as_1d_int_list(
                example["completion_mask"], key="completion_mask"
            )
            if len(input_ids) != len(completion_mask):
                raise ValueError(
                    "input_ids and completion_mask must have the same length"
                )

            row: dict[str, list[int]] = {}
            truncated_len = min(len(input_ids), self.max_seq_length)
            row["input_ids"] = input_ids[:truncated_len]
            row["completion_mask"] = completion_mask[:truncated_len]

            attention_mask = example.get("attention_mask")
            if attention_mask is None:
                row["attention_mask"] = [1] * truncated_len
            else:
                attention = _as_1d_int_list(attention_mask, key="attention_mask")
                if len(attention) != len(input_ids):
                    raise ValueError(
                        "input_ids and attention_mask must have the same length"
                    )
                row["attention_mask"] = attention[:truncated_len]

            for key in ("token_type_ids", "mm_token_type_ids"):
                if key not in example:
                    continue
                values = _as_1d_int_list(example[key], key=key)
                if len(values) != len(input_ids):
                    raise ValueError(f"input_ids and {key} must have the same length")
                row[key] = values[:truncated_len]

            row["labels"] = _build_completion_only_labels(
                input_ids=row["input_ids"],
                completion_mask=row["completion_mask"],
                attention_mask=row["attention_mask"],
            )
            rows.append(row)

        batch_length = max(len(row["input_ids"]) for row in rows)
        output: dict[str, Any] = {}
        output["input_ids"] = torch.tensor(
            [
                self._pad(
                    row["input_ids"], length=batch_length, value=self.pad_token_id
                )
                for row in rows
            ],
            dtype=torch.long,
        )
        output["attention_mask"] = torch.tensor(
            [
                self._pad(row["attention_mask"], length=batch_length, value=0)
                for row in rows
            ],
            dtype=torch.long,
        )
        output["labels"] = torch.tensor(
            [
                self._pad(row["labels"], length=batch_length, value=IGNORE_INDEX)
                for row in rows
            ],
            dtype=torch.long,
        )
        for key in ("token_type_ids", "mm_token_type_ids"):
            if not all(key in row for row in rows):
                continue
            output[key] = torch.tensor(
                [self._pad(row[key], length=batch_length, value=0) for row in rows],
                dtype=torch.long,
            )
        return output


def _make_completion_only_data_collator(
    tokenizer: Any, *, max_seq_length: int
) -> _CompletionOnlyDataCollator:
    return _CompletionOnlyDataCollator(
        pad_token_id=_pad_token_id(tokenizer),
        max_seq_length=max_seq_length,
    )


def _is_message_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
            for m in value
        )
    )


def _load_self_generated_sft_rows(
    training_data_paths: Sequence[Path],
    tokenizer: Any,
    *,
    max_seq_length: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped_fully_truncated = 0
    for training_data_path in training_data_paths:
        resolved_path = Path(training_data_path).expanduser().resolve()
        with open(resolved_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                if not payload.get("success"):
                    continue

                prompt = payload.get("prompt")
                raw_response = payload.get("raw_response")
                context = payload.get("messages")  # Afterlife: conversation before the response
                if not isinstance(raw_response, str):
                    continue
                if context is not None:
                    if not _is_message_list(context):
                        continue
                    prompt = None
                elif not isinstance(prompt, str):
                    continue

                # Keep one full-conversation encoding to avoid the Gemma 4
                # prompt/completion concat crash, but attach an explicit mask so
                # only response tokens contribute to the loss.
                row = _build_query_response_sft_row(
                    tokenizer,
                    prompt=prompt,
                    raw_response=raw_response,
                    context=context,
                )
                # Drop rows whose prompt alone fills the sequence window: after the
                # collator truncates to the first max_seq_length tokens, no response
                # token survives, so every label is IGNORE_INDEX. Such a row carries
                # zero training signal and, alone in a batch (bs=1), divides the loss
                # by zero valid tokens -> nan that poisons gradient accumulation.
                if not _completion_survives_truncation(
                    row,
                    max_seq_length=max_seq_length,
                ):
                    dropped_fully_truncated += 1
                    continue
                rows.append(row)

    if dropped_fully_truncated:
        logger.warning(
            "Dropped {} self-generated SFT row(s) whose response is fully truncated "
            "at max_seq_length={} (prompt alone fills the window); kept {}.",
            dropped_fully_truncated,
            max_seq_length,
            len(rows),
        )
    return rows


def _tokenized_length(tokenizer: Any, text: str) -> int:
    return len(_tokenize_text(tokenizer, text))


def _warn_if_dataset_would_truncate(
    dataset: Any,
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> None:
    longest_tokens = 0
    longest_index = -1
    truncated_count = 0
    fully_masked_after_truncation = 0
    for index, row in enumerate(dataset):
        completion_mask = None
        if isinstance(row, dict) and isinstance(row.get("input_ids"), list):
            token_count = len(row["input_ids"])
            if isinstance(row.get("completion_mask"), list):
                completion_mask = row["completion_mask"]
        else:
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                continue
            token_count = _tokenized_length(tokenizer, text)
        if token_count > longest_tokens:
            longest_tokens = token_count
            longest_index = index
        if token_count > max_seq_length:
            truncated_count += 1
            if (
                completion_mask is not None
                and 1 not in completion_mask[:max_seq_length]
            ):
                fully_masked_after_truncation += 1

    if truncated_count == 0:
        return

    message = (
        f"Training dataset would truncate {truncated_count}/{len(dataset)} rows with "
        f"max_seq_length={max_seq_length}; longest row is {longest_tokens} tokens "
        f"at dataset index {longest_index}. Set max_seq_length to at least "
        f"{longest_tokens} to preserve every row, or deliberately pre-clip/export "
        "shorter training examples before SFT."
    )
    if fully_masked_after_truncation:
        message += (
            f" {fully_masked_after_truncation} rows would have no response tokens "
            "left after truncation."
        )
    logger.warning(message)


def _load_rehearsal_sft_rows(
    *,
    tokenizer: Any,
    dataset_name: str,
    split: str,
    text_field: str,
    rows_per_epoch: int,
    virtual_epochs: int,
    seed: int,
    prompt: str,
    prompt_chars: int,
    max_chars: int,
    max_seq_length: int | None = None,
) -> list[list[dict[str, Any]]]:
    if rows_per_epoch <= 0:
        return [[] for _ in range(virtual_epochs)]

    rehearsal_dataset = _load_dataset(dataset_name, split=split)
    valid_texts: list[str] = []
    for row in rehearsal_dataset:
        value = row.get(text_field)
        if not isinstance(value, str):
            continue
        text = " ".join(value.split())
        if text:
            valid_texts.append(text[:max_chars])

    if not valid_texts:
        raise ValueError(
            f"No rehearsal text found in dataset '{dataset_name}' field '{text_field}'"
        )

    rows_by_epoch: list[list[dict[str, Any]]] = []
    dropped_fully_truncated = 0
    for epoch in range(virtual_epochs):
        rng = random.Random(seed + epoch)
        if rows_per_epoch <= len(valid_texts):
            selected_texts = rng.sample(valid_texts, rows_per_epoch)
        else:
            selected_texts = [rng.choice(valid_texts) for _ in range(rows_per_epoch)]

        epoch_rows: list[dict[str, Any]] = []
        for text in selected_texts:
            excerpt = text[:prompt_chars].rstrip()
            row_prompt = prompt.rstrip()
            if excerpt:
                row_prompt += f":\n\n{excerpt}"
            else:
                row_prompt += "."
            row = _build_query_response_sft_row(
                tokenizer,
                prompt=row_prompt,
                raw_response=text,
            )
            if not _completion_survives_truncation(
                row,
                max_seq_length=max_seq_length,
            ):
                dropped_fully_truncated += 1
                continue
            epoch_rows.append(row)
        rows_by_epoch.append(epoch_rows)

    if dropped_fully_truncated:
        kept = sum(len(epoch_rows) for epoch_rows in rows_by_epoch)
        logger.warning(
            "Dropped {} rehearsal SFT row(s) whose response is fully truncated "
            "at max_seq_length={} (prompt alone fills the window); kept {}.",
            dropped_fully_truncated,
            max_seq_length,
            kept,
        )
    return rows_by_epoch


def _load_sft_dataset(
    training_data_paths: Sequence[Path],
    tokenizer: Any,
    *,
    max_seq_length: int | None = None,
    virtual_epochs: int = 1,
    rehearsal_dataset: str | None = None,
    rehearsal_split: str = "train",
    rehearsal_text_field: str = "text",
    rehearsal_rows_per_epoch: int = 0,
    rehearsal_seed: int = 3407,
    rehearsal_prompt: str = "Continue the following passage",
    rehearsal_prompt_chars: int = 256,
    rehearsal_max_chars: int = 2048,
) -> Any:
    """Load generation JSONL files plus optional rehearsal rows into an SFT Dataset."""

    if virtual_epochs < 1:
        raise ValueError("virtual_epochs must be at least 1")

    rows = _load_self_generated_sft_rows(
        training_data_paths, tokenizer, max_seq_length=max_seq_length
    )

    if not rows:
        raise ValueError("No training rows found")

    rehearsal_rows_by_epoch: list[list[dict[str, Any]]]
    if rehearsal_dataset:
        rehearsal_rows_by_epoch = _load_rehearsal_sft_rows(
            tokenizer=tokenizer,
            dataset_name=rehearsal_dataset,
            split=rehearsal_split,
            text_field=rehearsal_text_field,
            rows_per_epoch=rehearsal_rows_per_epoch,
            virtual_epochs=virtual_epochs,
            seed=rehearsal_seed,
            prompt=rehearsal_prompt,
            prompt_chars=rehearsal_prompt_chars,
            max_chars=rehearsal_max_chars,
            max_seq_length=max_seq_length,
        )
    else:
        rehearsal_rows_by_epoch = [[] for _ in range(virtual_epochs)]

    expanded_rows: list[dict[str, Any]] = []
    for epoch in range(virtual_epochs):
        expanded_rows.extend(rows)
        expanded_rows.extend(rehearsal_rows_by_epoch[epoch])

    return _dataset_from_list(expanded_rows)


def train_sft(
    base_model: str,
    training_data_paths: Sequence[str],
    output_dir: str,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    max_steps: int = 50,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    weight_decay: float = 0.0,
    logging_steps: int = 1,
    num_train_epochs: int = 1,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    seed: int = 42,
    rehearsal_dataset: str | None = None,
    rehearsal_split: str = "train",
    rehearsal_text_field: str = "text",
    rehearsal_rows_per_epoch: int = 0,
    rehearsal_seed: int | None = None,
    rehearsal_prompt: str = "Continue the following passage",
    rehearsal_prompt_chars: int = 256,
    rehearsal_max_chars: int = 2048,
    wandb_project: str | None = None,
    export_format: TrainingExportFormat = "lora",
    gguf_quantize: str = "f16",
    save_steps: int = 0,
    resume_from_checkpoint: bool = False,
    group_by_length: bool = False,
    inner_loss: InnerLoss = "sft",
    dft_impl: DftImpl = "fused",
    asft_mode: AsftMode = "asft",
    asft_kl_weight: float = 0.0,
    asft_kl_direction: AsftKlDirection = "forward",
    asft_reference_policy: AsftReferencePolicy = "disable_adapter",
    asft_streaming: AsftStreamingMode = "off",
    asft_ref_microbatch_size: int | None = None,
    asft_seq_chunk_size: int | None = None,
    asft_normalize_by: AsftNormalizeBy = "tokens",
) -> Path:
    """Run SFT on the provided generation window and save a trained LoRA adapter."""

    if export_format != "lora" and (
        (save_steps and save_steps > 0) or resume_from_checkpoint
    ):
        raise ValueError(
            "Training checkpointing is only supported for LoRA adapter exports. "
            "Use export_format='lora' for checkpointed runs, then perform any "
            "larger merged_16bit or GGUF export from the final adapter."
        )

    if inner_loss == "dft" and dft_impl == "fused":
        os.environ["UNSLOTH_ENABLE_CCE"] = "0"
        # Pin the fused-loss chunk count. Forcing UNSLOTH_ENABLE_CCE=0 routes DFT through
        # Unsloth's DENSE chunked path (compute_fused_ce_loss), which materializes per-chunk
        # [chunk, vocab] fp32 logits. Its default auto-sizer targets ~50% of *free* VRAM,
        # which is wildly optimistic once the 31B-4bit shard fills the card → it barely chunks
        # and can OOM on 24 GB devices. The CCE/NLL path
        # never hit this because it streams over vocab. UNSLOTH_CE_LOSS_N_CHUNKS is read at
        # cross_entropy_loss import time, so it MUST be set before `import unsloth` below.
        # 16 chunks @ seq 8192 ⇒ 512 tok/chunk ⇒ ~0.5 GiB logits/chunk; setdefault lets the
        # launch override it. (gemma-4: vocab 262144.)
        os.environ.setdefault("UNSLOTH_CE_LOSS_N_CHUNKS", "16")

    # Import unsloth FIRST — before trl/transformers/peft are pulled in by
    # _load_sft_classes / _load_base_model below. Unsloth patches the gemma-4
    # modeling code (incl. the VLM image-token path) at import time; if trl or
    # transformers import first, the patches don't apply and the text-only SFT
    # forward index-asserts on device ("index out of bounds: 0 <= tmp < 1" in the
    # embedding lookup). This restores the proven nsl2 import order. See memory
    # sft-qlora-prompt-completion-crash.
    import unsloth  # noqa: F401

    if inner_loss == "dft" and dft_impl == "fused":
        _install_fused_dft()

    if wandb_project:
        if not os.environ.get("WANDB_API_KEY"):
            dotenv_api_key = _load_wandb_api_key_from_dotenv()
            if dotenv_api_key:
                os.environ["WANDB_API_KEY"] = dotenv_api_key
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "wandb_project is set but WANDB_API_KEY is not in the environment. "
                "Export WANDB_API_KEY, or unset wandb_project to disable W&B tracking."
            )
        os.environ["WANDB_PROJECT"] = wandb_project
        report_to = "wandb"
    else:
        report_to = "none"

    if inner_loss == "asft":
        sft_config_cls, sft_trainer_cls, asft_streaming_config_cls = (
            _load_asft_classes()
        )
    else:
        sft_config_cls, sft_trainer_cls = _load_sft_classes()
        asft_streaming_config_cls = None

    trainer_class = _qualified_name(sft_trainer_cls)
    package_versions = _training_package_versions()
    logger.info(
        "Training objective: inner_loss={}, dft_impl={}, trainer={}",
        inner_loss,
        dft_impl if inner_loss == "dft" else None,
        trainer_class,
    )
    logger.info(
        "Training package versions: unsloth={}, unsloth_zoo={}, trl={}, transformers={}, peft={}, torch={}",
        package_versions["unsloth_version"],
        package_versions["unsloth_zoo_version"],
        package_versions["trl_version"],
        package_versions["transformers_version"],
        package_versions["peft_version"],
        package_versions["torch_version"],
    )
    if inner_loss == "asft":
        asft_kl_active = asft_mode in ("sft+kl", "asft") and asft_kl_weight != 0.0
        logger.info(
            "ASFT active: mode={}, kl_weight={}, kl_active={}, kl_direction={}, reference_policy={}, streaming={}, normalize_by={}",
            asft_mode,
            asft_kl_weight,
            asft_kl_active,
            asft_kl_direction,
            asft_reference_policy,
            asft_streaming,
            asft_normalize_by,
        )
        if asft_mode == "asft" and asft_kl_weight == 0.0:
            logger.warning(
                "ASFT selected with kl_weight=0.0; this is a DFT-equivalent ablation."
            )

    model, tokenizer = _load_base_model(
        base_model,
        max_seq_length=max_seq_length,
    )
    if inner_loss == "dft" and dft_impl == "fused":
        # Backstop: the per-model compiled module imports the loss fns by value at model-load.
        # The source patch above runs first so it normally suffices; re-bind here in case the
        # compiled module was already imported. The fail-loud engagement log confirms which ran.
        from afterlife.trainer.dft_fused import _repatch_generated_modules

        _repatch_generated_modules()
    model = _attach_lora_adapter(
        model,
        {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout},
    )

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if export_format != "gguf":
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_training_paths = [
        str(Path(path).expanduser().resolve()) for path in training_data_paths
    ]
    dataset = _load_sft_dataset(
        [Path(path) for path in resolved_training_paths],
        tokenizer,
        max_seq_length=max_seq_length,
        virtual_epochs=num_train_epochs,
        rehearsal_dataset=rehearsal_dataset,
        rehearsal_split=rehearsal_split,
        rehearsal_text_field=rehearsal_text_field,
        rehearsal_rows_per_epoch=rehearsal_rows_per_epoch,
        rehearsal_seed=rehearsal_seed if rehearsal_seed is not None else seed,
        rehearsal_prompt=rehearsal_prompt,
        rehearsal_prompt_chars=rehearsal_prompt_chars,
        rehearsal_max_chars=rehearsal_max_chars,
    )
    _warn_if_dataset_would_truncate(
        dataset,
        tokenizer,
        max_seq_length=max_seq_length,
    )
    row_count = len(dataset)
    effective_warmup_steps = 0 if warmup_ratio > 0 else warmup_steps

    # Periodic checkpointing (crash insurance for long runs). save_steps>0 keeps the
    # last few step-checkpoints under output_dir/checkpoint-*; 0 preserves the
    # original "save only the final adapter" behaviour. The checkpoints carry the
    # optimizer/scheduler state so --resume-from-checkpoint can continue them.
    if save_steps and save_steps > 0:
        checkpoint_kwargs: dict[str, Any] = {
            "save_strategy": "steps",
            "save_steps": save_steps,
            "save_total_limit": 3,
        }
    else:
        checkpoint_kwargs = {"save_strategy": "no"}

    training_args = sft_config_cls(
        output_dir=str(resolved_output_dir),
        max_steps=max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=effective_warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        **checkpoint_kwargs,
        report_to=report_to,
        max_length=max_seq_length,
        # Pre-tokenized full-chat sequences with completion_mask. This keeps
        # the proven one-pass Gemma 4 encoding path while masking query tokens
        # out of the loss in the collator.
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        seed=seed,
        **_mixed_precision_kwargs(),
        optim="paged_adamw_8bit",
    )
    # Set post-init like group_by_length: this avoids depending on Unsloth's
    # compiled SFTConfig wrapper accepting every upstream TRL constructor kwarg.
    training_args.loss_type = (
        "dft" if inner_loss == "dft" and dft_impl == "trl" else "nll"
    )
    # Set group_by_length AFTER construction: Unsloth's compiled SFTConfig wrapper
    # rejects it as an __init__ kwarg (TypeError) even though it is a standard
    # TrainingArguments field. The base Trainer reads args.group_by_length when it
    # builds the sampler, so a post-init assignment takes effect; length_column_name
    # stays at its "length" default, which matches the per-row length we attach.
    training_args.group_by_length = group_by_length

    trainer_kwargs: dict[str, Any] = {}
    if inner_loss == "asft":
        assert asft_streaming_config_cls is not None
        trainer_kwargs.update(
            asft_enabled=True,
            asft_mode=asft_mode,
            kl_weight=asft_kl_weight,
            kl_direction=asft_kl_direction,
            reference_policy=asft_reference_policy,
            asft_streaming=asft_streaming_config_cls(
                mode=asft_streaming,
                enabled=asft_streaming != "off",
                ref_microbatch_size=asft_ref_microbatch_size,
                seq_chunk_size=asft_seq_chunk_size,
            ),
            normalize_by=asft_normalize_by,
        )

    trainer = sft_trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=_make_completion_only_data_collator(
            tokenizer,
            max_seq_length=max_seq_length,
        ),
        **trainer_kwargs,
    )
    last_checkpoint = None
    if resume_from_checkpoint:
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(str(resolved_output_dir))
        if last_checkpoint:
            logger.info("Resuming SFT from checkpoint {}", last_checkpoint)
        else:
            logger.warning(
                "--resume-from-checkpoint set but no checkpoint found in {}; starting fresh.",
                resolved_output_dir,
            )
    trainer.train(resume_from_checkpoint=last_checkpoint)

    artifact_path = _save_training_artifact(
        model,
        tokenizer,
        resolved_output_dir,
        export_format=export_format,
        gguf_quantize=gguf_quantize,
    )
    metadata: dict[str, Any] = {
        "base_model": base_model,
        "max_steps": max_steps,
        "configured_num_train_epochs": num_train_epochs,
        "virtual_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": effective_warmup_steps,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "weight_decay": weight_decay,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "seed": seed,
        "row_count": row_count,
        "completion_only_loss": True,
        "inner_loss": inner_loss,
        "dft_impl": dft_impl if inner_loss == "dft" else None,
        "masking": "completion_mask",
        "save_steps": save_steps,
        "group_by_length": group_by_length,
        "training_data_paths": resolved_training_paths,
        "rehearsal_dataset": rehearsal_dataset,
        "rehearsal_split": rehearsal_split,
        "rehearsal_text_field": rehearsal_text_field,
        "rehearsal_rows_per_epoch": rehearsal_rows_per_epoch,
        "rehearsal_seed": rehearsal_seed if rehearsal_seed is not None else seed,
        "export_format": export_format,
        "wandb_project": wandb_project,
        "exported_at": _export_timestamp(),
    }
    if inner_loss == "asft":
        metadata.update(
            {
                "asft_mode": asft_mode,
                "asft_kl_weight": asft_kl_weight,
                "asft_kl_direction": asft_kl_direction,
                "asft_reference_policy": asft_reference_policy,
                "asft_streaming": asft_streaming,
                "asft_ref_microbatch_size": asft_ref_microbatch_size,
                "asft_seq_chunk_size": asft_seq_chunk_size,
                "asft_normalize_by": asft_normalize_by,
                "trainer_class": trainer_class,
                "unsloth_asft_fork_commit": _unsloth_asft_fork_commit(),
                **package_versions,
            }
        )
    _write_metadata(_training_metadata_path(artifact_path), metadata)
    return artifact_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QLoRA training entry point (vendored from NSL)."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--training-data", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rehearsal-dataset", default=None)
    parser.add_argument("--rehearsal-split", default="train")
    parser.add_argument("--rehearsal-text-field", default="text")
    parser.add_argument("--rehearsal-rows-per-epoch", type=int, default=0)
    parser.add_argument("--rehearsal-seed", type=int, default=None)
    parser.add_argument(
        "--rehearsal-prompt",
        default="Continue the following passage",
    )
    parser.add_argument("--rehearsal-prompt-chars", type=int, default=256)
    parser.add_argument("--rehearsal-max-chars", type=int, default=2048)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--export-format",
        choices=("lora", "merged_16bit", "gguf"),
        default="lora",
    )
    parser.add_argument("--quantize", default="f16")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="checkpoint every N steps (0 = save only the final adapter)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="resume from the latest checkpoint-* in --output, if present",
    )
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="batch similar-length rows together (needs per-device batch size > 1 "
        "to reduce padding waste on long-tailed data)",
    )
    parser.add_argument(
        "--inner-loss",
        choices=("sft", "dft", "asft"),
        default="sft",
        help="inner-loop loss to use: standard SFT NLL, DFT, or ASFT",
    )
    parser.add_argument(
        "--dft-impl",
        choices=("fused", "trl"),
        default="fused",
        help=(
            "DFT implementation: memory-efficient Unsloth fused patch or "
            "TRL full-logit loss"
        ),
    )
    parser.add_argument(
        "--asft-mode",
        choices=("dft", "sft+kl", "asft"),
        default="asft",
        help="ASFT objective variant when --inner-loss asft is selected",
    )
    parser.add_argument("--asft-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--asft-kl-direction",
        choices=("forward", "reverse"),
        default="forward",
    )
    parser.add_argument(
        "--asft-reference-policy",
        choices=("disable_adapter", "frozen_copy"),
        default="disable_adapter",
    )
    parser.add_argument(
        "--asft-streaming",
        choices=("off", "auto", "batch", "seq", "hybrid"),
        default="off",
    )
    parser.add_argument("--asft-ref-microbatch-size", type=int, default=None)
    parser.add_argument("--asft-seq-chunk-size", type=int, default=None)
    parser.add_argument(
        "--asft-normalize-by",
        choices=("tokens", "weights"),
        default="tokens",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.training_data:
        parser.error("--training-data is required")

    output_path = train_sft(
        base_model=args.base_model,
        training_data_paths=args.training_data,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        rehearsal_dataset=args.rehearsal_dataset,
        rehearsal_split=args.rehearsal_split,
        rehearsal_text_field=args.rehearsal_text_field,
        rehearsal_rows_per_epoch=args.rehearsal_rows_per_epoch,
        rehearsal_seed=args.rehearsal_seed,
        rehearsal_prompt=args.rehearsal_prompt,
        rehearsal_prompt_chars=args.rehearsal_prompt_chars,
        rehearsal_max_chars=args.rehearsal_max_chars,
        wandb_project=args.wandb_project,
        export_format=args.export_format,
        gguf_quantize=args.quantize,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        group_by_length=args.group_by_length,
        inner_loss=args.inner_loss,
        dft_impl=args.dft_impl,
        asft_mode=args.asft_mode,
        asft_kl_weight=args.asft_kl_weight,
        asft_kl_direction=args.asft_kl_direction,
        asft_reference_policy=args.asft_reference_policy,
        asft_streaming=args.asft_streaming,
        asft_ref_microbatch_size=args.asft_ref_microbatch_size,
        asft_seq_chunk_size=args.asft_seq_chunk_size,
        asft_normalize_by=args.asft_normalize_by,
    )

    print(output_path)
    return 0


__all__ = [
    "resolve_training_export_format",
    "train_sft",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
