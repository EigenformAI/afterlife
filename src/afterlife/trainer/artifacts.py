from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast


TrainingExportFormat = Literal["lora", "merged_16bit", "gguf"]
ConfiguredTrainingExportFormat = Literal["auto", "lora", "merged_16bit", "gguf"]
LORA_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
LORA_SAFETENSORS_INDEX_FILENAME = "adapter_model.safetensors.index.json"


def resolve_training_export_format(
    model_name: str,
    configured_format: ConfiguredTrainingExportFormat = "auto",
) -> TrainingExportFormat:
    backend = model_name.strip().split(":", 1)[0]
    if configured_format == "auto":
        return "gguf" if backend == "llama" else "lora"
    if backend == "llama" and configured_format == "lora":
        raise ValueError(
            "llama.cpp cannot consume the PEFT LoRA directory produced by this "
            "trainer; use export_format='gguf'"
        )
    if backend == "llama" and configured_format == "merged_16bit":
        raise ValueError(
            "llama backend does not consume merged_16bit training artifacts; "
            "use export_format='gguf'"
        )
    if backend == "vllm" and configured_format == "gguf":
        raise ValueError(
            "vLLM does not consume GGUF training artifacts; use "
            "export_format='lora' or export_format='merged_16bit'"
        )
    return cast(TrainingExportFormat, configured_format)


def training_artifact_path(
    artifact_root: Path,
    generation_id: int,
    export_format: TrainingExportFormat,
) -> Path:
    if export_format == "gguf":
        return artifact_root / f"after_generation_{generation_id}.gguf"
    return artifact_root / f"after_generation_{generation_id}"


def training_metadata_path(
    artifact_path: Path,
    export_format: TrainingExportFormat,
) -> Path:
    if export_format == "gguf":
        return artifact_path.with_suffix(".training_info.json")
    return artifact_path / "training_info.json"


def _load_training_metadata(
    artifact_path: Path,
    export_format: TrainingExportFormat,
) -> dict[str, object]:
    metadata_path = training_metadata_path(artifact_path, export_format)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"training artifact metadata not found: {metadata_path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"training artifact metadata is unreadable: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"training artifact metadata must be an object: {metadata_path}")
    if metadata.get("export_format") != export_format:
        raise ValueError(
            "training artifact metadata export_format mismatch: "
            f"expected {export_format!r}, got {metadata.get('export_format')!r}"
        )
    return metadata


def validate_training_artifact(
    artifact_path: Path | str,
    export_format: TrainingExportFormat,
    *,
    expected_path: Path | str | None = None,
) -> Path:
    resolved = Path(artifact_path).expanduser().resolve()
    if expected_path is not None:
        expected = Path(expected_path).expanduser().resolve()
        if resolved != expected:
            raise ValueError(
                f"unexpected training artifact path: expected {expected}, got {resolved}"
            )

    if export_format == "gguf":
        if not resolved.is_file():
            raise FileNotFoundError(f"GGUF training artifact not found: {resolved}")
    elif not resolved.is_dir():
        raise FileNotFoundError(f"training artifact directory not found: {resolved}")

    _load_training_metadata(resolved, export_format)

    if export_format == "lora":
        config_path = resolved / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"LoRA adapter config not found: {config_path}")
        try:
            adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"LoRA adapter config is unreadable: {config_path}") from exc
        if not isinstance(adapter_config, dict):
            raise ValueError(f"LoRA adapter config must be an object: {config_path}")
        has_monolithic_weights = any(
            (resolved / filename).is_file() for filename in LORA_WEIGHT_FILENAMES
        )
        index_path = resolved / LORA_SAFETENSORS_INDEX_FILENAME
        has_sharded_weights = False
        if not has_monolithic_weights and index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"LoRA adapter weight index is unreadable: {index_path}"
                ) from exc
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    f"LoRA adapter weight index has no weight_map: {index_path}"
                )
            shard_values = list(weight_map.values())
            if not all(isinstance(name, str) and name for name in shard_values):
                raise ValueError(
                    f"LoRA adapter weight index has invalid shard names: {index_path}"
                )
            shard_names = {cast(str, name) for name in shard_values}
            shard_paths = [(resolved / name).resolve() for name in shard_names]
            if any(not path.is_relative_to(resolved) for path in shard_paths):
                raise ValueError(
                    f"LoRA adapter weight index references a shard outside: {resolved}"
                )
            missing_shards = [path for path in shard_paths if not path.is_file()]
            if missing_shards:
                raise FileNotFoundError(
                    f"LoRA adapter shard not found: {missing_shards[0]}"
                )
            has_sharded_weights = True
        if not has_monolithic_weights and not has_sharded_weights:
            raise FileNotFoundError(f"LoRA adapter weights not found under: {resolved}")
    elif export_format == "merged_16bit":
        weight_patterns = ("model*.safetensors", "pytorch_model*.bin")
        if not any(
            path.is_file()
            for pattern in weight_patterns
            for path in resolved.glob(pattern)
        ):
            raise FileNotFoundError(f"merged model weights not found under: {resolved}")

    return resolved


__all__ = [
    "ConfiguredTrainingExportFormat",
    "TrainingExportFormat",
    "resolve_training_export_format",
    "training_artifact_path",
    "training_metadata_path",
    "validate_training_artifact",
]
