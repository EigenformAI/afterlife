from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_lora(path: Path, *, export_format: str = "lora") -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "training_info.json").write_text(
        json.dumps({"export_format": export_format}),
        encoding="utf-8",
    )
    return path


def test_validate_training_artifact_accepts_complete_expected_lora(tmp_path: Path) -> None:
    from afterlife.trainer.artifacts import validate_training_artifact

    artifact = _write_lora(tmp_path / "after_generation_0")

    assert validate_training_artifact(
        artifact,
        "lora",
        expected_path=artifact,
    ) == artifact.resolve()


@pytest.mark.parametrize("missing", ["adapter_config.json", "adapter_model.safetensors", "training_info.json"])
def test_validate_training_artifact_rejects_incomplete_lora(
    tmp_path: Path,
    missing: str,
) -> None:
    from afterlife.trainer.artifacts import validate_training_artifact

    artifact = _write_lora(tmp_path / "after_generation_0")
    (artifact / missing).unlink()

    with pytest.raises((FileNotFoundError, ValueError), match="LoRA|metadata"):
        validate_training_artifact(artifact, "lora", expected_path=artifact)


def test_validate_training_artifact_rejects_wrong_path_or_format(tmp_path: Path) -> None:
    from afterlife.trainer.artifacts import validate_training_artifact

    artifact = _write_lora(tmp_path / "unexpected")

    with pytest.raises(ValueError, match="unexpected training artifact path"):
        validate_training_artifact(
            artifact,
            "lora",
            expected_path=tmp_path / "after_generation_0",
        )

    with pytest.raises(ValueError, match="metadata export_format"):
        validate_training_artifact(artifact, "merged_16bit")


def test_validate_training_artifact_rejects_lora_index_with_missing_shards(
    tmp_path: Path,
) -> None:
    from afterlife.trainer.artifacts import validate_training_artifact

    artifact = _write_lora(tmp_path / "after_generation_0")
    (artifact / "adapter_model.safetensors").unlink()
    (artifact / "adapter_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "adapter_model-00001-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="shard"):
        validate_training_artifact(artifact, "lora")


def test_validate_training_artifact_accepts_complete_sharded_lora(
    tmp_path: Path,
) -> None:
    from afterlife.trainer.artifacts import validate_training_artifact

    artifact = _write_lora(tmp_path / "after_generation_0")
    (artifact / "adapter_model.safetensors").unlink()
    shard = "adapter_model-00001-of-00001.safetensors"
    (artifact / shard).write_bytes(b"weights")
    (artifact / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard}}),
        encoding="utf-8",
    )

    assert validate_training_artifact(artifact, "lora") == artifact.resolve()


def test_validate_training_artifact_rejects_invalid_lora_shard_names(
    tmp_path: Path,
) -> None:
    from afterlife.trainer.artifacts import validate_training_artifact

    artifact = _write_lora(tmp_path / "after_generation_0")
    (artifact / "adapter_model.safetensors").unlink()
    (artifact / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": ["not", "a", "path"]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid shard names"):
        validate_training_artifact(artifact, "lora")


def test_validate_training_artifact_rejects_merged_weight_directory(
    tmp_path: Path,
) -> None:
    from afterlife.trainer.artifacts import validate_training_artifact

    artifact = tmp_path / "after_generation_0"
    artifact.mkdir()
    (artifact / "training_info.json").write_text(
        json.dumps({"export_format": "merged_16bit"}),
        encoding="utf-8",
    )
    (artifact / "model.safetensors").mkdir()

    with pytest.raises(FileNotFoundError, match="merged model weights"):
        validate_training_artifact(artifact, "merged_16bit")


def test_llama_merged_export_error_recommends_only_gguf() -> None:
    from afterlife.trainer.artifacts import resolve_training_export_format

    with pytest.raises(ValueError, match="use export_format='gguf'") as exc_info:
        resolve_training_export_format("llama:model", "merged_16bit")

    assert "export_format='lora'" not in str(exc_info.value)
