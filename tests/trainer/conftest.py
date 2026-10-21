# Trainer tests come from NSL2 and need the `[train]` extra (torch, loguru, ...).
# Skip them on CPU-only / agent-only installs; test_asft also needs unsloth.
import importlib.util


def _missing(*modules: str) -> bool:
    return any(importlib.util.find_spec(m) is None for m in modules)


# test_training_artifacts.py is stdlib-only, so it always runs.
collect_ignore = ["test_qlora.py", "test_asft.py", "test_dft_fused.py"] if _missing("torch", "loguru", "dotenv") else []
if _missing("loguru", "dotenv"):  # check-model reuses the trainer's masking, which needs these but not torch
    collect_ignore.append("test_modelcheck.py")
if _missing("unsloth"):
    collect_ignore.append("test_asft.py")
