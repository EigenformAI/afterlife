# Trainer tests come from NSL2 and need the `[train]` extra (torch, loguru, ...).
# Skip them on CPU-only / agent-only installs; test_asft also needs unsloth.
import importlib.util


def _missing(*modules: str) -> bool:
    return any(importlib.util.find_spec(m) is None for m in modules)


collect_ignore_glob = ["test_*.py"] if _missing("torch", "loguru", "dotenv") else []
if _missing("unsloth"):
    collect_ignore_glob.append("test_asft.py")
