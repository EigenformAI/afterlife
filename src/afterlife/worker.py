"""Operator-side training worker.

The agent only ever *requests* a run. This module, started by the operator
(`afterlife worker`), turns a pending request into a run directory, trains a
fresh LoRA adapter from the base model on the current selection, evaluates it,
and leaves it ready for explicit activation. Nothing is activated here.

Training and evaluation run in child processes (`python -m afterlife.worker
train|eval <run_dir>`) so the CLI never imports torch and GPU memory is freed
between steps.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from afterlife.config import Config
from afterlife.recipes import resolve_recipe, train_sft_kwargs
from afterlife.selection import build_training_set, to_trainer_rows
from afterlife.store import Store, new_id, utc_now


class WorkerError(Exception):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _minutes_since_last_run(store: Store) -> float | None:
    runs = [r for r in store.runs() if r.get("status") != "failed"]
    if not runs:
        return None
    last = datetime.strptime(runs[-1]["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds() / 60


def _set_request(store: Store, request: dict[str, Any], status: str, **extra: Any) -> None:
    request.update(status=status, updated_at=utc_now(), **extra)
    store.save_request(request)


def _frozen_training_set(store: Store, config: Config, request: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the training set the agent saw at request time, under current caps."""
    frozen = request.get("frozen_selection")
    if not frozen:
        return None
    by_id = {e["experience_id"]: e for e in store.experiences()}
    cap = config.operator.max_examples_per_run
    excluded = dict(frozen["excluded"])
    train = [by_id[i] for i in frozen["train_ids"] if i in by_id]
    for e in train[cap:]:
        excluded[e["experience_id"]] = f"over budget ({cap} per run)"
    train = train[:cap]
    heldout = [by_id[i] for i in frozen["heldout_ids"] if i in by_id]
    strategy = request.get("strategy_at_request", {})
    return {
        "mode": strategy.get("mode"),
        "policy_hash": strategy.get("policy_hash"),
        "train": train,
        "heldout": heldout,
        "excluded": excluded,
        "counts": {**request.get("counts_at_request", {}), "train": len(train), "heldout": len(heldout),
                   "train_chars": sum(e["chars"] for e in train)},
    }


def prepare_run(store: Store, config: Config, request: dict[str, Any]) -> Path:
    """Freeze the selection for a request into runs/<run-id>/. Returns the run dir."""
    recipe = resolve_recipe(config, request["method"], request.get("params", {}))
    ts = _frozen_training_set(store, config, request) or build_training_set(store, config)
    if not ts["train"]:
        raise WorkerError("nothing to train on: the current selection is empty after validation")

    run_id = new_id("run")
    run_dir = store.run_dir(run_id)
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "train.jsonl", [r for e in ts["train"] for r in to_trainer_rows(e)])
    _write_jsonl(run_dir / "heldout.jsonl", [r for e in ts["heldout"] for r in to_trainer_rows(e)])
    store.write_json(
        run_dir / "selection.json",
        {
            "train_ids": [e["experience_id"] for e in ts["train"]],
            "heldout_ids": [e["experience_id"] for e in ts["heldout"]],
            "excluded": ts["excluded"],
        },
    )
    active = store.active()
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "prepared",
        "request_id": request["request_id"],
        "request_note": request.get("note"),
        "base_model": config.base_model,
        "starts_from": "base",  # each run trains a fresh adapter from the base model
        "active_at_request": active.get("run_id"),
        "selection": {"mode": ts["mode"], "policy_hash": ts["policy_hash"], "counts": ts["counts"]},
        "recipe": recipe,
        "train_sft_kwargs": train_sft_kwargs(config, recipe),
        "datasets": {
            "train_sha256": _sha256_file(run_dir / "train.jsonl"),
            "heldout_sha256": _sha256_file(run_dir / "heldout.jsonl"),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "adapter_path": None,
        "eval": None,
        "error": None,
    }
    store.write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _update_manifest(store: Store, run_dir: Path, **fields: Any) -> dict[str, Any]:
    manifest = store.read_json(run_dir / "manifest.json")
    manifest.update(fields)
    store.write_json(run_dir / "manifest.json", manifest)
    return manifest


def _child(step: str, run_dir: Path) -> None:
    cmd = [sys.executable, "-m", "afterlife.worker", step, str(run_dir)]
    with open(run_dir / f"{step}.log", "a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise WorkerError(f"{step} failed (exit {proc.returncode}); see {run_dir / f'{step}.log'}")


def process_request(store: Store, request: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    config = store.config()
    if not config.operator.training_enabled:
        raise WorkerError("training is disabled by the operator ([operator].training_enabled)")
    gap = config.operator.min_minutes_between_runs
    since = _minutes_since_last_run(store)
    if gap and since is not None and since < gap:
        raise WorkerError(f"last run was {since:.0f} min ago; operator minimum is {gap} min")

    _set_request(store, request, "preparing")
    try:
        run_dir = prepare_run(store, config, request)
    except Exception as exc:
        _set_request(store, request, "failed", error=str(exc))
        raise
    run_id = run_dir.name
    _set_request(store, request, "running", run_id=run_id)

    if dry_run:
        manifest = _update_manifest(store, run_dir, status="dry_run")
        _set_request(store, request, "dry_run", run_id=run_id)
        return manifest

    started = time.time()
    try:
        _update_manifest(store, run_dir, status="training")
        _child("train", run_dir)
        _update_manifest(
            store, run_dir, status="evaluating", adapter_path=str(run_dir / "adapter"),
            train_seconds=round(time.time() - started, 1),
        )
        started = time.time()
        _child("eval", run_dir)
        eval_report = store.read_json(run_dir / "eval.json")
        manifest = _update_manifest(
            store, run_dir, status="ready", eval=eval_report.get("summary"),
            eval_seconds=round(time.time() - started, 1),
            peak_gpu_gb={step: (store.read_json(run_dir / f"{step}_stats.json") or {}).get("peak_gpu_gb")
                         for step in ("train", "eval")},
        )
    except Exception as exc:
        _update_manifest(store, run_dir, status="failed", error=str(exc))
        _set_request(store, request, "failed", error=str(exc))
        raise
    _set_request(store, request, "done")
    return manifest


def run_pending(store: Store, *, once: bool = True, dry_run: bool = False, poll_seconds: int = 30) -> list[dict[str, Any]]:
    """Process pending requests oldest first. One worker at a time per store."""
    results = []
    with open(store.root / ".worker.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerError("another worker is already running on this store") from exc
        while True:
            pending = [r for r in store.requests() if r["status"] == "pending"]
            for request in pending:
                try:
                    results.append({"request_id": request["request_id"], "ok": True,
                                    "manifest": process_request(store, request, dry_run=dry_run)})
                except WorkerError as exc:
                    results.append({"request_id": request["request_id"], "ok": False, "error": str(exc)})
            if once:
                return results
            time.sleep(poll_seconds)


# -- child-process entry points ------------------------------------------------

def _train_child(run_dir: Path) -> None:
    from afterlife.trainer.qlora import train_sft

    manifest = json.loads((run_dir / "manifest.json").read_text())
    train_sft(
        base_model=manifest["base_model"],
        training_data_paths=[str(run_dir / "train.jsonl")],
        output_dir=str(run_dir / "adapter"),
        **manifest["train_sft_kwargs"],
    )


def _eval_child(run_dir: Path) -> None:
    from afterlife.evaluator import evaluate_run

    evaluate_run(run_dir)


def _record_peak_memory(run_dir: Path, step: str) -> None:
    import torch

    if torch.cuda.is_available():
        peak = max(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count()))
        Store.write_json(run_dir / f"{step}_stats.json", {"peak_gpu_gb": round(peak / 2**30, 2)})


if __name__ == "__main__":
    step, target = sys.argv[1], Path(sys.argv[2])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    {"train": _train_child, "eval": _eval_child}[step](target)
    _record_peak_memory(target, step)
