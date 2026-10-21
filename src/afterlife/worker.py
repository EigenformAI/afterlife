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
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from afterlife.config import Config
from afterlife.recipes import resolve_recipe, rows_per_pass, train_sft_kwargs
from afterlife.selection import admission_problem, build_training_set, content_hash, to_trainer_rows
from afterlife.store import Store, new_id, utc_now


class WorkerError(Exception):
    pass


class CooldownWait(WorkerError):
    """The operator's minimum gap between runs has not passed; the request stays pending."""


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


def _last_run_at(store: Store) -> datetime | None:
    # Failed runs and dry runs trained nothing, so they do not start the cooldown.
    runs = [r for r in store.runs() if r.get("status") not in ("failed", "interrupted", "dry_run")]
    if not runs:
        return None
    return datetime.strptime(runs[-1]["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _set_request(store: Store, request: dict[str, Any], status: str, **extra: Any) -> None:
    request.update(status=status, updated_at=utc_now(), **extra)
    store.save_request(request)


def _frozen_training_set(store: Store, config: Config, request: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the training set the agent saw at request time, under the operator's current rules.

    Membership is fixed at request time: nothing is added from later store
    contents. A member the current rules no longer allow, or whose content no
    longer matches what was requested, is dropped with the reason, not replaced.
    """
    frozen = request.get("frozen_selection")
    if not frozen:
        return None
    op = config.operator
    by_id = {e["experience_id"]: e for e in store.experiences()}
    hashes = frozen.get("content_hashes", {})  # absent on requests queued by older versions
    excluded = dict(frozen["excluded"])

    def admit(ids: list[str]) -> list[dict[str, Any]]:
        kept = []
        for i in ids:
            e = by_id.get(i)
            if e is None:
                excluded[i] = "no longer in the store"
            elif i in hashes and content_hash(e) != hashes[i]:
                excluded[i] = "content changed since the request was queued"
            elif (problem := admission_problem(e, op)) is not None:
                excluded[i] = f"no longer allowed: {problem}"
            else:
                kept.append(e)
        return kept

    train = admit(frozen["train_ids"])
    for e in train[op.max_examples_per_run :]:
        excluded[e["experience_id"]] = f"over budget ({op.max_examples_per_run} per run)"
    train = train[: op.max_examples_per_run]
    heldout = admit(frozen["heldout_ids"])
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
    ts = _frozen_training_set(store, config, request) or build_training_set(store, config)
    if not ts["train"]:
        raise WorkerError("nothing to train on: the current selection is empty after validation")
    train_rows = [r for e in ts["train"] for r in to_trainer_rows(e)]
    recipe = resolve_recipe(config, request["method"], request.get("params", {}),
                            train_rows=rows_per_pass(config, len(train_rows)))

    run_id = new_id("run")
    run_dir = store.run_dir(run_id)
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "train.jsonl", train_rows)
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
        "created_ns": time.time_ns(),
        "status": "prepared",
        "request_id": request["request_id"],
        "request_note": request.get("note"),
        "base_model": config.base_model,
        "serve_model": config.serve_model,  # activation refuses this adapter if either model changes
        "starts_from": "base",  # each run trains a fresh adapter from the base model
        # Evaluated as "current": the adapter live when the run was prepared
        # (the request records the one live when it was queued).
        "active_at_run": active.get("run_id"),
        "selection": {"mode": ts["mode"], "policy_hash": ts["policy_hash"],
                      "counts": {**ts["counts"], "train_rows": len(train_rows)}},
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


def _child(step: str, run_dir: Path, lock_fd: int | None = None) -> None:
    # The child inherits the worker lock (flock follows the open file), so if
    # the worker dies mid-run, no new worker can start until the child is done
    # with the GPU. On an exception here (Ctrl-C included) subprocess.run kills it.
    cmd = [sys.executable, "-m", "afterlife.worker", step, str(run_dir)]
    with open(run_dir / f"{step}.log", "a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              pass_fds=(lock_fd,) if lock_fd is not None else ())
    if proc.returncode != 0:
        raise WorkerError(f"{step} failed (exit {proc.returncode}); see {run_dir / f'{step}.log'}")


def _ended(exc: BaseException) -> str:
    """A worker stopped by Ctrl-C or SIGTERM was interrupted; anything else failed."""
    return "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"


def recover_interrupted(store: Store) -> list[dict[str, Any]]:
    """Mark work left mid-flight by a worker that died as interrupted. Call with the worker lock held.

    Holding the lock means no worker or training child is still running, so
    anything still "preparing" or "running" was abandoned. It is never
    retrained automatically: `afterlife retry` queues it again.
    """
    recovered = []
    for request in store.requests():
        if request["status"] not in ("preparing", "running"):
            continue
        error = f"worker stopped while {request['status']}; `afterlife retry {request['request_id']}` to queue it again"
        if request.get("run_id"):
            run_dir = store.run_dir(request["run_id"])
            manifest = store.read_json(run_dir / "manifest.json")
            if manifest and manifest.get("status") in ("prepared", "training", "evaluating"):
                _update_manifest(store, run_dir, status="interrupted",
                                 error=f"worker stopped while {manifest['status']}")
        _set_request(store, request, "interrupted", error=error)
        recovered.append({"request_id": request["request_id"], "ok": False, "recovered": True, "error": error})
    return recovered


def process_request(store: Store, request: dict[str, Any], *, dry_run: bool = False,
                    lock_fd: int | None = None) -> dict[str, Any]:
    config = store.config()
    if not config.operator.training_enabled:
        raise WorkerError("training is disabled by the operator ([operator].training_enabled)")
    gap = config.operator.min_minutes_between_runs
    last = _last_run_at(store)
    if gap and last is not None and datetime.now(timezone.utc) < last + timedelta(minutes=gap):
        since = (datetime.now(timezone.utc) - last).total_seconds() / 60
        eligible_at = (last + timedelta(minutes=gap)).strftime("%Y-%m-%dT%H:%M:%SZ")
        reason = f"last run was {since:.0f} min ago; operator minimum is {gap} min"
        if request.get("eligible_at") != eligible_at:  # visible in `afterlife status`
            _set_request(store, request, "pending", blocked_reason=reason, eligible_at=eligible_at)
        raise CooldownWait(f"{reason}; waiting until {eligible_at}")

    _set_request(store, request, "preparing", blocked_reason=None, eligible_at=None)
    try:
        run_dir = prepare_run(store, config, request)
    except BaseException as exc:
        _set_request(store, request, _ended(exc), error=str(exc) or type(exc).__name__)
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
        _child("train", run_dir, lock_fd)
        _update_manifest(
            store, run_dir, status="evaluating", adapter_path=str(run_dir / "adapter"),
            train_seconds=round(time.time() - started, 1),
        )
        started = time.time()
        _child("eval", run_dir, lock_fd)
        eval_report = store.read_json(run_dir / "eval.json")
        manifest = _update_manifest(
            store, run_dir, status="ready", eval=eval_report.get("summary"),
            eval_seconds=round(time.time() - started, 1),
            peak_gpu_gb={step: (store.read_json(run_dir / f"{step}_stats.json") or {}).get("peak_gpu_gb")
                         for step in ("train", "eval")},
        )
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        _update_manifest(store, run_dir, status=_ended(exc), error=error)
        _set_request(store, request, _ended(exc), error=error)
        raise
    _set_request(store, request, "done")
    return manifest


def _process_pending(store: Store, waiting: set[str], *, dry_run: bool = False,
                     lock_fd: int | None = None) -> list[dict[str, Any]]:
    """One pass over the queue. ``waiting`` holds requests already reported as cooling down."""
    results = []
    for request in [r for r in store.requests() if r["status"] == "pending"]:
        try:
            results.append({"request_id": request["request_id"], "ok": True,
                            "manifest": process_request(store, request, dry_run=dry_run, lock_fd=lock_fd)})
        except CooldownWait as exc:
            # Report a wait once, not every poll; the request stays pending.
            if request["request_id"] not in waiting:
                waiting.add(request["request_id"])
                results.append({"request_id": request["request_id"], "ok": False,
                                "waiting": True, "error": str(exc)})
        except Exception as exc:  # one bad request must not stop the queue
            results.append({"request_id": request["request_id"], "ok": False, "error": str(exc)})
    return results


def run_pending(store: Store, *, once: bool = True, dry_run: bool = False, poll_seconds: int = 30) -> list[dict[str, Any]]:
    """Process pending requests oldest first. One worker at a time per store."""
    results = []
    waiting: set[str] = set()
    with open(store.root / ".worker.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerError("another worker, or a training process left by one, is running on this store") from exc
        # SIGTERM (e.g. a pod shutting down) unwinds like Ctrl-C, so the run is marked interrupted.
        previous = signal.signal(signal.SIGTERM, _raise_exit)
        try:
            results += recover_interrupted(store)
            while True:
                results += _process_pending(store, waiting, dry_run=dry_run, lock_fd=lock.fileno())
                if once:
                    return results
                time.sleep(poll_seconds)
        finally:
            signal.signal(signal.SIGTERM, previous)


def _raise_exit(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


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
