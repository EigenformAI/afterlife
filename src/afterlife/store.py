"""On-disk layout and append-only logs under `.afterlife/`.

    .afterlife/
      config.toml          operator config (agent: read-only)
      experiences.jsonl    everything offered as a possible memory
      decisions.jsonl      nominate / reject / defer / external-memory-only
      policy_events.jsonl  strategy changes (mode + policy hash)
      policies/<hash>.json every policy version ever set
      requests/<id>.json   training requests and their status
      runs/<run-id>/       manifest.json, train.jsonl, heldout.jsonl, eval.json, adapter/
      active.json          which adapter is live, plus history for rollback
      active-adapter       symlink to the live adapter, derived from active.json
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from afterlife.config import Config, load_config

STORE_DIRNAME = ".afterlife"
ENV_HOME = "AFTERLIFE_HOME"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def find_root(start: Path | None = None) -> Path:
    """Locate the store: $AFTERLIFE_HOME, else nearest `.afterlife/` upwards, else ./.afterlife."""
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / STORE_DIRNAME).is_dir():
            return candidate / STORE_DIRNAME
    return here / STORE_DIRNAME


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root

    # -- paths ---------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.root / "config.toml"

    @property
    def experiences_path(self) -> Path:
        return self.root / "experiences.jsonl"

    @property
    def decisions_path(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def policy_events_path(self) -> Path:
        return self.root / "policy_events.jsonl"

    @property
    def policies_dir(self) -> Path:
        return self.root / "policies"

    @property
    def requests_dir(self) -> Path:
        return self.root / "requests"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def exists(self) -> bool:
        return self.config_path.is_file()

    def config(self) -> Config:
        return load_config(self.config_path)

    def init_layout(self) -> None:
        for d in (self.root, self.policies_dir, self.requests_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)
        for f in (self.experiences_path, self.decisions_path, self.policy_events_path):
            f.touch(exist_ok=True)

    # -- append-only logs ----------------------------------------------------
    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the store lock across a read-check-write sequence.

        Not reentrant: append/append_many take the same lock, so do not call
        them inside.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def append(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.locked(), open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def append_many(self, path: Path, choose: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Append the records ``choose()`` returns, deciding and writing under one lock.

        ``choose`` runs with the lock held, so it sees every earlier write and
        no other writer can get in between; it may raise to write nothing.
        Returns what was written.
        """
        with self.locked():
            records = choose()
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
                handle.flush()
                os.fsync(handle.fileno())
        return records

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with open(path, encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: corrupt JSONL line") from exc
        return rows

    def experiences(self) -> list[dict[str, Any]]:
        return self.read_jsonl(self.experiences_path)

    def decisions(self) -> list[dict[str, Any]]:
        return self.read_jsonl(self.decisions_path)

    def policy_events(self) -> list[dict[str, Any]]:
        return self.read_jsonl(self.policy_events_path)

    def experience_ids(self) -> set[str]:
        return {e["experience_id"] for e in self.experiences()}

    def latest_decisions(self) -> dict[str, dict[str, Any]]:
        """Most recent decision per experience id (later lines win)."""
        latest: dict[str, dict[str, Any]] = {}
        for d in self.decisions():
            latest[d["experience_id"]] = d
        return latest

    # -- mutable JSON documents (atomic replace) -----------------------------
    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        # A temp file per writer: a shared one lets concurrent writers replace each other's.
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def read_json(path: Path, default: Any = None) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def requests(self) -> list[dict[str, Any]]:
        out = [self.read_json(p) for p in sorted(self.requests_dir.glob("*.json"))]
        return sorted(out, key=lambda r: (r["created_at"], r.get("created_ns", 0)))

    def save_request(self, request: dict[str, Any]) -> None:
        self.write_json(self.requests_dir / f"{request['request_id']}.json", request)

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError(f"invalid run id {run_id!r}")
        return self.runs_dir / run_id

    def runs(self) -> list[dict[str, Any]]:
        manifests = [self.read_json(p) for p in self.runs_dir.glob("*/manifest.json")]
        return sorted((m for m in manifests if m), key=lambda m: (m["created_at"], m.get("created_ns", 0)))

    def resolve_run(self, ref: str) -> dict[str, Any]:
        runs = self.runs()
        if ref == "latest":
            if not runs:
                raise LookupError("no runs yet")
            return runs[-1]
        for m in runs:
            if m["run_id"] == ref:
                return m
        raise LookupError(f"no run {ref!r}")

    def active(self) -> dict[str, Any]:
        return self.read_json(self.active_path, {"run_id": None, "adapter_path": None, "history": []})

    @property
    def active_link(self) -> Path:
        return self.root / "active-adapter"

    def sync_active_link(self) -> None:
        """Point active-adapter at what active.json says, repairing a crash between the two.

        active.json is the record; the symlink is what the model server reads.
        The swap is a rename, so a reader never sees the link missing mid-switch.
        """
        run_id = self.active().get("run_id")
        link = self.active_link
        if run_id is None:
            if link.is_symlink() or link.exists():
                link.unlink()
            return
        target = Path("runs") / run_id / "adapter"
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        tmp = self.root / f".active-adapter.{secrets.token_hex(4)}"
        tmp.symlink_to(target)
        os.replace(tmp, link)
