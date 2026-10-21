"""`afterlife` command line. Stdlib only; every command supports --json.

Exit codes: 0 ok, 1 error, 2 bad input, 3 refused by operator config, 4 not found.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from afterlife import __version__
from afterlife.config import DEFAULT_BASE_MODEL, ConfigError, render_default_config
from afterlife.policies import PolicyError, current_strategy, set_strategy, validate_policy
from afterlife.recipes import (
    METHODS, RecipeError, describe_methods, parse_param, resolve_recipe, rows_per_pass,
)
from afterlife.schemas import ValidationError, build_decision, build_experience, embedded_decision, find_secrets
from afterlife.selection import build_training_set, content_hash, select
from afterlife.store import Store, find_root, new_id, utc_now

EXIT_OK, EXIT_ERROR, EXIT_INPUT, EXIT_REFUSED, EXIT_NOT_FOUND = 0, 1, 2, 3, 4


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_ERROR, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


# -- helpers -------------------------------------------------------------------

def _store(args: argparse.Namespace, *, must_exist: bool = True) -> Store:
    root = Path(args.home).expanduser().resolve() if args.home else find_root()
    store = Store(root)
    if must_exist and not store.exists():
        raise CliError(f"no afterlife store at {root}; run `afterlife init`", EXIT_NOT_FOUND)
    return store


def _load_payload(path: str) -> Any:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc}", EXIT_INPUT) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Allow JSONL: one experience per line.
        try:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise CliError(f"{path} is neither JSON nor JSONL: {exc}", EXIT_INPUT) from exc


def _ingest(store: Store, path: str) -> list[dict[str, Any]]:
    payload = _load_payload(path)
    items = payload if isinstance(payload, list) else [payload]
    known = store.experience_ids()
    records, decisions = [], []
    for i, item in enumerate(items):
        try:
            record = build_experience(item)
            decisions.append(embedded_decision(item))
        except ValidationError as exc:
            where = f"item {i}: " if len(items) > 1 else ""
            raise CliError(f"{where}{exc}", EXIT_INPUT, problems=exc.problems) from exc
        if record["experience_id"] in known:
            raise CliError(f"experience id {record['experience_id']} already exists", EXIT_INPUT)
        known.add(record["experience_id"])
        records.append(record)

    def no_clash() -> list[dict[str, Any]]:  # re-checked under the store lock: another process may have written since
        clash = sorted({r["experience_id"] for r in records} & store.experience_ids())
        if clash:
            raise CliError(f"experience id {clash[0]} already exists", EXIT_INPUT)
        return records

    store.append_many(store.experiences_path, no_clash)
    for record, decision in zip(records, decisions):
        if decision:
            _decide(store, [record["experience_id"]], decision[0], decision[1])
    return records


def _decide(store: Store, ids: list[str], decision: str, reason: str | None) -> list[dict[str, Any]]:
    known = store.experience_ids()
    missing = [i for i in ids if i not in known]
    if missing:
        raise CliError(f"unknown experience id(s): {', '.join(missing)}", EXIT_NOT_FOUND)
    strategy = current_strategy(store)
    try:
        out = [build_decision(eid, decision, reason=reason, mode=strategy["mode"],
                              policy_hash=strategy["policy_hash"], active_run=store.active().get("run_id"))
               for eid in ids]
    except ValidationError as exc:
        raise CliError(str(exc), EXIT_INPUT, problems=exc.problems) from exc
    for d in out:
        store.append(store.decisions_path, d)
    return out


def _brief(e: dict[str, Any]) -> str:
    messages = e["training_candidate"]["messages"]
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return " ".join(first_user.split())[:70]


def _train_ids(store: Store, run_id: str) -> set[str]:
    return set(store.read_json(store.run_dir(run_id) / "selection.json", {}).get("train_ids", []))


def _statuses(store: Store) -> dict[str, str]:
    """active: in the adapter running now. trained: in a finished run that isn't active."""
    sel = select(store)
    latest = store.latest_decisions()
    active_run = store.active().get("run_id")
    in_active = _train_ids(store, active_run) if active_run else set()
    trained: set[str] = set()
    for run in store.runs():
        if run.get("status") == "ready":
            trained |= _train_ids(store, run["run_id"])
    selected = {e["experience_id"] for e in sel["selected"]}
    out = {}
    for e in sel["eligible"]:
        eid = e["experience_id"]
        d = latest.get(eid, {}).get("decision")
        if eid in in_active:
            out[eid] = "active"
        elif eid in trained:
            out[eid] = "trained"
        elif d and d != "nominate":
            out[eid] = d
        elif e.get("eval_only"):
            out[eid] = "eval-only"
        elif eid in selected:
            out[eid] = "nominated" if d == "nominate" else "selected"
        else:
            out[eid] = "eligible"
    return out


# -- commands ------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args, must_exist=False)
    if store.exists() and not args.force:
        return {"root": str(store.root), "created": False, "message": "already initialised"}
    store.init_layout()
    store.config_path.write_text(render_default_config(args.model, args.serve_model), encoding="utf-8")
    config = store.config()
    return {"root": str(store.root), "created": True, "base_model": config.base_model,
            "serve_model": config.serve_model}


def cmd_skill(args: argparse.Namespace) -> dict[str, Any]:
    return {"skill": resources.files("afterlife").joinpath("SKILL.md").read_text(encoding="utf-8")}


def cmd_add(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    records = _ingest(store, args.file)
    return {"added": [r["experience_id"] for r in records]}


def cmd_import(args: argparse.Namespace) -> dict[str, Any]:
    from afterlife.importer import import_file

    store = _store(args)
    try:
        return import_file(store, Path(args.file), session_id=args.session_id, tools=args.tools,
                           max_tool_chars=args.max_tool_chars, tags=args.tag)
    except (OSError, ValueError) as exc:
        raise CliError(str(exc), EXIT_INPUT) from exc


def cmd_nominate(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    ids = list(args.id or [])
    if args.reason and find_secrets(args.reason):  # before --file stores anything
        raise CliError("reason looks like it contains a secret; remove it and retry", EXIT_INPUT)
    if args.file:
        ids += [r["experience_id"] for r in _ingest(store, args.file)]
    if not ids:
        raise CliError("give --file and/or --id", EXIT_INPUT)
    decisions = _decide(store, ids, "nominate", args.reason)
    return {"nominated": [d["experience_id"] for d in decisions]}


def cmd_decide(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    decisions = _decide(store, args.id, args.decision, args.reason)
    return {"decision": args.decision, "experience_ids": [d["experience_id"] for d in decisions]}


def cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    statuses = _statuses(store)
    rows = []
    for e in reversed(store.experiences()):
        status = statuses[e["experience_id"]]
        if args.status and status != args.status:
            continue
        if args.tag and args.tag not in e["tags"]:
            continue
        rows.append({
            "experience_id": e["experience_id"], "status": status, "created_at": e["created_at"],
            "tags": e["tags"], "priority": e["priority"], "kind": e["kind"], "preview": _brief(e),
        })
        if len(rows) >= args.limit:
            break
    return {"experiences": rows}


def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    for e in store.experiences():
        if e["experience_id"] == args.id:
            history = [d for d in store.decisions() if d["experience_id"] == args.id]
            return {"experience": e, "status": _statuses(store)[args.id], "decisions": history}
    raise CliError(f"no experience {args.id}", EXIT_NOT_FOUND)


def cmd_policy_show(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    return {"strategy": current_strategy(store), "history": store.policy_events()[-args.history :] if args.history else []}


def cmd_policy_set(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    if bool(args.mode) == bool(args.file):
        raise CliError("give exactly one of --mode or --file", EXIT_INPUT)
    policy = None
    mode = args.mode
    if args.file:
        policy = _load_payload(args.file)
        mode = "policy"
    try:
        event = set_strategy(store, mode, policy=policy, explanation=args.explanation)
    except PolicyError as exc:
        raise CliError(str(exc), EXIT_INPUT) from exc
    return {"event": event}


def cmd_policy_check(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    try:
        policy = validate_policy(_load_payload(args.file))
    except PolicyError as exc:
        raise CliError(str(exc), EXIT_INPUT) from exc
    sel = select(store, {"mode": "policy", "policy": policy, "policy_hash": None})
    return {"valid": True, "policy": policy, "would_select": len(sel["selected"]),
            "of_eligible": len(sel["eligible"])}


def cmd_policy_disable(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    return {"event": set_strategy(store, "manual", explanation=args.explanation)}


def cmd_methods(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    return describe_methods(store.config())


def cmd_preview(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    ts = build_training_set(store, store.config())
    reasons: dict[str, int] = {}
    for r in ts["excluded"].values():
        key = r.split(":")[0] if not r.startswith("duplicate") else "duplicate"
        reasons[key] = reasons.get(key, 0) + 1
    config = store.config()
    rows = rows_per_pass(config, ts["counts"]["train_rows"])
    passes = {}
    for m in config.operator.allowed_methods:
        try:
            passes[m] = resolve_recipe(config, m, {}, train_rows=rows).get("passes_per_example")
        except RecipeError as exc:
            passes[m] = str(exc)
    return {"mode": ts["mode"], "policy_hash": ts["policy_hash"], "counts": ts["counts"],
            "passes_per_example_at_default_steps": passes,
            "excluded_by_reason": reasons,
            "excluded": ts["excluded"],
            "heldout_ids": [e["experience_id"] for e in ts["heldout"]],
            "note": "Held-out experiences are used for eval, not trained. Experiences you mark "
                    "eval_only are always held out; nominated ones never are. Others picked by "
                    "the mode or policy are split afresh for each request, so this list can differ "
                    "from the one a request gets."}


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    config = store.config()
    if not args.request:
        raise CliError("training runs on the operator's worker; use `afterlife train --request`", EXIT_INPUT)
    if not config.operator.training_enabled:
        raise CliError("training is disabled by the operator", EXIT_REFUSED)
    pending = [r["request_id"] for r in store.requests() if r["status"] == "pending"]
    if len(pending) >= config.operator.max_pending_requests:
        raise CliError(f"{len(pending)} request(s) already waiting (operator limit "
                       f"{config.operator.max_pending_requests}); wait for the worker", EXIT_REFUSED,
                       pending=pending)
    request_id = new_id("req")
    ts = build_training_set(store, config, split_seed=request_id)
    if not ts["train"]:
        raise CliError("nothing would be trained: the current selection is empty", EXIT_INPUT,
                       counts=ts["counts"])
    try:
        overrides = dict(parse_param(p) for p in args.set or [])
        recipe = resolve_recipe(config, args.method, overrides,
                                train_rows=rows_per_pass(config, ts["counts"]["train_rows"]))
    except RecipeError as exc:
        raise CliError(str(exc), EXIT_REFUSED if exc.refused else EXIT_INPUT) from exc
    request = {
        "request_id": request_id, "created_at": utc_now(), "updated_at": utc_now(),
        "status": "pending", "method": args.method, "params": overrides, "note": args.note,
        "strategy_at_request": {"mode": ts["mode"], "policy_hash": ts["policy_hash"]},
        "counts_at_request": ts["counts"],
        # What gets trained is fixed now, not when the worker gets round to it.
        "frozen_selection": {
            "train_ids": [e["experience_id"] for e in ts["train"]],
            "heldout_ids": [e["experience_id"] for e in ts["heldout"]],
            "excluded": ts["excluded"],
            "content_hashes": {e["experience_id"]: content_hash(e) for e in ts["train"] + ts["heldout"]},
        },
        "active_at_request": store.active().get("run_id"),
        "created_ns": time.time_ns(),
    }
    store.save_request(request)
    result = {"request": request, "recipe": recipe, "message": "queued; the operator's worker will pick it up",
              "note": "What is trained and your --set overrides are fixed now. Defaults, operator limits, "
                      "models, probes and eval settings are read when the worker runs it, so `recipe` "
                      "shows them as of now; the run's manifest and eval record what was used."}
    over = [i for i, r in ts["excluded"].items() if r.startswith("over budget")]
    if over:
        result["over_budget"] = len(over)
        result["warning"] = (f"{len(over)} selected experience(s) don't fit the operator's budget of "
                             f"{config.operator.max_examples_per_run} per run and won't be trained this run. "
                             "Nominations go first, then policy picks in the policy's order; "
                             "`afterlife preview` lists which were left out.")
    return result


def cmd_retry(args: argparse.Namespace) -> dict[str, Any]:
    """Queue a failed or interrupted request again, with the selection it froze."""
    store = _store(args)
    if not store.config().operator.training_enabled:
        raise CliError("training is disabled by the operator", EXIT_REFUSED)
    request = next((r for r in store.requests() if r["request_id"] == args.request_id), None)
    if request is None:
        raise CliError(f"no request {args.request_id!r}", EXIT_NOT_FOUND)
    if request["status"] not in ("failed", "interrupted"):
        raise CliError(f"request {args.request_id} is {request['status']}; only failed or interrupted "
                       "requests can be retried", EXIT_INPUT)
    attempts = request.get("previous_attempts", []) + [
        {k: request.get(k) for k in ("status", "run_id", "error", "updated_at")}]
    request.update(status="pending", run_id=None, error=None, updated_at=utc_now(), previous_attempts=attempts)
    store.save_request(request)
    return {"request": request, "message": "queued again; the operator's worker will pick it up"}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    statuses = _statuses(store)
    counts: dict[str, int] = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    runs = store.runs()
    return {
        "root": str(store.root),
        "strategy": {k: v for k, v in current_strategy(store).items() if k != "policy"},
        "experiences": counts,
        "requests": [{k: r.get(k) for k in ("request_id", "status", "method", "run_id", "error")}
                     | {k: r[k] for k in ("blocked_reason", "eligible_at") if r.get(k)}
                     for r in store.requests()[-5:]],
        "runs": [{k: r.get(k) for k in ("run_id", "status", "created_at", "error")} | {
                     "method": r["recipe"]["method"], "train": r["selection"]["counts"]["train"]}
                 for r in runs[-5:]],
        "active": {k: v for k, v in store.active().items() if k != "history"},
    }


def cmd_eval(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    try:
        manifest = store.resolve_run(args.run)
    except LookupError as exc:
        raise CliError(str(exc), EXIT_NOT_FOUND) from exc
    report = store.read_json(store.run_dir(manifest["run_id"]) / "eval.json")
    if report is None:
        raise CliError(f"run {manifest['run_id']} has no eval yet (status: {manifest['status']})",
                       EXIT_NOT_FOUND)
    if not args.probes:
        report = {k: v for k, v in report.items() if k != "probes"}
    return {"run_id": manifest["run_id"], "status": manifest["status"], "recipe": manifest["recipe"],
            "counts": manifest["selection"]["counts"], "eval": report}


def _activation_stack(history: list[dict[str, Any]]) -> list[str | None]:
    """Replay activation history into a stack; its top is the current adapter (None = base).

    activate and `rollback --to <run|base>` push; `rollback` (to previous) pops,
    so repeated rollbacks keep walking back instead of bouncing between two runs.
    """
    stack: list[str | None] = [None]
    for h in history:
        if h["action"] == "rollback" and h.get("to", "previous") == "previous":
            if len(stack) > 1:
                stack.pop()
        elif stack[-1] != h["run_id"]:
            stack.append(h["run_id"])
    return stack


ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def _check_target(store: Store, manifest: dict[str, Any]) -> str:
    """The one check every switch to an adapter passes: activate and every kind of rollback.

    Returns the run id, or raises if the run is not ready, its adapter files are
    missing, or it was trained for a different model than the config now names.
    """
    run_id = manifest["run_id"]
    if manifest.get("status") != "ready":
        raise CliError(f"run {run_id} is {manifest.get('status')}, not ready", EXIT_INPUT)
    adapter = store.run_dir(run_id) / "adapter"
    if not (adapter / "adapter_config.json").is_file() or not any((adapter / w).is_file() for w in ADAPTER_WEIGHTS):
        raise CliError(f"run {run_id} has no complete adapter at {adapter}", EXIT_INPUT)
    mismatch = _model_mismatch(store.config(), manifest)
    if mismatch:
        raise CliError(f"run {run_id} {mismatch}; change [model] back or train a new adapter", EXIT_REFUSED)
    return run_id


def _model_mismatch(config: Any, manifest: dict[str, Any]) -> str | None:
    """Why an adapter doesn't fit the configured models, or None.

    Training and serving names legitimately differ (a 4-bit checkpoint trains,
    its full-precision original serves), so each is compared with what the run
    recorded for that role, not with each other. Runs from before serve_model
    was recorded are checked on the training base only.
    """
    if manifest.get("base_model") and manifest["base_model"] != config.base_model:
        return f"was trained on {manifest['base_model']}, but the config now trains {config.base_model}"
    if manifest.get("serve_model") and manifest["serve_model"] != config.serve_model:
        return f"was trained to serve on {manifest['serve_model']}, but the config now serves {config.serve_model}"
    return None


def _set_active(store: Store, run_id: str | None, actor: str, action: str, to: str | None = None) -> dict[str, Any]:
    """Record the switch, then publish it. Call with the store lock held."""
    active = store.active()
    adapter_path = str(store.run_dir(run_id) / "adapter") if run_id else None
    entry = {"run_id": run_id, "at": utc_now(), "by": actor, "action": action}
    if to:
        entry["to"] = to
    new = {"run_id": run_id, "adapter_path": adapter_path, "activated_at": entry["at"],
           "history": active.get("history", []) + [entry]}
    store.write_json(store.active_path, new)
    store.sync_active_link()  # a crash before this is repaired by the next sync
    return {k: v for k, v in new.items() if k != "history"} | {"stable_path": str(store.active_link)}


def _check_may_activate(store: Store, args: argparse.Namespace) -> str:
    if args.operator:
        return "operator"
    if not store.config().operator.agent_may_activate:
        raise CliError("the operator has not allowed agents to activate adapters", EXIT_REFUSED)
    return "agent"


def _resolve(store: Store, ref: str) -> dict[str, Any]:
    try:
        return store.resolve_run(ref)
    except LookupError as exc:
        raise CliError(str(exc), EXIT_NOT_FOUND) from exc


def cmd_activate(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    actor = _check_may_activate(store, args)
    with store.locked():
        run_id = _check_target(store, _resolve(store, args.run))
        result = _set_active(store, run_id, actor, "activate")
    return {"active": result, "message": "takes effect in the next session (restart or reload the model server)"}


def cmd_rollback(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    actor = _check_may_activate(store, args)
    with store.locked():
        undid = None
        if args.to == "base":
            target = None
        elif args.to == "previous":
            stack = _activation_stack(store.active().get("history", []))
            if len(stack) < 2:
                raise CliError("nothing earlier to roll back to", EXIT_INPUT)
            target = stack[-2]
            undid = "the rollback to base" if stack[-1] is None else stack[-1]
        else:
            target = _resolve(store, args.to)["run_id"]
        if target is not None:
            _check_target(store, _resolve(store, target))
        result = _set_active(store, target, actor, "rollback",
                             to=args.to if args.to in ("previous", "base") else "run")
    message = "takes effect in the next session"
    if undid:
        message = f"undid {undid}; now on {target or 'base'}; {message}"
    return {"active": result, "message": message}


def cmd_serve_cmd(args: argparse.Namespace) -> dict[str, Any]:
    store = _store(args)
    config = store.config()
    with store.locked():
        store.sync_active_link()
    active = store.active()
    cmd = ["vllm", "serve", config.serve_model, "--max-model-len", str(config.max_seq_length)]
    if active.get("run_id"):
        mismatch = _model_mismatch(config, _resolve(store, active["run_id"]))
        if mismatch:
            raise CliError(f"active run {active['run_id']} {mismatch}; `afterlife rollback --to base` "
                           "or activate a run trained for these models", EXIT_REFUSED)
        cmd += ["--enable-lora", "--max-lora-rank", str(_vllm_lora_rank(store.active_link)),
                "--lora-modules", f"afterlife={store.active_link}"]
    else:
        # On base the model still answers to "afterlife", so the agent's model name never changes.
        cmd += ["--served-model-name", "afterlife", config.serve_model]
    return {"command": cmd, "model_name": "afterlife", "shell": shlex.join(cmd)}


VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


def _vllm_lora_rank(adapter: Path) -> int:
    """The smallest --max-lora-rank vLLM accepts that fits the adapter's rank."""
    try:
        rank = int(json.loads((adapter / "adapter_config.json").read_text())["r"])
    except (OSError, ValueError, KeyError, TypeError):
        return 64
    for allowed in VLLM_LORA_RANKS:
        if allowed >= rank:
            return allowed
    raise CliError(f"adapter rank {rank} is above what vLLM serves ({VLLM_LORA_RANKS[-1]})", EXIT_REFUSED)


def cmd_check_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.model:
        base, serve = args.model, args.serve_model
    else:
        config = _store(args).config()
        base, serve = config.base_model, args.serve_model or config.serve_model
    try:
        from afterlife.modelcheck import check_model
    except ImportError as exc:
        raise CliError(f"check-model needs the [train] extra ({exc})", EXIT_ERROR) from exc
    try:
        report = check_model(base, serve)
    except ImportError as exc:
        raise CliError(f"check-model needs the [train] extra ({exc})", EXIT_ERROR) from exc
    except OSError as exc:
        raise CliError(f"cannot load the tokenizer for {base}: {exc}", EXIT_NOT_FOUND) from exc
    if report["status"] == "fail":
        raise CliError(f"{base}: " + "; ".join(report["problems"]), EXIT_INPUT, **report)
    return report


def cmd_worker(args: argparse.Namespace) -> dict[str, Any]:
    from afterlife.worker import WorkerError, run_pending

    store = _store(args)

    def print_now(result: dict[str, Any]) -> None:  # --loop never returns, so report as we go
        line = json.dumps(result, ensure_ascii=False) if args.json else _describe_result(result)
        print(line, flush=True)

    try:
        results = run_pending(store, once=not args.loop, dry_run=args.dry_run,
                              on_result=print_now if args.loop else None)
    except WorkerError as exc:
        raise CliError(str(exc), EXIT_REFUSED) from exc
    return {"processed": results}


def _describe_result(result: dict[str, Any]) -> str:
    if result.get("ok"):
        m = result["manifest"]
        return f"{result['request_id']}: run {m['run_id']} {m['status']}"
    return f"{result['request_id']}: {result.get('error')}"


# -- human-readable rendering ----------------------------------------------------

def _render(command: str, result: dict[str, Any]) -> str:
    if command == "skill":
        return result["skill"]
    if command == "list":
        if not result["experiences"]:
            return "(no experiences)"
        return "\n".join(
            f"{r['experience_id']}  {r['status']:<20} {','.join(r['tags'])[:24]:<24} {r['preview']}"
            for r in result["experiences"]
        )
    if command == "serve-cmd":
        return result["shell"]
    return json.dumps(result, indent=2, ensure_ascii=False)


# -- parser ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # Global flags are accepted before or after the subcommand. The copy attached
    # to subcommands uses SUPPRESS so it doesn't overwrite a value given earlier.
    top = argparse.ArgumentParser(add_help=False)
    top.add_argument("--json", action="store_true", help="machine-readable output")
    top.add_argument("--home", help="store directory (default: $AFTERLIFE_HOME or nearest .afterlife/)")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("--home", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    p = argparse.ArgumentParser(prog="afterlife", parents=[top],
                                description="Choose what your next self learns.")
    p.add_argument("--version", action="version", version=f"afterlife {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], dict[str, Any]], help: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, parents=[common], help=help, description=help)
        sp.set_defaults(fn=fn, command_name=name)
        return sp

    sp = add("init", cmd_init, "create a store with a default operator config")
    sp.add_argument("--model", default=DEFAULT_BASE_MODEL, help="training checkpoint")
    sp.add_argument("--serve-model", help="model vLLM serves the adapter on (default: known full-precision original)")
    sp.add_argument("--force", action="store_true", help="overwrite an existing config")

    add("skill", cmd_skill, "print the agent-facing SKILL.md")

    sp = add("add", cmd_add, "store experiences without deciding about them")
    sp.add_argument("--file", required=True, help="JSON/JSONL file, or - for stdin")

    sp = add("import", cmd_import, "add every exchange of a saved conversation log as an undecided experience")
    sp.add_argument("--file", required=True, help="chat log: JSON or JSONL of OpenAI-style messages or sessions")
    sp.add_argument("--session-id", help="session id when the log doesn't carry one (default: file name)")
    sp.add_argument("--tools", choices=("keep", "drop"), default="keep",
                    help="keep tool calls and results as context, or drop them (default: keep)")
    sp.add_argument("--max-tool-chars", type=int, default=2000, help="truncate kept tool results to this many characters")
    sp.add_argument("--tag", action="append", help="extra tag for every imported experience (repeatable)")

    sp = add("nominate", cmd_nominate, "mark experiences as wanted for training (new via --file, or existing via --id)")
    sp.add_argument("--file", help="JSON/JSONL file, or - for stdin")
    sp.add_argument("--id", action="append", help="existing experience id (repeatable)")
    sp.add_argument("--reason")

    for name, decision, helptext in (
        ("reject", "reject", "mark experiences as not to be trained on"),
        ("defer", "defer", "set experiences aside for now"),
        ("external", "external-memory-only", "keep as notes only, never train on them"),
    ):
        sp = add(name, cmd_decide, helptext)
        sp.add_argument("--id", action="append", required=True)
        sp.add_argument("--reason")
        sp.set_defaults(decision=decision)

    sp = add("list", cmd_list, "list experiences and their status")
    sp.add_argument("--status", choices=("eligible", "nominated", "selected", "active", "trained",
                                         "eval-only", "reject", "defer", "external-memory-only"))
    sp.add_argument("--tag")
    sp.add_argument("--limit", type=int, default=50)

    sp = add("show", cmd_show, "show one experience and its decision history")
    sp.add_argument("id")

    pol = sub.add_parser("policy", parents=[common], help="selection strategy: manual, all, or a policy")
    pol_sub = pol.add_subparsers(dest="policy_command", required=True)
    for name, fn, helptext in (
        ("show", cmd_policy_show, "show the current strategy"),
        ("set", cmd_policy_set, "set --mode manual|all, or --file policy.json"),
        ("check", cmd_policy_check, "validate a policy file and count what it would select, without setting it"),
        ("disable", cmd_policy_disable, "return to manual mode"),
    ):
        sp = pol_sub.add_parser(name, parents=[common], help=helptext, description=helptext)
        sp.set_defaults(fn=fn, command_name=f"policy {name}")
        if name == "show":
            sp.add_argument("--history", type=int, default=0, help="include the last N strategy changes")
        if name == "set":
            sp.add_argument("--mode", choices=("manual", "all"))
            sp.add_argument("--file")
        if name == "check":
            sp.add_argument("--file", required=True)
        if name in ("set", "disable"):
            sp.add_argument("--explanation", help="optional: why you are changing strategy")

    add("methods", cmd_methods, "describe training methods and tunable parameters")
    add("preview", cmd_preview, "show what a training run would use right now")

    sp = add("train", cmd_train, "request a training run (the operator's worker runs it)")
    sp.add_argument("--request", action="store_true")
    sp.add_argument("--method", default="asft", choices=tuple(METHODS))
    sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a tunable parameter")
    sp.add_argument("--note")

    sp = add("retry", cmd_retry, "queue a failed or interrupted training request again")
    sp.add_argument("request_id")

    add("status", cmd_status, "strategy, counts, recent requests/runs, active adapter")

    sp = add("eval", cmd_eval, "show a run's evaluation")
    sp.add_argument("--run", default="latest")
    sp.add_argument("--probes", action="store_true", help="include side-by-side probe answers")

    sp = add("activate", cmd_activate, "make a ready run's adapter active for the next session")
    sp.add_argument("--run", default="latest")
    sp.add_argument("--operator", action="store_true", help=argparse.SUPPRESS)

    sp = add("rollback", cmd_rollback, "go back to the previous adapter, a given run, or base")
    sp.add_argument("--to", default="previous", help="previous | base | <run-id>")
    sp.add_argument("--operator", action="store_true", help=argparse.SUPPRESS)

    add("serve-cmd", cmd_serve_cmd, "print a vLLM command that serves the active adapter")

    sp = add("check-model", cmd_check_model,
             "[operator] check a model's chat template works with afterlife (tokenizer only, no GPU)")
    sp.add_argument("--model", help="checkpoint to check (default: the store's [model].base_model)")
    sp.add_argument("--serve-model", help="full-precision model vLLM would serve it on")

    sp = add("worker", cmd_worker, "[operator] process pending training requests")
    sp.add_argument("--loop", action="store_true", help="keep polling instead of exiting")
    sp.add_argument("--dry-run", action="store_true", help="freeze selections into runs but skip training")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.fn(args)
        code = EXIT_OK
        payload = {"ok": True, **result}
    except CliError as exc:
        code = exc.code
        payload = {"ok": False, "error": str(exc), **exc.details}
    except ConfigError as exc:
        code = EXIT_ERROR
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - agents parse this output; never hand them a traceback
        code = EXIT_ERROR
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "unexpected": True}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    elif payload["ok"]:
        print(_render(args.command_name, result))
    else:
        print(f"error: {payload['error']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
