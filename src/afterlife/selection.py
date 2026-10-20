"""From experience pool to training set.

    eligible   every stored experience
    selected   what the agent's decisions + current mode/policy pick
    validated  selected minus structural/length/verification failures and duplicates
    trained    validated, minus held-out, capped by budget

Held-out data comes from two places. Experiences the agent marks eval_only are
always held out and never trained. Experiences picked by the mode or policy
(not nominated one by one) are split by session, a fresh split per request, so
nothing is held out every time. Explicit nominations are never held out: if
the agent chose it, it gets trained.

Every excluded experience gets a reason, so `selected != trained` is visible.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from afterlife.config import Config
from afterlife.policies import current_strategy, evaluate_policy, order_experiences
from afterlife.schemas import check_messages, find_secrets, target_turns
from afterlife.store import Store

EXCLUDING_DECISIONS = {"reject", "defer", "external-memory-only"}


def select(store: Store, strategy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply decisions and the current strategy. Pure w.r.t. the store contents."""
    strategy = strategy or current_strategy(store)
    experiences = store.experiences()
    latest = store.latest_decisions()

    explicit_in, undecided, eval_only, excluded = [], [], [], {}
    for e in experiences:
        d = latest.get(e["experience_id"])
        if d is not None and d["decision"] in EXCLUDING_DECISIONS:
            excluded[e["experience_id"]] = f"decision: {d['decision']}"
        elif e.get("eval_only"):
            eval_only.append(e)
        elif d is None:
            undecided.append(e)
        else:
            explicit_in.append(e)

    mode = strategy["mode"]
    if mode == "manual":
        for e in undecided:
            excluded[e["experience_id"]] = "manual mode: not nominated"
        auto = []
    elif mode == "all":
        auto = undecided
    else:
        ids, policy_excluded = evaluate_policy(strategy["policy"], undecided)
        excluded.update(policy_excluded)
        by_id = {e["experience_id"]: e for e in undecided}
        auto = [by_id[i] for i in ids]

    return {
        "mode": mode,
        "policy_hash": strategy.get("policy_hash"),
        "eligible": experiences,
        "selected": explicit_in + auto,
        "explicit_ids": [e["experience_id"] for e in explicit_in],
        "eval_only": eval_only,
        "excluded": excluded,
    }


def _heldout_sessions(experiences: list[dict[str, Any]], fraction: float, seed: str = "") -> set[str]:
    """Pick ceil(fraction * n) sessions by hash; at least one when there are >= 2.

    Splitting by session keeps paraphrases of one episode on the same side.
    Experiences without a session_id count as their own session. The seed
    (the request id) rotates which sessions are held out from run to run.
    """
    keys = sorted({e["source"].get("session_id") or e["experience_id"] for e in experiences})
    if fraction <= 0 or len(keys) < 2:
        return set()
    n = min(len(keys) - 1, max(1, math.ceil(fraction * len(keys))))
    ranked = sorted(keys, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    return set(ranked[:n])


def build_training_set(store: Store, config: Config, split_seed: str = "") -> dict[str, Any]:
    sel = select(store)
    op = config.operator
    excluded = dict(sel["excluded"])
    seen: dict[str, str] = {}
    valid = []
    for e in sel["selected"] + sel["eval_only"]:
        eid = e["experience_id"]
        messages = e["training_candidate"]["messages"]
        problems = check_messages(messages)
        if problems:
            excluded[eid] = "invalid: " + "; ".join(problems)
        elif find_secrets("\n".join(m["content"] for m in messages)):
            excluded[eid] = "invalid: possible secret"
        elif e["chars"] > op.max_example_chars:
            excluded[eid] = f"too long: {e['chars']} chars > {op.max_example_chars}"
        elif op.require_verified and e["training_candidate"]["verification"]["status"] != "verified":
            excluded[eid] = "operator requires verified examples"
        elif e["dedup_key"] in seen:
            excluded[eid] = f"duplicate of {seen[e['dedup_key']]}"
        else:
            seen[e["dedup_key"]] = eid
            valid.append(e)

    explicit = set(sel["explicit_ids"])
    eval_ids = {e["experience_id"] for e in sel["eval_only"]}
    splittable = [e for e in valid if e["experience_id"] not in explicit | eval_ids]
    held = _heldout_sessions(splittable, op.eval_fraction, split_seed)
    train, heldout = [], []
    for e in valid:
        eid = e["experience_id"]
        key = e["source"].get("session_id") or eid
        if eid in eval_ids or (eid not in explicit and key in held):
            heldout.append(e)
        else:
            train.append(e)

    # Budget: explicit nominations first, then by priority / recency.
    ordered = order_experiences(train, "priority")
    ordered.sort(key=lambda e: e["experience_id"] not in explicit)
    for e in ordered[op.max_examples_per_run :]:
        excluded[e["experience_id"]] = f"over budget ({op.max_examples_per_run} per run)"
    train = ordered[: op.max_examples_per_run]

    return {
        "mode": sel["mode"],
        "policy_hash": sel["policy_hash"],
        "train": train,
        "heldout": heldout,
        "excluded": excluded,
        "counts": {
            "eligible": len(sel["eligible"]),
            "selected": len(sel["selected"]),
            "validated": len(valid),
            "train": len(train),
            "heldout": len(heldout),
            "heldout_eval_only": sum(1 for e in heldout if e["experience_id"] in eval_ids),
            "train_chars": sum(e["chars"] for e in train),
            "train_rows": sum(len(target_turns(e["training_candidate"]["messages"])) for e in train),
        },
    }


def to_trainer_rows(experience: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert to the row format trainer.qlora reads: one prompt, one response per row.

    Each target turn (marked "train": true, else the final assistant turn)
    becomes one row. Earlier turns are folded into its prompt as a plain
    transcript, so the loss lands only on that turn.
    """
    messages = experience["training_candidate"]["messages"]
    rows = []
    for i in target_turns(messages):
        context = [{"role": m["role"], "content": m["content"]} for m in messages[:i]]
        if len(context) == 1 and context[0]["role"] == "user":
            prompt = context[0]["content"]
        else:
            prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in context)
        rows.append({
            "success": True,
            "prompt": prompt,
            "raw_response": messages[i]["content"],
            "experience_id": experience["experience_id"],
            "turn": i,
        })
    return rows
