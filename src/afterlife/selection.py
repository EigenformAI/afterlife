"""From experience pool to training set.

    eligible   every stored experience
    selected   what the agent's decisions + current mode/policy pick
    validated  selected minus structural/length/verification failures and duplicates
    trained    validated, split by session into train vs held-out, capped by budget

Every excluded experience gets a reason, so `selected != trained` is visible.
"""

from __future__ import annotations

import hashlib
from typing import Any

from afterlife.config import Config
from afterlife.policies import current_strategy, evaluate_policy, order_experiences
from afterlife.schemas import check_messages, find_secrets
from afterlife.store import Store

EXCLUDING_DECISIONS = {"reject", "defer", "external-memory-only"}


def select(store: Store, strategy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply decisions and the current strategy. Pure w.r.t. the store contents."""
    strategy = strategy or current_strategy(store)
    experiences = store.experiences()
    latest = store.latest_decisions()

    explicit_in, undecided, excluded = [], [], {}
    for e in experiences:
        d = latest.get(e["experience_id"])
        if d is None:
            undecided.append(e)
        elif d["decision"] == "nominate":
            explicit_in.append(e)
        else:
            excluded[e["experience_id"]] = f"decision: {d['decision']}"

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
        "excluded": excluded,
    }


def _is_heldout(session_id: str | None, experience_id: str, fraction: float) -> bool:
    key = session_id or experience_id
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def build_training_set(store: Store, config: Config) -> dict[str, Any]:
    sel = select(store)
    op = config.operator
    excluded = dict(sel["excluded"])
    seen: dict[str, str] = {}
    valid = []
    for e in sel["selected"]:
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

    train, heldout = [], []
    for e in valid:
        if _is_heldout(e["source"].get("session_id"), e["experience_id"], op.eval_fraction):
            heldout.append(e)
        else:
            train.append(e)
    # Never hold out everything: tiny pools would otherwise train on nothing.
    if not train and heldout:
        train, heldout = heldout, []

    # Budget: explicit nominations first, then by priority / recency.
    explicit = set(sel["explicit_ids"])
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
            "train_chars": sum(e["chars"] for e in train),
        },
    }


def to_trainer_row(experience: dict[str, Any]) -> dict[str, Any]:
    """Convert to the row format trainer.qlora reads: one prompt, one response.

    Earlier turns are folded into the prompt as a plain transcript, so the loss
    lands only on the final assistant turn.
    """
    messages = experience["training_candidate"]["messages"]
    *context, final = messages
    if len(context) == 1 and context[0]["role"] == "user":
        prompt = context[0]["content"]
    else:
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in context)
    return {
        "success": True,
        "prompt": prompt,
        "raw_response": final["content"],
        "experience_id": experience["experience_id"],
    }
