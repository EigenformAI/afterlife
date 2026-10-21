"""Experience records and their validation.

An *experience* is something the agent might want its next self to learn from:
a short conversation whose target assistant turns are the behaviour to learn.
An experience marked ``eval_only`` is never trained; it is used to check
whether training on the others carries over to it. Whether it is actually selected is a separate *decision* (see decisions.jsonl),
and whether it is actually trained on is decided later still, by the worker.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from typing import Any

from afterlife.store import new_id, sha256_json, utc_now

SCHEMA_VERSION = "0.1"
ROLES = {"system", "user", "assistant", "tool"}  # tool turns are context only, never trained
# Extra message keys passed through to the chat template (tool use).
MESSAGE_EXTRAS = ("tool_calls", "tool_call_id", "name")
DECISIONS = ("nominate", "reject", "defer", "external-memory-only")
VERIFICATION_STATUSES = ("unverified", "verified", "failed")
EXPERIENCE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

# Things that should never land in a training set. Deliberately conservative:
# a false positive costs the agent a rewrite, a false negative leaks a key into
# weights it may later share.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    "password_assignment": re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S{4,}"),
}


class ValidationError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def find_secrets(text: str) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]


def messages_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        parts.append(str(m.get("content", "")))
        if m.get("tool_calls"):
            parts.append(json.dumps(m["tool_calls"], ensure_ascii=False))
    return "\n".join(parts)


def check_messages(messages: Any) -> list[str]:
    """Structural problems that make a candidate unusable as an SFT example."""
    if not isinstance(messages, list) or not messages:
        return ["messages must be a non-empty list"]
    problems = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            problems.append(f"messages[{i}] must be an object")
            continue
        if m.get("role") not in ROLES:
            problems.append(f"messages[{i}].role must be one of {sorted(ROLES)}")
        if not isinstance(m.get("content"), str):
            problems.append(f"messages[{i}].content must be a string")
        if "tool_calls" in m and (m.get("role") != "assistant" or not isinstance(m["tool_calls"], list)):
            problems.append(f"messages[{i}].tool_calls must be a list on an assistant turn")
        if "train" in m and not isinstance(m["train"], bool):
            problems.append(f"messages[{i}].train must be true or false")
        elif m.get("train") and m.get("role") != "assistant":
            problems.append(f"messages[{i}]: only assistant turns can be marked train")
        for key in ("tool_call_id", "name"):
            if key in m and not isinstance(m[key], str):
                problems.append(f"messages[{i}].{key} must be a string")
    if problems:
        return problems
    for i in target_turns(messages):
        if not messages[i]["content"].strip():
            problems.append(f"messages[{i}] is a training target but empty")
        if not any(m["role"] == "user" for m in messages[:i]):
            problems.append(f"messages[{i}] is a training target with no user turn before it")
    if not any(m.get("train") for m in messages) and messages[-1]["role"] != "assistant":
        problems.append(
            "last message must be an assistant turn, or mark the assistant turns to learn with \"train\": true"
        )
    return problems


def target_turns(messages: list[dict[str, Any]]) -> list[int]:
    """Indices of the assistant turns to learn: those marked train=true, else the last turn."""
    marked = [i for i, m in enumerate(messages) if m.get("train") is True]
    if marked:
        return marked
    return [len(messages) - 1] if messages and messages[-1].get("role") == "assistant" else []


def dedup_key(messages: list[dict[str, Any]]) -> str:
    """Hash of the conversation with case/whitespace/unicode-form normalised.

    Which turns are trained is part of the key: the same conversation learned
    at a different turn is a different example. Tool-use fields count too.
    """
    targets = set(target_turns(messages))
    norm = [
        {
            "role": m["role"],
            "content": " ".join(unicodedata.normalize("NFKC", m["content"]).lower().split()),
            "train": i in targets,
            **{k: m[k] for k in MESSAGE_EXTRAS if k in m},
        }
        for i, m in enumerate(messages)
    ]
    return sha256_json(norm)[:16]


def build_experience(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise agent input into a stored experience record.

    Accepts either the flat form::

        {"messages": [...], "tags": [...], "priority": 0.7, "session_id": "s1"}

    or the nested form (``training_candidate`` / ``selection`` / ``source``). Raises ValidationError on anything that must not
    be stored at all (bad structure, likely secrets).
    """
    if not isinstance(payload, dict):
        raise ValidationError(["experience must be a JSON object"])
    problems = [f"{key} must be an object" for key in ("training_candidate", "selection", "source", "verification")
                if payload.get(key) is not None and not isinstance(payload[key], dict)]
    if problems:
        raise ValidationError(problems)

    candidate = payload.get("training_candidate") or {}
    if candidate.get("verification") is not None and not isinstance(candidate["verification"], dict):
        raise ValidationError(["training_candidate.verification must be an object"])
    selection = payload.get("selection") or {}
    source = dict(payload.get("source") or {})
    if "session_id" in payload:
        source.setdefault("session_id", payload["session_id"])
    if "trajectory_ref" in payload:
        source.setdefault("trajectory_ref", payload["trajectory_ref"])
    if "url" in payload:
        source.setdefault("url", payload["url"])
    # Readers group, hash and print these, so anything but a string poisons the store.
    problems = [f"source.{k} must be a non-empty string" for k in ("session_id", "trajectory_ref", "url")
                if source.get(k) is not None and not (isinstance(source[k], str) and source[k])]
    problems += [f"source.{k} must be a string, number, true/false or null" for k, v in source.items()
                 if not isinstance(v, (str, int, float, bool, type(None)))]
    experience_id = payload.get("experience_id")
    if experience_id is not None and not (isinstance(experience_id, str) and EXPERIENCE_ID.fullmatch(experience_id)):
        problems.append("experience_id must be 1-128 letters, digits, '.', '_', ':' or '-'")
    if problems:
        raise ValidationError(problems)

    messages = candidate.get("messages", payload.get("messages"))
    problems = check_messages(messages)
    if problems:
        raise ValidationError(problems)

    secrets_found = find_secrets(messages_text(messages))
    if secrets_found:
        raise ValidationError(
            [f"looks like it contains a secret ({', '.join(secrets_found)}); remove it and retry"]
        )

    tags = payload.get("tags", selection.get("tags", []))
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValidationError(["tags must be a list of strings"])

    priority = payload.get("priority", selection.get("priority"))
    if priority is not None:
        if not isinstance(priority, (int, float)) or not 0 <= priority <= 1:
            raise ValidationError(["priority must be a number in [0, 1]"])

    verification = dict(candidate.get("verification") or payload.get("verification") or {})
    verification.setdefault("status", "unverified")
    if verification["status"] not in VERIFICATION_STATUSES:
        raise ValidationError([f"verification.status must be one of {VERIFICATION_STATUSES}"])

    kind = payload.get("kind")
    if kind is not None and not isinstance(kind, str):
        raise ValidationError(["kind must be a string"])

    eval_only = payload.get("eval_only", False)
    if not isinstance(eval_only, bool):
        raise ValidationError(["eval_only must be true or false"])
    decision = embedded_decision(payload)  # validate before anything is stored

    record = {
        "schema_version": SCHEMA_VERSION,
        "experience_id": experience_id or new_id("exp"),
        "created_at": utc_now(),
        "created_ns": time.time_ns(),  # orders records made within the same second
        "source": source,
        "kind": kind,
        "tags": sorted(set(tags)),
        "priority": priority,
        "eval_only": eval_only,
        "training_candidate": {"messages": messages, "verification": verification},
        "dedup_key": dedup_key(messages),
        "chars": len(messages_text(messages)),
    }
    # Messages were scanned above with a clearer message; this covers everything
    # else that is persisted: source, tags, verification notes, ids, the reason.
    secrets_found = find_secrets(json.dumps([record, decision], ensure_ascii=False))
    if secrets_found:
        raise ValidationError(
            [f"metadata looks like it contains a secret ({', '.join(secrets_found)}); remove it and retry"]
        )
    return record


def embedded_decision(payload: dict[str, Any]) -> tuple[str, str | None] | None:
    """A decision carried inside the experience itself (brief format: selection.decision/reason)."""
    selection = payload.get("selection") or {}
    decision = payload.get("decision", selection.get("decision"))
    reason = payload.get("reason", selection.get("reason"))
    if decision is None:
        return None
    if decision not in DECISIONS:
        raise ValidationError([f"decision must be one of {DECISIONS}"])
    if reason is not None and not isinstance(reason, str):
        raise ValidationError(["reason must be a string"])
    return decision, reason


def build_decision(
    experience_id: str,
    decision: str,
    *,
    reason: str | None,
    mode: str,
    policy_hash: str | None,
    active_run: str | None,
) -> dict[str, Any]:
    if decision not in DECISIONS:
        raise ValidationError([f"decision must be one of {DECISIONS}"])
    if reason and find_secrets(reason):
        raise ValidationError(["reason looks like it contains a secret; remove it and retry"])
    return {
        "decision_id": new_id("dec"),
        "experience_id": experience_id,
        "created_at": utc_now(),
        "decision": decision,
        "reason": reason,
        "mode_in_effect": mode,
        "policy_hash": policy_hash,
        "active_run": active_run,
    }
