"""Experience records and their validation.

An *experience* is something the agent might want its next self to learn from:
a short conversation whose final assistant turn is the behaviour to learn.
Whether it is actually selected is a separate *decision* (see decisions.jsonl),
and whether it is actually trained on is decided later still, by the worker.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from afterlife.store import new_id, sha256_json, utc_now

SCHEMA_VERSION = "0.1"
ROLES = {"system", "user", "assistant"}
DECISIONS = ("nominate", "reject", "defer", "external-memory-only")
VERIFICATION_STATUSES = ("unverified", "verified", "failed")

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


def _messages_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


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
    if problems:
        return problems
    if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
        problems.append("last message must be a non-empty assistant turn (that is what gets learned)")
    if not any(m["role"] == "user" for m in messages[:-1]):
        problems.append("need at least one user turn before the final assistant turn")
    return problems


def dedup_key(messages: list[dict[str, Any]]) -> str:
    """Hash of the conversation with case/whitespace/unicode-form normalised."""
    norm = [
        {
            "role": m["role"],
            "content": " ".join(unicodedata.normalize("NFKC", m["content"]).lower().split()),
        }
        for m in messages
    ]
    return sha256_json(norm)[:16]


def build_experience(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise agent input into a stored experience record.

    Accepts either the flat form::

        {"messages": [...], "tags": [...], "priority": 0.7, "session_id": "s1"}

    or the nested form from the engineering brief (``training_candidate`` /
    ``selection`` / ``source``). Raises ValidationError on anything that must not
    be stored at all (bad structure, likely secrets).
    """
    if not isinstance(payload, dict):
        raise ValidationError(["experience must be a JSON object"])

    candidate = payload.get("training_candidate") or {}
    selection = payload.get("selection") or {}
    source = dict(payload.get("source") or {})
    if "session_id" in payload:
        source.setdefault("session_id", payload["session_id"])
    if "trajectory_ref" in payload:
        source.setdefault("trajectory_ref", payload["trajectory_ref"])
    if "url" in payload:
        source.setdefault("url", payload["url"])

    messages = candidate.get("messages", payload.get("messages"))
    problems = check_messages(messages)
    if problems:
        raise ValidationError(problems)

    secrets_found = find_secrets(_messages_text(messages))
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

    return {
        "schema_version": SCHEMA_VERSION,
        "experience_id": payload.get("experience_id") or new_id("exp"),
        "created_at": utc_now(),
        "source": source,
        "kind": kind,
        "tags": sorted(set(tags)),
        "priority": priority,
        "training_candidate": {"messages": messages, "verification": verification},
        "dedup_key": dedup_key(messages),
        "chars": len(_messages_text(messages)),
    }


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
