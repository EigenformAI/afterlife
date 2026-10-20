"""Import conversation logs into the experience pool.

Without this, every experience has to be written out by hand, so keeping
everything costs the agent as much effort as choosing one at a time. Import
adds each exchange of a saved conversation as an *undecided* experience. It
selects nothing by itself: in manual mode imported experiences just sit in the
pool until nominated; under `--mode all` or a policy they are picked up like
any other.

Accepted input (OpenAI-style chat messages, which most local servers and agent
frameworks can export):

* a JSON object ``{"messages": [...], "session_id": ...}`` (one session)
* a JSON list of such objects, or a JSON list of messages (one session)
* JSONL with one session object per line, or one message per line (one session)

An *exchange* is a user turn and everything up to the next user turn. Its
target is the last assistant turn in it with text content (the answer after
any tool calls). Earlier exchanges of the session come along as context,
oldest dropped first, to stay within a character budget.

Re-importing the same log is a no-op: experience ids are derived from the
session id, the exchange's position and its content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from afterlife.schemas import ValidationError, build_experience, messages_text
from afterlife.store import Store, sha256_json

ROLE_ALIASES = {"developer": "system", "function": "tool", "human": "user", "ai": "assistant", "model": "assistant"}
TOOL_MODES = ("keep", "drop")


def _text(content: Any) -> str:
    """Message content as text; content-part lists keep only their text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p["text"] if isinstance(p, dict) and isinstance(p.get("text"), str) else p
            for p in content
            if isinstance(p, str) or (isinstance(p, dict) and isinstance(p.get("text"), str))
        )
    return str(content)


def _is_message(item: Any) -> bool:
    return isinstance(item, dict) and "role" in item and "messages" not in item


def load_sessions(path: Path, default_session: str) -> list[dict[str, Any]]:
    """Parse a log file into [{"session_id", "messages"}]."""
    text = path.read_text(encoding="utf-8")
    try:
        items: Any = json.loads(text)
    except json.JSONDecodeError:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a session object, a list of sessions, or a list of messages")
    if all(_is_message(i) for i in items):
        return [{"session_id": default_session, "messages": items}]
    sessions = []
    for n, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
            raise ValueError(f"{path}: item {n} has no messages list")
        sid = item.get("session_id") or item.get("id")
        sessions.append({
            "session_id": str(sid) if sid else (default_session if len(items) == 1 else f"{default_session}#{n}"),
            "messages": item["messages"],
        })
    return sessions


def _normalise(raw: list[Any], tools: str, max_tool_chars: int) -> tuple[list[dict[str, Any]], int]:
    """Clean messages; returns (messages, number of messages dropped)."""
    out, dropped = [], 0
    for m in raw:
        if not isinstance(m, dict):
            dropped += 1
            continue
        role = ROLE_ALIASES.get(m.get("role"), m.get("role"))
        if role not in ("system", "user", "assistant", "tool"):
            dropped += 1
            continue
        msg: dict[str, Any] = {"role": role, "content": _text(m.get("content"))}
        if tools == "keep":
            if role == "assistant" and isinstance(m.get("tool_calls"), list) and m["tool_calls"]:
                msg["tool_calls"] = m["tool_calls"]
            if role == "tool":
                for key in ("tool_call_id", "name"):
                    if isinstance(m.get(key), str):
                        msg[key] = m[key]
                if len(msg["content"]) > max_tool_chars:
                    msg["content"] = msg["content"][:max_tool_chars] + "\n…[truncated]"
        elif role == "tool" or (role == "assistant" and not msg["content"].strip()):
            dropped += 1
            continue
        if role != "tool" and not msg["content"].strip() and "tool_calls" not in msg:
            dropped += 1
            continue
        out.append(msg)
    return out, dropped


def _exchanges(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(start, end) index ranges, each starting at a user turn."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    return [(s, starts[k + 1] if k + 1 < len(starts) else len(messages)) for k, s in enumerate(starts)]


def session_experiences(
    session: dict[str, Any], *, tools: str, max_tool_chars: int, max_chars: int, source: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Turn one session into experience payloads (one per exchange that has an answer)."""
    messages, dropped = _normalise(session["messages"], tools, max_tool_chars)
    stats = {"messages_dropped": dropped, "exchanges_without_answer": 0}
    system = [m for m in messages if m["role"] == "system"][:1]
    ranges = _exchanges(messages)
    payloads = []
    for k, (start, end) in enumerate(ranges):
        answers = [i for i in range(start, end) if messages[i]["role"] == "assistant" and messages[i]["content"].strip()]
        if not answers:
            stats["exchanges_without_answer"] += 1
            continue
        target = answers[-1]
        # Earlier exchanges as context, newest kept first, within the character budget.
        context_start = start
        for prev_start, _ in reversed(ranges[:k]):
            candidate = system + messages[prev_start : target + 1]
            if len(messages_text(candidate)) > max_chars:
                break
            context_start = prev_start
        convo = system + [m for m in messages[context_start : target + 1] if m["role"] != "system"]
        sid = session["session_id"]
        payloads.append({
            # Stable across re-imports even if the context budget changes.
            "experience_id": "imp-" + sha256_json([sid, k, messages[start : target + 1]])[:16],
            "messages": convo,
            "session_id": sid,
            "trajectory_ref": f"{source}#session={sid}&exchange={k}",
            "kind": "imported",
            "tags": ["imported"],
        })
    return payloads, stats


def import_file(
    store: Store, path: Path, *, session_id: str | None = None, tools: str = "keep",
    max_tool_chars: int = 2000, tags: list[str] | None = None,
) -> dict[str, Any]:
    if tools not in TOOL_MODES:
        raise ValueError(f"tools must be one of {TOOL_MODES}")
    config = store.config()
    sessions = load_sessions(path, session_id or path.stem)
    known = store.experience_ids()
    added, refused = [], []
    totals = {"sessions": len(sessions), "already_imported": 0, "messages_dropped": 0, "exchanges_without_answer": 0}
    for session in sessions:
        payloads, stats = session_experiences(
            session, tools=tools, max_tool_chars=max_tool_chars,
            max_chars=config.operator.max_example_chars, source=str(path),
        )
        for key, value in stats.items():
            totals[key] += value
        for payload in payloads:
            if payload["experience_id"] in known:
                totals["already_imported"] += 1
                continue
            payload["tags"] = sorted(set(payload["tags"]) | set(tags or []))
            try:
                record = build_experience(payload)
            except ValidationError as exc:
                refused.append({"trajectory_ref": payload["trajectory_ref"], "problems": exc.problems})
                continue
            store.append(store.experiences_path, record)
            known.add(record["experience_id"])
            added.append(record["experience_id"])
    return {"added": added, "refused": refused, **totals}
