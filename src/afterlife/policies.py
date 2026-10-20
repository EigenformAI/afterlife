"""Selection modes and declarative selection policies.

Three modes decide what happens to experiences the agent has *not* explicitly
decided about:

    manual  - nothing is selected unless nominated
    all     - everything eligible is selected
    policy  - a declarative JSON policy decides

Explicit decisions always win, in every mode: `nominate` selects, `reject`,
`defer` and `external-memory-only` exclude.

A policy is data, never code. It is evaluated deterministically here; nothing
from it is executed. Shape::

    {
      "name": "optional label",
      "include": [ {predicates...}, ... ],   # OR of clauses; each clause ANDs its predicates
      "exclude": [ {predicates...}, ... ],   # same shape; any match excludes
      "order_by": "priority" | "recent" | "oldest" | "random",
      "limit": 200,
      "seed": 0
    }

An empty or missing `include` means "include everything not excluded".
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from afterlife.store import Store, sha256_json, utc_now

MODES = ("manual", "all", "policy")
ORDER_BY = ("priority", "recent", "oldest", "random")
MAX_LIMIT = 100_000


class PolicyError(Exception):
    pass


def _text(exp: dict[str, Any]) -> str:
    return "\n".join(m["content"] for m in exp["training_candidate"]["messages"]).lower()


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _str_list(name: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise PolicyError(f"{name} must be a non-empty list of strings")
    return value


def _number(name: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
        raise PolicyError(f"{name} must be a number in [{lo}, {hi}]")
    return float(value)


def _int(name: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise PolicyError(f"{name} must be an integer in [{lo}, {hi}]")
    return value


# Each predicate: (validate(value) -> normalised, build(value, context) -> exp -> bool)
Context = dict[str, Any]
Predicate = Callable[[dict[str, Any]], bool]


def _p_tags_any(v: list[str], _: Context) -> Predicate:
    s = set(v)
    return lambda e: bool(s & set(e["tags"]))


def _p_tags_all(v: list[str], _: Context) -> Predicate:
    s = set(v)
    return lambda e: s <= set(e["tags"])


def _p_tags_none(v: list[str], _: Context) -> Predicate:
    s = set(v)
    return lambda e: not (s & set(e["tags"]))


def _p_min_priority(v: float, _: Context) -> Predicate:
    return lambda e: e.get("priority") is not None and e["priority"] >= v


def _p_max_priority(v: float, _: Context) -> Predicate:
    return lambda e: e.get("priority") is not None and e["priority"] <= v


def _p_kind_in(v: list[str], _: Context) -> Predicate:
    s = set(v)
    return lambda e: e.get("kind") in s


def _p_verification_in(v: list[str], _: Context) -> Predicate:
    s = set(v)
    return lambda e: e["training_candidate"]["verification"]["status"] in s


def _p_contains_any(v: list[str], _: Context) -> Predicate:
    needles = [n.lower() for n in v]
    return lambda e: any(n in _text(e) for n in needles)


def _p_created_within_days(v: float, ctx: Context) -> Predicate:
    cutoff = ctx["now"] - timedelta(days=v)
    return lambda e: _parse_ts(e["created_at"]) >= cutoff


def _p_recent_sessions(v: int, ctx: Context) -> Predicate:
    # NSL's sliding window, expressed over sessions instead of generations.
    keep = set(ctx["sessions_newest_first"][:v])
    return lambda e: e["source"].get("session_id") in keep


def _p_min_chars(v: int, _: Context) -> Predicate:
    return lambda e: e["chars"] >= v


def _p_max_chars(v: int, _: Context) -> Predicate:
    return lambda e: e["chars"] <= v


PREDICATES: dict[str, tuple[Callable[[str, Any], Any], Callable[[Any, Context], Predicate]]] = {
    "tags_any": (_str_list, _p_tags_any),
    "tags_all": (_str_list, _p_tags_all),
    "tags_none": (_str_list, _p_tags_none),
    "min_priority": (lambda n, v: _number(n, v, 0, 1), _p_min_priority),
    "max_priority": (lambda n, v: _number(n, v, 0, 1), _p_max_priority),
    "kind_in": (_str_list, _p_kind_in),
    "verification_in": (_str_list, _p_verification_in),
    "contains_any": (_str_list, _p_contains_any),
    "created_within_days": (lambda n, v: _number(n, v, 0, 36500), _p_created_within_days),
    "recent_sessions": (lambda n, v: _int(n, v, 1, MAX_LIMIT), _p_recent_sessions),
    "min_chars": (lambda n, v: _int(n, v, 0, 10**7), _p_min_chars),
    "max_chars": (lambda n, v: _int(n, v, 0, 10**7), _p_max_chars),
}


def _validate_clauses(name: str, clauses: Any) -> list[dict[str, Any]]:
    if clauses is None:
        return []
    if not isinstance(clauses, list):
        raise PolicyError(f"{name} must be a list of clauses")
    out = []
    for i, clause in enumerate(clauses):
        if not isinstance(clause, dict) or not clause:
            raise PolicyError(f"{name}[{i}] must be a non-empty object")
        norm = {}
        for key, value in clause.items():
            if key not in PREDICATES:
                raise PolicyError(
                    f"{name}[{i}]: unknown predicate {key!r}; known: {', '.join(sorted(PREDICATES))}"
                )
            norm[key] = PREDICATES[key][0](f"{name}[{i}].{key}", value)
        out.append(norm)
    return out


def validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise PolicyError("policy must be a JSON object")
    allowed = {"name", "description", "include", "exclude", "order_by", "limit", "seed"}
    unknown = sorted(set(policy) - allowed)
    if unknown:
        raise PolicyError(f"unknown policy keys: {', '.join(unknown)}")
    order_by = policy.get("order_by", "priority")
    if order_by not in ORDER_BY:
        raise PolicyError(f"order_by must be one of {ORDER_BY}")
    norm: dict[str, Any] = {
        "include": _validate_clauses("include", policy.get("include")),
        "exclude": _validate_clauses("exclude", policy.get("exclude")),
        "order_by": order_by,
    }
    for key in ("name", "description"):
        if key in policy:
            if not isinstance(policy[key], str):
                raise PolicyError(f"{key} must be a string")
            norm[key] = policy[key]
    if "limit" in policy:
        norm["limit"] = _int("limit", policy["limit"], 1, MAX_LIMIT)
    if "seed" in policy:
        norm["seed"] = _int("seed", policy["seed"], 0, 2**31)
    return norm


def policy_hash(policy: dict[str, Any]) -> str:
    return sha256_json(policy)[:12]


def _context(experiences: list[dict[str, Any]]) -> Context:
    sessions: list[str] = []
    for e in sorted(experiences, key=lambda e: e["created_at"], reverse=True):
        sid = e["source"].get("session_id")
        if sid and sid not in sessions:
            sessions.append(sid)
    return {"now": datetime.now(timezone.utc), "sessions_newest_first": sessions}


def _compile(clauses: list[dict[str, Any]], ctx: Context) -> list[list[tuple[str, Predicate]]]:
    return [[(k, PREDICATES[k][1](v, ctx)) for k, v in clause.items()] for clause in clauses]


def order_experiences(
    experiences: list[dict[str, Any]], order_by: str, seed: int = 0
) -> list[dict[str, Any]]:
    if order_by == "priority":
        # Highest priority first; unscored last; newest first within ties.
        newest = sorted(experiences, key=lambda e: e["created_at"], reverse=True)
        return sorted(newest, key=lambda e: -(e["priority"] if e.get("priority") is not None else -1))
    if order_by == "recent":
        return sorted(experiences, key=lambda e: e["created_at"], reverse=True)
    if order_by == "oldest":
        return sorted(experiences, key=lambda e: e["created_at"])
    if order_by == "random":
        def key(e: dict[str, Any]) -> str:
            return hashlib.sha256(f"{seed}:{e['experience_id']}".encode()).hexdigest()
        return sorted(experiences, key=key)
    raise PolicyError(f"unknown order_by {order_by!r}")


def evaluate_policy(
    policy: dict[str, Any], experiences: list[dict[str, Any]]
) -> tuple[list[str], dict[str, str]]:
    """Apply a validated policy. Returns (selected ids in order, {excluded id: reason})."""
    ctx = _context(experiences)
    include = _compile(policy["include"], ctx)
    exclude = _compile(policy["exclude"], ctx)
    passed, excluded = [], {}
    for e in experiences:
        hit = next(
            (f"exclude[{i}]" for i, c in enumerate(exclude) if all(p(e) for _, p in c)), None
        )
        if hit:
            excluded[e["experience_id"]] = f"policy {hit} matched"
            continue
        if include and not any(all(p(e) for _, p in c) for c in include):
            excluded[e["experience_id"]] = "policy: no include clause matched"
            continue
        passed.append(e)
    ordered = order_experiences(passed, policy["order_by"], policy.get("seed", 0))
    limit = policy.get("limit")
    if limit is not None:
        for e in ordered[limit:]:
            excluded[e["experience_id"]] = f"policy limit {limit}"
        ordered = ordered[:limit]
    return [e["experience_id"] for e in ordered], excluded


# -- strategy state ----------------------------------------------------------

def current_strategy(store: Store) -> dict[str, Any]:
    events = store.policy_events()
    if not events:
        return {"mode": "manual", "policy_hash": None, "policy": None, "since": None}
    last = events[-1]
    policy = None
    if last.get("policy_hash"):
        policy = store.read_json(store.policies_dir / f"{last['policy_hash']}.json")
    return {
        "mode": last["new_mode"],
        "policy_hash": last.get("policy_hash"),
        "policy": policy,
        "since": last["created_at"],
    }


def set_strategy(
    store: Store,
    mode: str,
    *,
    policy: dict[str, Any] | None = None,
    explanation: str | None = None,
    actor: str = "agent",
) -> dict[str, Any]:
    if mode not in MODES:
        raise PolicyError(f"mode must be one of {MODES}")
    if (mode == "policy") != (policy is not None):
        raise PolicyError("a policy file is required for mode 'policy' and only for it")
    old = current_strategy(store)
    phash = None
    if policy is not None:
        policy = validate_policy(policy)
        phash = policy_hash(policy)
        path = store.policies_dir / f"{phash}.json"
        if not path.exists():
            store.write_json(path, policy)
    event = {
        "event": "strategy_changed",
        "created_at": utc_now(),
        "actor": actor,
        "old_mode": old["mode"],
        "new_mode": mode,
        "old_policy_hash": old["policy_hash"],
        "policy_hash": phash,
        "explanation": explanation,
        "active_run": store.active().get("run_id"),
    }
    store.append(store.policy_events_path, event)
    return event
