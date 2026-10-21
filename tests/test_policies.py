import pytest
from conftest import experience

from afterlife.policies import PolicyError, evaluate_policy, validate_policy
from afterlife.schemas import build_experience


def _pool(*specs):
    out = []
    for i, (text, kw) in enumerate(specs):
        e = build_experience(experience(text, **kw))
        e["created_at"] = f"2026-01-{i + 1:02d}T00:00:00Z"
        out.append(e)
    return out


def _select(policy, pool):
    ids, excluded = evaluate_policy(validate_policy(policy), pool)
    by_id = {e["experience_id"]: e for e in pool}
    return [by_id[i]["training_candidate"]["messages"][1]["content"].split()[-1] for i in ids], excluded


POOL = (
    ("cats", {"tags": ["fun", "animals"], "priority": 0.9, "session": "s1"}),
    ("tax", {"tags": ["boring"], "priority": 0.2, "session": "s1"}),
    ("comets", {"tags": ["fun", "space"], "priority": 0.5, "session": "s2"}),
    ("rust", {"tags": ["code"], "session": "s3"}),
)


def test_empty_policy_includes_everything():
    assert len(_select({}, _pool(*POOL))[0]) == 4


def test_include_is_or_of_ands():
    policy = {"include": [{"tags_all": ["fun", "space"]}, {"min_priority": 0.8}]}
    assert sorted(_select(policy, _pool(*POOL))[0]) == ["cats", "comets"]


def test_exclude_wins_and_is_explained():
    selected, excluded = _select({"exclude": [{"tags_any": ["boring"]}]}, _pool(*POOL))
    assert "tax" not in selected
    assert list(excluded.values()) == ["policy exclude[0] matched"]


def test_order_and_limit():
    selected, excluded = _select({"order_by": "priority", "limit": 2}, _pool(*POOL))
    assert selected == ["cats", "comets"]
    assert sum(1 for r in excluded.values() if r.startswith("policy limit")) == 2
    assert _select({"order_by": "oldest"}, _pool(*POOL))[0] == ["cats", "tax", "comets", "rust"]


def test_random_order_is_deterministic_per_seed():
    pool = _pool(*POOL)
    a = _select({"order_by": "random", "seed": 1}, pool)[0]
    assert a == _select({"order_by": "random", "seed": 1}, pool)[0]
    assert sorted(a) == sorted(_select({"order_by": "random", "seed": 2}, pool)[0])


def test_recent_sessions_is_a_sliding_window():
    assert sorted(_select({"include": [{"recent_sessions": 2}]}, _pool(*POOL))[0]) == ["comets", "rust"]


def test_contains_any_is_case_insensitive():
    assert _select({"include": [{"contains_any": ["COMETS"]}]}, _pool(*POOL))[0] == ["comets"]


@pytest.mark.parametrize("policy, message", [
    ({"include": [{"exec": "x"}]}, "unknown predicate"),
    ({"include": [{"min_priority": 2}]}, "min_priority"),
    ({"include": [{"tags_any": "fun"}]}, "tags_any"),
    ({"include": {"tags_any": ["fun"]}}, "list of clauses"),
    ({"order_by": "vibes"}, "order_by"),
    ({"limit": 0}, "limit"),
    ({"code": "print(1)"}, "unknown policy keys"),
    ({"include": [{}]}, "non-empty"),
])
def test_validation_errors(policy, message):
    with pytest.raises(PolicyError, match=message):
        validate_policy(policy)


def test_recent_sessions_counts_sessions_the_agent_already_decided_on():
    pool = _pool(("old", {"session": "s1"}), ("mid", {"session": "s2"}), ("new", {"session": "s3"}))
    undecided = pool[:2]  # the agent has already nominated the newest session's experience
    ids, _ = evaluate_policy(validate_policy({"include": [{"recent_sessions": 1}]}), undecided, pool)
    assert ids == []  # s3 is the most recent session, not s2


def test_same_second_ties_go_to_the_later_record():
    pool = _pool(("old", {"session": "s1"}), ("new", {"session": "s2"}))
    for e in pool:
        e["created_at"] = "2026-01-01T00:00:00Z"  # one batch, one second
    assert _select({"include": [{"recent_sessions": 1}]}, pool)[0] == ["new"]
    assert _select({"order_by": "recent"}, pool)[0] == ["new", "old"]


def test_random_order_depends_on_the_seed():
    pool = _pool(*POOL)
    for i, e in enumerate(pool):
        e["experience_id"] = f"exp-{i}"  # fixed ids: random ones make the two orders coincide 1 time in 24
    assert _select({"order_by": "random", "seed": 1}, pool)[0] != _select({"order_by": "random", "seed": 2}, pool)[0]
