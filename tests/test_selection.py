import pytest
from conftest import experience

from afterlife.schemas import ValidationError, build_experience, dedup_key
from afterlife.selection import _is_heldout, build_training_set, to_trainer_row
from afterlife.store import Store


def test_heldout_split_keeps_sessions_together():
    for session in ("s1", "s2", "s3", "alpha", "beta"):
        assert len({_is_heldout(session, f"exp-{i}", 0.5) for i in range(20)}) == 1


def test_budget_prefers_explicit_nominations(home, cli, write_json):
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("max_examples_per_run = 500", "max_examples_per_run = 2")
                   .replace("eval_fraction = 0.15", "eval_fraction = 0.0"))
    cli("add", "--file", write_json([experience(f"auto{i}", priority=0.99) for i in range(3)]))
    _, out = cli("nominate", "--file", write_json(experience("mine", priority=0.1)))
    cli("policy", "set", "--mode", "all")

    store = Store(home)
    ts = build_training_set(store, store.config())
    train_ids = [e["experience_id"] for e in ts["train"]]
    assert out["nominated"][0] in train_ids
    assert len(train_ids) == 2
    assert sum(1 for r in ts["excluded"].values() if r.startswith("over budget")) == 2


def test_require_verified(home, cli, write_json):
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("require_verified = false", "require_verified = true"))
    cli("nominate", "--file", write_json([
        experience("checked", verification={"status": "verified"}), experience("hunch"),
    ]))
    store = Store(home)
    ts = build_training_set(store, store.config())
    assert ts["counts"]["validated"] == 1


def test_multi_turn_folds_context_into_prompt():
    exp = build_experience({"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and?"},
        {"role": "assistant", "content": "that's all"},
    ]})
    row = to_trainer_row(exp)
    assert row["raw_response"] == "that's all"
    assert "[system]\nbe brief" in row["prompt"] and "[assistant]\nhello" in row["prompt"]


def test_dedup_key_normalises_case_and_whitespace():
    a = [{"role": "user", "content": "Hi  there"}, {"role": "assistant", "content": "Yo"}]
    b = [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": " yo "}]
    assert dedup_key(a) == dedup_key(b)


@pytest.mark.parametrize("payload, message", [
    ({"messages": []}, "non-empty list"),
    ({"messages": [{"role": "assistant", "content": "x"}]}, "user turn"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "tool", "content": "y"}]}, "role"),
    ({"messages": experience("a")["messages"], "priority": 3}, "priority"),
    ({"messages": experience("a")["messages"], "tags": "fun"}, "tags"),
    ({"messages": [{"role": "user", "content": "-----BEGIN RSA PRIVATE KEY-----"},
                   {"role": "assistant", "content": "ok"}]}, "secret"),
])
def test_experience_validation(payload, message):
    with pytest.raises(ValidationError, match=message):
        build_experience(payload)


def test_nested_brief_format_is_accepted():
    exp = build_experience({
        "source": {"session_id": "session-17"},
        "selection": {"tags": ["debugging"], "priority": 0.85, "reason": "ignored here"},
        "training_candidate": {"messages": experience("a")["messages"],
                               "verification": {"status": "unverified"}},
    })
    assert exp["tags"] == ["debugging"] and exp["priority"] == 0.85
    assert exp["source"]["session_id"] == "session-17"
