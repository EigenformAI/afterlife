import pytest
from conftest import experience

from afterlife.schemas import ValidationError, build_experience, dedup_key
from afterlife.selection import _heldout_sessions, build_training_set, to_trainer_rows
from afterlife.store import Store


def _exps(sessions):
    return [build_experience(experience(f"e{i}", session=s)) for i, s in enumerate(sessions)]


def test_heldout_holds_out_whole_sessions():
    held = _heldout_sessions(_exps(["s1", "s1", "s2", "s2", "s3"]), 0.3)
    assert len(held) == 1 and held <= {"s1", "s2", "s3"}


def test_heldout_always_holds_out_something_and_never_everything():
    sessions = [f"s{i:02d}" for i in range(12)]
    assert len(_heldout_sessions(_exps(sessions), 0.15)) == 2  # ceil(0.15 * 12)
    assert len(_heldout_sessions(_exps(["a", "b"]), 0.9)) == 1
    assert _heldout_sessions(_exps(["only"] * 5), 0.5) == set()
    assert _heldout_sessions(_exps(sessions), 0.0) == set()


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


def _split(home, seed=""):
    store = Store(home)
    ts = build_training_set(store, store.config(), split_seed=seed)
    return {e["experience_id"] for e in ts["train"]}, {e["experience_id"] for e in ts["heldout"]}


def test_nominations_are_never_held_out(home, cli, write_json):
    _, out = cli("nominate", "--file", write_json([experience(f"n{i}", session=f"s{i}") for i in range(10)]))
    train, heldout = _split(home)
    assert heldout == set() and train == set(out["nominated"])


def test_eval_only_is_always_held_out_and_never_trained(home, cli, write_json):
    cli("nominate", "--file", write_json(experience("mine")))
    _, out = cli("add", "--file", write_json(experience("check", session="s2", eval_only=True)))
    for mode in ("manual", "all"):
        cli("policy", "set", "--mode", mode)
        train, heldout = _split(home)
        assert heldout == set(out["added"]) and not train & heldout


def test_automatic_split_rotates_between_requests(home, cli, write_json):
    cli("add", "--file", write_json([experience(f"a{i}", session=f"s{i}") for i in range(10)]))
    cli("policy", "set", "--mode", "all")
    splits = [_split(home, seed=f"req-{i}")[1] for i in range(5)]
    assert all(len(h) == 2 for h in splits)  # ceil(0.15 * 10)
    assert len({frozenset(h) for h in splits}) > 1
    assert _split(home, seed="req-0")[1] == splits[0]  # stable for a given request


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
    [row] = to_trainer_rows(exp)
    assert row["raw_response"] == "that's all"
    assert "[system]\nbe brief" in row["prompt"] and "[assistant]\nhello" in row["prompt"]


def test_marked_turns_are_learned_instead_of_the_last():
    exp = build_experience({"messages": [
        {"role": "user", "content": "how do I find old pages?"},
        {"role": "assistant", "content": "use the Wayback Machine", "train": True},
        {"role": "user", "content": "and RSS?"},
        {"role": "assistant", "content": "look for a link rel=alternate tag", "train": True},
        {"role": "user", "content": "thanks!"},
        {"role": "assistant", "content": "you're welcome!"},
    ]})
    rows = to_trainer_rows(exp)
    assert [r["raw_response"] for r in rows] == ["use the Wayback Machine", "look for a link rel=alternate tag"]
    assert rows[0]["prompt"] == "how do I find old pages?"
    assert "[assistant]\nuse the Wayback Machine" in rows[1]["prompt"]
    assert all("welcome" not in r["prompt"] + r["raw_response"] for r in rows)


def test_marked_conversation_may_end_on_a_user_turn():
    exp = build_experience({"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "train": True},
        {"role": "user", "content": "thanks!"},
    ]})
    assert [r["raw_response"] for r in to_trainer_rows(exp)] == ["a"]


def test_eval_only_must_be_bool():
    with pytest.raises(ValidationError, match="eval_only"):
        build_experience(experience("a", eval_only="yes"))


def test_dedup_key_normalises_case_and_whitespace():
    a = [{"role": "user", "content": "Hi  there"}, {"role": "assistant", "content": "Yo"}]
    b = [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": " yo "}]
    assert dedup_key(a) == dedup_key(b)


@pytest.mark.parametrize("payload, message", [
    ({"messages": []}, "non-empty list"),
    ({"messages": [{"role": "assistant", "content": "x"}]}, "user turn"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "tool", "content": "y"}]}, "role"),
    ({"messages": [{"role": "user", "content": "x", "train": True},
                   {"role": "assistant", "content": "y"}]}, "only assistant turns"),
    ({"messages": [{"role": "user", "content": "x"},
                   {"role": "assistant", "content": "y", "train": "yes"}]}, "true or false"),
    ({"messages": [{"role": "assistant", "content": "y", "train": True},
                   {"role": "user", "content": "x"}]}, "no user turn"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
                   {"role": "user", "content": "thanks"}]}, "mark the assistant turns"),
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
        "selection": {"tags": ["debugging"], "priority": 0.85},
        "training_candidate": {"messages": experience("a")["messages"],
                               "verification": {"status": "unverified"}},
    })
    assert exp["tags"] == ["debugging"] and exp["priority"] == 0.85
    assert exp["source"]["session_id"] == "session-17"
