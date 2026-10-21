import json

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


def test_multi_turn_keeps_the_conversation_as_messages():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and?"},
        {"role": "assistant", "content": "that's all"},
    ]
    [row] = to_trainer_rows(build_experience({"messages": messages}))
    assert row["raw_response"] == "that's all"
    assert row["messages"] == messages[:-1]
    assert "prompt" not in row  # the trainer renders messages with the chat template


def test_single_question_row_also_carries_prompt():
    [row] = to_trainer_rows(build_experience(experience("a")))
    assert row["prompt"] == "prompt for a"
    assert row["messages"] == [{"role": "user", "content": "prompt for a"}]


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
    assert rows[0]["messages"] == [{"role": "user", "content": "how do I find old pages?"}]
    assert [m["role"] for m in rows[1]["messages"]] == ["user", "assistant", "user"]
    assert all("train" not in m for r in rows for m in r["messages"])
    assert all("welcome" not in json.dumps(r) for r in rows)


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


def _convo(train_turn):
    turns = [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]
    return [{"role": r, "content": c, **({"train": True} if i == train_turn else {})}
            for i, (r, c) in enumerate(turns)]


def test_dedup_key_counts_which_turns_are_trained():
    assert dedup_key(_convo(1)) != dedup_key(_convo(3))
    assert dedup_key(_convo(3)) == dedup_key(_convo(None))  # unmarked = last turn trained


def test_dedup_key_counts_tool_calls():
    call = lambda name: [{"role": "user", "content": "go"},
                         {"role": "assistant", "content": "", "tool_calls": [{"name": name}]},
                         {"role": "tool", "content": "ok", "tool_call_id": "1"},
                         {"role": "assistant", "content": "done"}]
    assert dedup_key(call("ls")) != dedup_key(call("rm"))


def test_same_conversation_at_different_turns_is_not_a_duplicate(home, cli, write_json):
    cli("nominate", "--file", write_json([{"messages": _convo(1)}, {"messages": _convo(3)}]))
    store = Store(home)
    ts = build_training_set(store, store.config())
    assert len(ts["train"]) == 2 and ts["excluded"] == {}


def test_dedup_uses_current_hash_not_the_stored_one(home, cli, write_json):
    cli("nominate", "--file", write_json([{"messages": _convo(1)}, {"messages": _convo(3)}]))
    store = Store(home)
    # Simulate records written before the key included train markers.
    records = [dict(e, dedup_key="stale") for e in store.experiences()]
    store.experiences_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert len(build_training_set(store, store.config())["train"]) == 2


@pytest.mark.parametrize("payload, message", [
    ({"messages": []}, "non-empty list"),
    ({"messages": [{"role": "assistant", "content": "x"}]}, "user turn"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "function", "content": "y"}]}, "role"),
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


def _set_created(home, stamps):
    path = home / "experiences.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for row, ts in zip(rows, stamps):
        row["created_at"] = ts
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_budget_follows_the_policy_order(home, cli, write_json):
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("max_examples_per_run = 500", "max_examples_per_run = 2")
                   .replace("eval_fraction = 0.15", "eval_fraction = 0.0"))
    _, out = cli("add", "--file", write_json([
        experience("old", priority=0.9), experience("mid", priority=0.1), experience("new", priority=0.5),
    ]))
    old, mid, new = out["added"]
    _set_created(home, ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"])
    cli("policy", "set", "--file", write_json({"order_by": "recent"}))
    train, _ = _split(home)
    assert train == {new, mid}  # not the highest-priority ones


@pytest.mark.parametrize("payload, message", [
    ({"training_candidate": "not-an-object"}, "training_candidate must be an object"),
    ({"source": ["x"]}, "source must be an object"),
    ({"session_id": ["bad-session"]}, "session_id must be a non-empty string"),
    ({"source": {"note": {"nested": 1}}}, "source.note"),
    ({"experience_id": "../escape"}, "experience_id"),
    ({"experience_id": 7}, "experience_id"),
    ({"verification": "yes"}, "verification must be an object"),
])
def test_malformed_metadata_is_refused(payload, message):
    base = experience("a")
    base.pop("session_id")
    with pytest.raises(ValidationError, match=message):
        build_experience({**base, **payload})


@pytest.mark.parametrize("payload", [
    {"url": "https://example.com/?key=sk-" + "a" * 30},
    {"tags": ["sk-" + "b" * 30]},
    {"verification": {"status": "verified", "note": "password: hunter22"}},
    {"selection": {"decision": "nominate", "reason": "token ghp_" + "c" * 36}},
])
def test_secrets_in_metadata_are_refused(payload):
    with pytest.raises(ValidationError, match="secret"):
        build_experience({**experience("a"), **payload})


def test_effective_heldout_fraction_reflects_whole_sessions(home, cli, write_json):
    cli("add", "--file", write_json([experience(f"e{i}", session=f"s{i % 2}") for i in range(4)]))
    cli("policy", "set", "--mode", "all")
    store = Store(home)
    counts = build_training_set(store, store.config())["counts"]
    assert counts["heldout_fraction_target"] == 0.15
    assert counts["heldout_fraction_effective"] == 0.5  # one of two sessions


@pytest.mark.parametrize("credential", [
    "AIzaSy" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "hf_" + "a" * 34,
    "github_pat_" + "11ABCDEFG0" + "x" * 50,
    "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "postgres://user:hunter22@db.example.com/prod",
    "api_key = 9f8e7d6c5b4a39281706f5e4d3c2b1a0",
    'export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',
])
def test_common_credentials_are_refused(credential):
    exp = experience("a")
    exp["messages"][1]["content"] = f"Here you go: {credential}"
    with pytest.raises(ValidationError, match="secret"):
        build_experience(exp)


@pytest.mark.parametrize("benign", [
    "token = get_token()",
    "Set api_key in your config file before running.",
    "The secret to good bread is time.",
    "https://example.com/docs/page",
    "def check(password): return len(password) > 8",
])
def test_ordinary_text_is_not_mistaken_for_a_credential(benign):
    exp = experience("a")
    exp["messages"][1]["content"] = benign
    build_experience(exp)


TOOLS = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}]


def test_tool_definitions_are_stored_and_reach_trainer_rows():
    exp = build_experience({**experience("a"), "tools": TOOLS})
    assert exp["training_candidate"]["tools"] == TOOLS
    assert all(r["tools"] == TOOLS for r in to_trainer_rows(exp))
    assert "tools" not in to_trainer_rows(build_experience(experience("a")))[0]
    msgs = exp["training_candidate"]["messages"]
    assert dedup_key(msgs, TOOLS) != dedup_key(msgs)


def test_tools_must_be_a_list_of_objects():
    with pytest.raises(ValidationError, match="tools must be"):
        build_experience({**experience("a"), "tools": "add"})
