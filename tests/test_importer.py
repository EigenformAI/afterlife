import json

from afterlife.selection import build_training_set, to_trainer_rows
from afterlife.store import Store

SESSION = [
    {"role": "developer", "content": "You like exploring."},
    {"role": "user", "content": [{"type": "text", "text": "What's on the Wayback Machine for my site?"},
                                 {"type": "image_url", "image_url": {"url": "data:..."}}]},
    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "fetch", "arguments": "{\"url\": \"web.archive.org\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "snapshot list " + "x" * 5000},
    {"role": "assistant", "content": "There are 40 snapshots, the oldest from 2009."},
    {"role": "user", "content": "thanks!"},
    {"role": "assistant", "content": "you're welcome!"},
    {"role": "user", "content": "bye"},
]


def _write(tmp_path, payload, name="log.json"):
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return str(path)


def _messages(home, eid):
    return next(e for e in Store(home).experiences() if e["experience_id"] == eid)["training_candidate"]["messages"]


def test_import_splits_exchanges_and_keeps_tool_context(home, cli, tmp_path):
    code, out = cli("import", "--file", _write(tmp_path, {"session_id": "s9", "messages": SESSION}))
    assert code == 0 and len(out["added"]) == 2 and out["exchanges_without_answer"] == 1
    first, second = (_messages(home, e) for e in out["added"])
    assert [m["role"] for m in first] == ["system", "user", "assistant", "tool", "assistant"]
    assert first[1]["content"] == "What's on the Wayback Machine for my site?"
    assert first[2]["tool_calls"][0]["function"]["name"] == "fetch"
    assert first[3]["content"].endswith("…[truncated]") and len(first[3]["content"]) < 2100
    assert second[-1]["content"] == "you're welcome!"
    assert second[1]["role"] == "user" and second[-2]["content"] == "thanks!"  # earlier exchange as context

    [row] = to_trainer_rows(next(e for e in Store(home).experiences() if e["experience_id"] == out["added"][0]))
    assert row["raw_response"] == "There are 40 snapshots, the oldest from 2009."
    assert row["messages"][2]["tool_calls"] and row["messages"][3]["tool_call_id"] == "c1"


def test_import_adds_undecided_experiences(home, cli, tmp_path):
    _, out = cli("import", "--file", _write(tmp_path, {"messages": SESSION}))
    store = Store(home)
    assert build_training_set(store, store.config())["counts"]["train"] == 0  # manual mode: nothing selected
    cli("policy", "set", "--mode", "all")
    assert build_training_set(store, store.config())["counts"]["validated"] == 2
    assert all(e["source"]["session_id"] == "log" for e in store.experiences())  # from the file name


def test_reimport_is_a_no_op(home, cli, tmp_path):
    path = _write(tmp_path, {"messages": SESSION})
    cli("import", "--file", path)
    _, out = cli("import", "--file", path)
    assert out["added"] == [] and out["already_imported"] == 2


def test_drop_tools(home, cli, tmp_path):
    _, out = cli("import", "--file", _write(tmp_path, {"messages": SESSION}), "--tools", "drop")
    first = _messages(home, out["added"][0])
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    assert "tool_calls" not in first[2]


def test_jsonl_formats_and_tags(home, cli, tmp_path):
    lines = "\n".join(json.dumps(m) for m in SESSION[1:5])
    _, out = cli("import", "--file", _write(tmp_path, lines, "msgs.jsonl"), "--tag", "browsing")
    assert len(out["added"]) == 1
    assert Store(home).experiences()[0]["tags"] == ["browsing", "imported"]

    sessions = "\n".join(json.dumps({"id": f"chat-{i}", "messages": SESSION[1:5]}) for i in range(2))
    _, out = cli("import", "--file", _write(tmp_path, sessions, "sessions.jsonl"))
    assert out["sessions"] == 2 and len(out["added"]) == 2


def test_exchange_with_a_secret_is_refused_not_fatal(home, cli, tmp_path):
    session = [{"role": "user", "content": "my key is AKIAABCDEFGHIJKLMNOP"}, {"role": "assistant", "content": "ok"},
               {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    code, out = cli("import", "--file", _write(tmp_path, {"messages": session}), "--max-tool-chars", "100")
    assert code == 0 and len(out["refused"]) == 2 and out["added"] == []  # the key is context for the second too


def test_context_budget_drops_oldest_exchanges(home, cli, tmp_path):
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("max_example_chars = 12000", "max_example_chars = 300"))
    session = []
    for i in range(5):
        session += [{"role": "user", "content": f"q{i} " + "y" * 50}, {"role": "assistant", "content": f"a{i} " + "z" * 50}]
    _, out = cli("import", "--file", _write(tmp_path, {"messages": session}))
    last = _messages(home, out["added"][-1])
    assert last[0]["content"].startswith("q3") and len(last) == 4  # 6 messages would be 323 chars
