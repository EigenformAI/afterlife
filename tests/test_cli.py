import json
from pathlib import Path

from conftest import experience

from afterlife.store import Store


def _ids(cli, status=None):
    args = ["list"] + (["--status", status] if status else [])
    return {r["experience_id"]: r["status"] for r in cli(*args)[1]["experiences"]}


def test_nominate_and_add_track_status(home, cli, write_json):
    code, out = cli("nominate", "--file", write_json(experience("a")), "--reason", "liked it")
    assert code == 0
    nominated = out["nominated"][0]
    code, out = cli("add", "--file", write_json([experience("b"), experience("c")]))
    assert code == 0 and len(out["added"]) == 2

    statuses = _ids(cli)
    assert statuses[nominated] == "nominated"
    assert sorted(statuses.values()) == ["eligible", "eligible", "nominated"]

    decision = Store(home).decisions()[0]
    assert decision["reason"] == "liked it"
    assert decision["mode_in_effect"] == "manual"


def test_secret_is_refused_and_not_stored(home, cli, write_json):
    exp = experience("x")
    exp["messages"][1]["content"] = "use ghp_" + "a" * 36
    code, out = cli("add", "--file", write_json(exp))
    assert code == 2 and "secret" in out["error"]
    assert Store(home).experiences() == []


def test_bad_batch_stores_nothing(home, cli, write_json):
    bad = {"messages": [{"role": "user", "content": "no answer"}]}
    code, _ = cli("add", "--file", write_json([experience("ok"), bad]))
    assert code == 2
    assert Store(home).experiences() == []


def test_explicit_decisions_override_save_all(home, cli, write_json):
    _, out = cli("add", "--file", write_json([experience("a"), experience("b")]))
    a, b = out["added"]
    cli("reject", "--id", a, "--reason", "not me")
    cli("policy", "set", "--mode", "all")
    statuses = _ids(cli)
    assert statuses[a] == "reject"
    assert statuses[b] == "selected"


def test_strategy_changes_are_logged_and_policies_versioned(home, cli, write_json):
    policy = {"include": [{"tags_any": ["fun"]}], "order_by": "recent"}
    cli("policy", "set", "--mode", "all", "--explanation", "trying it")
    code, out = cli("policy", "set", "--file", write_json(policy))
    assert code == 0
    phash = out["event"]["policy_hash"]
    cli("policy", "disable")

    events = Store(home).policy_events()
    assert [(e["old_mode"], e["new_mode"]) for e in events] == [
        ("manual", "all"), ("all", "policy"), ("policy", "manual")
    ]
    assert events[0]["explanation"] == "trying it"
    assert events[1]["policy_hash"] == phash
    assert (home / "policies" / f"{phash}.json").is_file()


def test_policy_check_does_not_change_strategy(home, cli, write_json):
    cli("add", "--file", write_json([experience("a", tags=["fun"]), experience("b")]))
    code, out = cli("policy", "check", "--file", write_json({"include": [{"tags_any": ["fun"]}]}))
    assert code == 0 and out["would_select"] == 1
    assert cli("policy", "show")[1]["strategy"]["mode"] == "manual"


def test_invalid_policy_is_rejected(home, cli, write_json):
    code, out = cli("policy", "set", "--file", write_json({"include": [{"eval": "rm -rf /"}]}))
    assert code == 2 and "unknown predicate" in out["error"]


def test_train_request_validation(home, cli, write_json):
    code, out = cli("train", "--request")
    assert code == 2 and "empty" in out["error"]

    cli("nominate", "--file", write_json(experience("a")))
    code, out = cli("train", "--request", "--method", "sft", "--set", "max_steps=100000")
    assert code == 2  # outside PARAM_BOUNDS
    code, out = cli("train", "--request", "--method", "sft", "--set", "max_steps=5000")
    assert code == 3 and "operator cap" in out["error"]
    code, out = cli("train", "--request", "--method", "sft", "--set", "asft_kl_weight=0.1")
    assert code == 2 and "only applies" in out["error"]

    code, out = cli("train", "--request", "--method", "asft", "--set", "asft_kl_weight=0.2")
    assert code == 0
    assert out["recipe"]["params"]["asft_kl_weight"] == 0.2
    assert out["request"]["status"] == "pending"


def test_brief_format_decision_and_reason_are_kept(home, cli, write_json):
    def brief(text, decision):
        return {"selection": {"decision": decision, "reason": f"because {text}"},
                "training_candidate": {"messages": experience(text)["messages"]}}

    _, out = cli("add", "--file", write_json([brief("no", "reject"), brief("yes", "nominate")]))
    rejected, nominated = out["added"]
    cli("policy", "set", "--mode", "all")
    statuses = _ids(cli)
    assert statuses[rejected] == "reject" and statuses[nominated] == "nominated"
    _, out = cli("show", rejected)
    assert out["decisions"][-1]["reason"] == "because no"
    assert cli("add", "--file", write_json(brief("x", "maybe")))[0] == 2


def test_disabled_method_is_refused(home, cli, write_json):
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace('allowed_methods = ["sft", "dft", "asft"]', 'allowed_methods = ["asft"]'))
    cli("nominate", "--file", write_json(experience("a")))
    code, out = cli("train", "--request", "--method", "sft")
    assert code == 3


def test_worker_dry_run_freezes_selection(home, cli, write_json):
    _, out = cli("nominate", "--file", write_json([
        experience("same"), experience("same"), experience("other", session="s2"),
    ]))
    dup = out["nominated"][1]
    cli("train", "--request", "--method", "dft")
    code, out = cli("worker", "--dry-run")
    assert code == 0
    result = out["processed"][0]
    assert result["ok"]
    manifest = result["manifest"]
    assert manifest["status"] == "dry_run"
    assert manifest["train_sft_kwargs"]["inner_loss"] == "dft"
    assert manifest["selection"]["counts"]["selected"] == 3
    assert manifest["selection"]["counts"]["validated"] == 2

    run_dir = home / "runs" / manifest["run_id"]
    rows = [json.loads(l) for l in (run_dir / "train.jsonl").read_text().splitlines()]
    rows += [json.loads(l) for l in (run_dir / "heldout.jsonl").read_text().splitlines()]
    assert all(r["success"] and r["messages"] and r["raw_response"] for r in rows)
    selection = json.loads((run_dir / "selection.json").read_text())
    assert selection["excluded"][dup].startswith("duplicate of")

    # A dry run is not a trained adapter: nothing counts as trained, nothing can be activated.
    assert "trained" not in _ids(cli).values()
    code, out = cli("activate", "--run", manifest["run_id"])
    assert code == 2


def test_one_bad_request_does_not_stop_the_worker(home, cli, write_json):
    cli("nominate", "--file", write_json(experience("a")))
    cli("train", "--request", "--method", "sft")
    cli("train", "--request", "--method", "asft")
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace('allowed_methods = ["sft", "dft", "asft"]', 'allowed_methods = ["asft"]'))
    code, out = cli("worker", "--dry-run")
    assert code == 0
    assert [r["ok"] for r in out["processed"]] == [False, True]
    assert [r["status"] for r in Store(home).requests()] == ["failed", "dry_run"]


def _fake_ready_run(home: Path, run_id: str, created_at: str) -> None:
    run_dir = home / "runs" / run_id
    (run_dir / "adapter").mkdir(parents=True)
    Store(home).write_json(run_dir / "manifest.json", {
        "run_id": run_id, "created_at": created_at, "status": "ready",
        "recipe": {"method": "asft"}, "selection": {"counts": {"train": 1}},
    })


def test_activate_and_rollback(home, cli):
    _fake_ready_run(home, "run-a", "2026-01-01T00:00:00Z")
    _fake_ready_run(home, "run-b", "2026-01-02T00:00:00Z")

    assert cli("activate", "--run", "run-a")[0] == 0
    assert cli("activate")[1]["active"]["run_id"] == "run-b"  # latest
    assert (home / "active-adapter").resolve() == (home / "runs" / "run-b" / "adapter").resolve()

    code, out = cli("rollback")
    assert code == 0 and out["active"]["run_id"] == "run-a"
    code, out = cli("rollback", "--to", "base")
    assert out["active"]["run_id"] is None
    assert not (home / "active-adapter").exists()
    assert "--enable-lora" not in cli("serve-cmd")[1]["command"]


def test_rollback_to_a_run_that_is_not_ready_is_refused(home, cli, write_json):
    cli("nominate", "--file", write_json(experience("a")))
    cli("train", "--request", "--method", "asft")
    run_id = cli("worker", "--dry-run")[1]["processed"][0]["manifest"]["run_id"]
    assert cli("rollback", "--to", run_id)[0] == 2
    assert Store(home).active().get("run_id") is None


def test_repeated_rollback_walks_back_through_history(home, cli):
    for i, run in enumerate(("run-a", "run-b", "run-c")):
        _fake_ready_run(home, run, f"2026-01-0{i + 1}T00:00:00Z")
        cli("activate", "--run", run)
    assert [cli("rollback")[1]["active"]["run_id"] for _ in range(3)] == ["run-b", "run-a", None]
    assert cli("rollback")[0] == 2  # nothing earlier than base

    cli("activate", "--run", "run-c")
    cli("rollback", "--to", "base")
    assert cli("rollback")[1]["active"]["run_id"] == "run-c"  # undoes the jump to base


def test_status_active_means_in_the_adapter_running_now(home, cli, write_json):
    _, out = cli("nominate", "--file", write_json([experience("a"), experience("b", session="s2")]))
    a, b = out["nominated"]
    for run, ids in (("run-a", [a]), ("run-b", [b])):
        _fake_ready_run(home, run, "2026-01-01T00:00:00Z")
        Store(home).write_json(home / "runs" / run / "selection.json", {"train_ids": ids})
    cli("activate", "--run", "run-a")
    assert _ids(cli) == {a: "active", b: "trained"}
    cli("rollback", "--to", "base")
    assert _ids(cli) == {a: "trained", b: "trained"}


def test_operator_can_forbid_agent_activation(home, cli):
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("agent_may_activate = true", "agent_may_activate = false"))
    _fake_ready_run(home, "run-a", "2026-01-01T00:00:00Z")
    assert cli("activate", "--run", "run-a")[0] == 3
    assert cli("activate", "--run", "run-a", "--operator")[0] == 0


def test_json_flag_before_subcommand(home, capsys):
    from afterlife.cli import main

    capsys.readouterr()
    assert main(["--json", "status"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_skill_is_packaged(home, cli):
    code, out = cli("skill")
    assert code == 0 and out["skill"].startswith("---\nname: afterlife")
