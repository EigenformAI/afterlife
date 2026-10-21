import json
import re
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


def _fake_ready_run(home: Path, run_id: str, created_at: str, **manifest) -> None:
    run_dir = home / "runs" / run_id
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "adapter" / "adapter_config.json").write_text("{}")
    (run_dir / "adapter" / "adapter_model.safetensors").write_text("")
    Store(home).write_json(run_dir / "manifest.json", {
        "run_id": run_id, "created_at": created_at, "status": "ready",
        "recipe": {"method": "asft"}, "selection": {"counts": {"train": 1}}, **manifest,
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
    out = cli("rollback")[1]
    assert out["active"]["run_id"] == "run-c"  # undoes the jump to base
    assert out["message"].startswith("undid the rollback to base; now on run-c")


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


def test_ingest_rechecks_ids_under_the_lock(home, cli, write_json, monkeypatch):
    import afterlife.cli as cli_mod

    real_build = cli_mod.build_experience

    def build_then_race(item):
        record = real_build(item)
        # Another process writes the same id after our first check, before our write.
        Store(home).append(Store(home).experiences_path, dict(record))
        return record

    monkeypatch.setattr(cli_mod, "build_experience", build_then_race)
    code, out = cli("add", "--file", write_json(experience("a", experience_id="exp-fixed")))
    assert code == 2 and "already exists" in out["error"]
    ids = [e["experience_id"] for e in Store(home).experiences()]
    assert ids.count("exp-fixed") == 1


def _set_cooldown(home: Path, minutes: int) -> None:
    cfg = home / "config.toml"
    cfg.write_text(re.sub(r"min_minutes_between_runs = \S+", f"min_minutes_between_runs = {minutes}", cfg.read_text()))


def test_cooldown_leaves_request_pending_and_says_why(home, cli, write_json):
    from afterlife.store import utc_now

    _set_cooldown(home, 60)
    _fake_ready_run(home, "run-a", utc_now())
    cli("nominate", "--file", write_json(experience("a")))
    cli("train", "--request", "--method", "sft")

    code, out = cli("worker", "--dry-run")
    assert code == 0 and out["processed"][0]["waiting"]
    request = Store(home).requests()[0]
    assert request["status"] == "pending" and "operator minimum is 60 min" in request["blocked_reason"]

    shown = cli("status")[1]["requests"][0]
    assert shown["eligible_at"] == request["eligible_at"] and shown["blocked_reason"]
    cli("worker", "--dry-run")
    assert Store(home).requests()[0]["eligible_at"] == request["eligible_at"]  # stable across polls


def test_dry_runs_do_not_start_the_cooldown(home, cli, write_json):
    _set_cooldown(home, 60)
    cli("nominate", "--file", write_json(experience("a")))
    cli("train", "--request", "--method", "sft")
    cli("train", "--request", "--method", "sft")
    code, out = cli("worker", "--dry-run")
    assert [r["ok"] for r in out["processed"]] == [True, True]
    assert "blocked_reason" not in cli("status")[1]["requests"][0]


def test_worker_loop_reports_a_cooldown_once(home, cli, write_json):
    from afterlife.store import utc_now
    from afterlife.worker import _process_pending

    _set_cooldown(home, 60)
    _fake_ready_run(home, "run-a", utc_now())
    cli("nominate", "--file", write_json(experience("a")))
    cli("train", "--request", "--method", "sft")

    store, waiting = Store(home), set()
    assert [r["waiting"] for r in _process_pending(store, waiting, dry_run=True)] == [True]
    assert _process_pending(store, waiting, dry_run=True) == []  # later polls stay quiet

    _set_cooldown(home, 0)  # cooldown over: the request runs and its wait note is cleared
    assert [r["ok"] for r in _process_pending(store, waiting, dry_run=True)] == [True]
    request = store.requests()[0]
    assert request["status"] == "dry_run" and request["blocked_reason"] is None


# -- activation: one checked, serialized, recoverable transition -----------------

def _activate_many(home: str, runs: list[str], n: int) -> None:
    import os

    from afterlife.cli import main as run_cli
    os.environ["AFTERLIFE_HOME"] = home
    for i in range(n):
        run_cli(["activate", "--run", runs[i % len(runs)], "--json"])


def test_concurrent_activations_keep_every_history_entry(home):
    import multiprocessing

    for run in ("run-a", "run-b"):
        _fake_ready_run(home, run, "2026-01-01T00:00:00Z")
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_activate_many, args=(str(home), ["run-a", "run-b"][i % 2:] + ["run-a"], 15))
             for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    store = Store(home)
    assert len(store.active()["history"]) == 60
    assert (home / "active-adapter").resolve() == Path(store.active()["adapter_path"]).resolve()
    assert not list(home.glob(".active.json.*")) and not list(home.glob(".active-adapter.*"))


def test_a_lost_symlink_is_repaired_from_active_json(home, cli):
    _fake_ready_run(home, "run-a", "2026-01-01T00:00:00Z")
    cli("activate", "--run", "run-a")
    (home / "active-adapter").unlink()  # crash between recording and publishing
    assert cli("serve-cmd")[0] == 0
    assert (home / "active-adapter").resolve() == (home / "runs" / "run-a" / "adapter").resolve()


def test_rollback_checks_the_adapter_is_still_there(home, cli):
    import shutil

    for run in ("run-a", "run-b"):
        _fake_ready_run(home, run, "2026-01-01T00:00:00Z")
        cli("activate", "--run", run)
    shutil.rmtree(home / "runs" / "run-a" / "adapter")
    code, out = cli("rollback")
    assert code == 2 and "no complete adapter" in out["error"]
    assert Store(home).active()["run_id"] == "run-b"


def test_an_adapter_is_refused_for_a_different_model(home, cli):
    _fake_ready_run(home, "run-old", "2026-01-01T00:00:00Z", base_model="other/base", serve_model="other/serve")
    code, out = cli("activate", "--run", "run-old")
    assert code == 3 and "other/base" in out["error"]

    config = Store(home).config()
    _fake_ready_run(home, "run-a", "2026-01-02T00:00:00Z",
                    base_model=config.base_model, serve_model=config.serve_model)
    _fake_ready_run(home, "run-legacy", "2026-01-03T00:00:00Z", base_model=config.base_model)
    assert cli("activate", "--run", "run-legacy")[0] == 0  # recorded before serve_model was
    assert cli("activate", "--run", "run-a")[0] == 0

    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace(f'serve_model = "{config.serve_model}"', 'serve_model = "some/other"'))
    code, out = cli("serve-cmd")
    assert code == 3 and "rollback --to base" in out["error"]
    assert cli("rollback", "--to", "base")[0] == 0  # always possible
    assert cli("serve-cmd")[0] == 0


def test_serve_command_survives_shell_quoting(tmp_path, monkeypatch, cli):
    import shlex

    root = tmp_path / "my store" / ".afterlife"
    monkeypatch.setenv("AFTERLIFE_HOME", str(root))
    cli("init")
    config = Store(root).config()
    _fake_ready_run(root, "run-a", "2026-01-01T00:00:00Z", base_model=config.base_model)
    cli("activate", "--run", "run-a")
    out = cli("serve-cmd")[1]
    assert shlex.split(out["shell"]) == out["command"]


# -- frozen requests under current operator rules ------------------------------

def test_frozen_members_must_still_meet_current_rules(home, cli, write_json):
    cli("nominate", "--file", write_json([experience("a"), experience("b", session="s2")]))
    cli("add", "--file", write_json(experience("later", session="s3")))
    cli("train", "--request", "--method", "sft")
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("require_verified = false", "require_verified = true"))
    code, out = cli("worker", "--dry-run")
    result = out["processed"][0]
    assert not result["ok"] and "nothing to train" in result["error"]  # refused, not replaced


def test_frozen_member_is_dropped_when_its_content_changed(home, cli, write_json):
    _, out = cli("nominate", "--file", write_json([experience("a"), experience("b", session="s2")]))
    a, b = out["nominated"]
    cli("train", "--request", "--method", "sft")
    store = Store(home)
    records = store.experiences()
    records[0]["training_candidate"]["messages"][1]["content"] = "swapped in later"
    store.experiences_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = cli("worker", "--dry-run")[1]["processed"][0]["manifest"]
    selection = json.loads((home / "runs" / manifest["run_id"] / "selection.json").read_text())
    assert selection["train_ids"] == [b]
    assert selection["excluded"][a] == "content changed since the request was queued"


# -- interruption and recovery ---------------------------------------------------

def test_ctrl_c_mid_training_marks_the_request_interrupted(home, cli, write_json, monkeypatch):
    import pytest

    from afterlife import worker

    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "_child", interrupted)
    cli("nominate", "--file", write_json(experience("a")))
    cli("train", "--request", "--method", "sft")
    with pytest.raises(KeyboardInterrupt):
        worker.run_pending(Store(home))
    request = Store(home).requests()[0]
    assert request["status"] == "interrupted"
    assert Store(home).resolve_run(request["run_id"])["status"] == "interrupted"


def test_work_left_by_a_dead_worker_is_recovered_then_retried(home, cli, write_json):
    cli("nominate", "--file", write_json(experience("a")))
    request_id = cli("train", "--request", "--method", "sft")[1]["request"]["request_id"]
    run_id = cli("worker", "--dry-run")[1]["processed"][0]["manifest"]["run_id"]
    store = Store(home)
    # Rewind to what a SIGKILLed worker leaves behind.
    request = store.requests()[0]
    request["status"] = "running"
    store.save_request(request)
    manifest = store.resolve_run(run_id)
    store.write_json(home / "runs" / run_id / "manifest.json", {**manifest, "status": "training"})

    out = cli("worker", "--dry-run")[1]
    assert out["processed"] == [{"request_id": request_id, "ok": False, "recovered": True,
                                 "error": out["processed"][0]["error"]}]  # not retrained by itself
    assert store.requests()[0]["status"] == "interrupted"
    assert store.resolve_run(run_id)["status"] == "interrupted"

    assert cli("retry", "done-or-missing")[0] == 4
    code, out = cli("retry", request_id)
    assert code == 0 and out["request"]["previous_attempts"][0]["status"] == "interrupted"
    assert cli("worker", "--dry-run")[1]["processed"][0]["ok"]
    assert cli("retry", request_id)[0] == 2  # only failed or interrupted ones


def test_preview_says_which_experience_was_excluded_and_why(home, cli, write_json):
    _, out = cli("nominate", "--file", write_json([experience("same"), experience("same")]))
    excluded = cli("preview")[1]["excluded"]
    assert excluded == {out["nominated"][1]: f"duplicate of {out['nominated'][0]}"}


def test_train_request_warns_when_the_budget_cuts(home, cli, write_json):
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("max_examples_per_run = 500", "max_examples_per_run = 2")
                   .replace("eval_fraction = 0.15", "eval_fraction = 0.0"))
    cli("nominate", "--file", write_json([experience(f"n{i}", session=f"s{i}") for i in range(3)]))
    out = cli("train", "--request", "--method", "sft")[1]
    assert out["over_budget"] == 1 and "budget of 2" in out["warning"]

    cfg.write_text(cfg.read_text().replace("max_examples_per_run = 2", "max_examples_per_run = 500"))
    out = cli("train", "--request", "--method", "sft")[1]
    assert "over_budget" not in out and "warning" not in out
