import json

from afterlife.evaluator import _mean, _per_memory, load_probes


def test_per_memory_pools_turns_of_one_experience():
    rows = [{"experience_id": "a"}, {"experience_id": "a"}, {"experience_id": "b"}]
    scored = {"base": [(2.0, 1), (4.0, 3), (1.0, 2)], "updated": [(1.0, 1), (1.0, 3), None]}
    memories = {m["experience_id"]: m for m in _per_memory(rows, scored)}
    assert memories["a"]["turns"] == 2
    assert memories["a"]["loss"] == {"base": 3.5, "updated": 1.0}  # token-weighted
    assert memories["b"]["loss"] == {"base": 1.0, "updated": None}


def test_per_memory_ignores_rows_past_the_scored_limit():
    rows = [{"experience_id": "a"}, {"experience_id": "b"}]
    assert [m["experience_id"] for m in _per_memory(rows, {"base": [(1.0, 1)]})] == ["a"]


def test_mean_of_nothing_is_none():
    assert _mean([None]) is None and _mean([]) is None


def test_agent_probes_cannot_take_a_shipped_probe_id(tmp_path):
    custom = [{"id": "reg-arith", "prompt": "custom one"}, {"id": "reg-arith", "prompt": "custom two"},
              {"prompt": "no id"}, {"id": "bad", "prompt": ["not", "a", "string"]}, "not an object"]
    (tmp_path / "probes.jsonl").write_text("\n".join(json.dumps(p) for p in custom))
    probes = load_probes(tmp_path)
    ids = [p["id"] for p in probes]
    assert len(ids) == len(set(ids))
    by_id = {p["id"]: p for p in probes}
    assert by_id["reg-arith"]["source"] == "regression" and "17 * 23" in by_id["reg-arith"]["prompt"]
    assert by_id["agent:reg-arith"]["prompt"] == "custom one"
    assert by_id["agent:reg-arith-2"]["prompt"] == "custom two"
    assert sum(p["source"] == "agent" for p in probes) == 3  # malformed ones dropped
