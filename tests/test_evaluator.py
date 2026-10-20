from afterlife.evaluator import _mean, _per_memory


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
