import pytest
from conftest import experience

from afterlife.recipes import (
    RecipeError, parse_param, passes_per_example, resolve_recipe, steps_per_pass, train_sft_kwargs,
)
from afterlife.store import Store


def test_passes_depend_on_selection_size():
    assert steps_per_pass(24) == 3 and steps_per_pass(500) == 63 and steps_per_pass(1) == 1
    assert passes_per_example(40, 24) == 13.33  # the GPU smoke test's SFT run
    assert passes_per_example(100, 500) == 1.59


def test_epochs_set_the_steps(home):
    config = Store(home).config()
    recipe = resolve_recipe(config, "sft", {"epochs": 2.0}, train_rows=24)
    assert recipe["params"]["max_steps"] == 6 and recipe["passes_per_example"] == 2.0
    assert train_sft_kwargs(config, recipe)["max_steps"] == 6


def test_epochs_over_the_operator_cap_say_what_would_fit(home):
    with pytest.raises(RecipeError, match=r"at most 4\.76 epochs"):
        resolve_recipe(Store(home).config(), "sft", {"epochs": 10.0}, train_rows=500)  # cap 300 / 63


def test_epochs_and_max_steps_conflict(home):
    with pytest.raises(RecipeError, match="not both"):
        resolve_recipe(Store(home).config(), "sft", {"epochs": 1.0, "max_steps": 10}, train_rows=8)


def test_num_train_epochs_is_not_tunable():
    with pytest.raises(RecipeError, match="not tunable"):
        parse_param("num_train_epochs=3")


def test_request_and_preview_report_passes(home, cli, write_json):
    cli("nominate", "--file", write_json([experience(f"e{i}", session=f"s{i}") for i in range(16)]))
    _, out = cli("preview")
    assert out["passes_per_example_at_default_steps"]["asft"] == 50.0  # 100 steps, 2 steps per pass
    _, out = cli("train", "--request", "--method", "asft", "--set", "epochs=3")
    assert out["recipe"]["params"]["max_steps"] == 6 and out["recipe"]["passes_per_example"] == 3.0
