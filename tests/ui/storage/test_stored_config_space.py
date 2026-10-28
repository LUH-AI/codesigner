"""An `.ihpo` may carry its search space as ConfigSpace's serialized dict, and
every reader of a space checks it before asking the model.
"""

import json
from typing import cast

import pytest
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter

from core import io

from tests.conftest import FIXTURES_DIR


def _a_space():
    """`_a_space` serializes a space with log scale and category."""
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

    space = ConfigurationSpace(
        space={"C": Float("C", (0.01, 100.0), log=True),
               "depth": Integer("depth", (1, 20)),
               "kernel": Categorical("kernel", ["rbf", "linear"])},
        seed=0,
    )
    return space.to_serialized_dict()


def test_a_space_decodes_back_to_the_same_hyperparameters():
    decoded = io.config_space_from_serialized(_a_space(), seed=7)
    assert decoded is not None

    assert list(decoded.keys()) == ["C", "depth", "kernel"]
    assert cast(UniformFloatHyperparameter, decoded["C"]).log is True
    assert list(cast(CategoricalHyperparameter, decoded["kernel"]).choices) == ["rbf", "linear"]


def test_decoding_does_not_consume_the_stored_dict():
    """`from_serialized_dict` pops keys, and `config_space_for` decodes several
    times per render.
    """
    stored = _a_space()

    first = io.config_space_from_serialized(stored)
    second = io.config_space_from_serialized(stored)
    assert first is not None and second is not None

    assert list(first.keys()) == list(second.keys())
    assert stored["hyperparameters"][0]["type"] == "uniform_float"


def test_no_space_is_not_an_error():
    """None and {} mean an older file."""
    assert io.config_space_from_serialized(None) is None
    assert io.config_space_from_serialized({}) is None


def test_the_space_survives_the_file():
    snapshot = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    snapshot["space"] = _a_space()

    again = io.parse(json.dumps(snapshot).encode("utf-8"))

    assert again["space"] == snapshot["space"]


@pytest.mark.django_db
def test_the_space_survives_the_row():
    from experiments.services import snapshot as adapter

    original = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    original["space"] = _a_space()

    row = adapter.experiment_from_snapshot(original)
    again = adapter.snapshot_from_experiment(row)

    assert row.data.config_space == original["space"]
    assert again["space"] == original["space"]


@pytest.mark.django_db
def test_an_experiment_without_a_space_writes_no_section():
    """A `"space": null` would claim older experiments answer a question they
    cannot.
    """
    from experiments.services import snapshot as adapter

    original = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    assert "space" not in original

    row = adapter.experiment_from_snapshot(original)

    assert row.data.config_space is None
    assert "space" not in adapter.snapshot_from_experiment(row)


def test_the_page_prefers_the_stored_space_over_the_model(models):
    """A widened model space may differ, and the trials belong to the space that
    produced them.
    """
    from experiments.services.snapshot import config_space_for

    model = next(iter(models.values()))
    built = {"model": model, "seed": 0, "config_space": _a_space()}

    space = config_space_for(built)
    assert space is not None
    assert list(space.keys()) == ["C", "depth", "kernel"]


def test_the_page_falls_back_to_the_model(models):
    from experiments.services.snapshot import config_space_for

    model = next(iter(models.values()))
    built = {"model": model, "seed": 0, "config_space": None}

    from_model = config_space_for(built)

    assert from_model is not None
    assert list(from_model.keys()) == list(model.get_config_space(seed=0).keys())


def test_a_space_with_no_model_still_answers():
    """An imported run or a read-only custom-model experiment has no importable
    model.
    """
    from experiments.services.snapshot import config_space_for

    built = {"model": None, "seed": 0, "config_space": _a_space()}

    space = config_space_for(built)
    assert space is not None
    assert list(space.keys()) == ["C", "depth", "kernel"]


def test_nothing_to_ask_is_still_None():
    """None is the figures' supported empty answer."""
    from experiments.services.snapshot import config_space_for

    assert config_space_for({"model": None, "seed": 0}) is None
    assert config_space_for(None) is None


def test_a_newer_format_is_refused_rather_than_lifted():
    """The format 1 lifter would return a future file with most sections empty.
    """
    snapshot = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    snapshot["format"] = io.SNAPSHOT_FORMAT + 1

    with pytest.raises(ValueError, match="newer than"):
        io.parse(json.dumps(snapshot).encode("utf-8"))
