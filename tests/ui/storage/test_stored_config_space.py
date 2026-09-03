"""The search space travels with the experiment, not only with the model.

*What:* the optional `space` section — a `.ihpo` may carry the search space its
trials were drawn from, as ConfigSpace's own serialized dict, and everything
that needs a space reads it before falling back to asking the model.

*How:* a space is built from a registry model and put through each seam in
turn — the file (`io.save`/`io.parse`), the row (`experiment_from_snapshot` and
back), and the page (`_config_space_for`) — including the case these exist for:
an experiment with **no resolvable model at all**, which before this could
answer nothing and now answers from the file.
"""

import json

import pytest

from core import io

from tests.conftest import FIXTURES_DIR


def _a_space():
    """A small space with a log scale and a category, as a serialized dict."""
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

    space = ConfigurationSpace(
        space={"C": Float("C", (0.01, 100.0), log=True),
               "depth": Integer("depth", (1, 20)),
               "kernel": Categorical("kernel", ["rbf", "linear"])},
        seed=0,
    )
    return space.to_serialized_dict()


def test_a_space_decodes_back_to_the_same_hyperparameters():
    """`config_space_from_serialized` is the inverse of `to_serialized_dict`."""
    decoded = io.config_space_from_serialized(_a_space(), seed=7)

    assert list(decoded.keys()) == ["C", "depth", "kernel"]
    assert decoded["C"].log is True
    assert list(decoded["kernel"].choices) == ["rbf", "linear"]


def test_decoding_does_not_consume_the_stored_dict():
    """Decoding twice works — `from_serialized_dict` pops keys off its input.

    The stored dict belongs to a snapshot the caller keeps using, and
    `_config_space_for` runs several times per page render, so a decode that
    destroyed its argument would work once and then raise.
    """
    stored = _a_space()

    first = io.config_space_from_serialized(stored)
    second = io.config_space_from_serialized(stored)

    assert list(first.keys()) == list(second.keys())
    assert stored["hyperparameters"][0]["type"] == "uniform_float"


def test_no_space_is_not_an_error():
    """None and {} both mean "this file predates the section", not a failure."""
    assert io.config_space_from_serialized(None) is None
    assert io.config_space_from_serialized({}) is None


def test_the_space_survives_the_file():
    """`space` written by `io.save` comes back out of `io.parse` unchanged."""
    snapshot = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    snapshot["space"] = _a_space()

    again = io.parse(json.dumps(snapshot).encode("utf-8"))

    assert again["space"] == snapshot["space"]


@pytest.mark.django_db
def test_the_space_survives_the_row():
    """snapshot → Experiment row → snapshot keeps `space` byte-for-byte."""
    from ui.services import snapshot as adapter

    original = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    original["space"] = _a_space()

    row = adapter.experiment_from_snapshot(original)
    again = adapter.snapshot_from_experiment(row)

    assert row.config_space == original["space"]
    assert again["space"] == original["space"]


@pytest.mark.django_db
def test_an_experiment_without_a_space_writes_no_section():
    """An absent space is absent, not null.

    Every experiment predating this has none, and a `"space": null` in each of
    their files would claim they answer a question they do not.
    """
    from ui.services import snapshot as adapter

    original = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    assert "space" not in original

    row = adapter.experiment_from_snapshot(original)

    assert row.config_space is None
    assert "space" not in adapter.snapshot_from_experiment(row)


def test_the_page_prefers_the_stored_space_over_the_model(models):
    """`_config_space_for` reads the file's space before asking the model.

    The two can legitimately differ — a model whose search space was widened
    since the run — and the trials belong to the one that produced them.
    """
    from ui.views import _config_space_for

    model = next(iter(models.values()))
    built = {"model": model, "seed": 0, "config_space": _a_space()}

    assert list(_config_space_for(built).keys()) == ["C", "depth", "kernel"]


def test_the_page_falls_back_to_the_model(models):
    """With no stored space, the model answers — exactly as it always did."""
    from ui.views import _config_space_for

    model = next(iter(models.values()))
    built = {"model": model, "seed": 0, "config_space": None}

    from_model = _config_space_for(built)

    assert from_model is not None
    assert list(from_model.keys()) == list(model.get_config_space(seed=0).keys())


def test_a_space_with_no_model_still_answers():
    """The case the section exists for: trials, a space, and no model at all.

    A run read out of somebody else's output has no model, and a custom-model
    experiment viewed read-only cannot import the one it has. Before the stored
    space, both answered None here and lost every surrogate-backed figure.
    """
    from ui.views import _config_space_for

    built = {"model": None, "seed": 0, "config_space": _a_space()}

    assert list(_config_space_for(built).keys()) == ["C", "depth", "kernel"]


def test_nothing_to_ask_is_still_None():
    """No space and no model stays None — the figures' supported empty answer."""
    from ui.views import _config_space_for

    assert _config_space_for({"model": None, "seed": 0}) is None
    assert _config_space_for(None) is None


def test_a_newer_format_is_refused_rather_than_lifted():
    """A format this build does not know is an error, not a format 1 file.

    `normalize` dispatched on exact equality, so anything that was not format 2
    fell to the format 1 lifter — which does not fail on a future file, it
    quietly returns a snapshot with most of its sections empty.
    """
    snapshot = io.parse((FIXTURES_DIR / "test.ihpo").read_bytes())
    snapshot["format"] = io.SNAPSHOT_FORMAT + 1

    with pytest.raises(ValueError, match="newer than"):
        io.parse(json.dumps(snapshot).encode("utf-8"))
