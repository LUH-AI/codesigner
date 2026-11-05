"""An export keeps each game's Möbius terms and leaves out the importance,
interactions and total they imply (`games.without_derived`), and `io.parse`
rebuilds them (`games.with_derived`)."""

import copy
import json

import pytest

from core import io
from core.analytics import games
from core.optimizers.base import HP_GAME_FIELDS

from tests.conftest import FIXTURES_DIR

DERIVED = [name for game in HP_GAME_FIELDS for name in games._derived_fields(game)]


@pytest.fixture
def snapshot():
    return json.loads((FIXTURES_DIR / "analytics.ihpo").read_bytes())


def _close(want, got):
    if isinstance(want, dict):
        assert list(want) == list(got)
        for key in want:
            _close(want[key], got[key])
    else:
        assert got == pytest.approx(want, abs=1e-9)


def test_the_derived_values_come_back_from_the_moebius_terms(snapshot):
    """What: every game's importance, interactions and total, as HyperSHAP
    computed them, return from the Möbius terms alone. How: a fixture's result
    without them, parsed, compared value by value and in key order."""
    slim = {**snapshot, "result": games.without_derived(snapshot["result"])}
    assert not set(DERIVED) & set(slim["result"])

    back = io.parse(json.dumps(slim).encode())["result"]

    for name in DERIVED:
        _close(snapshot["result"][name], back[name])


def test_the_terms_and_warnings_stay(snapshot):
    """What: leaving out the derived values keeps the Möbius terms and every
    warning. How: the fixture's result before and after."""
    slim = games.without_derived(snapshot["result"])

    for fields in HP_GAME_FIELDS.values():
        assert slim[fields["moebius"]] == snapshot["result"][fields["moebius"]]
        assert slim[fields["warning"]] == snapshot["result"][fields["warning"]]


def test_a_game_without_terms_keeps_its_values(snapshot):
    """What: a game that fell back to forest importances or uniform weights has
    no Möbius terms, so its importance travels as it is. How: sensitivity for
    f1 given uniform weights and no terms, stripped and parsed."""
    result = copy.deepcopy(snapshot["result"])
    names = list(result["hyperparameter_sensitivity"]["f1"])
    uniform = {p: 1 / len(names) for p in names}
    result["hyperparameter_sensitivity"]["f1"] = uniform
    result["hyperparameter_sensitivity_interactions"]["f1"] = {}
    result["hyperparameter_sensitivity_moebius"]["f1"] = []

    slim = games.without_derived(result)
    back = io.parse(json.dumps({**snapshot, "result": slim}).encode())["result"]

    assert slim["hyperparameter_sensitivity"] == {"f1": uniform}
    assert back["hyperparameter_sensitivity"]["f1"] == uniform
    _close(snapshot["result"]["hyperparameter_sensitivity"]["accuracy"],
           back["hyperparameter_sensitivity"]["accuracy"])


def test_an_older_file_keeps_what_it_states(snapshot):
    """What: a file that still carries the derived values is read as written.
    How: the fixture parsed directly, compared exactly."""
    back = io.parse(json.dumps(snapshot).encode())["result"]

    for name in DERIVED:
        assert back[name] == snapshot["result"][name]
