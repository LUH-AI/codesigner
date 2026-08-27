"""Tunability says what was achievable. The ablation says what was achieved.

Tunability is the Shapley value of "the best you could reach by being allowed to
tune this, starting from the search space's default". That is the right question
*before* a search and the wrong one during it: by the time it reads 98%, the
optimizer has usually already taken most of that 98%, and the figure is spending
itself on a settled question while the hyperparameter with the remaining gains
reads under 1%.

The other half is already computed — the local explanation decomposes (this
trial − default) against the same baseline, in the same units. So the ratio is
well posed, and the difference is what is still on the table. These cover the
arithmetic, the two ways it can degenerate, and that asking for it changes
nothing until it is asked for.

**The measure is switched off** (`ui.views.TUNING_PROGRESS_ENABLED`), because the
tunability it subtracts from is not stable across processes. So every test here
but the last turns it back on: they are what says the code still works when the
flag is flipped, which is the point of leaving it in place rather than deleting
it. The last one is the only test of the shipped behaviour, and it asserts the
page offers none of this.
"""

import json

import pytest
from django.urls import reverse

from core import io
from core.optimizers import OptimizationResult, TrialResult
from ui.services import snapshot as adapter
from ui.views import SETTLED, _tuning_progress

from tests.conftest import FIXTURES_DIR


@pytest.fixture
def enabled(monkeypatch):
    """Turn "still to gain" back on for one test."""
    monkeypatch.setattr("ui.views.TUNING_PROGRESS_ENABLED", True)


def _result(shares, total, **kw):
    trial = TrialResult(trial=1, config={}, scores={"accuracy": 0.8}, score=0.8,
                        incumbent_score=0.8, incumbent_config={})
    return OptimizationResult(
        trials=[trial], primary_metric="accuracy", best_config={}, best_score=0.8,
        hyperparameter_importance={"accuracy": shares},
        hyperparameter_importance_warning={},
        hyperparameter_tunability_total={"accuracy": total}, **kw)


def test_the_share_and_the_scale_reconstruct_the_raw_value(enabled):
    """The whole reason the scale is stored. A share cannot be compared against
    the ablation, which is raw and signed; `share x total` can."""
    rows, _settled = _tuning_progress(
        _result({"a": 0.75, "b": 0.25}, 0.08), "accuracy", {"a": 0.03, "b": 0.01})
    by = {row["name"]: row for row in rows}

    assert by["a"]["achievable"] == 0.06      # 0.75 of 0.08
    assert by["b"]["achievable"] == 0.02
    assert by["a"]["banked"] == 0.03
    assert by["a"]["remaining"] == 0.03
    assert by["a"]["banked_share"] == 0.5


def test_it_is_ordered_by_what_is_left_not_by_size(enabled):
    """The actionable ordering, and deliberately not importance's. The point of
    the view is that the biggest hyperparameter is often the one with nothing
    left in it."""
    rows, _settled = _tuning_progress(
        _result({"big": 0.9, "small": 0.1}, 0.10), "accuracy",
        {"big": 0.089, "small": 0.001})

    assert [row["name"] for row in rows] == ["small", "big"]
    assert rows[0]["remaining"] > rows[1]["remaining"]


def test_an_axis_the_trial_made_worse_keeps_its_sign(enabled):
    """A negative banked value means this trial's setting is worse than the
    default — drift picked up while chasing whichever hyperparameter mattered.
    It is floored out of the share, because "you have overshot" is not somewhere
    to spend budget, but kept in the row, because it is the one finding here
    that is actionable on its own."""
    rows, _settled = _tuning_progress(
        _result({"a": 1.0}, 0.05), "accuracy", {"a": -0.02})

    assert rows[0]["banked"] == -0.02, "the sign survives"
    assert rows[0]["remaining"] == 0.07, "and it counts as headroom, not as loss"


def test_beating_the_estimated_ceiling_is_marked_rather_than_floored(enabled):
    """Measured, not hypothetical: on a real run the best-tuned hyperparameter
    banked more than tunability said was achievable.

    That is not an error. The ceiling comes from 10,000 *random* draws of the
    configuration space, and a real optimizer beats random search — that is what
    it is for. So it means the search found something the estimate could not
    see, which is the strongest evidence an axis is finished, and it is marked
    rather than quietly rounded to zero.
    """
    rows, _settled = _tuning_progress(
        _result({"a": 1.0}, 0.05), "accuracy", {"a": 0.09})

    assert rows[0]["remaining"] == 0.0, "no negative amount left"
    assert rows[0]["beyond"] is True


def test_falling_short_of_the_ceiling_is_not_marked(enabled):
    rows, _settled = _tuning_progress(
        _result({"a": 1.0}, 0.05), "accuracy", {"a": 0.02})

    assert rows[0]["beyond"] is False
    assert rows[0]["remaining"] == pytest.approx(0.03)


def test_a_trial_that_has_taken_everything_says_so(enabled):
    """Rather than drawing a pie of rounding — the same lesson as the all-zero
    guard in `_compute_hp_game`: an empty answer is a finding, and a normalised
    empty answer is a picture of noise."""
    rows, settled = _tuning_progress(
        _result({"a": 0.5, "b": 0.5}, 0.10), "accuracy", {"a": 0.0499, "b": 0.0499})

    assert settled
    assert sum(row["remaining"] for row in rows) < SETTLED * 0.10 + 1e-9


def test_a_result_with_no_scale_offers_nothing_rather_than_a_wrong_ratio(enabled):
    """Every .ihpo written before the scale was stored has shares and no units.
    A ratio against the wrong denominator would be worse than no ratio."""
    rows, settled = _tuning_progress(
        _result({"a": 1.0}, 0.0), "accuracy", {"a": 0.03})

    assert rows == [] and not settled


# ── through the endpoint ─────────────────────────────────────────────────────

def _experiment():
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))


def test_the_endpoint_carries_both_halves_and_the_renderings(client, enabled):
    """What is left cannot be precomputed — it depends on which trial is
    selected — so it rides back with the ablation that produced it, both
    renderings at once, and switching rendering afterwards costs no request."""
    exp = _experiment()
    body = json.loads(client.get(
        f"/experiments/{exp.pk}/trial-ablation/?metric=accuracy&idx=0").content)

    assert body["rows"], "one row per hyperparameter"
    assert set(body["rows"][0]) == {"name", "achievable", "banked", "remaining",
                                    "banked_share", "beyond"}
    assert set(body["split"]) == {"pie", "bar"}
    assert body["split"]["pie"]["data"][0]["type"] == "pie"
    assert body["split"]["bar"]["layout"]["barmode"] == "stack"
    if not body["settled"]:
        assert set(body["headroom"]) == {"pie", "bar"}
        assert body["headroom"]["pie"]["data"][0]["type"] == "pie"


def test_the_page_offers_it_without_computing_it(client, enabled):
    """The box is on the page; what fills it is not, until it is ticked. A
    reader who never asks pays nothing."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="importance-remaining"' in html
    assert "Still to gain" in html
    assert 'id="imp-settled"' in html
    # and the columns are there but empty — the page fills them from the fetch
    table = html.split('class="importance-table"', 1)[1].split("</table>", 1)[0]
    assert table.count('class="progress-col"') >= 3
    assert "<td class=\"progress-col\"></td>" in table


def test_the_stored_scale_is_what_the_shares_are_shares_of():
    """The field exists so the two halves can be compared at all, so it has to
    be the actual denominator the normalisation used — not a recomputation that
    might drift from it."""
    from core.models import RandomForestModel
    from core.optimizers import RandomOptimizer

    trials = [
        TrialResult(trial=i, config={"n_estimators": 10 + 40 * i, "max_depth": 2 + i,
                                     "min_samples_split": 0.01 + 0.05 * i,
                                     "max_features": 0.2 + 0.1 * i},
                    scores={"accuracy": 0.5 + 0.03 * i}, score=0.5 + 0.03 * i,
                    incumbent_score=0.5 + 0.03 * i, incumbent_config={})
        for i in range(6)]
    space = RandomForestModel().get_config_space(seed=0)
    shares, _warn, _inter, _moebius, total = RandomOptimizer()._compute_hp_game(
        space, trials, "accuracy", "tunability", 0)

    assert total > 0
    assert sum(shares.values()) == pytest.approx(1.0)
    # every share times the scale is a value in the metric's own units, and they
    # add up to the scale
    assert sum(v * total for v in shares.values()) == pytest.approx(total)


def test_it_reads_tunability_and_not_whichever_game_is_showing(enabled):
    """Both halves are tunability's. The achievable side is its own game, and
    the banked side is the ablation, which measures against the same baseline
    the max game measures from — so the subtraction is only well posed there.

    Under mistunability the achievable value is downside rather than gain, and
    nothing "achieves" sensitivity's variance. Asserted at the arithmetic: a
    result whose other games disagree with tunability still produces tunability's
    rows.
    """
    result = _result({"a": 1.0, "b": 0.0}, 0.10)
    result.hyperparameter_sensitivity = {"accuracy": {"a": 0.0, "b": 1.0}}
    result.hyperparameter_mistunability = {"accuracy": {"a": 0.0, "b": 1.0}}

    rows, _settled = _tuning_progress(result, "accuracy", {"a": 0.02, "b": 0.0})
    by = {row["name"]: row for row in rows}

    assert by["a"]["achievable"] == 0.10, "tunability's share, not the others'"
    assert by["b"]["achievable"] == 0.0


def test_the_control_can_be_taken_away_as_one_thing(client, enabled):
    """The checkbox and its explanation are one control, so the page hides them
    together when the game is not tunability — the same way the cube's axis
    pickers go when a projection is showing, rather than being left there
    offering an answer they cannot give."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    toggle = html.split("data-remaining-toggle", 1)[1].split("</span>\n        </p>", 1)[0]

    assert 'id="importance-remaining"' in toggle
    assert "importance_remaining_help" in toggle, "the explanation goes with it"
    assert 'currentGame() === "tunability"' in html
    assert "toggle.hidden = !tunability" in html
    assert "impRemaining.checked = false" in html, "and it is cleared on the way out"


def test_both_games_are_measured_from_the_same_zero():
    """The subtraction is only meaningful if the two halves share an origin.
    They do, and it is not an assumption: each game returns 0 for the empty
    coalition, so both are gains over the same baseline configuration rather
    than absolute scores."""
    import numpy as np
    import hypershap.games as hs_games
    import hypershap.task as hs_task
    from hypershap.utils import Aggregation, RandomConfigSpaceSearcher

    from core.models import RandomForestModel
    from core.optimizers import RandomOptimizer
    from core.optimizers.base import _HPO_SIMULATION_SAMPLES

    trials = [
        TrialResult(trial=i, config={"n_estimators": 10 + 40 * i, "max_depth": 2 + i,
                                     "min_samples_split": 0.01 + 0.05 * i,
                                     "max_features": 0.2 + 0.1 * i},
                    scores={"accuracy": 0.5 + 0.03 * i}, score=0.5 + 0.03 * i,
                    incumbent_score=0.5 + 0.03 * i, incumbent_config={})
        for i in range(6)]
    space = RandomForestModel().get_config_space(seed=0)
    explainer = RandomOptimizer()._build_explainer(space, trials, "accuracy", 0)
    surrogate = explainer.hs.explanation_task.surrogate_model
    if isinstance(surrogate, list):
        surrogate = surrogate[0]
    n = len(list(space.keys()))
    baseline = space.get_default_configuration()

    task = hs_task.TunabilityExplanationTask(
        config_space=space, surrogate_model=surrogate, baseline_config=baseline)
    tunability = hs_games.TunabilityGame(
        explanation_task=task,
        cs_searcher=RandomConfigSpaceSearcher(
            explanation_task=task, n_samples=_HPO_SIMULATION_SAMPLES,
            mode=Aggregation.MAX, seed=0))
    ablation = hs_games.AblationGame(explanation_task=hs_task.AblationExplanationTask(
        config_space=space, surrogate_model=surrogate, baseline_config=baseline,
        config_of_interest=baseline))

    assert tunability(np.zeros((1, n), bool))[0] == pytest.approx(0.0)
    assert ablation(np.zeros((1, n), bool))[0] == pytest.approx(0.0)


def test_the_box_does_not_come_back_ticked(client, enabled):
    """A browser restores a checkbox across a reload the way it restores a
    select. But this one is a *request* — it asks for a computation that has not
    been made — and a request is not a preference to remember. Restored, it comes
    back ticked with nothing behind it, claiming a view the figure is not
    showing, which is exactly the state it was reported in.
    """
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    box = html.split('id="importance-remaining"', 1)[1].split(">", 1)[0]
    assert "checked" not in box
    assert "if (startup && impRemaining) impRemaining.checked = false;" in html
    assert "syncRemainingControl(true)" in html, "and the page says so on load"


# ── switched off, which is what ships ────────────────────────────────────────

def test_none_of_it_reaches_the_page_while_the_measure_is_off(client):
    """The shipped behaviour, and the only test here that does not turn the flag
    back on.

    Off has to mean absent rather than empty. The server returning no rows would
    on its own leave the box and the table's three columns sitting there offering
    an answer that never arrives, which reads as a broken feature rather than as
    one that is not being offered — so the markup goes too, and both halves are
    checked here: nothing to tick, and no columns to fill.
    """
    from ui import views

    assert views.TUNING_PROGRESS_ENABLED is False, "this is what ships"

    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="importance-remaining"' not in html
    assert "Still to gain" not in html
    assert 'id="imp-settled"' not in html
    table = html.split('class="importance-table"', 1)[1].split("</table>", 1)[0]
    assert "progress-col" not in table

    # And the endpoint that fed them returns the local explanation alone.
    payload = json.loads(client.get(
        f"/experiments/{exp.pk}/trial-ablation/?metric=accuracy&idx=0").content)
    assert payload["rows"] == []
    assert payload["split"] == {} and payload["headroom"] == {}
    assert payload["settled"] is False
