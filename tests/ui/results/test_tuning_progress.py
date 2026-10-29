"""Tunability gives what was achievable and the trial ablation gives what was
achieved, both measured from the default. The difference of the two is the gain
still left. The flag `TUNING_PROGRESS_ENABLED` is off because tunability varies
across processes, and every test except the last turns it back on.
"""

import json

import pytest
from tests.conftest import FIXTURES_DIR

from django.urls import reverse

from core import analytics, io
from core.analytics import SETTLED, games, tuning_progress
from core.optimizers import OptimizationResult, TrialResult
from experiments.services import snapshot as adapter


@pytest.fixture
def enabled(monkeypatch):
    """`enabled` turns the flag on."""
    monkeypatch.setattr("ui.views.analytics.TUNING_PROGRESS_ENABLED", True)


def _result(shares, total, **kw):
    trial = TrialResult(trial=1, config={}, scores={"accuracy": 0.8}, score=0.8,
                        incumbent_score=0.8, incumbent_config={})
    return OptimizationResult(
        trials=[trial], primary_metric="accuracy", best_config={}, best_score=0.8,
        hyperparameter_importance={"accuracy": shares},
        hyperparameter_importance_warning={},
        hyperparameter_tunability_total={"accuracy": total}, **kw)


def test_the_share_and_the_scale_reconstruct_the_raw_value(enabled):
    """Raw `share x total` makes the share comparable with the signed ablation."""
    rows, _settled = tuning_progress(
        _result({"a": 0.75, "b": 0.25}, 0.08), "accuracy", {"a": 0.03, "b": 0.01})
    by = {row["name"]: row for row in rows}

    assert by["a"]["achievable"] == 0.06      # 0.75×0.08
    assert by["b"]["achievable"] == 0.02
    assert by["a"]["banked"] == 0.03
    assert by["a"]["remaining"] == 0.03
    assert by["a"]["banked_share"] == 0.5


def test_it_is_ordered_by_what_is_left_not_by_size(enabled):
    """The biggest hyperparameter often has nothing left, so the rows sort by
    remaining gain."""
    rows, _settled = tuning_progress(
        _result({"big": 0.9, "small": 0.1}, 0.10), "accuracy",
        {"big": 0.089, "small": 0.001})

    assert [row["name"] for row in rows] == ["small", "big"]
    assert rows[0]["remaining"] > rows[1]["remaining"]


def test_an_axis_the_trial_made_worse_keeps_its_sign(enabled):
    """A negative banked value means drift from the default, and the row keeps
    the negative value while the share drops it to zero."""
    rows, _settled = tuning_progress(
        _result({"a": 1.0}, 0.05), "accuracy", {"a": -0.02})

    assert rows[0]["banked"] == -0.02, "the sign survives"
    assert rows[0]["remaining"] == 0.07, "and it counts as headroom, not as loss"


def test_beating_the_estimated_ceiling_is_marked_rather_than_floored(enabled):
    """The ceiling comes from 10,000 random draws, so a real optimizer can beat
    it, and the row marks that instead of rounding to zero.
    """
    rows, _settled = tuning_progress(
        _result({"a": 1.0}, 0.05), "accuracy", {"a": 0.09})

    assert rows[0]["remaining"] == 0.0, "no negative amount left"
    assert rows[0]["beyond"] is True


def test_falling_short_of_the_ceiling_is_not_marked(enabled):
    rows, _settled = tuning_progress(
        _result({"a": 1.0}, 0.05), "accuracy", {"a": 0.02})

    assert rows[0]["beyond"] is False
    assert rows[0]["remaining"] == pytest.approx(0.03)


def test_a_trial_that_has_taken_everything_says_so(enabled):
    """`tuning_progress` flags `settled` when the remaining gain is under
    `SETTLED`, instead of charting rounding error."""
    rows, settled = tuning_progress(
        _result({"a": 0.5, "b": 0.5}, 0.10), "accuracy", {"a": 0.0499, "b": 0.0499})

    assert settled
    assert sum(row["remaining"] for row in rows) < SETTLED * 0.10 + 1e-9


def test_a_result_with_no_scale_offers_nothing_rather_than_a_wrong_ratio(enabled):
    """`tuning_progress` returns no rows for an older `.ihpo` without a scale,
    rather than a misscaled ratio."""
    rows, settled = tuning_progress(
        _result({"a": 1.0}, 0.0), "accuracy", {"a": 0.03})

    assert rows == [] and not settled


# trial-ablation endpoint

def _experiment():
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))


def test_the_endpoint_carries_both_halves_and_the_renderings(client, enabled):
    """The ablation response carries the remaining gain in both renderings,
    because that gain depends on the selected trial."""
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
    """The box renders, and nothing computes until it is ticked."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="importance-remaining"' in html
    assert "Still to gain" in html
    assert 'id="imp-settled"' in html
    # columns present but empty until the fetch
    table = html.split('class="importance-table"', 1)[1].split("</table>", 1)[0]
    assert table.count('class="progress-col"') >= 3
    assert "<td class=\"progress-col\"></td>" in table


def test_the_stored_scale_is_what_the_shares_are_shares_of():
    """The stored scale must be the normalisation's own denominator."""
    from core.models import RandomForestModel

    trials = [
        TrialResult(trial=i, config={"n_estimators": 10 + 40 * i, "max_depth": 2 + i,
                                     "min_samples_split": 0.01 + 0.05 * i,
                                     "max_features": 0.2 + 0.1 * i},
                    scores={"accuracy": 0.5 + 0.03 * i}, score=0.5 + 0.03 * i,
                    incumbent_score=0.5 + 0.03 * i, incumbent_config={})
        for i in range(6)]
    space = RandomForestModel().get_config_space(seed=0)
    shares, _warn, _inter, _moebius, total = analytics.compute_hp_game(
        space, trials, "accuracy", "tunability", 0)

    assert total > 0
    assert sum(shares.values()) == pytest.approx(1.0)
    # shares times scale sum back to the scale
    assert sum(v * total for v in shares.values()) == pytest.approx(total)


def test_it_reads_tunability_and_not_whichever_game_is_showing(enabled):
    """The rows use tunability, because only the max game shares the ablation's
    baseline.
    """
    result = _result({"a": 1.0, "b": 0.0}, 0.10)
    result.hyperparameter_sensitivity = {"accuracy": {"a": 0.0, "b": 1.0}}
    result.hyperparameter_mistunability = {"accuracy": {"a": 0.0, "b": 1.0}}

    rows, _settled = tuning_progress(result, "accuracy", {"a": 0.02, "b": 0.0})
    by = {row["name"]: row for row in rows}

    assert by["a"]["achievable"] == 0.10, "tunability's share, not the others'"
    assert by["b"]["achievable"] == 0.0


def test_the_control_can_be_taken_away_as_one_thing(client, enabled):
    """The checkbox and its help hide together, and uncheck, when another game
    is showing."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    toggle = html.split("data-remaining-toggle", 1)[1].split("</span>\n        </p>", 1)[0]

    assert 'id="importance-remaining"' in toggle
    assert "importance_remaining_help" in toggle, "the explanation goes with it"
    assert 'currentGame() === "tunability"' in html
    assert "toggle.hidden = !tunability" in html
    assert "impRemaining.checked = false" in html, "and it is cleared on the way out"


def test_both_games_are_measured_from_the_same_zero():
    """Each game returns 0 for the empty coalition, so both halves are gains
    over one baseline."""
    import numpy as np
    import hypershap.games as hs_games
    import hypershap.task as hs_task
    from hypershap.utils import Aggregation, RandomConfigSpaceSearcher

    from core.models import RandomForestModel

    trials = [
        TrialResult(trial=i, config={"n_estimators": 10 + 40 * i, "max_depth": 2 + i,
                                     "min_samples_split": 0.01 + 0.05 * i,
                                     "max_features": 0.2 + 0.1 * i},
                    scores={"accuracy": 0.5 + 0.03 * i}, score=0.5 + 0.03 * i,
                    incumbent_score=0.5 + 0.03 * i, incumbent_config={})
        for i in range(6)]
    space = RandomForestModel().get_config_space(seed=0)
    explainer = games.build_explainer(space, trials, "accuracy", 0)
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
            explanation_task=task, n_samples=10_000,
            mode=Aggregation.MAX, seed=0))
    ablation = hs_games.AblationGame(explanation_task=hs_task.AblationExplanationTask(
        config_space=space, surrogate_model=surrogate, baseline_config=baseline,
        config_of_interest=baseline))

    assert tunability(np.zeros((1, n), bool))[0] == pytest.approx(0.0)
    assert ablation(np.zeros((1, n), bool))[0] == pytest.approx(0.0)


def test_the_box_does_not_come_back_ticked(client, enabled):
    """A reload must restore the box unticked, because it is a request for a
    computation.
    """
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    box = html.split('id="importance-remaining"', 1)[1].split(">", 1)[0]
    assert "checked" not in box
    assert "if (startup && impRemaining) impRemaining.checked = false;" in html
    assert "syncRemainingControl(true)" in html, "and the page says so on load"


# switched off, as shipped

def test_none_of_it_reaches_the_page_while_the_measure_is_off(client):
    """With the flag off the markup goes too, since an empty box and columns
    would look broken.
    """
    from ui.views import analytics

    assert analytics.TUNING_PROGRESS_ENABLED is False, "this is what ships"

    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="importance-remaining"' not in html
    assert "Still to gain" not in html
    assert 'id="imp-settled"' not in html
    table = html.split('class="importance-table"', 1)[1].split("</table>", 1)[0]
    assert "progress-col" not in table

    # the endpoint returns the local explanation alone
    payload = json.loads(client.get(
        f"/experiments/{exp.pk}/trial-ablation/?metric=accuracy&idx=0").content)
    assert payload["rows"] == []
    assert payload["split"] == {} and payload["headroom"] == {}
    assert payload["settled"] is False
