"""The audit runs an experiment, imports its exported file into a fresh one, and
compares the values both pages are built from. Facts about the local instance
and its reader stay out of the comparison and the file.
"""

import json

import pytest

from django.urls import reverse

from core import io
from experiments.services import snapshot as adapter
from ui.views.dashboard import _detail_context

from tests.conftest import DATASETS_DIR, export_ihpo, post_new_experiment

#: Every value the page draws from.
DISPLAYED = [
    "summary",            # identity strip
    "panels",             # configurations and game tables
    "metric_plots",       # per-metric plots
    "static_plots",       # metric-independent figures
    "trial_rows",
    "run_summary",        # last run's trials, time and stop reason
    "hp_names",
    "incumbent_targets",  # run form's opening target
    "incumbent_target",
    "metric_names",
    "run_default_metric",
    "trials_exhausted",
    "figure_options",
    "figure_views",
]

#: The copy is a different row and says so.
LOCAL_TO_THIS_COPY = ("pk", "identifier", "name", "given_name")


def _ran_experiment(client):
    """`_ran_experiment` runs three trials, leaving a run summary and optimizer
    state to carry."""
    from experiments.services.run import create_run, execute_run

    post_new_experiment(client, {
        "name": "round-trip", "model_name": "Random Forest",
        "optimizer_name": "Random Search", "seed": "7",
        "evaluation_scheme": "kfold", "evaluation_value": "3",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
    })
    from experiments.models import Experiment
    exp = Experiment.objects.get(data__name="round-trip")
    execute_run(create_run(exp, {"max_trials": 3}, "accuracy").id)
    exp.refresh_from_db()
    return exp


def _reimported(client, exp):
    body = json.loads(
        export_ihpo(client, exp.pk).content)
    body["name"] = "reopened"
    return adapter.experiment_from_snapshot(io.parse(json.dumps(body).encode()))


def _normalised(context, key):
    value = context.get(key)
    if key == "summary":
        value = {k: v for k, v in value.items() if k not in LOCAL_TO_THIS_COPY}
    return json.dumps(value, default=str, sort_keys=True)


def test_the_page_is_built_from_the_same_values_after_a_round_trip(client, rf):
    exp = _ran_experiment(client)
    copy = _reimported(client, exp)
    request = rf.get("/")
    request.user = None

    original = _detail_context(request, exp)
    reopened = _detail_context(request, copy)

    assert original["run_summary"], "the run is in the summary, so comparing it means something"
    for key in DISPLAYED:
        assert _normalised(original, key) == _normalised(reopened, key), key


def test_the_deferred_figures_come_back_too(client, rf):
    """Deferred figures recompute from the model and dataset, the one input the
    file assumes, so they must match after the trip."""
    exp = _ran_experiment(client)
    copy = _reimported(client, exp)

    def fetched(pk):
        base = f"/experiments/{pk}/"
        assert exp.data.result is not None
        hp = next(iter(exp.data.result["configs"].values()))
        first = list(hp.keys())[0]
        return [
            client.get(f"{base}partial-dependence/?metric=accuracy&hp={first}").json(),
            client.get(f"{base}trial-ablation/?metric=accuracy&idx=0").json(),
            client.get(f"{base}local-effects/?metric=accuracy").json(),
        ]

    assert fetched(exp.pk) == fetched(copy.pk)


def test_what_is_left_out_is_about_the_reader_not_the_experiment(client, rf):
    """The audit names the keys it skips, all rebuilt from the local instance
    and kept out of shared files.
    """
    exp = _ran_experiment(client)
    request = rf.get("/")
    request.user = None
    context = _detail_context(request, exp)

    about_the_instance = {"experiment", "may", "ownership", "env", "can_run",
                          "model_refusal", "run_error", "active_run", "result",
                          # The machine's own patience, meaningless elsewhere.
                          "default_trial_timeout",
                          # From trial durations on this machine.
                          "poll_seconds",
                          # An importing reader lacks the model file.
                          "model_export_unavailable",
                          # Kept models are local files that an import lacks.
                          "kept_trial_indices", "trial_models_reason",
                          "export_default_kept"}
    declarations = {"figures", "sidebar_figures", "selectable_figures",
                    "selection_colors",
                    # This reader's page arrangement.
                    "figure_tabs", "figure_layout",
                    # Labels for this browser's kept answers.
                    "computed_version",
                    # Two constants.
                    "beta_min", "beta_max",
                    "autocompute", "explanation_games", "explanation_game_help",
                    "has_result", "supports_confidence", "can_requery", "trials_page_size",
                    "tuning_progress",
                    # The prior controls show the optimizer's capability and a
                    # setting with its bounds.
                    "judges_priors", "prior_top_k", "prior_top_k_bounds",
                    # A built-in constant.
                    "default_trial_factor",
                    "live_figures",
                    "game_figures",
                    # This instance's ability to fill the result's gaps.
                    "analytics_absent",
                    # This render's trial count, echoed by the status poll.
                    "trial_count"}

    assert set(context) - set(DISPLAYED) == about_the_instance | declarations


def test_a_stated_prior_survives_a_round_trip(client):
    """The file must carry a stated prior, since the prior steers the next run.
    The file omits the knots, and `core.priors` rebuilds the density from `kind`
    and `params`."""
    exp = _ran_experiment(client)
    assert exp.data.result is not None
    hp = next(iter(exp.data.result["configs"].values()))
    name = list(hp.keys())[0]

    exp.data.priors = {name: {"kind": "normal", "params": {"mu": 0.6, "sigma": 0.2},
                         "decay": {"shape": "quadratic", "beta": 7.5},
                         "at_trial": 3}}
    exp.data.save(update_fields=["priors"])

    copy = _reimported(client, exp)

    assert copy.data.priors == exp.data.priors
    assert "knots" not in copy.data.priors[name]

    # The rebuilt density has its peak at the stated mean.
    from core.priors import density_from

    stated = copy.data.priors[name]
    grid = [i / 256 for i in range(257)]
    values = density_from(stated["kind"], stated["params"], grid)
    assert values is not None

    assert max(values, key=lambda point: point[1])[0] == pytest.approx(0.6, abs=1 / 256)
