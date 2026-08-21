"""Every number on the experiment page is in the .ihpo, or comes from it.

The file is the experiment. If the page can show something the file cannot say,
then exporting an experiment loses it and an experiment reopened elsewhere is a
different experiment — quietly, and only in the places nobody checked.

So rather than reading the file's schema against the page's markup, this runs an
experiment, exports it, imports the file into a fresh experiment, and asserts the
two pages are built from the same values. Anything the file failed to carry shows
up as a difference.

What is deliberately *not* compared: who may edit it, whether its model's
environment is ready, whether a run is in flight. Those are facts about this
instance and this reader, not about the experiment, and they have no business in
a file that gets sent to someone else.
"""

import json

from django.urls import reverse

from core import io
from ui.services import snapshot as adapter
from ui.views import _detail_context

from tests.conftest import DATASETS_DIR, export_ihpo

#: Everything the page draws a number, a label or a plot from.
DISPLAYED = [
    "summary",            # the identity strip: model, optimizer, metric, evaluation, seed
    "panels",             # best/selected configuration, and every game's warning and table
    "metric_plots",       # every per-metric figure's plot, for every view
    "static_plots",       # the figures that are the same for every metric
    "trial_rows",         # the trials table
    "run_summary",        # last run: trials, time in trials, overhead, why it stopped
    "hp_names",
    "incumbent_targets",  # what the run form's target opens at
    "incumbent_target",
    "metric_names",
    "run_default_metric",
    "trials_exhausted",
    "figure_options",
    "figure_views",
]

#: Keys that differ by construction — the copy is a different row in a different
#: database, and says so.
LOCAL_TO_THIS_COPY = ("pk", "identifier", "name")


def _ran_experiment(client):
    """An experiment with a real run behind it, so there is a run summary and a
    stored optimizer state to carry as well as trials."""
    from ui.services.run import create_run, execute_run

    client.post(reverse("ui:new_experiment"), {
        "name": "round-trip", "model_name": "Random Forest",
        "optimizer_name": "Random Search", "seed": "7",
        "evaluation_scheme": "kfold", "evaluation_value": "3",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
    })
    from ui.models import Experiment
    exp = Experiment.objects.get(name="round-trip")
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
    """The audit itself, over every displayed value at once."""
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
    """Partial dependence, the local explanation and the beeswarm are not in the
    file — they are computed on request from the model and the dataset, which is
    the one thing the file is allowed to assume you have. So they have to come
    out the same on the far side, and that is what makes leaving them out of the
    file correct rather than lossy."""
    exp = _ran_experiment(client)
    copy = _reimported(client, exp)

    def fetched(pk):
        base = f"/experiments/{pk}/"
        hp = next(iter(exp.result["configs"].values()))
        first = list(hp.keys())[0]
        return [
            client.get(f"{base}partial-dependence/?metric=accuracy&hp={first}").json(),
            client.get(f"{base}trial-ablation/?metric=accuracy&idx=0").json(),
            client.get(f"{base}local-effects/?metric=accuracy").json(),
        ]

    assert fetched(exp.pk) == fetched(copy.pk)


def test_what_is_left_out_is_about_the_reader_not_the_experiment(client, rf):
    """The keys this audit skips, named, so that skipping them stays a decision.

    Whether you may edit an experiment, whether its model's environment has been
    built here, whether a run is in flight — none of that belongs in a file that
    gets sent to someone else, and all of it is rebuilt from this instance.
    """
    exp = _ran_experiment(client)
    request = rf.get("/")
    request.user = None
    context = _detail_context(request, exp)

    about_the_instance = {"experiment", "may", "ownership", "env", "can_run",
                          "model_refusal", "run_error", "active_run", "result"}
    declarations = {"figures", "grid_figures", "column_figures",
                    "sidebar_figures", "selectable_figures", "selection_colors",
                    "autocompute", "explanation_games", "explanation_game_help",
                    "has_result", "supports_confidence", "trials_page_size"}

    assert set(context) - set(DISPLAYED) == about_the_instance | declarations
