import copy
import json
import logging
import statistics
import tempfile
from importlib.metadata import version as dist_version
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_not_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.http import Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from core import io, provenance, smac_import
from core.metrics import metric_for
from core.modelhost import deadline
from core.optimizers.base import FAILURE_CRITERIA

from .figures import (
    FIGURES, FIGURES_BY_KEY, HP_GAME_FIELDS, HP_GAME_HELP, HP_GAME_LABELS,
    MARKER_COLOR, NEGATIVE_COLOR, SELECTION_COLOR, UNCERTAINTY_SCALE,
    autocompute_key,
    deferred_computations,
    acquisition_slice_plot,
    hyperparameter_ablation_plot, hyperparameter_importance_plot,
    hyperparameter_progress_plot, local_effects_plot, partial_dependence_plot,
)
from .forms import (
    EVALUATION_SCHEMES, DefaultExperimentSettingsForm, ExperimentSettingsForm,
    NewExperimentForm)
from .models import Experiment, GlobalSettings
from . import permissions
from .permissions import DELETE, EDIT, EXPORT, RUN, VIEW, experiment_view
from .registry import METRICS, MODELS, OPTIMIZERS
from django.utils import timezone

from .services import run as run_service
from .services import modelenv
from .services import snapshot as snapshot_adapter
from .services.run import resolve_seed
from .services.run_logic import decide_run, resolve_metric_change
from .optimizer_labels import describe_all, grouped
from .services.settings import (
    SETTING_BOUNDS, SETTING_DEFAULTS, global_defaults, resolve_settings)
from .validators import dataset_upload_error, model_upload_error, oversized

_ACTIVE = ["pending", "running"]


#: Reaches the console handler configured in settings.LOGGING. Used only where
#: something went wrong that the reader cannot be told about, because the thing
#: they asked for succeeded anyway — see `_abandon_cluster_job`.
logger = logging.getLogger(__name__)


def _model_available(exp):
    """Whether *exp*'s model can be built to run it.

    Registry models are always available. A custom model needs its .py present,
    the feature enabled, and either a prepared environment or permission to be
    imported into this process. Reads `env_status` — a column — so rendering a
    page never shells out.
    """
    if exp.model_name in MODELS:
        return True
    if not (exp.model_file and settings.ALLOW_CUSTOM_MODELS):
        return False
    if exp.env_status == Experiment.ENV_READY:
        return True
    return not exp.env_pending and not settings.REQUIRE_LOGIN


def _owner(request):
    """The account creating an experiment, or None with no accounts.

    `AnonymousUser` is not a row, so it cannot be stored; None is the column's
    way of saying nobody's.
    """
    user = request.user
    return user if user.is_authenticated else None


def _ownership(request, exp):
    """Who this experiment belongs to, or None when the instance has no
    accounts and there is nothing to say."""
    if not settings.REQUIRE_LOGIN:
        return None
    return {"owner": exp.owner, "shared": exp.shared,
            "mine": exp.owner_id == request.user.pk}


# How each stopping criterion is read off the Run form: its parser and the range
# it is held to. Every criterion is here, including the trial cap — the form has
# no required field, only a requirement that at least one be filled in.
_STOPPING_FIELDS = {
    "max_trials": (int, 1, 100_000),
    # None, None: a target is compared against a score, so its range is the
    # metric's and not a constant here — 0 to 1 is accuracy's range, not every
    # metric's. Resolved per submission; see `_posted_stopping`.
    "target_score": (float, None, None),
    "max_seconds": (float, 1.0, None),
    "max_trial_seconds": (float, 1.0, None),
    "no_improvement_trials": (int, 1, None),
    "incumbent_confidence": (float, 0.0, 1.0),
    "max_failures": (int, 1, None),
    "max_consecutive_failures": (int, 1, None),
}


#: What ended a run, for the summary line. Every criterion is named: with no
#: privileged default, "it stopped" no longer implies the trial count.
STOPPED_BY_LABELS = {
    "max_trials": _("all the requested trials ran"),
    "target_score": _("performance surpassed the target"),
    "max_seconds": _("the time limit was reached"),
    "max_trial_seconds": _("the compute budget was used up"),
    "no_improvement_trials": _("the score had stopped improving"),
    "incumbent_confidence": _("the search was confident nothing better remained"),
    "cancelled": _("it was interrupted"),
    "all_failing": _("too many trials were failing"),
}


def surpass_target(score: float, places: int = 4) -> str:
    """The incumbent's score, as a target for the next run to beat.

    Rounded for the field it goes in, but never *below* the score itself: the
    criterion is a strict improvement, so a target a hair under what is already
    in hand would be met before the run started and end it after one trial.
    """
    shown = round(score, places)
    if shown < score:
        shown = round(shown + 10 ** -places, places)
    return f"{shown:g}"


def _posted_stopping(request, metric_name: str = "") -> dict:
    """The stopping criteria as submitted, ignoring the blanks.

    A criterion left empty is absent rather than zero: zero would mean "stop
    immediately", which is never what an empty box asks for. Unparseable input
    is dropped the same way — and if that leaves nothing at all, the caller
    refuses the run rather than starting one that cannot end.

    *metric_name* is the metric the run will optimize, which is what bounds
    `target_score`: a target is compared against a score, so what counts as out
    of range is whatever that metric's range is, and an unbounded metric bounds
    it not at all.
    """
    stopping = {}
    for key, (parse, low, high) in _STOPPING_FIELDS.items():
        raw = (request.POST.get(key) or "").strip()
        if not raw:
            continue
        try:
            value = parse(raw)
        except ValueError:
            continue
        if key == "target_score":
            low, high = metric_for(metric_name).bounds
            # Clamped at both ends rather than dropped below the low one.
            # Dropping made an out-of-range target *absent*, so a run asking for
            # nothing else was then refused for having no criterion at all —
            # which names a different problem from the one the reader created.
            if low is not None:
                value = max(value, low)
        elif value < low:
            continue
        stopping[key] = min(value, high) if high is not None else value
    return stopping


def _posted_trial_timeout(request) -> dict:
    """How long one call to the model may take, as submitted.

    Read separately from `_posted_stopping` because it is not a stopping
    criterion: nothing here ends a run, it ends a trial, and the run carries on
    with that trial recorded as failed. Storing it among the criteria would also
    mean the collector silently dropped it, since it filters to
    `STOPPING_CRITERIA`.

    Unlike a criterion, this one always has a value — a blank or unreadable box
    falls back to the deployment's `MODEL_TRIAL_TIMEOUT` rather than being
    absent, because "no deadline at all" is a thing to ask for deliberately (0)
    and not a thing to arrive at by mistyping.
    """
    mode = request.POST.get("trial_timeout_mode")
    if mode not in deadline.MODES:
        mode = deadline.MODE_FIXED
    return {
        "mode": mode,
        "seconds": _posted_number(request, "trial_timeout_seconds",
                                  settings.MODEL_TRIAL_TIMEOUT, low=0.0),
        "factor": _posted_number(request, "trial_timeout_factor",
                                 deadline.DEFAULT_FACTOR, low=1.0),
    }


def _posted_number(request, key: str, default: float, *, low: float) -> float:
    raw = (request.POST.get(key) or "").strip()
    try:
        return max(low, float(raw.replace(",", ".")))
    except ValueError:
        return float(default)


#: Optimizer settings are posted with this prefix, so a parameter can be named
#: whatever the optimizer calls it without colliding with `n_trials` or a
#: stopping criterion on the same form.
_OPTIMIZER_PREFIX = "opt_"


def _posted_optimizer_params(request, optimizer) -> dict:
    """The optimizer's settings as submitted, read through its own schema.

    Types and bounds come from the declaration rather than a table here, which
    is the difference between this and `_posted_stopping`: an optimizer that
    declares a new setting gets it parsed without this function changing.

    A blank numeric field means "leave it to the strategy" and is stored as
    None, not dropped — the difference matters, because a stored value that is
    simply absent would be filled by the constructor default on the next read
    and the two would disagree.
    """
    params = {}
    for param in optimizer.params_schema:
        raw = request.POST.get(f"{_OPTIMIZER_PREFIX}{param.name}")
        if param.type == "bool":
            params[param.name] = bool(raw)
            continue
        raw = (raw or "").strip()
        if not raw:
            params[param.name] = None if param.default is None else param.default
            continue
        if param.type == "select":
            params[param.name] = raw if raw in param.choices else param.default
            continue
        try:
            value = (int if param.type == "int" else float)(raw)
        except ValueError:
            params[param.name] = param.default
            continue
        if param.min is not None:
            value = max(param.min, value)
        if param.max is not None:
            value = min(param.max, value)
        params[param.name] = value
    return params


def _optimizer_param_context(optimizer, stored=None) -> dict:
    """What the settings partial needs, for whichever optimizer this is.

    Settings that belong to a group are handed over separately, keyed by name,
    so the group's own template can place each one rather than take them in
    declaration order.
    """
    layout, advanced = grouped(describe_all(optimizer, stored))
    return {
        "optimizer_layout": layout,
        "advanced_params": advanced,
        "has_advanced_params": bool(advanced),
    }


def _optimizer_panels(request, selected: str) -> list:
    """One settings panel per optimizer that has settings.

    All of them, not just the chosen one: the page hides the rest and disables
    their inputs, so the dropdown swaps panels without a request and only the
    chosen optimizer's values are submitted. An optimizer with an empty schema
    contributes nothing rather than an empty box.

    On a re-render after a validation error each panel is filled from what was
    posted. For the hidden ones that is nothing, which the parser reads as their
    own defaults — the same answer as never having touched them.
    """
    panels = []
    for key, optimizer in OPTIMIZERS.items():
        if not optimizer.params_schema:
            continue
        posted = _posted_optimizer_params(request, optimizer) if request.method == "POST" else None
        panels.append({"key": key, "selected": key == selected,
                       # Inert as rendered, not merely hidden: with no
                       # JavaScript the dropdown cannot swap panels anyway, and
                       # an enabled one would still post its values.
                       "disabled": key != selected,
                       **_optimizer_param_context(optimizer, posted)})
    return panels


def _selected_optimizer(request) -> str:
    """The optimizer the create form is showing — posted, or the first offered."""
    posted = request.POST.get("optimizer_name")
    return posted if posted in OPTIMIZERS else next(iter(OPTIMIZERS))


def _optimizer_for(name):
    """The registry optimizer an experiment names, or None. Aliases included, so
    an experiment stored under an optimizer's older name still resolves."""
    return OPTIMIZERS.get(name) or next(
        (o for o in OPTIMIZERS.values()
         if o.name == name or name in getattr(o, "aliases", ())), None)


def metric_label(current_metric, original_metric):
    """The label shown for an experiment's metric.

    "~" when no metric has been committed by a run yet, "Inconsistent" when
    the current metric no longer matches the one the results were produced
    with, otherwise the metric name itself.
    """
    if original_metric is None:
        return "~"
    if current_metric != original_metric:
        return _("Inconsistent")
    return current_metric


def home(request):
    return render(request, "ui/home.html")


#: Items per page on the full experiment list — the sidebar only ever shows
#: the most recent `context_processors.SIDEBAR_LIMIT`; this is where the rest
#: of an instance's experiments are reachable.
EXPERIMENT_LIST_PAGE_SIZE = 50


def experiment_list(request):
    """Every experiment this request may see, paginated.

    Through the same policy-scoped queryset the sidebar uses
    (`permissions.visible_experiments`), so this never shows an experiment the
    sidebar wouldn't and a page would then refuse to open.
    """
    paginator = Paginator(permissions.visible_experiments(request), EXPERIMENT_LIST_PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "ui/experiment_list.html", {"page": page})


@login_not_required
def healthz(request):
    """Liveness probe for the container healthcheck — no DB, no template.

    Exempt from the login wall: the container runtime has no session, and a
    healthcheck that 302s to a login page reports a healthy instance as down.
    """
    return HttpResponse("ok", content_type="text/plain")


def _dataset_path_from(form, tmp_paths):
    """Resolve the chosen dataset to a filesystem path (temp file for uploads)."""
    demo = form.cleaned_data.get("demo_dataset")
    if demo:
        return demo
    upload = form.cleaned_data["dataset_file"]
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    for chunk in upload.chunks():
        tmp.write(chunk)
    tmp.close()
    tmp_paths.append(tmp.name)
    return tmp.name


def _evaluation_schemes():
    """`EVALUATION_SCHEMES` as the create form's script needs it.

    Shipped rather than duplicated in the template: the label, bounds and
    default the number takes are the form's to decide, and a copy in JavaScript
    is a second source of truth that drifts silently. Lazy translation strings
    are resolved here, since `json_script` cannot serialize them.
    """
    return {name: {**spec, "label": str(spec["label"]), "help": str(spec["help"])}
            for name, spec in EVALUATION_SCHEMES.items()}


def new_experiment(request):
    """Set up (but do not run) an experiment, then go to its detail page.

    Persists the experiment with no result and no committed metric; the dataset
    is copied into MEDIA. Running is a separate action on the detail page.
    """
    may_upload = permissions.policy().may_upload_models(request)
    if request.method != "POST":
        return render(request, "ui/new_experiment.html", {
            "form": NewExperimentForm(may_upload_models=may_upload),
            "optimizer_panels": _optimizer_panels(request, _selected_optimizer(request)),
            "evaluation_schemes": _evaluation_schemes(),
        })

    form = NewExperimentForm(request.POST, request.FILES,
                             may_upload_models=may_upload)
    if not form.is_valid():
        return render(request, "ui/new_experiment.html", {
            "form": form,
            "optimizer_panels": _optimizer_panels(request, _selected_optimizer(request)),
            "evaluation_schemes": _evaluation_schemes(),
        })

    cleaned = form.cleaned_data
    seed = resolve_seed(cleaned["seed"])
    tmp_paths = []
    try:
        dataset_path = _dataset_path_from(form, tmp_paths)
        chosen = OPTIMIZERS[cleaned["optimizer_name"]]
        optimizer = type(chosen)(**type(chosen).known_params(
            _posted_optimizer_params(request, chosen)))
        # A mounted model is adopted from its server-side path (unless an upload
        # was given, which takes precedence); the adapter copies it into MEDIA.
        mounted = cleaned.get("mounted_model") or ""
        model_path = mounted if (mounted and not cleaned.get("model_file")) else ""
        folds = int(cleaned.get("cv_folds") or 0)
        test_size = cleaned.get("test_size")
        snapshot = {
            "format": io.SNAPSHOT_FORMAT,
            "version": dist_version("codesigner"),
            "name": cleaned["name"],
            "seed": seed,
            "dataset": {"filename": Path(dataset_path).name, "path": dataset_path},
            "model": {"kind": "file" if model_path else "registry",
                      "name": cleaned["model_name"], "path": model_path},
            "evaluation": provenance.evaluation(folds, test_size=test_size),
            "metrics": {"names": list(METRICS), "current": None, "original": None},
            "optimizer": {"name": optimizer.name, "params": optimizer.get_params()},
            "result": None,
        }
        # This snapshot's paths were built right here: a validated demo/mounted
        # choice, or a temp file written above — so they are ours to adopt.
        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, model_file=cleaned.get("model_file"), adopt_paths=True,
            owner=_owner(request),
        )
        # A custom model needs an environment before it can run. Resolving it
        # takes minutes, so it happens in the background and the detail page
        # reports progress.
        modelenv.start_preparation(exp, cleaned.get("model_source"))
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)

    return redirect("ui:experiment_detail", pk=exp.pk)


@experiment_view(VIEW)
def experiment_detail(request, exp):
    """Show a saved experiment: its config, results/figures, run state, Run form."""
    return render(request, "ui/experiment_detail.html", _detail_context(request, exp))


@experiment_view(VIEW)
def trial_panel(request, exp):
    """Render the selected-config panel for one (metric, trial index).

    Backs click-to-select on the performance figure (curve 0 only, point index
    into result.trials) — the browser fetches this fragment and swaps it into
    the panel for the metric currently being viewed.
    """
    metric = request.GET.get("metric", "")
    idx_raw = request.GET.get("idx", "")

    if metric not in exp.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    result = _rebuild_result(exp)
    idx = int(idx_raw)
    if result is None or not (0 <= idx < len(result.trials)):
        return HttpResponseBadRequest("invalid trial index")

    return render(request, "ui/_selected_config_inner.html",
                  {"sel": _selected_panel_data(result, metric, idx),
                   # The traceback link needs to know which experiment it is
                   # asking about; on the first render the page context has it,
                   # and this partial is also served on its own.
                   "experiment": exp})


@experiment_view(VIEW)
def trial_traceback(request, exp):
    """The stored traceback for one failed trial, as plain text.

    A file to open rather than a panel to read: a traceback is a dozen lines of
    paths and frames, which is the wrong shape for a sidebar and the right shape
    for something you scroll, search and paste elsewhere. Served from the stored
    result — the traceback was recorded when the trial failed, so this reads and
    never computes.

    404 for a trial that did not fail and for one that failed without a
    traceback: a timeout has none worth keeping, an old result predates them
    being recorded, and an export can have been asked to leave them out.
    """
    idx_raw = request.GET.get("idx", "")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    result = _rebuild_result(exp)
    idx = int(idx_raw)
    if result is None or not (0 <= idx < len(result.trials)):
        return HttpResponseBadRequest("invalid trial index")

    trial = result.trials[idx]
    if not trial.traceback:
        raise Http404("no traceback for this trial")

    text = f"Trial {trial.trial} — {trial.failure}\n\n{trial.traceback}"
    response = HttpResponse(text, content_type="text/plain; charset=utf-8")
    # Inline, so a click opens it rather than downloading it. A reader wanting
    # the file can still save it from there.
    response["Content-Disposition"] = (
        f'inline; filename="{exp.identifier}-trial-{trial.trial}-traceback.txt"')
    return response


@experiment_view(VIEW)
def trial_ablation(request, exp):
    """The local-explanation (HyperSHAP ablation) figure for one (metric,
    trial index) — fetched lazily, only when the importance figure's game
    selector is on "Local" and the selection changes, since (unlike the
    other three games) this depends on which trial is selected rather than
    just the experiment's stored result, so it cannot be precomputed for
    every trial the way the rest of the page is.
    """
    metric = request.GET.get("metric", "")
    idx_raw = request.GET.get("idx", "")

    if metric not in exp.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    built = _rebuild_experiment(exp)
    idx = int(idx_raw)
    if (built is None or built["result"] is None
            or not (0 <= idx < len(built["result"].trials))):
        return HttpResponseBadRequest("invalid trial index")

    figure, warning, ablation = _local_ablation_data(built, metric, idx)
    rows, settled = _tuning_progress(built["result"], metric, ablation)
    # The headroom renderings cannot be precomputed — they need this trial's
    # ablation — so they come back with it, all three at once, and switching
    # rendering afterwards costs no second request.
    remaining = {row["name"]: row["remaining"] for row in rows}
    return JsonResponse({
        "figure": figure,
        "warning": warning,
        "rows": rows,
        "settled": settled,
        # Two ways of showing the same pair. `split` sizes each hyperparameter as
        # the plain figure does and divides it into what has been banked and what
        # has not — the honest version of the default view. `headroom` sizes them
        # by what is left alone, which reorders the figure and is what the "still
        # to gain" box asks for.
        "split": {
            rendering: _plot_json(hyperparameter_progress_plot(rows, rendering))
            for rendering in ("pie", "bar")
        } if rows else {},
        "headroom": {
            rendering: _plot_json(hyperparameter_importance_plot(
                remaining, rendering, title=str(_("Still to gain"))))
            for rendering in ("pie", "bar")
        } if rows and not settled else {},
    })


def _plot_json(plot):
    return json.loads(plot.to_json()) if plot is not None else None


@experiment_view(VIEW)
def partial_dependence(request, exp):
    """The partial-dependence (PDP/ICE) figure for one (metric,
    hyperparameter) — fetched lazily on every load and every hyperparameter
    switch, since which hyperparameter is being explained isn't one of a
    small precomputable set the rest of the page ships upfront (unlike the
    three global HyperSHAP games), and fitting a surrogate + predicting
    across a grid for every hyperparameter on every page load, for a figure
    that only ever shows one at a time, would be pure waste.
    """
    metric = request.GET.get("metric", "")
    hp_name = request.GET.get("hp", "")

    if metric not in exp.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = _rebuild_experiment(exp)
    if (built is None or built["result"] is None or not built["result"].trials
            or hp_name not in built["result"].trials[0].config):
        return HttpResponseBadRequest("invalid hyperparameter")

    figure, warning = _partial_dependence_data(
        built, metric, hp_name, resolve_settings(exp)["ice_max_curves"])
    return JsonResponse({"figure": figure, "warning": warning})


@experiment_view(VIEW)
def acquisition_slice(request, exp):
    """The acquisition-and-beliefs figure for one (metric, hyperparameter).

    Fetched per hyperparameter rather than shipped, for `partial_dependence`'s
    reason exactly — it fits a surrogate and predicts across a grid, and only
    one hyperparameter is ever on screen.

    The belief and acquisition traces come back empty; the browser fills them
    (`ui/static/ui/acquisition.js`). Everything it needs to do that rides in the
    figure's own `layout.meta`, so the response is the figure plus a warning,
    the same shape `partial_dependence` returns.
    """
    metric = request.GET.get("metric", "")
    hp_name = request.GET.get("hp", "")

    if metric not in exp.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = _rebuild_experiment(exp)
    result = built["result"] if built else None
    if (result is None or not result.trials
            or hp_name not in result.trials[0].config):
        return HttpResponseBadRequest("invalid hyperparameter")

    config_space = _config_space_for(built)
    if config_space is None:
        return JsonResponse({"figure": None, "warning": _(
            "The model this experiment used is not available here, so its "
            "search space cannot be rebuilt.")})

    grid, positions, mu, sigma, eta, warning = built["optimizer"].compute_incumbent_slice(
        config_space, result.trials, metric, hp_name, seed=built["seed"])
    if not positions:
        return JsonResponse({"figure": None, "warning": warning})

    figure = acquisition_slice_plot(
        hp_name, positions, grid, mu, sigma, metric,
        eta=eta, higher_is_better=metric_for(metric).higher_is_better,
        kind="categorical" if hasattr(config_space[hp_name], "choices") else "continuous",
    )
    return JsonResponse({"figure": _plot_json(figure), "warning": warning})


@experiment_view(VIEW)
def surrogate_uncertainty(request, exp):
    """The uncertainty field under the configuration cube's axes view, for one
    (metric, x, y) — how much the surrogate's own trees disagree across the
    plane those two hyperparameters span.

    **The axes view only, and only at two axes.** The two projections have no
    way back: `core.projection.project` returns coordinates and discards the
    fitted PCA/PLS, so there is no inverse to map a grid in component space
    back to configurations — and even with one, the points it landed on need not
    be valid configurations. Three axes would be a volume through a point cloud,
    which is a worse picture than none. One is a strip, and a field under a strip
    has no second dimension to vary in.

    Fetched rather than shipped, like partial dependence and for the same
    reason: it is per *pair*, and a model with six hyperparameters has fifteen
    pairs of which the reader is looking at one.
    """
    metric = request.GET.get("metric", "")
    x_hp = request.GET.get("x", "")
    y_hp = request.GET.get("y", "")
    if metric not in exp.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = _rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")

    config_space = _config_space_for(built)
    if config_space is None:
        return JsonResponse({
            "x": [], "y": [], "z": [],
            "warning": _("The model this experiment used is not available here, "
                         "so its search space cannot be rebuilt."),
        })

    x_grid, y_grid, z, warning = built["optimizer"].compute_surrogate_uncertainty(
        config_space, result.trials, metric, x_hp, y_hp, seed=built["seed"])
    return JsonResponse({
        "x": x_grid, "y": y_grid, "z": z, "warning": warning,
        # The ramp travels with the numbers, so what the colour *means* stays
        # declared once in `plots.py` beside every other colour on the page —
        # see UNCERTAINTY_SCALE for why it is neither green nor blue.
        "colorscale": UNCERTAINTY_SCALE,
    })


@experiment_view(VIEW)
def local_effects(request, exp):
    """The beeswarm of every sampled trial's local effect, for one metric.

    Fetched rather than shipped, and for a stronger reason than the other two
    deferred computations: this one is a whole ablation game per trial. Capped
    by the experiment's `local_effects_max_trials`, and (like every deferred
    computation now) it waits to be asked unless its autocompute setting says
    otherwise.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = _rebuild_experiment(exp)
    if built is None or built["result"] is None or not built["result"].trials:
        return HttpResponseBadRequest("no result")

    figure, warning = _local_effects_data(
        built, metric, resolve_settings(exp)["local_effects_max_trials"])
    return JsonResponse({"figure": figure, "warning": warning})


@experiment_view(VIEW)
def metric_figures(request, exp):
    """Every per-metric figure's plot payload for one metric.

    The page ships only the metric it opens on (see `_detail_context`), so this
    answers for the rest — once each, the first time one is selected. Unlike
    `trial_ablation` and `partial_dependence` this computes nothing new: the
    numbers behind these figures were all worked out at run completion and
    stored, so the cost here is building the Plotly objects and serializing
    them, which is exactly the cost the page used to pay for every metric on
    every load whether or not anyone looked.

    Honours the visibility settings through `_shown_figures`, so a figure
    switched off is absent here too rather than reachable by asking directly.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = _rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")

    per_metric = [f for f in _shown_figures(exp) if f.per_metric]
    return JsonResponse(_figure_plots(result, per_metric, metric,
                                      config_space=_config_space_for(built)))


@experiment_view(RUN)
def experiment_run(request, exp):
    """Launch a background run of an experiment (or confirm a metric change).

    The Run form supplies the stopping criteria and the metric to optimize. A
    run that would change the optimized metric first shows a confirmation; the
    confirmation posts back with a `decision` of "new" or "old".
    """
    if (request.method != "POST" or not exp.dataset or exp.is_running
            or not _model_available(exp)):
        return redirect("ui:experiment_detail", pk=exp.pk)

    chosen = request.POST.get("optimize_metric")
    stopping = _posted_stopping(request, chosen or exp.current_metric or "")
    trial_timeout = _posted_trial_timeout(request)
    decision = request.POST.get("decision")

    if not any(k not in FAILURE_CRITERIA for k in stopping):
        # A run has to be able to end. Back to the page with the reason rather
        # than a started run that never finishes. The failure limits do not
        # count: they are how a run notices it is broken, and a run that is
        # working would never reach them.
        context = _detail_context(request, exp)
        context["run_error"] = _("Set at least one stopping criterion, so the "
                                 "run has something to end on.")
        return render(request, "ui/experiment_detail.html", context)

    if decision:
        optimize_metric = resolve_metric_change(decision, exp.current_metric, chosen)
        if optimize_metric is None:
            return redirect("ui:experiment_detail", pk=exp.pk)
    else:
        action, optimize_metric = decide_run(exp.original_metric, exp.current_metric, chosen)
        if action == "warn":
            return render(request, "ui/metric_change.html", {
                "experiment": exp, "chosen": chosen,
                # Carried through the confirmation, or answering it would
                # silently drop the limits the run was set up with.
                "stopping": stopping,
                "trial_timeout": trial_timeout,
            })

    run = run_service.create_run(exp, stopping, optimize_metric,
                                 started_by=_owner(request),
                                 trial_timeout=trial_timeout)
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _stored_trial_count(exp) -> int:
    """How many trials *exp*'s saved result holds, without rebuilding it.

    Read straight off the JSON rather than through `_rebuild_experiment`,
    because this is polled every two seconds and the answer is usually "the same
    as last time" — the whole point of asking is to decide whether it is worth
    doing any of the expensive work at all.
    """
    return len((exp.result or {}).get("data") or [])


#: How often the run-status fragment is polled, in seconds. The floor is what a
#: poll can usefully be — the run writes at most every
#: `PARTIAL_RESULT_SECONDS`, so anything faster is a request that can only
#: answer "nothing yet". Above it the interval follows the trials themselves:
#: polling ten times per trial is ten times the traffic for one new row, and on
#: a real dataset a trial is minutes rather than milliseconds. The ceiling keeps
#: a long-trial run from feeling stopped.
POLL_MIN_SECONDS = 5
POLL_MAX_SECONDS = 30


def _poll_seconds(exp) -> int:
    """How long the page should wait before asking again.

    Read off the trials already stored — their own recorded durations — so it
    adapts as a run goes and needs nothing measured on the page. The median of
    the last few rather than the mean of all of them: a run's first trials are
    not like its later ones, and one pathological trial should not slow the poll
    for the rest of the run.

    Rendered into the fragment's own `hx-trigger`, which poll.js re-reads off
    each replacement — so the interval adapts per swap with no state anywhere
    and no second mechanism.
    """
    data = (exp.result or {}).get("data") or []
    recent = [entry.get("time") or 0.0 for entry in data[-10:]]
    typical = statistics.median(recent) if recent else 0.0
    return int(min(POLL_MAX_SECONDS, max(POLL_MIN_SECONDS, round(typical))))


def _live_payloads(exp, since: int = 0):
    """Fresh plots for every live figure of *exp*, for every metric, and the
    table rows for the trials the page has not seen.

    Only the figures declaring `Figure.live` — the ones that read nothing but
    the trials. Every metric rather than only the one being viewed, because the
    poll has no way to know which that is and the page switches between them
    without asking the server (see `metric_figures`); a live figure is four
    cheap Plotly builds, against the surrogate fits this deliberately excludes.

    Shaped to match the page's own two payload maps, so the browser merges
    rather than translates: `metric_plots[metric][key]` and `static_plots[key]`.
    """
    built = _rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return {}

    shown = _shown_figures(exp)
    live = [f for f in shown if f.live]
    config_space = _config_space_for(built)
    per_metric = [f for f in live if f.per_metric]
    static = [f for f in live if not f.per_metric]
    payload = {
        "metric_plots": {
            metric: _figure_plots(result, per_metric, metric, config_space=config_space)
            for metric in exp.metric_names
        } if per_metric else {},
        "static_plots": _figure_plots(result, static, None, config_space=config_space),
    }

    # The trials table is server-rendered HTML rather than a plot, so it cannot
    # ride in the payload maps above. It gets the rows it is missing instead —
    # rendered from the same partial the page built the table from, and appended
    # rather than swapped, so a sort, a page, a scroll position and a selected
    # row all survive an update that only adds to the end.
    if any(f.key == "trials" for f in shown) and len(result.trials) > since:
        payload["rows_html"] = render_to_string(
            "ui/figures/_trial_rows.html",
            {"trial_rows": _trial_rows(result, exp.metric_names,
                                       _hp_names(result), start=since)})
    return payload


@experiment_view(VIEW)
def run_status(request, exp):
    """HTMX poll target: the current run's status, or a refresh when finished.

    Also the channel for figures that move while the run is still going. There
    was never a rendering constraint on those — the trial-based figures need
    nothing but the trials — only a persistence one: the result was written once,
    at the end. `ui/services/run.py` now writes it periodically, and this hands
    the page whatever has arrived since it last said what it had.

    No new endpoint and no second poll loop: this fragment is already fetched
    every two seconds and already replaced wholesale, so the payload rides along
    inside it. `?trials=` is the page saying how many trials it has drawn, which
    is what keeps the cost down — build the plots only when that disagrees with
    what is stored, which under a five-second write interval is at most every
    third poll.

    A page showing "No results yet" has no figure grid to draw into, so it is
    sent the count alone and reloads itself once — see `experiment_detail.html`.
    """
    run_service.sweep_orphaned_runs()
    active = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    if active is None:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response

    stored = _stored_trial_count(exp)
    try:
        seen = int(request.GET.get("trials", stored))
    except (TypeError, ValueError):
        seen = stored

    context = {"experiment": exp, "run": active, "trial_count": stored,
               "poll_seconds": _poll_seconds(exp)}
    if stored != seen:
        live = {"trials": stored}
        # Nothing to merge into on a page that has no figures — it is about to
        # reload, so building payloads for it would be work thrown away.
        if seen > 0:
            live.update(_live_payloads(exp, since=seen))
        context["live"] = live
    return render(request, "ui/_run_status.html", context)


@experiment_view(VIEW)
def env_status(request, exp):
    """Poll target while a model environment is being built.

    Shaped exactly like `run_status`: the partial while there is something to
    wait for, then an empty response with `HX-Refresh` so the page reloads into
    whatever the outcome was. poll.js already understands both.
    """
    if not exp.env_pending:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response
    return render(request, "ui/_env_status.html", {"experiment": exp})


def _analytics_absent(result) -> bool:
    """True when a stored result has trials but no explanation games in it.

    Two ways to arrive here, both deliberate. A **cancelled** run skips them:
    pressing Cancel used to still buy the full analytics bill, so on a wide
    model you waited minutes for a run you had just stopped. A **partial** write
    (`ui/services/run.py`) leaves them out for the same reason at finer grain —
    2^n_hp coalitions per game per metric is not something to pay per trial.

    Both used to be permanent: the numbers were computed once, at run
    completion, or never. `experiment_compute_analytics` is the way to ask for
    them afterwards, so declining to compute them eagerly no longer means
    declining to compute them at all.
    """
    return bool(result.trials) and not any(result.hyperparameter_importance.values())


@require_POST
@experiment_view(RUN)
def experiment_compute_analytics(request, exp):
    """Compute the explanation games for an experiment that has none.

    Synchronous, and bounded by the same budget the eager computation is held
    to: `eager_analytics_budget_exceeded` turns away anything wide enough to be
    a problem, which leaves this in the same range as the local-effects fetch
    the page already makes. Anything it declines is declined with a reason, on
    the page, rather than started and waited on.

    Written back onto the result rather than returned as a payload — which is
    where the three fetched figures differ, and why. Those answer a question
    that has no small precomputable set of answers (which trial, which
    hyperparameter), so each one is a fresh request. These are exactly the
    fields the result already has, computed from the same trials that are
    already stored, so filling them in makes the result what a completed run
    would have written — the page renders it with no live-update path of its
    own, the export carries it, and nobody pays for it twice.

    Refused while a run is in flight: it would race that run's own partial
    writes, and the run will compute them itself when it finishes.
    """
    if exp.is_running:
        return redirect("ui:experiment_detail", pk=exp.pk)

    built = _rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return redirect("ui:experiment_detail", pk=exp.pk)

    optimizer = built["optimizer"]
    optimizer.analytics_max_coalitions = settings.ANALYTICS_EAGER_MAX_COALITIONS or None
    games = optimizer.compute_hp_games(
        _config_space_for(built), result.trials, built["metrics"], seed=built["seed"])

    for game, fields in HP_GAME_FIELDS.items():
        importance, warning, interactions, moebius, total = games[game]
        setattr(result, fields["importance"], importance)
        setattr(result, fields["warning"], warning)
        setattr(result, fields["interactions"], interactions)
        setattr(result, fields["moebius"], moebius)
    result.hyperparameter_interactions_warning = games["tunability"][1]
    result.hyperparameter_tunability_total = games["tunability"][4]

    Experiment.objects.filter(pk=exp.pk).update(
        result=optimizer.serialize_result(result))
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def prepare_env(request, exp):
    """Build (or rebuild) this experiment's model environment.

    For the retry button. Helps when the cause was transient — no network, an
    index down — or when uv has since been installed. A model file with a bad
    dependency name needs a new experiment, since there is no way to replace the
    file in place.
    """
    if exp.model_file and not exp.is_running:
        modelenv.start_preparation(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(EDIT)
def experiment_share(request, exp):
    """Turn read access for everyone else on or off.

    Only reachable on an instance with accounts, and only by the owner (EDIT is
    refused on someone else's), so an experiment cannot be shared out from under
    the person it belongs to.
    """
    exp.shared = bool(request.POST.get("shared"))
    exp.save(update_fields=["shared"])
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_cancel(request, exp):
    """Request cancellation of the experiment's active run.

    Cooperative: this sets a flag the running optimizer polls between trials, so
    the trials already finished are kept and the result is still written. It
    needs something to be reading the flag — see `run_force_stop` for when
    nothing is.

    POST only: it changes something, and on GET a prefetcher or an <img> tag
    pointing here would cancel someone's run without CSRF ever being consulted.
    """
    exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_force_stop(request, exp):
    """Give up on a run that is not answering, and say so on the row.

    Cancelling is a request to whatever is executing the run. When nothing is —
    the process was restarted, the consumer died — there is nobody to read it,
    and the experiment sits at "running" with no way out. This is the way out.

    For a run in this process it does not kill anything, and does not pretend
    to: the web process has no handle on the worker, which is in another thread
    or another process entirely. What it does is stop the experiment waiting on
    it. If the worker turns out to be alive after all, it finishes as it always
    would have — its last act is a filtered update of this same row, which will
    put back whatever actually happened.

    A run on a cluster is the exception, because there *is* a handle: the job
    id. Abandoning the row alone would leave the job running to its wall-clock
    limit with nobody listening, holding an allocation nobody wants — so it is
    cancelled with the scheduler too. That loses whatever it had not written,
    which is what this escalation always costs.

    Offered only once cancelling has been asked for and has not worked, so it is
    the escalation rather than the first thing to hand: a cooperative cancel
    keeps the trials that already ran, and this cannot.
    """
    stuck = exp.runs.filter(status__in=_ACTIVE, cancel_requested=True)
    for job_id in stuck.exclude(job_id="").values_list("job_id", flat=True):
        _abandon_cluster_job(job_id)
    stuck.update(
        status="error", finished_at=timezone.now(),
        error=_("Given up on by hand: it stopped answering, and cancelling it "
                "had no effect."),
    )
    return redirect("ui:experiment_detail", pk=exp.pk)


def _abandon_cluster_job(job_id: str) -> None:
    """`scancel` a job whose run is being given up on.

    Best-effort and deliberately quiet. This runs inside a request, and the
    cluster being unreachable is a likely reason the run stopped answering in
    the first place — so failing here must not stop the row being freed, which
    is the thing the reader actually asked for. A job that outlives this is
    bounded by its own wall-clock limit.
    """
    from .services import cluster

    try:
        cluster.kill(cluster.SshTransport(settings.CLUSTER_HOST, timeout=15.0), job_id)
    except Exception:  # noqa: BLE001 — see above
        logger.warning("could not scancel job %s", job_id, exc_info=True)


#: What a ticked box on the export page posts. One value for both of its
#: questions: they are asked the same way and answered the same way, and a
#: second constant would only be a second place to change it.
KEEP = "keep"


@experiment_view(EXPORT)
def experiment_export(request, exp):
    """Download a saved experiment as an .ihpo file, having asked what goes in it.

    Server paths are always blanked. Two things are worth a question, because
    both are facts about a person or a machine rather than about a search, and
    the file is complete as a record of the search without either:

    - per-trial `starttime`/`endtime`, which say what time of day someone was
      working and on which days. The durations that make the timing figures
      readable survive without them.
    - a failed trial's stored traceback, which names absolute paths on this
      machine, the packages installed on it and their versions. The failure
      itself survives without it, with its one-line reason.

    Asked at the point of export rather than kept as a setting. A setting is
    answered once, by whoever set it up, for every file the instance ever sends
    — and the answer depends on the file and on who is about to receive it. It
    also has to be found and understood before the question has been raised,
    which is the wrong order for a question about what you are handing over.

    Scrubbed here, not in the adapter, so the detail page's own reconstruction
    and the run engine are unaffected; deserialize ignores the keys on
    re-import.
    """
    if request.method != "POST":
        # How many tracebacks there are to ask about, and so whether to ask.
        # Counted off the stored result rather than rebuilt through the
        # optimizer: this needs one key per trial, not an OptimizationResult.
        entries = ((exp.result or {}).get("data") or [])
        count = sum(1 for e in entries
                    if (e.get("additional_info") or {}).get("traceback"))
        return render(request, "ui/export_confirm.html",
                      {"experiment": exp, "has_tracebacks": bool(count),
                       "traceback_count": count})

    snapshot = snapshot_adapter.snapshot_from_experiment(exp, provenance=True)
    # The paths name files on this server, which is of no use to whoever opens
    # the file and tells them how the instance is laid out.
    snapshot["dataset"]["path"] = ""
    snapshot["model"]["path"] = ""
    keep_times = request.POST.get("timestamps") == KEEP
    keep_tracebacks = request.POST.get("tracebacks") == KEEP
    if not (keep_times and keep_tracebacks) and snapshot.get("result"):
        snapshot["result"] = copy.deepcopy(snapshot["result"])
        for entry in snapshot["result"].get("data", []):
            if not keep_times:
                entry.pop("starttime", None)
                entry.pop("endtime", None)
            # The reason stays either way. It is the trial's own result — this
            # configuration does not work — while the traceback is a description
            # of the machine it did not work on.
            if not keep_tracebacks:
                (entry.get("additional_info") or {}).pop("traceback", None)
    body = io.to_bytes(snapshot)
    response = HttpResponse(body, content_type="application/octet-stream")
    response["Content-Disposition"] = f'attachment; filename="{exp.name}.ihpo"'
    return response


def _posted_settings(request):
    """The settings as submitted, coerced to the type of each default.

    Almost every setting is a checkbox, where absent means off. The exceptions
    are the numeric ones (`ice_max_curves`), which are read as integers and
    clamped to `SETTING_BOUNDS` — a display preference typed into a box is not
    worth failing a form over, and anything unreadable falls back to the
    built-in default rather than to zero, which would mean something specific
    and wrong ("no limit").

    Keyed off `type(default)` rather than a second list of which settings are
    numbers, so declaring one in `SETTING_DEFAULTS` is all it takes.
    """
    posted = {}
    for key, default in SETTING_DEFAULTS.items():
        if isinstance(default, bool):
            posted[key] = bool(request.POST.get(key))
            continue
        try:
            value = int(request.POST.get(key, ""))
        except (TypeError, ValueError):
            value = default
        low, high = SETTING_BOUNDS.get(key, (None, None))
        if low is not None:
            value = max(low, min(high, value))
        posted[key] = value
    return posted


@experiment_view(EDIT)
def experiment_settings(request, exp):
    """One experiment's settings: inherit the defaults, override, reset, or
    promote its own settings to be the defaults (which asks first).

    How the search runs is not here. An optimizer's settings are chosen when the
    experiment is created and fixed for its life, like `cv_folds` — see
    `_posted_optimizer_params`.
    """
    if request.method == "POST":
        if "reset" in request.POST or request.POST.get("use_default_settings"):
            exp.use_default_settings = True
            exp.settings = {}
            exp.save(update_fields=["use_default_settings", "settings"])
            return redirect("ui:experiment_settings", pk=exp.pk)

        posted = _posted_settings(request)

        if ("save_as_default" in request.POST
                or "confirm_save_as_default" in request.POST):
            if not permissions.policy().may_change_defaults(request):
                raise PermissionDenied

        if "save_as_default" in request.POST:
            # Ask before changing what every inheriting experiment shows.
            return render(request, "ui/save_as_default_confirm.html", {
                "experiment": exp,
                "pending": [key for key, on in posted.items() if on],
            })

        if "confirm_save_as_default" in request.POST:
            gs = GlobalSettings.get_solo()
            gs.default_experiment_settings = posted
            gs.save(update_fields=["default_experiment_settings"])
            return redirect("ui:experiment_settings", pk=exp.pk)

        exp.use_default_settings = False
        exp.settings = posted
        exp.save(update_fields=["use_default_settings", "settings"])
        return redirect("ui:experiment_settings", pk=exp.pk)

    form = ExperimentSettingsForm(initial={
        **resolve_settings(exp),
        "use_default_settings": exp.use_default_settings,
    })
    return render(request, "ui/experiment_settings.html",
                  {"experiment": exp, "form": form})


def appearance(request):
    """Appearance settings (display/theme options; currently a placeholder)."""
    return render(request, "ui/appearance.html", {})


def account(request):
    """Who you are signed in as, and what you may do here.

    Assembled rather than looked up: the answer now comes from four separate
    grants, held directly or through a group, and before this there was no page
    anywhere that could say what they added up to. An operator reading a bug
    report needs it as much as the person does.

    Absent without accounts — `REQUIRE_LOGIN` off means there is no account to
    describe and every answer would be an unconditional yes.
    """
    if not settings.REQUIRE_LOGIN:
        raise Http404
    policy = permissions.policy()
    return render(request, "ui/account.html", {
        # Each paired with what it lets you do rather than its codename, which
        # names the grant and not the consequence.
        "powers": [
            (_("See every experiment on this instance"),
             request.user.has_perm("access.view_all_experiments")),
            (_("Run, edit and delete other people's experiments"),
             request.user.has_perm("access.manage_experiments")),
            (_("Change the settings every experiment inherits"),
             policy.may_change_defaults(request)),
            (_("Upload and run custom models"),
             policy.may_upload_models(request)),
        ],
        # Its own line because it is an instance-wide floor rather than
        # something about this account: off means off, for everyone.
        "custom_models_off": not settings.ALLOW_CUSTOM_MODELS,
        "is_administrator": request.user.is_superuser,
    })


def default_experiment_settings(request):
    """Edit the default experiment settings that inheriting experiments use.

    Every key in the schema is a checkbox, so an absent key means unchecked.
    """
    if not permissions.policy().may_change_defaults(request):
        raise PermissionDenied
    gs = GlobalSettings.get_solo()
    if request.method == "POST":
        gs.default_experiment_settings = _posted_settings(request)
        gs.save(update_fields=["default_experiment_settings"])
        return redirect("ui:default_experiment_settings")

    form = DefaultExperimentSettingsForm(initial=global_defaults())
    return render(request, "ui/default_experiment_settings.html", {"form": form})


@experiment_view(DELETE)
def experiment_delete(request, exp):
    """Confirm (GET) then delete (POST) a saved experiment, stopping any run."""
    if request.method == "POST":
        exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
        exp.delete()
        return redirect("ui:home")
    return render(request, "ui/delete_confirm.html", {"experiment": exp})


def import_experiment(request):
    """Upload an .ihpo file to create a saved experiment.

    A dataset and — when custom models are enabled — a model .py may be
    attached to make the imported experiment runnable; without them it loads
    read-only.
    """
    may_upload = permissions.policy().may_upload_models(request)
    context = {"allow_custom_models": may_upload}
    upload = request.FILES.get("file")
    if request.method == "POST" and request.FILES.getlist("smac_dir"):
        return _import_smac_directory(request, context)
    if request.method == "POST" and upload is not None:
        # No Form here to hang a validator off, so the same checks the create
        # form's file fields run are called directly — see ui/validators.py.
        size_error = oversized(upload)
        if size_error:
            context["error"] = size_error
            return render(request, "ui/import.html", context)
        try:
            snapshot = io.parse(upload.read())
        except ValueError as exc:
            context["error"] = _("Invalid or unreadable experiment file.") + f" ({exc})"
            return render(request, "ui/import.html", context)

        # The file records which dataset produced its trials. Attaching a
        # different one here is the one way an experiment could go on adding
        # trials to a history they do not belong to, and nothing downstream —
        # not the incumbent, not the surrogate, not the importance — could tell.
        dataset_upload = request.FILES.get("dataset")
        if dataset_upload is not None:
            dataset_error = dataset_upload_error(dataset_upload)
            if dataset_error:
                context["error"] = dataset_error
                return render(request, "ui/import.html", context)
            mismatch = provenance.dataset_mismatch(
                snapshot.get("dataset"), provenance.sha256_stream(dataset_upload))
            if mismatch:
                context["error"] = mismatch
                return render(request, "ui/import.html", context)

        model_upload = request.FILES.get("model") if may_upload else None
        if model_upload is not None:
            model_error = model_upload_error(model_upload)
            if model_error:
                context["error"] = model_error
                return render(request, "ui/import.html", context)
        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, dataset_file=dataset_upload, model_file=model_upload,
            owner=_owner(request),
        )
        # An imported model is re-locked here rather than trusting pins chosen by
        # whoever exported it.
        modelenv.start_preparation(exp)
        return redirect("ui:experiment_detail", pk=exp.pk)

    return render(request, "ui/import.html", context)


def _import_smac_directory(request, context):
    """Create a read-only experiment from an uploaded SMAC output directory.

    Its own branch rather than a third optional file on the `.ihpo` form: this
    reads a *set* of files with no `.ihpo` among them, and neither a dataset nor
    a model may be attached to it. A SMAC scenario records nothing about what
    was being optimized, so there is nothing here to check an attached dataset
    against — see `core/smac_import.py`, and `import_experiment` on why a
    dataset that cannot be checked is not accepted.
    """
    files = {}
    for upload in request.FILES.getlist("smac_dir"):
        size_error = oversized(upload)
        if size_error:
            context["error"] = size_error
            return render(request, "ui/import.html", context)
        # Django's uploader keeps only the basename, so a directory picker's
        # relative paths do not survive the post — which is why the importer
        # matches on basenames, and why two run directories at once cannot be
        # told apart. Refused rather than blended: SMAC writes one run per
        # `<name>/<seed>` directory, and a runhistory read against another
        # run's config space is not a run that happened.
        name = upload.name
        if not name.endswith(".json"):
            continue
        if name in files:
            context["error"] = _(
                "This looks like more than one run — two files here are called "
                "%(name)s. Choose a single run directory, the one holding its "
                "runhistory.json.") % {"name": name}
            return render(request, "ui/import.html", context)
        try:
            files[name] = json.loads(upload.read().decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            context["error"] = _("Could not read %(name)s.") % {"name": name} + f" ({exc})"
            return render(request, "ui/import.html", context)

    try:
        snapshot = io.parse(json.dumps(
            smac_import.snapshot_from_smac(files)).encode("utf-8"))
    except ValueError as exc:
        context["error"] = _("This is not a readable SMAC run.") + f" ({exc})"
        return render(request, "ui/import.html", context)

    exp = snapshot_adapter.experiment_from_snapshot(snapshot, owner=_owner(request))
    return redirect("ui:experiment_detail", pk=exp.pk)


def _rebuild_experiment(exp):
    """Rebuild the full `build_experiment` dict from a saved experiment's
    snapshot (read-only — no dataset needed), or None if it can't be rebuilt
    (e.g. it names a metric/optimizer no longer available).

    For callers that need more than the result — the model (for its config
    space) and the optimizer instance, both needed to compute a local
    explanation on demand, which nothing pre-stored on the result can answer.
    Read-only still resolves a registry model (only a custom upload's file
    goes unread), so `built["model"]` is None only for those.
    """
    try:
        _, built = io.build_experiment(
            snapshot_adapter.snapshot_from_experiment(exp),
            METRICS, MODELS, OPTIMIZERS, read_only=True,
        )
    except ValueError:
        return None
    return built


def _rebuild_result(exp):
    """Rebuild the OptimizationResult from a saved experiment's snapshot
    (read-only — no dataset or live model needed), or None if it can't be
    rebuilt (e.g. it names a metric/optimizer no longer available)."""
    built = _rebuild_experiment(exp)
    return built["result"] if built else None


def _local_ablation_data(built, metric, idx):
    """The local-ablation figure + warning for one (metric, trial index) —
    HyperSHAP's `ablation` game, explaining that one trial's configuration
    against the config space's default, rather than a global share like the
    other three games. `built` is `_rebuild_experiment`'s full dict, since
    this needs the search space and the optimizer instance, neither of which
    the stored result carries.

    Returns (figure_json_or_None, warning_or_None, raw_effects). The effects
    are the signed per-hyperparameter values the figure is drawn from, in the
    metric's own units, which `_tuning_progress` needs and which cannot be read
    back out of the figure. The figure is None (with a warning) when there is no
    search space to be had — see `_config_space_for` for the two places one can
    come from.
    """
    config_space = _config_space_for(built)
    if config_space is None:
        return None, _("Local explanation needs the search space, which this "
                       "experiment does not carry and has no model to ask."), {}
    result = built["result"]
    ablation, warning = built["optimizer"].compute_hp_ablation(
        config_space, result.trials, metric, result.trials[idx].config,
        seed=built["seed"],
        # From the result, not the registry: an imported run's objective is
        # declared inside its own file, and the explainer needs its direction
        # (see `_pair_trials_with_oriented_scores`).
        metric=result.metric(metric))
    fig = hyperparameter_ablation_plot(ablation)
    return (json.loads(fig.to_json()) if fig is not None else None), warning, ablation


#: Whether the page offers "still to gain" at all.
#:
#: Off: the measure is not ready to be read. The presentation problems were the
#: visible half, but the numbers are not established either — the same run,
#: same seed, gives tunability +0.096 in one process and +0.070 in another,
#: while being bit-identical within a process. `_tuning_progress` subtracts one
#: of those quantities from another, so whatever moves between processes lands
#: in its result, and a reader would be comparing a difference against a
#: baseline that does not hold still. The tests below it check the arithmetic is
#: self-consistent, which is a weaker claim than the measure being sound.
#:
#: Everything stays in place and stops being reachable from the page, so the
#: switch back is this line — once the cross-process drift is understood.
TUNING_PROGRESS_ENABLED = False

#: Below this share of the achievable gain, "what is left" is rounding rather
#: than headroom, and a pie of it would be a picture of noise.
SETTLED = 0.02


def _tuning_progress(result, metric, ablation):
    """What each hyperparameter was worth, and how much of it has been banked.

    Tunability decomposes (best achievable − default); the ablation decomposes
    (this trial − default). Both are FSII order-1 terms of an order-2 fit,
    against the same baseline, and both games are zeroed at that baseline —
    measured, `v(none) = 0` for each — so the two are in the same units from the
    same origin and the subtraction is well posed.

    **Two things it is not**, both measured rather than assumed, and both
    reasons to read the ordering rather than the digits:

    - *Achievable is a lower bound.* The tunability game's maximum comes from
      10,000 random draws of the configuration space, and a real optimizer beats
      random search — that is what it is for. So a well-tuned hyperparameter can
      show more banked than achievable, and it is not an error: it means the
      search found something the estimate did not. `beyond` marks those, and
      they are the strongest evidence an axis is finished, not a contradiction.
    - *Order-1 is not the whole game.* Measured on a four-hyperparameter run, the
      order-1 terms account for about 70-100% of each game's own endpoint, the
      rest living in order-2 interactions that neither number carries. Both
      sides are deflated by the same mechanism, so the comparison survives it
      better than either figure alone does.

    Both halves are needed in the metric's own units, and only one of them is
    stored that way: `hyperparameter_importance` is a share of 100%, and
    `hyperparameter_tunability_total` is what it is a share *of*. A result
    written before that field existed has no scale to multiply by, and gets
    nothing rather than a ratio computed against the wrong denominator.

    Returns `(rows, settled)`. Each row is one hyperparameter's achievable,
    banked and remaining value; *settled* is True when the remaining total is
    rounding, which is a finding — this trial has taken essentially everything
    there was — and not a figure.
    """
    if not TUNING_PROGRESS_ENABLED:
        return [], False

    shares = result.hyperparameter_importance.get(metric, {})
    total = result.hyperparameter_tunability_total.get(metric, 0.0)
    if not shares or not total or not ablation:
        return [], False

    rows = []
    for name, share in shares.items():
        achievable = share * total
        # Signed on purpose. A negative one says this trial's value for that
        # hyperparameter is *worse* than the default — drift picked up while
        # chasing whichever one mattered — and flooring it here would hide the
        # one finding that is actionable on its own.
        banked = ablation.get(name, 0.0)
        rows.append({
            "name": name,
            "achievable": achievable,
            "banked": banked,
            # Floored, because a negative amount left is not an amount left.
            "remaining": max(0.0, achievable - banked),
            "banked_share": (banked / achievable) if achievable else 0.0,
            # Past the estimated ceiling: the search found something 10,000
            # random draws did not. Marked rather than quietly floored, because
            # it says something — this axis is not merely finished, it is
            # finished past where the estimate could see.
            "beyond": banked > achievable > 0,
        })
    rows.sort(key=lambda row: row["remaining"], reverse=True)
    left = sum(row["remaining"] for row in rows)
    return rows, left <= SETTLED * (total or 1.0)


def _partial_dependence_data(built, metric, hp_name, max_ice_curves=0):
    """The partial-dependence figure + warning for one (metric,
    hyperparameter) — same shape as `_local_ablation_data`, and for the same
    reason: fitting a surrogate and predicting across a grid needs the search
    space, not just the stored result, and depends on which hyperparameter is
    picked rather than being one of a small precomputable set.

    *max_ice_curves* is the experiment's `ice_max_curves` setting: how many
    trials get predicted and drawn, 0 for all of them. It caps the work, not
    just the picture — see `compute_partial_dependence`.

    Returns (figure_json_or_None, warning_or_None). The figure is None (with
    a warning) when there is no search space to be had.
    """
    config_space = _config_space_for(built)
    if config_space is None:
        return None, _("Partial dependence needs the search space, which this "
                       "experiment does not carry and has no model to ask.")
    result = built["result"]
    grid, ice_lines, pdp, warning = built["optimizer"].compute_partial_dependence(
        config_space, result.trials, metric, hp_name, seed=built["seed"],
        max_ice_curves=max_ice_curves)
    fig = partial_dependence_plot(hp_name, grid, ice_lines, pdp)
    return (json.loads(fig.to_json()) if fig is not None else None), warning


def _shown_figures(exp):
    """The figures *exp*'s settings switch on, in catalog order.

    Shared by the detail page and the `metric_figures` endpoint so a figure
    switched off is absent from both — the endpoint must not become a way to
    fetch what the page deliberately did not build.
    """
    shown = resolve_settings(exp)
    return [figure for figure in FIGURES if shown[figure.setting_key]]


def _figure_options(figure, result, metric_names):
    """One figure's display behaviour, for the page script.

    `absoluteScale` is the relayout its absolute/relative toggle pins to. For a
    per-metric figure it is keyed by metric first, because a range that is right
    for an accuracy is a fiction for a cost — and None when no metric offers one,
    which is how the toggle knows to stay hidden rather than appear and lie.
    """
    scale = figure.absolute_scale
    if figure.per_metric and scale is not None:
        scale = {name: figure.absolute_scale_for(result.metric(name))
                 for name in metric_names}
        scale = {name: views for name, views in scale.items() if views} or None
    return {"absoluteScale": scale, "perMetric": figure.per_metric}


def _config_space_for(built):
    """*built*'s search space, or None when there is nothing to build one from.

    The stored `space` section first, the model second. Two sources because
    neither covers everything: an experiment run here can always ask its model,
    while a run read out of somebody else's output has no model at all — and a
    custom-model experiment viewed read-only cannot ask either, which is what
    the stored section fixes for it.

    None is a supported answer throughout. The figures declaring
    `needs_config_space` fall back to linear axes rather than refusing to draw
    (see `Figure.needs_config_space`), and the three that genuinely cannot
    proceed say so in a caption.
    """
    built = built or {}
    stored = io.config_space_from_serialized(built.get("config_space"),
                                             seed=built.get("seed", 0))
    if stored is not None:
        return stored
    model = built.get("model")
    return model.get_config_space(seed=built["seed"]) if model is not None else None


def _hp_names(result) -> list:
    """The hyperparameters a result's trials carry, in config order."""
    return list(result.trials[0].config.keys()) if result.trials else []


def _trial_rows(result, metric_names, hp_names, start: int = 0) -> list:
    """The trials table's rows, from *start* onwards.

    Module-level and sliceable because it is rendered twice: the whole table
    when the page is built, and the tail of it when a run adds trials and the
    poll appends them — from the same partial, so the two cannot drift into
    formatting a duration or a failure differently.
    """
    return [
        {
            # Its position in the result, which is what everything that names a
            # trial uses — the panel endpoint, the local-ablation cache, every
            # plot's selection meta. `n` is the number shown.
            "idx": i,
            "n": t.trial,
            "config_values": [t.config.get(h) for h in hp_names],
            # .get, not [m]: a row stored before io.parse checked for it can be
            # missing a metric, and the table's job is to show what is there
            # rather than to be the thing that 500s the page.
            "scores": [t.scores.get(m) for m in metric_names],
            "duration": t.duration,
            # A trial that produced no measurement. Its scores are all 0.0 and
            # read as a bad trial rather than as no trial, so the row has to say
            # which it is — and the reason, since a table has room for it where
            # a figure does not.
            "failed": t.failed,
            "failure": t.failure,
        }
        for i, t in enumerate(result.trials)
        if i >= start
    ]


def _figure_plots(result, figures, metric=None, config_space=None):
    """`{figure key: plot JSON}` for *figures* at *metric*.

    A figure with declared views gets a dict of view key -> plot JSON (or None)
    instead of a single plot, one entry per view — the browser picks which to
    show; see experiment_detail.html's `payloadFor`.

    A figure whose `plot()` returns None gets None, which `draw()` reads as "no
    data yet" — that is how the three fetched figures (partial dependence, local
    explanation, local effects) end up here without being computed.

    Module-level, and taking the metric rather than closing over one, because
    `metric_figures` answers with exactly this for a metric the page did not
    ship — see `_detail_context` for why it only ships one.
    """
    def fig_json(plot):
        return json.loads(plot.to_json()) if plot is not None else None

    def one(figure):
        extra = {"config_space": config_space} if figure.needs_config_space else {}
        if figure.views:
            return {view: fig_json(figure.plot(result, metric, view=view, **extra))
                    for view in figure.views}
        return fig_json(figure.plot(result, metric, **extra))

    return {figure.key: one(figure) for figure in figures}


def _local_effects_data(built, metric, max_trials=0):
    """The beeswarm figure + warning for one metric — every sampled trial's
    local ablation, from `compute_local_effects`.

    Same shape and same reasoning as `_partial_dependence_data`: it needs the
    search space, not just the stored result, and it is the one figure whose
    cost is per *trial*, which is what `max_trials` bounds.
    """
    config_space = _config_space_for(built)
    if config_space is None:
        return None, _("Local effects need the search space, which this "
                       "experiment does not carry and has no model to ask.")
    result = built["result"]
    hp_names, rows, warning = built["optimizer"].compute_local_effects(
        config_space, result.trials, metric,
        seed=built["seed"], max_trials=max_trials, metric=result.metric(metric))
    fig = local_effects_plot(hp_names, rows)
    return (json.loads(fig.to_json()) if fig is not None else None), warning


def _selected_panel_data(result, metric, idx):
    """The selected-config panel's data for one (metric, trial index).

    The trial's score for *metric*, and its distance from the metric's best —
    always, including when the selected trial *is* the best one, where the
    distance is zero and `is_best` says why. The panel used to leave the row out
    in that case, which made it change height as you clicked from one trial to
    another; and "no difference" and "a difference of zero" are the same fact
    said two ways, only one of which you can read off the page.
    """
    trials = result.trials
    best_idx = result.best_index(metric)
    trial = trials[idx]
    score, best = trial.scores[metric], trials[best_idx].scores[metric]
    return {
        "metric": metric,
        "trial_n": trial.trial,
        "score": score,
        "delta": score - best,
        # Whether that difference is in the good direction, decided here rather
        # than by the template comparing it to zero. Against the *best* trial it
        # never is — but which sign means worse depends on the metric, and on
        # one where lower wins the sign is the other way round, so a template
        # reading `delta > 0` as "better" gets it exactly backwards.
        "delta_better": result.metric(metric).better(score, best),
        "is_best": idx == best_idx,
        "config": list(trial.config.items()),
        # A trial that produced no measurement: the score above is a
        # placeholder, and the panel says so rather than presenting a 0.0 as a
        # result. `idx`, not `trial_n`, because that is what the traceback
        # endpoint takes — the same index every figure's selection speaks in.
        "failed": trial.failed,
        "failure": trial.failure,
        "idx": idx,
        "has_traceback": bool(trial.traceback),
    }


def _evaluation_label(exp):
    """How a trial was scored, for the run-configuration box.

    "Not recorded" for an experiment imported from somebody else's output: a
    SMAC scenario says nothing about what was being optimized, so there is no
    split to report. The columns still hold their defaults — they have to hold
    something — and reporting those as fact would put a 20% holdout on the page
    for a run that may have used no such thing.
    """
    if exp.model_name not in MODELS and not exp.model_file:
        return _("Not recorded")
    if exp.cv_folds >= 2:
        return _("%(k)s-fold CV") % {"k": exp.cv_folds}
    return _("%(pct)s%% held out") % {"pct": round(exp.test_size * 100)}


def _detail_context(request, exp):
    """Detail-page context: identity, run state, and the figures to draw.

    Rebuilds the OptimizationResult from the stored snapshot (read-only) and,
    when present, builds the panels and per-metric figures for whichever figures
    the settings have switched on; the browser switches metrics client-side.
    """
    built = _rebuild_experiment(exp)
    result = built["result"] if built else None
    metric_names = list(exp.metric_names)
    active_run = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    last_run = exp.runs.order_by("-id").first()

    # Why this viewer may not run this experiment's custom model, if they
    # may not. Shown rather than silently disabling the form.
    model_refusal = permissions.policy().custom_model_refusal(exp, request.user)

    run_summary = None
    if last_run and last_run.status in ("done", "cancelled") and last_run.duration is not None:
        total = last_run.duration
        trials = last_run.trial_seconds or 0.0
        run_summary = {"total": total, "trials": trials, "overhead": max(0.0, total - trials),
                       "count": last_run.trial_count or 0,
                       "stopped_by": STOPPED_BY_LABELS.get(last_run.stopped_by)}

    context = {
        "experiment": exp,  # the _run_status.html include reverses URLs from experiment.pk
        "summary": {
            "pk": exp.pk,
            "name": exp.name,
            "identifier": exp.identifier,
            "model_name": exp.model_name,
            "optimizer_name": exp.optimizer_name,
            "current_metric": exp.current_metric,
            "metric_label": metric_label(exp.current_metric, exp.original_metric),
            "seed": exp.seed,
            "evaluation": _evaluation_label(exp),
        },
        "metric_names": metric_names,
        # A result with no trials counts as no result: it is what the early
        # return below stops building panels and plots for, so the template must
        # take its "No results yet" branch rather than render the figure script
        # with nothing for it to read.
        "has_result": result is not None and bool(result.trials),
        "active_run": active_run,
        "can_run": (bool(exp.dataset) and _model_available(exp)
                    and permissions.policy().may(request, exp, RUN)
                    and not model_refusal),
        # What this viewer may do, for the buttons. Read access got them here;
        # the rest depends on whose experiment it is.
        "may": {action: permissions.policy().may(request, exp, action)
                for action in (RUN, EDIT, DELETE, EXPORT)},
        "ownership": _ownership(request, exp),
        "run_error": last_run.error if (last_run and last_run.status == "error") else None,
        "model_refusal": model_refusal,
        # The model's environment, for the branches on the detail page.
        "env": {
            "status": exp.env_status,
            "pending": exp.env_pending,
            "in_process": exp.env_in_process,
            "ready": exp.env_status == Experiment.ENV_READY,
            "failed": exp.env_status == Experiment.ENV_FAILED,
            "error": exp.env_error,
            "summary": modelenv.prepared_summary(exp),
        },
        "run_summary": run_summary,
        # Only an optimizer that fits a model of the objective can answer the
        # confidence criterion, so only then is it offered.
        "supports_confidence": getattr(
            _optimizer_for(exp.optimizer_name), "supports_confidence_stopping", False),
        "run_default_metric": exp.current_metric or (metric_names[0] if metric_names else None),
        # What the deadline fields open on. The deployment's number, so an
        # instance that has tuned `MODEL_TRIAL_TIMEOUT` for its own hardware
        # offers that rather than making everyone retype it.
        "default_trial_timeout": int(settings.MODEL_TRIAL_TIMEOUT),
        "default_trial_factor": deadline.DEFAULT_FACTOR,
        # How many trials this page has drawn, which the run-status poll sends
        # back so the server can tell whether anything has moved since. 0 here
        # and overwritten below: the early return means "no figure grid", and
        # that is exactly what 0 tells `run_status`.
        "trial_count": 0,
        # How long the run-status poll waits between asks. Follows the trials'
        # own durations — see `_poll_seconds`.
        "poll_seconds": _poll_seconds(exp),
    }
    # A result with no trials is the same story as no result at all — there is
    # nothing to plot and no best trial to describe — so it takes the same early
    # return. Not a hypothetical: `io.parse` accepts a snapshot whose result has
    # an empty `data`, and importing one used to 500 the detail page (an argmax
    # over an empty range). A run never stores one (`run.py` only writes a
    # result that has trials), which is why this went unnoticed.
    if result is None or not result.trials:
        return context

    # Past the early return, so there is a grid: say how much of the run it is
    # showing. `run_status` compares this against what is stored and only builds
    # fresh plots when the two disagree.
    context["trial_count"] = len(result.trials)
    # And whether the explanations are missing and could be asked for — a
    # cancelled run's are skipped by design, and a partial write leaves them out
    # for the same reason. Not while a run is in flight: it will compute them
    # itself when it ends, and asking now would race its own writes.
    context["analytics_absent"] = _analytics_absent(result) and not active_run

    # Which figures to draw. A figure that is switched off is not rendered and
    # its plot is not built, so nothing is computed or shipped to go unused.
    figures = _shown_figures(exp)
    shown = resolve_settings(exp)

    panels = []
    for m in metric_names:
        best_idx = result.best_index(m)
        best = result.trials[best_idx]
        panels.append({
            "metric": m,
            "best_n": best.trial,
            # The array index (as opposed to best_n, the trial's own number)
            # of the metric's best trial — what the local-explanation fetch
            # asks for by default, before any click. The same index
            # click-to-select already works in (Plotly's pointIndex).
            "best_idx": best_idx,
            "best_score": best.scores[m],
            "best_config": list(best.config.items()),
            # No selection has been clicked yet, so it defaults to the best trial.
            "selected": _selected_panel_data(result, m, best_idx),
            # One entry per HyperSHAP game: its warning, and its "table"
            # rendering's rows (rendered straight from here rather than from a
            # plot, like best_config above, so it needs no JSON round-trip
            # through the page script). Keyed by game because the page's game
            # selector picks which applies, independently of the metric.
            #
            # The warning covers the interaction figures too. They are extracted
            # from the same HyperSHAP call, not a separate one, so whatever made
            # that call fail explains their emptiness as well — which is why
            # there is no second warning to keep in step with this one.
            "importance_by_game": {
                game: {
                    "warning": getattr(result, fields["warning"]).get(m),
                    "table": sorted(getattr(result, fields["importance"]).get(m, {}).items(),
                                    key=lambda kv: kv[1], reverse=True),
                }
                for game, fields in HP_GAME_FIELDS.items()
            },
            # The partial-dependence figure's hyperparameter picker defaults
            # to this metric's own top-tunability hyperparameter (falling
            # back to the first one in config order if importance isn't
            # available) — the same "most interesting thing first" instinct
            # `best_idx` already applies to which trial ablation explains by
            # default.
            "top_hp": (
                sorted(result.hyperparameter_importance.get(m, {}).items(),
                       key=lambda kv: kv[1], reverse=True)[0][0]
                if result.hyperparameter_importance.get(m)
                else (list(best.config.keys())[0] if best.config else "")
            ),
        })

    # Plots, keyed by figure, built straight off the catalog. Only the metric
    # the page opens on: building all of them was 72 `plot()` calls and 101ms
    # of a 151ms render on a 4-metric run, for six payloads that end up on
    # screen — the rest sat in the page as ~300KB of JSON (over a megabyte at
    # 500 trials) against the chance the metric selector was touched. The
    # browser fetches a metric the first time it is selected and keeps it, the
    # same lazy-fetch-and-cache shape partial dependence and local ablation
    # already use; see `metric_figures` and experiment_detail.html's `show`.
    # Metric-independent figures are still drawn once, here.
    opening_metric = context["run_default_metric"]
    config_space = _config_space_for(built)
    metric_plots = {
        opening_metric: _figure_plots(
            result, [f for f in figures if f.per_metric], opening_metric,
            config_space=config_space),
    } if opening_metric else {}
    static_plots = _figure_plots(result, [f for f in figures if not f.per_metric],
                                 config_space=config_space)
    static_plots = {key: plot for key, plot in static_plots.items() if plot is not None}

    hp_names = list(result.trials[0].config.keys()) if result.trials else []
    # The Run form's target starts at what there is to beat. Per metric, so the
    # page can follow the metric dropdown; the one for the metric the form opens
    # on is what the field is filled with.
    incumbent_targets = {p["metric"]: surpass_target(p["best_score"]) for p in panels}
    context.update(
        result=result,
        panels=panels,
        metric_plots=metric_plots,
        static_plots=static_plots,
        # Each figure's declared display behavior, for the page script.
        figure_options={f.key: _figure_options(f, result, metric_names)
                        for f in figures},
        # Which figures have alternate views, and what they're called — only
        # for those, so the script can tell a plain payload from one keyed by
        # view without guessing from its shape.
        figure_views={f.key: list(f.views) for f in figures if f.views},
        # Three lists, all in catalog order: the grid lays out the first,
        # reading each figure's own width to decide whether it tiles or spans;
        # the second gets a column beside it once the window is wide enough
        # (Figure.in_side_column); the sidebar renders the third
        # (Figure.in_sidebar).
        figures=figures,
        grid_figures=[f for f in figures
                      if not f.in_sidebar and not f.in_side_column],
        column_figures=[f for f in figures if f.in_side_column],
        sidebar_figures=[f for f in figures if f.in_sidebar],
        # Which of them take a click naming a trial and show which one is
        # selected (Figure.selects_trials), so the page's selection bus iterates
        # a declaration instead of naming figures.
        selectable_figures=[f.key for f in figures if f.selects_trials],
        # And which of them a running run can redraw from its trials alone
        # (Figure.live) — the same declaration `run_status` builds payloads for,
        # so the page and the poll cannot disagree about which figures move.
        live_figures=[f.key for f in figures if f.live],
        # The two colours the highlight is drawn in, from the builders that draw
        # everything else in them — the script used to restate them, which is
        # one place for the page and the plots to disagree about what "selected"
        # looks like.
        # `failure` for the one place the page has to draw a failure mark
        # itself: a 3D scene draws no marker outline, so the red cannot ride on
        # the glyph the way it does everywhere else. Sent rather than repeated
        # in the script, so the page restates no colour — see NEGATIVE_COLOR.
        selection_colors={"base": MARKER_COLOR, "selected": SELECTION_COLOR,
                          "failure": NEGATIVE_COLOR},
        # The one game selector, and what each game asks — resolved here rather
        # than in the template because the labels are lazy translations and
        # `json_script` cannot serialize those.
        explanation_games=[(game, str(label)) for game, label in HP_GAME_LABELS.items()],
        explanation_game_help={game: str(text) for game, text in HP_GAME_HELP.items()},
        hp_names=hp_names,
        # Whether the importance figure offers "still to gain" at all. The
        # server already returns nothing for it when off, which would leave the
        # box and the table's three columns there offering an answer that never
        # arrives, so the markup goes too.
        tuning_progress=TUNING_PROGRESS_ENABLED,
        incumbent_targets=incumbent_targets,
        incumbent_target=incumbent_targets.get(context["run_default_metric"], ""),
        trial_rows=_trial_rows(result, metric_names, hp_names),
        # The table's own page size, so its field and the script that reads it
        # start from the figure's number rather than each restating one.
        trials_page_size=FIGURES_BY_KEY["trials"].page_size,
        trials_exhausted=(
            result.trials_limit is not None
            and len(result.trials) >= result.trials_limit
        ),
        # Which on-request analytics may fetch themselves as soon as their figure
        # needs them, and which wait for a button. Keyed by the computation's name
        # (see Figure.deferred), for the page script.
        autocompute={name: shown[autocompute_key(name)]
                     for name, _label in deferred_computations()},
    )
    return context
