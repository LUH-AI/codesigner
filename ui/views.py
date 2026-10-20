import copy
import json
import logging
import statistics
import tempfile
import threading
from collections import OrderedDict
from dataclasses import replace
from importlib.metadata import version as dist_version
from pathlib import Path

import numpy as np

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_not_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db import transaction
from django.http import Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone
from django.utils.http import content_disposition_header, url_has_allowed_host_and_scheme
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy, ngettext
from django.views.decorators.http import require_POST

from access.policy import (
    DELETE, EDIT, EXPORT, RENAME, RUN, SHARE, VIEW, groups_to_choose_from, policy,
    visible_experiments,
)
from core import io, projected_prior, provenance, smac_import, tasks
from core import priors as prior_math
from core.metrics import metric_for, metrics_for
from core.modelhost import deadline
from core.optimizers.base import FAILURE_CRITERIA
from experiments.models import Experiment, ExperimentData, ExperimentShare, GlobalSettings
from experiments.services import bin as bin_service
from experiments.services import history, model_export, modelenv, processing, storage, timestamps
from experiments.services import priors as prior_service
from experiments.services import run as run_service
from experiments.services import snapshot as snapshot_adapter
from experiments.services.run import resolve_seed
from experiments.services.snapshot import config_space_for, rebuild_experiment, rebuild_result
from experiments.services.run_logic import decide_run, resolve_metric_change
from experiments.services.settings import SETTING_BOUNDS, SETTING_CHOICES

from . import optimizer_form, projected, times
from .experiment_settings import SETTING_DEFAULTS, global_defaults, resolve_settings
from .figures import (
    FIGURES, FIGURES_BY_KEY, HP_GAME_FIELDS, HP_GAME_HELP, HP_GAME_LABELS,
    ACCENT_COLOR, MARKER_COLOR, NEGATIVE_COLOR, PRIOR_COLOR, PROPOSAL_COLOR, SELECTION_COLOR,
    UNCERTAINTY_SCALE,
    autocompute_key,
    deferred_computations,
    acquisition_slice_plots,
    hyperparameter_ablation_plot, hyperparameter_importance_plot,
    forecast_plot, hyperparameter_progress_plot, local_effects_plot, partial_dependence_plot,
    slice_span,
)
from .figures.data_handling import BY_SLUG, SECTIONS, ProcessingStep
from .formatting import human, sigfigs
from .forms import (
    EVALUATION_SCHEMES, DefaultExperimentSettingsForm, ExperimentSettingsForm,
    NewExperimentForm)
from .layout import data_tabs
from .optimizer_labels import describe_all, grouped
from .permissions import BOTH, experiment_view
from .registry import METRICS, MODELS, OPTIMIZERS
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
    if exp.data.model_name in MODELS:
        return True
    if not (exp.data.model_file and settings.ALLOW_CUSTOM_MODELS):
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


def _group_by_pk(groups, pk):
    return next((g for g in groups if g.pk == pk), None)


def _chosen_group(request, context):
    """The group an import goes in, or an error, for the pages with no Form.

    `(group, None)` when it was chosen, or there was nothing to choose — then
    `Experiment.save` files it in the one group there is. `(None, message)`
    when a choice was needed and none of this person's groups was given.
    """
    groups = groups_to_choose_from(request.user)
    context["groups"] = groups
    if not groups:
        return None, None
    try:
        group = _group_by_pk(groups, int(request.POST.get("group", "")))
    except ValueError:
        group = None
    if group is None:
        return None, _("Choose which group this experiment is in.")
    return group, None


def _ownership(request, exp):
    """Who this experiment belongs to, or None when the instance has no
    accounts and there is nothing to say."""
    if not settings.REQUIRE_LOGIN:
        return None
    from access.policy import membership_in, share_of
    # Where sharing is done, for somebody who can open it — a member of the
    # experiment's group. Anybody else would be linked to a 404.
    in_group = membership_in(request.user, exp.group_id) is not None
    return {"owner": exp.owner, "mine": exp.owner_id == request.user.pk,
            "shared_with": exp.shares.count(),
            "level": share_of(request.user, exp),
            "group": exp.group if in_group else None,
            "group_url": (reverse("ui:group_detail", args=[exp.group_id])
                          + f"#exp-{exp.pk}") if in_group else None}


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
    "max_trials": gettext_lazy("all the requested trials ran"),
    "target_score": gettext_lazy("performance surpassed the target"),
    "max_seconds": gettext_lazy("the time limit was reached"),
    "max_trial_seconds": gettext_lazy("the compute budget was used up"),
    "no_improvement_trials": gettext_lazy("the score had stopped improving"),
    "incumbent_confidence": gettext_lazy("the search was confident nothing better remained"),
    "cancelled": gettext_lazy("it was interrupted"),
    "all_failing": gettext_lazy("too many trials were failing"),
    "exhausted": gettext_lazy("every configuration had been tried"),
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


def _optimizer_for(name):
    """The registry optimizer an experiment names, or None. Aliases included, so
    an experiment stored under an optimizer's older name still resolves."""
    return OPTIMIZERS.get(name) or next(
        (o for o in OPTIMIZERS.values()
         if o.name == name or name in getattr(o, "aliases", ())), None)


def _judges_priors(exp):
    """Whether the experiment's optimizer, as configured now, can judge a
    prior — which decides whether "Evaluate Prior" is offered."""
    optimizer = _optimizer_for(exp.data.optimizer_name)
    return bool(optimizer) and optimizer.judges_priors(exp.data.optimizer_params or {})


def _prior_judging_context(exp):
    """What the prior figure's "Evaluate Prior" controls open on."""
    return {"judges_priors": _judges_priors(exp),
            "prior_top_k": resolve_settings(exp)["prior_acceptance_top_k"],
            "prior_top_k_bounds": SETTING_BOUNDS["prior_acceptance_top_k"]}


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
    (`visible_experiments`), so this never shows an experiment the
    sidebar wouldn't and a page would then refuse to open.
    """
    paginator = Paginator(visible_experiments(request), EXPERIMENT_LIST_PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "ui/experiment_list.html", {"page": page})


@login_not_required
def healthz(request):
    """Liveness probe for the container healthcheck — no DB, no template.

    Exempt from the login wall: the container runtime has no session, and a
    healthcheck that 302s to a login page reports a healthy instance as down.
    """
    return HttpResponse("ok", content_type="text/plain")


def _settle_storage(request, exp) -> bool:
    """Make room in *exp*'s group for what creating it stored, or undo it.

    Datasets and model files come first: kept trial models are deleted, the
    stalest first, until the group is within its limit, and the person is told
    how many. Only when the datasets and model files alone are over is the
    experiment refused — deleted again, files and all. See
    experiments/services/storage.py.
    """
    group = exp.group
    if group is None or (not exp.data.dataset and not exp.data.model_file):
        return True
    within, deleted = storage.make_room(group)
    now = storage.usage(group)
    numbers = {"group": group.name, "used": human(now["total"]),
               "limit": human(now["limit"])}
    if not within:
        storage.delete_experiment_files(exp)
        messages.error(request, _(
            "%(group)s has no room for this: its datasets and model files alone would take "
            "%(used)s of its %(limit)s. Delete experiments it no longer needs, or ask a site "
            "administrator for more.") % numbers)
        return False
    if deleted:
        messages.warning(request, ngettext(
            "To make room in %(group)s, %(n)s kept trial model was deleted, the stalest "
            "first. It now uses %(used)s of %(limit)s.",
            "To make room in %(group)s, %(n)s kept trial models were deleted, the stalest "
            "first. It now uses %(used)s of %(limit)s.", deleted) % dict(numbers, n=deleted))
    return True


def _dataset_path_from(form, tmp_paths):
    """Resolve the chosen dataset to a filesystem path (temp file for uploads)."""
    demo = form.cleaned_data.get("demo_dataset")
    if demo:
        return demo
    upload = form.cleaned_data["dataset_file"]
    # Under its own name, in a directory of its own: the stored copy, the
    # file and everything exported from it are called what the person
    # uploaded, not what a temporary file happened to be called.
    folder = Path(tempfile.mkdtemp(prefix="codesigner-upload-"))
    path = folder / (Path(upload.name).name or "dataset.csv")
    with open(path, "wb") as out:
        for chunk in upload.chunks():
            out.write(chunk)
    tmp_paths.append(str(path))
    return str(path)


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
    """New: make a draft, and set it up (ui/setup_steps.py).

    A draft at once rather than a form first, so the experiment is in
    Experiment Selection from the moment New is pressed and can be left and
    picked up again. POST, because it creates something; a GET — a link, a
    bookmark — gets a page with the one button.
    """
    if request.method != "POST":
        return render(request, "ui/new_experiment.html")
    exp = Experiment.objects.create(
        name="", model_name="", optimizer_name="",
        metric_names=[], seed=0, owner=_owner(request), draft=True)
    return redirect("ui:setup", pk=exp.pk)


@experiment_view(VIEW)
def experiment_detail(request, exp):
    """Show a saved experiment: its config, results/figures, run state, Run form."""
    return _dashboard(request, exp, _detail_context(request, exp))


def _dashboard(request, exp, context):
    """The dashboard: its figures once there is a trial; before, the page for
    what can still be decided — the optimizer and the prior."""
    if context["trial_count"]:
        return render(request, "ui/experiment_detail.html", context)
    fixed = optimizer_form.why_fixed(exp)
    may_edit = policy().may(request, exp, EDIT)
    context.update(optimizer_form.context(exp, locked=bool(fixed) or not may_edit))
    context["optimizer_locked_reason"] = fixed
    return render(request, "ui/experiment_prerun.html", context)


@require_POST
@experiment_view(RENAME)
def experiment_rename(request, exp):
    """Give the experiment a name, or take it away: an empty one shows as
    "Untitled Experiment". Back to the page it was renamed from."""
    exp.data.name = (request.POST.get("name") or "").strip()[:200]
    exp.data.save(update_fields=["name"])
    back = request.POST.get("next")
    if back and url_has_allowed_host_and_scheme(back, allowed_hosts={request.get_host()},
                                                require_https=request.is_secure()):
        return redirect(back)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(EDIT)
def experiment_optimizer(request, exp):
    """Save the optimizer and its settings, before the experiment's first
    trial."""
    refusal = optimizer_form.save(request, exp)
    if refusal:
        messages.error(request, refusal)
    else:
        messages.success(request, _("Saved."))
    return redirect("ui:experiment_detail", pk=exp.pk)


@experiment_view(VIEW)
def trial_panel(request, exp):
    """Render the selected-config panel for one (metric, trial index).

    Backs click-to-select on the performance figure (curve 0 only, point index
    into result.trials) — the browser fetches this fragment and swaps it into
    the panel for the metric currently being viewed.
    """
    metric = request.GET.get("metric", "")
    idx_raw = request.GET.get("idx", "")

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    result = rebuild_result(exp)
    idx = int(idx_raw)
    if result is None or not (0 <= idx < len(result.trials)):
        return HttpResponseBadRequest("invalid trial index")

    return render(request, "ui/_selected_config_inner.html",
                  {"sel": _selected_panel_data(result, metric, idx),
                   # The traceback link needs to know which experiment it is
                   # asking about; on the first render the page context has it,
                   # and this partial is also served on its own.
                   "experiment": exp})


def _proposals(exp, metric):
    """The sampled configurations a proposal is picked from, and what the
    surrogate predicts of each for *metric* — the configuration cube's point
    cloud, out of the same cache. `(built, sample)`, or `(built, None)` when
    there is no sample to pick from."""
    built, _config_space, sample = _cube_sample(exp, metric)
    return built, sample


def _cube_sample(exp, metric):
    """`(built, config_space, sample)`: the configuration cube's shared sample
    for *metric*, with what it was drawn from — any of the last two None when
    there is nothing to draw from."""
    built = rebuild_experiment(exp)
    if not built or built["result"] is None or not built["result"].trials:
        return built, None, None
    config_space = config_space_for(built)
    if config_space is None:
        return built, None, None
    _surrogate, sample = _slice_model(exp, built, config_space, metric)
    return built, config_space, sample if sample and sample.get("configurations") else None


# ── priors drawn in the configuration cube ──────────────────────────────────

def _projected_exponents(exp, records, trials, inside_design):
    """How strongly each stated projected prior still counts — the same decay
    every prior has, see `experiments.services.priors.decayed`."""
    return [(prior_service.decayed(exp, record, trials, inside_design) or {}).get("exponent", 1.0)
            for record in records]


def _projected_weights(exp, built, config_space, X):
    """Every stated projected prior's weight at the vectorized rows *X*, as the
    page multiplies it in — rounded for the wire, or None with nothing stated."""
    records = list(exp.data.projected_priors or [])
    if not records or config_space is None or X is None or not len(X):
        return None
    result = built["result"]
    exponents = _projected_exponents(exp, records, len(result.trials),
                                     prior_math.inside_initial_design(result))
    weights = projected.stated_weights(records, exponents, config_space, np.asarray(X, float))
    return projected.rounded(weights) if weights is not None else None


def _projected_answer(exp, built, config_space, sample):
    """What the card and the cube are told after a projected prior changes: the
    stated priors as the card lists them, and their weight over the sample."""
    return {"stated": [projected.summary(r, config_space) for r in exp.data.projected_priors or []],
            "weights": _projected_weights(exp, built, config_space,
                                          projected.sample_vectors(sample) if sample else None)}


def _drawn(request):
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


@require_POST
@experiment_view(VIEW)
def projected_prior_preview(request, exp):
    """A prior being drawn in the configuration cube, before it is stated: the
    record it would be, its weight over the cube's sample, and the shapes the
    cube draws for it. VIEW: looking at a belief changes nothing."""
    body = _drawn(request)
    if body is None or body.get("metric") not in exp.data.metric_names:
        return HttpResponseBadRequest("invalid request")
    built, config_space, sample = _cube_sample(exp, body["metric"])
    if sample is None:
        return JsonResponse({"error": _("There is no sample of this space to draw on.")}, status=400)
    try:
        record, Z = projected.draft_record(config_space, sample, body.get("draft") or {},
                                           body["metric"])
    except (projected_prior.PriorError, KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error), "point": getattr(error, "point", None)},
                            status=400)
    return JsonResponse({
        "record": {k: v for k, v in record.items() if k != "map"},
        "weights": projected.rounded(projected_prior.density(Z, record)),
        "geometry": projected.geometry(record, Z, config_space),
        "profile": projected_prior.profile(record),
    })


@require_POST
@experiment_view(EDIT)
def state_projected_prior(request, exp):
    """State the prior drawn in the cube, or change a stated one's parameters
    and decay. Its points and projection are what it was drawn in and stay put:
    a different shape is a different belief."""
    body = _drawn(request)
    if body is None or body.get("metric") not in exp.data.metric_names:
        return HttpResponseBadRequest("invalid request")
    built, config_space, sample = _cube_sample(exp, body["metric"])
    if sample is None:
        return JsonResponse({"error": _("There is no sample of this space to draw on.")}, status=400)
    result = built["result"]
    records = list(exp.data.projected_priors or [])
    try:
        if body.get("id"):
            at = next((i for i, r in enumerate(records) if r.get("id") == body["id"]), None)
            if at is None:
                return HttpResponseBadRequest("no such prior")
            old = records[at]
            new = projected.with_params(old, body.get("params") or {})
        else:
            old = None
            new, _Z = projected.draft_record(config_space, sample, body.get("draft") or {},
                                             body["metric"])
            new["id"] = projected.new_id()
            new["at_trial"] = len(result.trials)
        new["decay"] = prior_math.decay_record(
            body.get("decay"), prior_service.run_budget(exp, int(new.get("at_trial") or 0)))
        new["delay_decay"] = bool(body.get("delay_decay") and prior_math.inside_initial_design(result))
        projected_prior.check(new, config_space)
    except (projected_prior.PriorError, KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)

    if old is None:
        records.append(new)
    else:
        records[at] = new
    exp.data.projected_priors = records
    exp.data.save(update_fields=["projected_priors"])
    history.record_projected_prior(exp, new["id"], old, new, user=request.user)
    return JsonResponse(_projected_answer(exp, built, config_space, sample))


@require_POST
@experiment_view(EDIT)
def withdraw_projected_prior(request, exp):
    """Take a stated projected prior back."""
    body = _drawn(request)
    if body is None or body.get("metric") not in exp.data.metric_names:
        return HttpResponseBadRequest("invalid request")
    records = list(exp.data.projected_priors or [])
    old = next((r for r in records if r.get("id") == body.get("id")), None)
    if old is None:
        return HttpResponseBadRequest("no such prior")
    exp.data.projected_priors = [r for r in records if r is not old]
    exp.data.save(update_fields=["projected_priors"])
    history.record_projected_prior(exp, old["id"], old, None, user=request.user)
    built, config_space, sample = _cube_sample(exp, body["metric"])
    return JsonResponse(_projected_answer(exp, built, config_space, sample))


def _proposal_index(source):
    """`n` from *source* as an index into the sample, or None."""
    raw = source.get("n", "")
    return int(raw) if raw.isdigit() else None


def _run_refusal(request, exp):
    """Why this viewer cannot start a run of *exp* now, or None."""
    if not policy().may(request, exp, RUN):
        return _("You can view this experiment but not run it.")
    if exp.is_running:
        return _("A run is under way.")
    if not exp.data.has_dataset or not _model_available(exp):
        return _("This experiment cannot be run here.")
    return policy().custom_model_refusal(exp, request.user) or None


@experiment_view(VIEW)
def proposal_panel(request, exp):
    """The selected-config panel for a proposal: one of the configuration
    cube's sampled configurations, picked off its point cloud.

    A configuration nobody has run, so what it shows is the surrogate's
    prediction for it rather than a score, and it offers to run it.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    n = _proposal_index(request.GET)
    built, sample = _proposals(exp, metric)
    if sample is None or n is None or n >= len(sample["configurations"]):
        return HttpResponseBadRequest("invalid proposal")

    result = built["result"]
    scoring = result.metric(metric)
    best = result.trials[result.best_index(metric)].scores[metric]
    mu, sigma = sample["mu"][n], sample["sigma"][n]
    return render(request, "ui/_selected_proposal_inner.html", {
        "proposal": {
            "metric": metric,
            "metric_title": scoring.title,
            "number": n + 1,
            "of": len(sample["configurations"]),
            "index": n,
            "mu": mu,
            "sigma": sigma,
            "delta": None if mu is None else mu - best,
            "delta_better": mu is not None and scoring.better(mu, best),
            "config": list(sample["configurations"][n].items()),
        },
        "experiment": exp,
        "run_refusal": _run_refusal(request, exp),
    })


@require_POST
@experiment_view(RUN)
def run_proposal(request, exp):
    """Run one proposal as a trial: a run of one, told which configuration to
    evaluate rather than asked to search for one, optimizing the metric the
    experiment already optimizes so that nothing else about the search moves.
    The run is recorded like any other, with the proposal it ran."""
    metric = request.POST.get("metric", "")
    n = _proposal_index(request.POST)
    if metric not in exp.data.metric_names or n is None or _run_refusal(request, exp):
        return redirect("ui:experiment_detail", pk=exp.pk)
    _built, sample = _proposals(exp, metric)
    if sample is None or n >= len(sample["configurations"]):
        return redirect("ui:experiment_detail", pk=exp.pk)

    number = n + 1
    run = run_service.create_run(
        exp, {"max_trials": 1}, exp.data.current_metric or metric,
        started_by=_owner(request),
        configurations=[{"config": dict(sample["configurations"][n]),
                         "origin": _("Proposal %(n)s") % {"n": number},
                         "proposal": number}])
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


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

    result = rebuild_result(exp)
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

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    built = rebuild_experiment(exp)
    idx = int(idx_raw)
    if (built is None or built["result"] is None
            or not (0 <= idx < len(built["result"].trials))):
        return HttpResponseBadRequest("invalid trial index")
    try:
        baseline = _baseline_index(request, built["result"])
    except ValueError:
        return HttpResponseBadRequest("invalid baseline")

    figure, warning, ablation = _local_ablation_data(built, metric, idx, baseline)
    # What a trial has banked is a gain over the default, and so is what
    # tunability says was there to gain, so the two only subtract from there.
    rows, settled = (_tuning_progress(built["result"], metric, ablation)
                     if baseline is None else ([], False))
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

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    if (built is None or built["result"] is None or not built["result"].trials
            or hp_name not in built["result"].trials[0].config):
        return HttpResponseBadRequest("invalid hyperparameter")

    figure, warning = _partial_dependence_data(
        built, metric, hp_name, resolve_settings(exp)["ice_max_curves"])
    return JsonResponse({"figure": figure, "warning": warning})


@experiment_view(VIEW)
def forecast_figure(request, exp):
    """The Forecast figure for one trial: the series, and what the trial's
    configuration forecast at every backtest.

    Deferred like partial dependence, and for a stronger reason — it runs the
    model, once per backtest, as the trial did. *idx* names the trial (the
    selected one on the page), *series* which series to draw when there are
    several.
    """
    metric = request.GET.get("metric", "")
    if not exp.data.forecasts or metric not in exp.data.metric_names:
        return HttpResponseBadRequest("not a forecast")
    result = rebuild_result(exp)
    try:
        idx = int(request.GET.get("idx", ""))
        trial = result.trials[idx]
    except (TypeError, ValueError, IndexError, AttributeError):
        return HttpResponseBadRequest("unknown trial")
    if exp.data.model_file:
        return JsonResponse({"figure": None, "series": [], "warning": _(
            "The forecast is drawn for built-in models only: an uploaded model runs in "
            "its own environment, which this page does not start.")})
    try:
        built = io.build_experiment(snapshot_adapter.snapshot_from_experiment(exp),
                                    METRICS, MODELS, OPTIMIZERS, read_only=False)[1]
    except ValueError as exc:
        return JsonResponse({"figure": None, "series": [], "warning": str(exc)})

    splits, model = built["splits"], built["model"]
    labels = exp.data.task == tasks.CLASSIFICATION
    names = sorted({str(s) for s in splits.series}) if splits.series is not None else []
    chosen = request.GET.get("series") or (names[0] if names else "")
    rows = (np.arange(len(splits.y)) if not names
            else np.flatnonzero(np.asarray(splits.series).astype(str) == chosen))
    backtests = []
    try:
        for train_idx, val_idx in splits.folds:
            predicted = np.asarray(model.fit_predict(trial.config, splits.X[train_idx],
                                                     splits.y[train_idx], splits.X[val_idx],
                                                     seed=built["seed"]),
                                   dtype=object if labels else float)
            mine = np.isin(val_idx, rows)
            backtests.append((built["times"][val_idx[mine]], predicted[mine]))
    except Exception as exc:  # noqa: BLE001 — the model's failure is the figure's message
        return JsonResponse({"figure": None, "series": names, "warning": _(
            "This configuration failed to forecast: %(error)s") % {"error": exc}})
    figure = forecast_plot(built["times"][rows], splits.y[rows], backtests,
                           target=_target_name(exp), series=chosen, labels=labels)
    return JsonResponse({"figure": _plot_json(figure), "series": names, "warning": None})


def _target_name(exp):
    """The dataset's target column, for the forecast's axis."""
    try:
        return io.target_of(exp.data.dataset_path())[0]
    except Exception:  # noqa: BLE001 — an axis title is not worth an error
        return ""


#: Fitted slice surrogates, keyed by everything they depend on.
#:
#: A slice's model is a function of the experiment, the metric and the trials —
#: **not** of which hyperparameter is on the axis. Without this, clicking through
#: a six-hyperparameter model paid six identical fits, measured at roughly 300ms
#: for a Gaussian process and 750ms for a random forest over fifteen trials.
#:
#: Process-local and small rather than Django's cache: a fitted GP or forest is
#: derived data no other process needs, cheap to rebuild, and awkward to pickle.
#: The trial count is in the key, which is what makes a live run invalidate
#: itself as results arrive. Both figures ask for the same key as the page
#: opens, so the second waits on the first's fit rather than fitting again —
#: one lock per key, so nothing else waits.
_SLICE_SURROGATES = OrderedDict()
_SLICE_SURROGATE_CACHE = 8
_SLICE_LOCKS = {}
_SLICE_LOCKS_GUARD = threading.Lock()


def _slice_model(exp, built, config_space, metric):
    """The fitted surrogate for this (experiment, metric, trials), and an even
    sample of the whole space through it — computed once, and read by both
    figures that need them.

    Returns `(surrogate, sample)`, the sample being
    `BaseOptimizer.sample_prediction_space`'s answer. Neither depends on which
    hyperparameter the acquisition figure has on its axis, nor on which view
    the configuration cube is in, nor on the priors — a prior weights the
    acquisition, which both figures compute in the browser from the sample's
    mean and spread. So one fit and one sample serve the acquisition figure's
    cloud (`_sample_cloud`) and the cube's prediction layer alike.

    `(None, None)` when a surrogate could not be fitted, which leaves the caller
    to fit and to report why — a failure is not cached, so the reader gets the
    reason rather than silence.
    """
    result = built["result"]
    key = (exp.pk, metric, len(result.trials), built["seed"])
    with _SLICE_LOCKS_GUARD:
        lock = _SLICE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        cached = _SLICE_SURROGATES.get(key)
        if cached is not None:
            _SLICE_SURROGATES.move_to_end(key)
            return cached

        optimizer = built["optimizer"]
        surrogate, _warning = optimizer.slice_surrogate(
            config_space, result.trials, metric, built["seed"])
        if surrogate is None:
            return None, None

        sample = optimizer.sample_prediction_space(
            config_space, result.trials, metric, seed=built["seed"], surrogate=surrogate)
        _SLICE_SURROGATES[key] = (surrogate, sample)
        while len(_SLICE_SURROGATES) > _SLICE_SURROGATE_CACHE:
            evicted, _ = _SLICE_SURROGATES.popitem(last=False)
            with _SLICE_LOCKS_GUARD:
                _SLICE_LOCKS.pop(evicted, None)
        return surrogate, sample


def _sample_cloud(sample):
    """The shared sample as the acquisition figure's cloud: `(rows, mu, sigma)`
    as arrays, with what could not be predicted or placed as NaN — see
    `compute_incumbent_slice`, which leaves those out per hyperparameter."""
    if not sample or not sample.get("vectors"):
        return None

    def array(values):
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    rows = np.array([[np.nan if v is None else v for v in row] for row in sample["vectors"]],
                    dtype=float)
    return rows, array(sample["mu"]), array(sample["sigma"])


@require_POST
@experiment_view(EDIT, drafts=BOTH)
def save_prior(request, exp):
    """Record what the reader believes about one hyperparameter.

    `EDIT` rather than `VIEW`: a prior is not a way of looking at a run, it is a
    statement that changes what the next one searches.

    The body carries the shape and its parameters *and* the density already
    evaluated on the grid. Only the second reaches SMAC — see
    `Experiment.priors` — and the first exists so the figure can put the
    controls back where they were left. The server does not evaluate a density
    itself, deliberately: two implementations of "a Normal with this σ" would
    have to agree forever, and the one in the browser is the one the reader saw.
    """
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("invalid json")

    hp_name = body.get("hp") or ""
    if not hp_name:
        return HttpResponseBadRequest("no hyperparameter named")

    priors = dict(exp.data.priors or {})
    prior = body.get("prior")
    if prior is None:
        # Uniform states nothing, so it is stored as nothing rather than as a
        # flat table the optimizer would multiply by for no reason.
        priors.pop(hp_name, None)
    else:
        # The trial this belief was stated at. A decay schedule is an exponent
        # over the trials *since* a prior was supplied, not since the run began
        # — a belief stated at trial 120 arrives at full strength rather than
        # inheriting the near-flat exponent an older one has decayed to.
        built = rebuild_experiment(exp)
        at_trial = len((built["result"].trials if built and built["result"] else []))
        previous = (exp.data.priors or {}).get(hp_name) or {}
        result_now = built["result"] if built else None
        priors[hp_name] = {
            "kind": prior.get("kind") or "uniform",
            "params": prior.get("params") or {},
            "decay": prior_math.decay_record(prior.get("decay"),
                                             prior_service.run_budget(exp, at_trial)),
            # Whether the decay waits for the initial design to finish. Only
            # ever true while there is a design still to finish: stated after
            # it, there is nothing to wait for and carrying the flag would
            # invite a later reader to think there was. The page offers the
            # choice in the same situation and defaults it on.
            "delay_decay": bool(prior.get("delay_decay")
                                and prior_math.inside_initial_design(result_now)),
            # Kept across an edit of the same shape: re-typing a sigma is not
            # restating the belief, and re-anchoring on every keystroke would
            # hold the decay at zero forever.
            "at_trial": (previous.get("at_trial")
                         if previous.get("kind") == prior.get("kind") else at_trial),
        }
        # No `knots`. The browser used to evaluate the density and send it, so
        # that what SMAC weighted by was literally what had been drawn; now the
        # server evaluates it for both, which makes that guarantee structural
        # rather than something two implementations had to keep agreeing on.

    stated_before = (exp.data.priors or {}).get(hp_name)
    exp.data.priors = priors
    exp.data.save(update_fields=["priors"])
    history.record_prior(exp, hp_name, stated_before, priors.get(hp_name), user=request.user)

    # The density for what was just stated, so the debounced save doubles as the
    # redraw. One round trip, and the curve the reader ends up looking at is the
    # one the optimizer will weight by, because a single function produced both.
    density = None
    if body.get("meta"):
        density = prior_math.prior_densities(priors.get(hp_name), body["meta"])
    return JsonResponse({"saved": sorted(priors), "density": density})


@experiment_view(EDIT, drafts=BOTH)
@require_POST
def reset_prior(request, exp):
    """Put one hyperparameter's belief back to what the last run searched under.

    The undo for an afternoon of dragging. A prior on this figure is edited
    live — every drag saves — so there is no other way back to the statement the
    trials on screen actually came out of, and "what I had before I started
    fiddling" is the thing a reader wants and cannot otherwise reconstruct.

    To uniform when there is nothing to go back to: no run yet, or a last run
    that stated nothing for this hyperparameter. Uniform *is* stating nothing —
    it multiplies the acquisition by a constant, which cannot change a ranking —
    so it is stored as absence, the same as withdrawing one by hand.

    `EDIT`, like `save_prior`, and for the same reason: this changes what the
    next run searches.
    """
    hp_name = (request.POST.get("hp") or "").strip()
    if not hp_name:
        return HttpResponseBadRequest("no hyperparameter named")

    # The last run that got as far as recording anything. A pending one has
    # searched under nothing yet, and going back to its priors would be going
    # back to the very edits this is meant to undo.
    last = (exp.runs.exclude(status="pending")
            .order_by("-id").values_list("priors", flat=True).first())
    restored = (last or {}).get(hp_name)

    priors = dict(exp.data.priors or {})
    stated_before = priors.get(hp_name)
    if restored:
        priors[hp_name] = restored
    else:
        priors.pop(hp_name, None)
    exp.data.priors = priors
    exp.data.save(update_fields=["priors"])
    if stated_before != priors.get(hp_name):
        history.record(exp, "prior_reset", user=request.user, hyperparameter=hp_name,
                       old=stated_before, new=priors.get(hp_name))

    # Handed straight back, so the figure redraws from the same answer every
    # other request gives it rather than recomputing the decay in the browser.
    # The curve too, where the page said where it wants it — as a save does.
    # Without it the restored knots would be drawn over the curve of the belief
    # they replaced.
    result = rebuild_result(exp)
    density = None
    try:
        meta = json.loads(request.POST.get("meta") or "null")
    except ValueError:
        meta = None
    if isinstance(meta, dict):
        density = prior_math.prior_densities(priors.get(hp_name), meta)
    return JsonResponse({
        "restored": bool(restored),
        "prior": prior_service.decayed_prior(exp, hp_name,
                                len(result.trials) if result else 0,
                                prior_math.inside_initial_design(result)),
        "density": density,
    })


@experiment_view(RUN, drafts=BOTH)
@require_POST
def evaluate_prior(request, exp):
    """Ask this run's optimizer whether the stated belief is worth acting on.

    DynaBO's safeguard, on demand. It builds the real facade, replays the
    recorded trials and offers the prior to `add_prior` behind a
    `ClimbingComparisonPolicy` — so the answer is the one the run itself would
    give, made by SMAC rather than by a lookalike here.

    *top_k* may be posted with the request, from the field beside the button;
    clamped to the setting's bounds, and the setting itself when it is absent
    or not a number.

    The tolerance is a share of the metric's scale (`experiments.services.priors.metric_scale`), so one
    setting means the same for an accuracy and an RMSE; SMAC is handed it in
    the metric's own units, which is what its threshold is compared in.

    `RUN` rather than `VIEW`, for `experiment_compute_analytics`' reason: it
    costs a model fit, thousands of acquisition evaluations and a local search
    per climb, and a diagnostic that anybody who can read an experiment can
    make it spend seconds on is a diagnostic that reading an experiment pays
    for.

    Nothing is written either way. A refused belief stays exactly where the
    reader left it — whether to keep it is their decision, and the page asks.
    """
    hp_name = (request.POST.get("hp") or "").strip()
    metric = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    settings_now = resolve_settings(exp)
    low, high = SETTING_BOUNDS["prior_acceptance_tolerance"]
    tolerance = min(max(float(settings_now.get("prior_acceptance_tolerance", 0.15)), low), high)
    top_k = settings_now.get("prior_acceptance_top_k",
                             SETTING_DEFAULTS["prior_acceptance_top_k"])
    try:
        top_k = int(request.POST["top_k"])
    except (KeyError, ValueError):
        pass
    low, high = SETTING_BOUNDS["prior_acceptance_top_k"]
    top_k = min(max(top_k, low), high)

    def unjudged(message):
        """One shape for every answer, so the page has one thing to read."""
        return JsonResponse({"verdict": "unjudged", "message": message,
                             "tolerance": tolerance})

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    config_space = config_space_for(built)
    if result is None or not result.trials or config_space is None:
        return unjudged(_("Nothing has been run yet, so there is no model to "
                          "judge this against. A belief stated now is acted on "
                          "in full."))

    optimizer = built["optimizer"]
    if not hasattr(optimizer, "evaluate_prior"):
        return unjudged(_("This optimizer fits no model of the objective, so it "
                          "has no opinion about where the optimum is."))
    if not _judges_priors(exp):
        return unjudged(_("Needs the Gaussian process search strategy."))
    scale = prior_service.metric_scale(exp, metric, result)
    if scale is None:
        return unjudged(_("This metric has no scale to measure the tolerance against yet."))
    margin = float(tolerance) * scale

    outcome = optimizer.evaluate_prior(
        config_space, result.trials, metric, built["seed"],
        priors=exp.data.priors or None,
        projected_priors=exp.data.projected_priors or None,
        tolerance=margin,
        top_k=top_k,
        previous_result=result,
        # The same anchor the figure draws with, so the belief is judged at the
        # strength it is actually being applied at.
        budget=prior_service.run_budget(exp, int(((exp.data.priors or {}).get(hp_name) or {})
                                    .get("at_trial") or 0)))
    # The wording is made here, not in `core`, which has no Django and so no
    # catalog to translate against. What crosses from there is the verdict and a
    # reason for the log; what reaches the page is a sentence.
    verdict = outcome.get("verdict", "unjudged")
    # Said as a finding about the belief, because that is what a reader asked
    # for. What the model actually computes — a mean acquisition over samples
    # from the belief against samples from the incumbent's neighbourhood — is
    # the reason for the finding and not the finding itself, and leading with it
    # left the answer to the question buried at the end of a sentence about
    # regions and neighbourhoods.
    if verdict == "accepted":
        message = _("This prior is potentially informative: the model rates the "
                    "region it points at as promising as where the search is "
                    "already headed.")
    elif verdict == "rejected":
        message = _("This prior is unlikely to be helpful: the model rates the "
                    "region it points at more than %(margin)s worse than "
                    "where the search is already headed, so acting on it would "
                    "cost trials.") % {"margin": sigfigs(margin)}
    else:
        logger.info("A prior could not be judged: %s", outcome.get("reason", ""))
        message = _("This prior could not be judged, so it stands as stated.")
    return JsonResponse({"verdict": verdict, "message": message,
                         "tolerance": tolerance, "margin": margin, "top_k": top_k})


def _next_trials(exp, anchor_hp=None):
    """What this run's optimizer would ask for next, in its own rank order, as
    `(config_space, [(config, origin, walk), ...])` — or None where it cannot be
    asked. *walk* is the path its local search climbed to it, or None.

    Not a recomputation that resembles the search: `slice_challengers` builds the
    facade with `_facade`, replays the recorded trials, applies the stated
    priors and calls `ask()`. The first is the configuration the next trial
    would use.

    *anchor_hp* is the hyperparameter whose prior the budget is anchored at, so
    the walk weights by the same exponent that prior's figure is drawing. Left
    out, it is the most recently stated prior, or the run as it stands.
    """
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return None
    config_space = config_space_for(built)
    if config_space is None:
        return None

    metric = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    # Not every optimizer can be asked. Random and grid search have no
    # surrogate and no acquisition function, so there is no ranking to report —
    # and the reader gets told that rather than a 500.
    optimizer = built["optimizer"]
    if not hasattr(optimizer, "slice_challengers"):
        return None

    priors = exp.data.priors or {}
    if anchor_hp is not None:
        at_trial = (priors.get(anchor_hp) or {}).get("at_trial")
    else:
        anchors = [p.get("at_trial") for p in priors.values() if p and p.get("at_trial")]
        at_trial = max(anchors) if anchors else None
    asked = optimizer.slice_challengers(
        config_space, result.trials, metric, built["seed"],
        priors=priors or None, previous_result=result,
        projected_priors=exp.data.projected_priors or None,
        budget=prior_service.run_budget(exp, int(at_trial or len(result.trials))), with_walks=True)
    if not asked:
        return None
    configs, origins, walks = asked
    return config_space, list(zip(configs, origins, walks))


def _candidate(rank, config, origin, walk=None, config_space=None):
    """One proposed configuration as the page draws it: its rank, where it came
    from, every value, the whole of it pre-formatted for a hover — a point
    differs from the incumbent in every coordinate, not only the ones on show —
    and the path its local search climbed to get there, start first."""
    values = {k: _plain(v) for k, v in sorted(config.items())}
    return {"rank": rank, "origin": origin, "config": values,
            "label": "<br>".join(f"{k} = {sigfigs(v)}" for k, v in values.items()),
            # Where it sits on every hyperparameter's vectorized axis, so either
            # figure can place it on whichever axis it is drawing — the
            # acquisition figure's rug as well as the cube.
            "positions": _vector_positions(config_space, config),
            "walk": ([{k: _plain(v) for k, v in step.items()} for step in walk]
                     if walk else None)}


def _vector_positions(config_space, config):
    """*config* in ConfigSpace's vectorized form, keyed by hyperparameter, or
    None when there is no space or it will not encode. An inactive
    hyperparameter has no position and is left out."""
    if config_space is None:
        return None
    from ConfigSpace import Configuration

    try:
        vector = Configuration(config_space, values=config).get_array()
    except Exception:  # noqa: BLE001 — a candidate that will not encode has no place
        return None
    return {name: float(v) for name, v in zip(config_space.keys(), vector) if np.isfinite(v)}


@experiment_view(VIEW)
def next_trials(request, exp):
    """What this run's optimizer would run next, as whole configurations, under
    the priors as saved. Both the acquisition figure and the configuration cube
    draw the answer, each placing it on its own axes. *hp*, when given, is the
    hyperparameter whose prior the trial budget is anchored at — see
    `_next_trials`. Not cached: it rebuilds and asks the optimizer, which takes
    a few seconds, and only when someone asks for it.
    """
    hp_name = request.GET.get("hp") or None
    asked = _next_trials(exp, anchor_hp=hp_name)
    if asked is None:
        return JsonResponse({"candidates": None})
    config_space, proposals = asked
    return JsonResponse({"candidates": [_candidate(rank, config, origin, walk, config_space)
                                        for rank, (config, origin, walk)
                                        in enumerate(proposals, start=1)]})


def _plain(value):
    """A configuration value as JSON sees it — numpy scalars are not."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


#: Figures built but not sent. The surrogate panel is switched off — see the
#: acquisition template — and its traces are a few thousand points per request
#: that nothing draws. Still *built*, because the model behind it is fitted
#: either way (the acquisition curve is computed from the same mean and spread)
#: and because the figure's own behaviour is still under test; only the sending
#: is skipped. Putting the panel back is this set and the template's div.
_NOT_SHIPPED = frozenset({"surrogate"})


def _slice_figures(exp, built, config_space, hp_name, metric):
    """The three slice figures, their `meta`, and the slice they came from.

    Split out of the view so that what is *built* and what is *shipped* are two
    decisions in two places. The surrogate figure is no longer sent, and a test
    that asks whether its spread is stepped or its ghost band faint is asking
    about the machinery rather than about the payload — so it asks here.

    Returns `(None, None, sliced)` when the slice has no positions, which is the
    one case the caller answers differently.
    """
    surrogate, sample = _slice_model(exp, built, config_space, metric)
    cloud = _sample_cloud(sample)
    sliced = built["optimizer"].compute_incumbent_slice(
        config_space, built["result"].trials, metric, hp_name, seed=built["seed"],
        surrogate=surrogate, cloud=cloud)
    if not sliced.positions:
        return None, None, sliced

    figures, meta = acquisition_slice_plots(
        hp_name, sliced.positions, sliced.grid, sliced.mu, sliced.sigma, metric,
        eta=sliced.eta, incumbent=sliced.incumbent, cloud=sliced.cloud,
        higher_is_better=metric_for(metric).higher_is_better,
        kind="categorical" if hasattr(config_space[hp_name], "choices") else "continuous",
    )
    meta["acquisitionFunction"] = _acquisition_function(built["optimizer"])
    # Priors drawn in the cube weigh whole configurations, so they are
    # evaluated at the configurations this figure predicted at: along the
    # slice, and over its cloud.
    slice_weights = _projected_weights(exp, built, config_space, sliced.slice_rows)
    meta["projectedWeights"] = (None if slice_weights is None else {
        "slice": slice_weights,
        "cloud": _projected_weights(exp, built, config_space, sliced.cloud_rows) or []})
    return figures, meta, sliced


def _acquisition_function(optimizer):
    """Which acquisition function the page should compute, and its margin.
    Expected improvement for an optimizer that names none."""
    return optimizer.acquisition_function() or {"kind": "ei", "xi": 0.0}


@experiment_view(VIEW, drafts=BOTH)
def acquisition_slice(request, exp):
    """The acquisition-and-priors figure for one (metric, hyperparameter).

    Fetched per hyperparameter rather than shipped, for `partial_dependence`'s
    reason exactly — it fits a surrogate and predicts across a grid, and only
    one hyperparameter is ever on screen.

    The prior and acquisition traces come back empty; the browser fills them
    (`ui/static/ui/acquisition.js`). Everything it needs to do that rides in the
    figure's own `layout.meta`, so the response is the figure plus a warning,
    the same shape `partial_dependence` returns.
    """
    metric = request.GET.get("metric", "")
    hp_name = request.GET.get("hp", "")

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    config_space = config_space_for(built)
    if config_space is None:
        return JsonResponse({"figure": None, "warning": _(
            "The model this experiment used is not available here, so its "
            "search space cannot be rebuilt.")})

    # Checked against the search space rather than against a trial's
    # configuration. A prior is a statement about the space, and the space
    # exists before the search does — asked for on an experiment that has never
    # been run there is no trial to read a name off, and that is exactly when
    # somebody is most likely to be stating a belief.
    if hp_name not in config_space:
        return HttpResponseBadRequest("invalid hyperparameter")

    # Everything else on this figure is the surrogate's, and there is no
    # surrogate without trials. The prior panel is the one part that needs none.
    if (result is None or not result.trials) and not request.GET.get("prior_only"):
        return HttpResponseBadRequest("no result")

    # A prior is a statement about the search space, so the space is all it
    # takes to draw one. Asked for on its own, it skips the surrogate entirely —
    # which is what a reader waits on — so the panel they can actually edit is
    # there immediately, and stays there when "compute automatically" is off.
    if request.GET.get("prior_only"):
        grid, positions, kind = built["optimizer"].slice_axis(config_space, hp_name)
        if not positions:
            return JsonResponse({"figures": None, "meta": None, "warning": ""})
        zeros = [0.0] * len(positions)
        figures, meta = acquisition_slice_plots(
            hp_name, positions, grid, zeros, zeros, metric,
            eta=0.0, incumbent=None, cloud=None,
            higher_is_better=metric_for(metric).higher_is_better, kind=kind)
        stated = (exp.data.priors or {}).get(hp_name)
        return JsonResponse({
            # Only the panel that needed nothing. The other two are what the
            # surrogate is for, and claiming them here with zeros would draw a
            # flat acquisition function as though it were a finding.
            "figures": {"prior": _plot_json(figures["prior"])},
            "meta": {k: v for k, v in meta.items() if k != "cloud"},
            "insideDesign": prior_math.inside_initial_design(result),
            "warning": "",
            # Zero trials when there has never been a run, which is the honest
            # step count: nothing has happened to this belief yet.
            "prior": prior_service.decayed_prior(exp, hp_name,
                                    len(result.trials) if result else 0,
                                    prior_math.inside_initial_design(result)),
            "betaRatioDefault": prior_math.BETA_RATIO_DEFAULT,
            "betaRatiosAblated": list(prior_math.BETA_RATIOS_ABLATED),
            "density": {"grid": prior_math.prior_density(stated, positions,
                                               meta.get("span") or [0.0, 1.0]),
                        "cloud": None},
        })

    figures, meta, sliced = _slice_figures(exp, built, config_space, hp_name, metric)
    if figures is None:
        return JsonResponse({"figure": None, "warning": sliced.warning})

    # Three figures and one `meta`, rather than the meta repeated on each: it
    # carries the sampled cloud, which is two thousand points, and all three are
    # drawn from the same numbers anyway.
    return JsonResponse({"figures": {name: _plot_json(fig)
                                     for name, fig in figures.items()
                                     if name not in _NOT_SHIPPED},
                         "meta": meta,
                         "warning": sliced.warning or _weighting_warning(exp),
                         "initialDesign": prior_math.initial_design_notice(result),
                         "insideDesign": prior_math.inside_initial_design(result),
                         "prior": prior_service.decayed_prior(exp, hp_name, len(result.trials),
                                                 prior_math.inside_initial_design(result)),
                         # What the β field should open on when nothing is
                         # stated yet: DynaBO's own initialisation.
                         "betaRatioDefault": prior_math.BETA_RATIO_DEFAULT,
                         "betaRatiosAblated": list(prior_math.BETA_RATIOS_ABLATED),
                         # So the first draw needs no second request.
                         "density": prior_math.prior_densities(
                             (exp.data.priors or {}).get(hp_name), meta)})


def _weighting_warning(exp):
    """Told to the reader, not just the log, when a stated prior cannot bite.

    A prior on a SMAC without the acquisition weight layer is drawn, saved, and
    ignored by both the search and the candidate walk. Nothing on the page would
    say so otherwise, and a belief that silently does nothing is worse than one
    that is refused.
    """
    if not (exp.data.priors or {}):
        return None
    try:
        import smac.acquisition.weight  # noqa: F401
    except ImportError:
        return _("This SMAC has no acquisition weight layer, so a stated prior "
                 "does not affect the search or the candidates below. Install "
                 "the version requirements.txt pins.")
    return None


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
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")

    config_space = config_space_for(built)
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
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    if built is None or built["result"] is None or not built["result"].trials:
        return HttpResponseBadRequest("no result")
    try:
        baseline = _baseline_index(request, built["result"])
    except ValueError:
        return HttpResponseBadRequest("invalid baseline")

    figure, warning = _local_effects_data(
        built, metric, resolve_settings(exp)["local_effects_max_trials"], baseline)
    return JsonResponse({"figure": figure, "warning": warning})


def _baseline_index(request, result):
    """The `baseline` query parameter as a trial index, or None when absent.

    The trial every HyperSHAP explanation on the page is measured from, when it
    is not the config space's default. Raises ValueError for a value naming no
    trial, which the caller answers with a 400 like any other bad index.
    """
    raw = request.GET.get("baseline", "")
    if raw == "":
        return None
    if not raw.isdigit() or int(raw) >= len(result.trials):
        raise ValueError(raw)
    return int(raw)


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
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")

    per_metric = [f for f in _shown_figures(exp) if f.per_metric]
    return JsonResponse(_figure_plots(result, per_metric, metric,
                                      config_space=config_space_for(built)))


@experiment_view(VIEW)
def surrogate_prediction(request, exp):
    """What the surrogate predicts across the search space, for one metric: the
    configuration cube's prediction layer.

    One answer per metric, whichever view the cube is in. The sample, the
    predictions and every view's coordinates for it come back together, so
    switching view, dimensionality or axes redraws without asking again; only a
    different metric means a different model. See
    `BaseOptimizer.sample_prediction_space`.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")
    config_space = config_space_for(built)
    if config_space is None:
        return JsonResponse({
            "values": [], "warning": _("The model this experiment used is not "
                                       "available here, so its search space "
                                       "cannot be rebuilt."),
        })

    optimizer = built["optimizer"]
    # The same sample the acquisition figure's cloud is drawn from, out of the
    # same cache, so whichever figure asks second costs nothing.
    _surrogate, shared = _slice_model(exp, built, config_space, metric)
    answer = dict(shared) if shared else optimizer.sample_prediction_space(
        config_space, result.trials, metric, seed=built["seed"])
    # The page draws the rounded `values`; the exact ones stay here, for
    # `run_proposal`.
    answer.pop("configurations", None)
    # Only an optimizer that fits a model of the objective has an acquisition
    # function to speak of; the stand-in forest for the others predicts, and
    # that is all.
    answer["acquisition"] = bool(getattr(optimizer, "fits_surrogate", False))
    answer["acquisition_function"] = _acquisition_function(optimizer)
    # Every stated prior, as the weight the acquisition is multiplied by: its
    # density over the same grid the acquisition figure draws it on, and the
    # exponent its decay has reached. The page interpolates each sample's own
    # position from these, and swaps one in as the acquisition figure edits it.
    priors = {}
    inside = prior_math.inside_initial_design(result)
    for name in config_space.keys():
        decayed = prior_service.decayed_prior(exp, name, len(result.trials), inside)
        if not decayed or decayed.get("kind") in (None, "uniform"):
            continue
        _grid, positions, kind = optimizer.slice_axis(config_space, name)
        density = prior_math.prior_density(decayed, positions, slice_span(positions, kind))
        if density:
            priors[name] = {"positions": positions, "density": density,
                            "exponent": decayed["exponent"],
                            "categorical": kind == "categorical"}
    answer["priors"] = priors
    # Each numeric hyperparameter's range, which is HP Space's legal box: a
    # point drawn for a prior outside it is no configuration at all.
    answer["bounds"] = {name: [float(hp.lower), float(hp.upper)]
                        for name, hp in config_space.items()
                        if hasattr(hp, "lower") and getattr(hp, "choices", None) is None}
    # And every prior drawn in the cube itself, as one weight per sample.
    answer["projected_weights"] = _projected_weights(
        exp, built, config_space, projected.sample_vectors(answer) if answer.get("vectors") else None)
    return JsonResponse(answer)


@experiment_view(VIEW)
def hypershap_figures(request, exp):
    """Every HyperSHAP game figure for one metric, measured from one trial.

    The stored games are gains over the config space's default, worked out at
    run completion. This answers for the sidebar's other baselines, the
    incumbent and the selected trial, which depend on the metric and on the
    click and so cannot be stored ahead of time. All three games at once, so
    the game selector switches between them without asking again; and as the
    same plots and warnings the page draws the stored games from, so it draws
    these the same way. `importance` carries the shares themselves, for the
    importance table's columns, which set every baseline side by side.
    """
    metric = request.GET.get("metric", "")
    idx_raw = request.GET.get("idx", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.isdigit():
        return HttpResponseBadRequest("invalid trial index")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    idx = int(idx_raw)
    if result is None or idx >= len(result.trials):
        return HttpResponseBadRequest("invalid trial index")

    config_space = config_space_for(built)
    if config_space is None:
        reason = _("HyperSHAP needs the search space, which this experiment "
                   "does not carry and has no model to ask.")
        games = {game: ({metric: {}}, {metric: reason}, {metric: {}}, {metric: []}, {metric: 0.0})
                 for game in HP_GAME_FIELDS}
    else:
        optimizer = built["optimizer"]
        optimizer.analytics_max_coalitions = settings.ANALYTICS_EAGER_MAX_COALITIONS or None
        games = optimizer.compute_hp_games(
            config_space, result.trials, {metric: result.metric(metric)},
            seed=built["seed"], baseline=result.trials[idx].config)

    parts = ("importance", "warning", "interactions", "moebius")
    measured = replace(result, **{
        fields[part]: {metric: games[game][i][metric]}
        for game, fields in HP_GAME_FIELDS.items() for i, part in enumerate(parts)})
    figures = [f for f in _shown_figures(exp) if f.explains_games]
    return JsonResponse({
        "plots": _figure_plots(measured, figures, metric, config_space=config_space),
        "warnings": {game: games[game][1][metric] for game in HP_GAME_FIELDS},
        "importance": {game: games[game][0][metric] for game in HP_GAME_FIELDS},
    })


@experiment_view(RUN)
def experiment_run(request, exp):
    """Launch a background run of an experiment (or confirm a metric change).

    The Run form supplies the stopping criteria and the metric to optimize. A
    run that would change the optimized metric first shows a confirmation; the
    confirmation posts back with a `decision` of "new" or "old".
    """
    if (request.method != "POST" or not exp.data.has_dataset or exp.is_running
            or not _model_available(exp)):
        return redirect("ui:experiment_detail", pk=exp.pk)

    chosen = request.POST.get("optimize_metric")
    stopping = _posted_stopping(request, chosen or exp.data.current_metric or "")
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
        return _dashboard(request, exp, context)

    if decision:
        optimize_metric = resolve_metric_change(decision, exp.data.current_metric, chosen)
        if optimize_metric is None:
            return redirect("ui:experiment_detail", pk=exp.pk)
    else:
        action, optimize_metric = decide_run(exp.data.original_metric, exp.data.current_metric, chosen)
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

    Read straight off the JSON rather than through `rebuild_experiment`,
    because this is polled every two seconds and the answer is usually "the same
    as last time" — the whole point of asking is to decide whether it is worth
    doing any of the expensive work at all.
    """
    return len((exp.data.result or {}).get("data") or [])


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
    data = (exp.data.result or {}).get("data") or []
    recent = [entry.get("time") or 0.0 for entry in data[-10:]]
    typical = statistics.median(recent) if recent else 0.0
    return int(min(POLL_MAX_SECONDS, max(POLL_MIN_SECONDS, round(typical))))


def _best_panels(result, metric_names):
    """Each metric's best trial so far, as the best-configuration cards show
    it. A metric no trial has scored has no card — see `_detail_context`."""
    panels = []
    for m in metric_names:
        best_idx = result.best_index(m)
        if best_idx is None:
            continue
        best = result.trials[best_idx]
        panels.append({"metric": m, "best_n": best.trial, "best_idx": best_idx,
                       "best_score": best.scores[m],
                       "best_config": list(best.config.items())})
    return panels


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
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return {}

    shown = _shown_figures(exp)
    live = [f for f in shown if f.live]
    config_space = config_space_for(built)
    per_metric = [f for f in live if f.per_metric]
    static = [f for f in live if not f.per_metric]
    payload = {
        "metric_plots": {
            metric: _figure_plots(result, per_metric, metric, config_space=config_space)
            for metric in exp.data.metric_names
        } if per_metric else {},
        "static_plots": _figure_plots(result, static, None, config_space=config_space),
    }

    # The trials table is server-rendered HTML rather than a plot, so it cannot
    # ride in the payload maps above. It gets the rows it is missing instead —
    # rendered from the same partial the page built the table from, and appended
    # rather than swapped, so a sort, a page, a scroll position and a selected
    # row all survive an update that only adds to the end.
    # The best configuration is server-rendered too, and moves whenever a trial
    # beats it. Rendered whole, every metric's card, from the same partial the
    # page drew them from; the page swaps each card's contents in place, so the
    # one being viewed stays the one showing.
    if any(f.key == "best_configuration" for f in shown):
        payload["best_html"] = render_to_string(
            "ui/figures/best_configuration.html",
            {"panels": _best_panels(result, exp.data.metric_names),
             "figure": FIGURES_BY_KEY["best_configuration"]})

    if any(f.key == "trials" for f in shown) and len(result.trials) > since:
        payload["rows_html"] = render_to_string(
            "ui/figures/_trial_rows.html",
            {"trial_rows": _trial_rows(result, exp.data.metric_names,
                                       _hp_names(result), start=since)})
    return payload


@experiment_view(VIEW)
def run_status(request, exp):
    """HTMX poll target: the current run's status, or a refresh when finished.

    Also the channel for figures that move while the run is still going. There
    was never a rendering constraint on those — the trial-based figures need
    nothing but the trials — only a persistence one: the result was written once,
    at the end. `experiments/services/run.py` now writes it periodically, and this hands
    the page whatever has arrived since it last said what it had.

    No new endpoint and no second poll loop: this fragment is already fetched
    every two seconds and already replaced wholesale, so the payload rides along
    inside it. `?trials=` is the page saying how many trials it has drawn, which
    is what keeps the cost down — build the plots only when that disagrees with
    what is stored, which under a five-second write interval is at most every
    third poll.

    A page with no trials yet has no figure grid to draw into, so it is sent
    the count alone and reloads itself once — see `experiment_prerun.html`.
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
    (`experiments/services/run.py`) leaves them out for the same reason at finer grain —
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

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return redirect("ui:experiment_detail", pk=exp.pk)

    optimizer = built["optimizer"]
    optimizer.analytics_max_coalitions = settings.ANALYTICS_EAGER_MAX_COALITIONS or None
    games = optimizer.compute_hp_games(
        config_space_for(built), result.trials, built["metrics"], seed=built["seed"])

    for game, fields in HP_GAME_FIELDS.items():
        importance, warning, interactions, moebius, total = games[game]
        setattr(result, fields["importance"], importance)
        setattr(result, fields["warning"], warning)
        setattr(result, fields["interactions"], interactions)
        setattr(result, fields["moebius"], moebius)
    result.hyperparameter_interactions_warning = games["tunability"][1]
    result.hyperparameter_tunability_total = games["tunability"][4]

    ExperimentData.objects.filter(experiment__pk=exp.pk).update(
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
    if exp.data.model_file and not exp.is_running:
        modelenv.start_preparation(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _back_to_sharing(request, exp):
    """Wherever the sharing controls were — the group page, as a rule — or the
    experiment itself when the form did not say."""
    from .navigation import safe_next
    return redirect(safe_next(request) or reverse("ui:experiment_detail", args=[exp.pk]))


def _colleague(request, exp):
    """The account a sharing form names, from the experiment's own group.

    Looked up among the group's members and nowhere else, so an id from outside
    the group is simply not found — the same answer as one that does not exist.
    """
    from django.contrib.auth import get_user_model
    if not settings.REQUIRE_LOGIN or exp.group_id is None:
        raise Http404
    try:
        pk = int(request.POST.get("user", ""))
    except ValueError:
        raise Http404
    colleague = (get_user_model().objects
                 .filter(pk=pk, memberships__group_id=exp.group_id).first())
    if colleague is None or colleague.pk == exp.owner_id:
        raise Http404
    return colleague


@require_POST
@experiment_view(SHARE)
def experiment_share(request, exp):
    """Set one colleague's access to this experiment: none, viewer or contributor.

    One person at a time, and only somebody in the experiment's group. The
    owner's alone to decide — a contributor can do everything to the work except
    choose who else may.
    """
    colleague = _colleague(request, exp)
    level = request.POST.get("level", "")
    if level not in dict(ExperimentShare.LEVELS) and level != "none":
        return HttpResponseBadRequest(_("Unknown sharing level."))
    if level == "none":
        exp.shares.filter(user=colleague).delete()
    else:
        ExperimentShare.objects.update_or_create(
            experiment=exp, user=colleague,
            defaults={"level": level, "granted_by": request.user})
    return _back_to_sharing(request, exp)


@require_POST
@experiment_view(SHARE)
def experiment_transfer(request, exp):
    """Hand this experiment to one of its contributors.

    Only a contributor: somebody who can already run and delete it, so the one
    new power ownership gives them is deciding who else may. The previous owner
    is left a contributor, not shut out — handing over the work is not the same
    decision as walking away from it — and both happen together or not at all,
    so there is never a moment with no owner or with two.
    """
    colleague = _colleague(request, exp)
    grant = exp.shares.filter(user=colleague, level=ExperimentShare.CONTRIBUTOR).first()
    # The page offers this for contributors only, so anybody else is a request
    # it never made — not found, the same answer as a stranger.
    if grant is None:
        raise Http404
    previous = exp.owner
    with transaction.atomic():
        grant.delete()
        exp.owner = colleague
        exp.save(update_fields=["owner"])
        if previous is not None:
            ExperimentShare.objects.update_or_create(
                experiment=exp, user=previous,
                defaults={"level": ExperimentShare.CONTRIBUTOR,
                          "granted_by": colleague})
    messages.success(request, _("%(name)s now belongs to %(user)s. You are a "
                                "contributor.")
                     % {"name": exp.title, "user": colleague.get_username()})
    return _back_to_sharing(request, exp)


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
    exp.runs.filter(status__in=_ACTIVE, cancel_requested=False).update(
        cancel_requested=True, cancel_requested_at=timezone.now(),
        cancel_requested_by=_owner(request))
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
    run_service.fail_runs(stuck, _("Given up on by hand: it stopped answering, and "
                                   "cancelling it had no effect."))
    return redirect("ui:experiment_detail", pk=exp.pk)


def _abandon_cluster_job(job_id: str) -> None:
    """`scancel` a job whose run is being given up on.

    Best-effort and deliberately quiet. This runs inside a request, and the
    cluster being unreachable is a likely reason the run stopped answering in
    the first place — so failing here must not stop the row being freed, which
    is the thing the reader actually asked for. A job that outlives this is
    bounded by its own wall-clock limit.
    """
    from remote import jobs

    try:
        jobs.kill(jobs.SshTransport(settings.CLUSTER_HOST, timeout=15.0), job_id)
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
    return _export(request, exp)


@experiment_view(VIEW)
def experiment_timeline(request, exp):
    """Everything that changed how this experiment searches, newest first —
    see ui/timeline.py."""
    from . import timeline

    entries = timeline.build(exp)
    return render(request, "ui/experiment_timeline.html", {
        **experiment_header(request, exp),
        "scheduled": [e for e in entries if e["scheduled"]],
        "entries": [e for e in entries if not e["scheduled"]],
        "time_basis": exp.data.time_basis})


@experiment_view(EXPORT)
def experiment_export_model(request, exp):
    """Download the experiment's model as a runnable script, tuned to one trial.

    The trial is the one the page has selected, named the way every figure
    names it — `?metric=&idx=`, the index into the result's trials. Without
    them it is the best trial on the metric being viewed. See
    core/model_export.py for what the file holds.
    """
    reason = model_export.unavailable(exp)
    if reason:
        raise Http404(reason)
    picked = _picked_trial(request, exp)
    if isinstance(picked, HttpResponse):
        return picked
    result, trial, metric = picked
    body = model_export.build(exp, trial, metric, result)
    response = HttpResponse(body, content_type="text/x-python; charset=utf-8")
    response["Content-Disposition"] = (
        f'attachment; filename="{model_export.filename(exp, trial.trial)}"')
    return response


def _picked_trial(request, exp):
    """The trial a download names — `?metric=&idx=`, the index into the
    result's trials, as every figure names one; without them the best on the
    metric being viewed — as (result, trial, metric), or the response refusing
    the request."""
    result = rebuild_result(exp)
    if result is None or not result.trials:
        raise Http404("no trials to export")
    metric = request.GET.get("metric") or exp.data.current_metric or exp.data.metric_names[0]
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    idx_raw = request.GET.get("idx")
    if idx_raw is None:
        idx = result.best_index(metric)
    elif idx_raw.lstrip("-").isdigit() and 0 <= int(idx_raw) < len(result.trials):
        idx = int(idx_raw)
    else:
        return HttpResponseBadRequest("invalid trial index")
    return result, result.trials[idx], metric


@experiment_view(EXPORT)
def experiment_export_trial_model(request, exp):
    """Download the selected trial's fitted model — its parameters, in its
    library's own format, with a note of what it is — as a zip.

    Only for a trial whose model the run kept (see experiments/services/trial_models.py):
    nothing is trained here. What comes back is partially trained, fitted on
    one fold's training rows, and its note says so.
    """
    import io as bytes_io
    import zipfile

    from experiments.models import TrialModel

    picked = _picked_trial(request, exp)
    if isinstance(picked, HttpResponse):
        return picked
    _result, trial, _metric = picked
    kept = TrialModel.objects.filter(experiment=exp, trial=trial.trial).first()
    folder = Path(kept.directory) if kept else None
    if folder is None or not folder.is_dir():
        raise Http404(trial_models_reason(exp))

    buffer = bytes_io.BytesIO()
    stem = model_export.filename(exp, trial.trial).removesuffix(".py") + "-parameters"
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                archive.write(path, f"{stem}/{path.name}")
    TrialModel.objects.filter(pk=kept.pk).update(last_used_at=timezone.now())
    response = HttpResponse(buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{stem}.zip"'
    return response


def _kept_trial_indices(exp, result):
    """The indices into *result*'s trials of those whose model was kept."""
    from experiments.models import TrialModel

    if result is None or not result.trials:
        return []
    kept = set(TrialModel.objects.filter(experiment=exp).values_list("trial", flat=True))
    return [i for i, trial in enumerate(result.trials) if trial.trial in kept]


def trial_models_reason(exp):
    """Why a trial of *exp* has no kept model to export."""
    from core.optimizers.trial import keeps_its_fit

    model = MODELS.get(exp.data.model_name)
    if exp.data.model_file or model is None or not keeps_its_fit(model):
        return _("Trial parameters are kept for the built-in models only: an uploaded model "
                 "is fitted in its own environment, which nothing is brought back from.")
    if not getattr(model, "has_parameters", True):
        return _("%(model)s has no fitted parameters to keep: its forecast is its history "
                 "replayed.") % {"model": model.name}
    best = resolve_settings(exp).get("keep_best_trial_models") or 0
    if not best and not resolve_settings(exp).get("keep_last_run_trial_models"):
        return _("This experiment keeps no trial models. Turn it on in Settings; a new run "
                 "keeps them from then on.")
    return ngettext(
        "This trial's model was not kept: a run keeps only the best %(n)s trial's model "
        "(and with the option on, its own), or its group had no room. Change it in Settings.",
        "This trial's model was not kept: a run keeps only the best %(n)s trials' models "
        "(and with the option on, its own), or its group had no room. Change it in Settings.",
        best) % {"n": best}


def _export(request, exp, cancel_url=None):
    if request.method != "POST":
        # How many tracebacks there are to ask about, and so whether to ask.
        # Counted off the stored result rather than rebuilt through the
        # optimizer: this needs one key per trial, not an OptimizationResult.
        entries = ((exp.data.result or {}).get("data") or [])
        count = sum(1 for e in entries
                    if (e.get("additional_info") or {}).get("traceback"))
        return render(request, "ui/export_confirm.html",
                      {"experiment": exp, "has_tracebacks": bool(count),
                       "traceback_count": count, "cancel_url": cancel_url,
                       "timestamp_modes": _timestamp_choices(exp),
                       "timestamp_mode": times.for_export(exp, request.user)})

    snapshot = snapshot_adapter.snapshot_from_experiment(exp, provenance=True)
    # The paths name files on this server, which is of no use to whoever opens
    # the file and tells them how the instance is laid out.
    snapshot["dataset"]["path"] = ""
    snapshot["model"]["path"] = ""
    # Every time in the file, in the form asked for — see
    # experiments/services/timestamps.py. "keep" is what the question used to post, when
    # the choice was all or nothing; no answer at all is still nothing.
    posted = request.POST.get("timestamps")
    mode = timestamps.clamp(exp, timestamps.ABSOLUTE if posted == KEEP
                            else posted if posted in timestamps.MODES else timestamps.NONE)
    keep_tracebacks = request.POST.get("tracebacks") == KEEP
    if snapshot.get("result"):
        snapshot["result"] = copy.deepcopy(snapshot["result"])
    timestamps.apply(snapshot, mode, timestamps.origin(exp))
    if not keep_tracebacks and snapshot.get("result"):
        for entry in snapshot["result"].get("data", []):
            # The reason stays either way. It is the trial's own result — this
            # configuration does not work — while the traceback is a description
            # of the machine it did not work on.
            (entry.get("additional_info") or {}).pop("traceback", None)
    body = io.to_bytes(snapshot)
    response = HttpResponse(body, content_type="application/octet-stream")
    # Its name when it has one, else its identifier: an unnamed experiment's
    # file is not a dotfile.
    response["Content-Disposition"] = content_disposition_header(
        True, f"{exp.name or exp.identifier}.ihpo")
    return response


def _timestamp_choices(exp):
    """The forms *exp*'s times can leave in, each with what it means."""
    return [(mode, *times.LABELS[mode]) for mode in timestamps.allowed(exp)]


def _posted_settings(request):
    """The settings as submitted, coerced to the type of each default.

    Almost every setting is a checkbox, where absent means off. The exceptions
    are the numeric ones — counts like `ice_max_curves`, and
    `prior_acceptance_tolerance`, which is a fraction of an objective's own
    scale — clamped to `SETTING_BOUNDS`. A display preference typed into a box
    is not worth failing a form over, and anything unreadable falls back to the
    built-in default rather than to zero, which would mean something specific
    and wrong ("no limit", or "refuse every prior").

    Keyed off `type(default)` rather than a second list of which settings are
    numbers, so declaring one in `SETTING_DEFAULTS` is all it takes — and that
    type is what each is read as, so a count stays a count and a tolerance
    keeps its decimals.
    """
    posted = {}
    for key, default in SETTING_DEFAULTS.items():
        if isinstance(default, bool):
            posted[key] = bool(request.POST.get(key))
            continue
        try:
            value = type(default)(request.POST.get(key, ""))
        except (TypeError, ValueError):
            value = default
        low, high = SETTING_BOUNDS.get(key, (None, None))
        if low is not None:
            value = max(low, min(high, value))
        if key in SETTING_CHOICES and value not in SETTING_CHOICES[key]:
            value = default
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
            if not policy().may_change_defaults(request):
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
    """How the pages look in this browser: light, dark, or as the system is.

    Kept in a cookie for a year — see `context_processors.theme` on why there
    and not on the account. A choice that is not one of the three is ignored.
    """
    from .context_processors import THEME_COOKIE, THEMES
    if request.method == "POST":
        response = redirect("ui:appearance")
        chosen = request.POST.get("theme")
        if chosen in THEMES:
            response.set_cookie(THEME_COOKIE, chosen, max_age=365 * 24 * 3600,
                                samesite="Lax", secure=request.is_secure())
        return response
    return render(request, "ui/appearance.html", {
        "themes": [("light", _("Light")), ("dark", _("Dark")),
                   ("system", _("Same as my system"))],
    })


def export_preferences(request):
    """How this reader's exported files carry times, unless an experiment says
    otherwise (ui/times.py). Kept for the reader rather than for
    the browser: it is a decision about what they send, wherever they send it
    from."""
    if request.method == "POST":
        times.set_reader_preference(request.user, request.POST.get("export_timestamps"))
        return redirect("ui:export_preferences")
    return render(request, "ui/export_preferences.html", {
        "modes": [(mode, *times.LABELS[mode]) for mode in timestamps.MODES],
        "chosen": times.reader_preference(request.user),
    })


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
    current = policy()
    return render(request, "ui/account.html", {
        # Each paired with what it lets you do rather than its codename, which
        # names the grant and not the consequence.
        "powers": [
            (_("See every experiment on this instance"),
             request.user.has_perm("access.view_all_experiments")),
            (_("Run, edit and delete other people's experiments"),
             request.user.has_perm("access.manage_experiments")),
            (_("Change the settings every experiment inherits"),
             current.may_change_defaults(request)),
            (_("Upload and run custom models"),
             current.may_upload_models(request)),
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
    if not policy().may_change_defaults(request):
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
    """Confirm (GET) then delete (POST) a saved experiment, stopping any run.

    Into its people's bins, not gone — see `experiments/services/bin.py`.
    """
    if request.method == "POST":
        bin_service.delete(exp, by=request.user)
        messages.success(request, _("Moved %(name)s to the bin.") % {"name": exp.title})
        return redirect("ui:home")
    return render(request, "ui/delete_confirm.html", {
        "experiment": exp,
        "people": _bin_holders(exp),
    })


def _bin_holders(exp):
    """Who else will find it in their bin, for the confirmation to name."""
    from django.contrib.auth import get_user_model
    if not settings.REQUIRE_LOGIN:
        return []
    return list(get_user_model().objects.filter(pk__in=bin_service.its_people(exp))
                .order_by("username"))


# ── data handling ────────────────────────────────────────────────────────────

def tabs_for(exp, url_for, setup=False):
    """The Data Handling page's tabs for *exp*: each section, its address
    (`url_for`), and its slots — every panel with what it needs to render — and
    whether the tab holds controls to save. With *setup*, a draft's, whose
    Overview is its own (`ui.layout.data_tabs`)."""
    tabs = []
    for tab in data_tabs(exp, setup=setup):
        slots = []
        for slot in tab["slots"]:
            figure = slot["figure"]
            control = figure.control(exp) if issubclass(figure, ProcessingStep) else None
            slots.append({**slot, "control": control})
        tabs.append({**tab, "url": url_for(tab["section"]), "slots": slots,
                     "has_controls": any(s["control"] for s in slots)})
    return tabs


def _data_url(exp, section):
    if section is SECTIONS[0]:
        return reverse("ui:data_overview", args=[exp.pk])
    return reverse("ui:data_section", args=[exp.pk, section.slug])


def _data_page(request, exp, section):
    may_edit = policy().may(request, exp, EDIT)
    if request.method == "POST":
        if not may_edit:
            raise Http404
        refusal = processing.save(exp, request.POST, request.user)
        if refusal:
            messages.error(request, refusal)
        else:
            messages.success(request, _("Saved."))
        return redirect(request.path)

    # One page, every section a tab of it: the experiment's own heading above
    # (as on its dashboard and timeline), the section asked for showing.
    fixed = processing.why_fixed(exp)
    return render(request, "ui/data_handling/section.html", {
        **experiment_header(request, exp),
        "tabs": tabs_for(exp, lambda s: _data_url(exp, s)),
        "section": section,
        "locked": bool(fixed) or not may_edit,
        "locked_reason": fixed,
        "uploaded_model": bool(exp.data.model_file),
    })


@experiment_view(VIEW)
def data_overview(request, exp):
    """The first of the sections, at the section's own root."""
    return _data_page(request, exp, SECTIONS[0])


@experiment_view(VIEW)
def data_section(request, exp, section):
    """Any other section, by its slug."""
    found = BY_SLUG.get(section)
    if found is None or found is SECTIONS[0]:
        raise Http404
    return _data_page(request, exp, found)


# ── the bin ──────────────────────────────────────────────────────────────────
#
# Addressed by pk like any experiment, but not through `@experiment_view`: that
# resolves through `experiments()`, which a deleted experiment is deliberately
# not in. `bin()` is the other queryset that decides existence, and a pk that
# is not in this person's bin is a 404 exactly as it would be elsewhere.

def _binned(request, pk):
    from django.shortcuts import get_object_or_404
    return get_object_or_404(policy().bin(request), pk=pk)


def experiment_bin(request):
    """What this person deleted, or had deleted from under them."""
    binned = (policy().bin(request)
              .select_related("owner", "deleted_by").order_by("-deleted_at"))
    return render(request, "ui/bin.html", {
        "binned": [{"experiment": exp,
                    # Without accounts there is no site bin for it to move to,
                    # so it stays until somebody removes it.
                    "expires_at": (bin_service.expires_at(exp)
                                   if settings.REQUIRE_LOGIN else None)}
                   for exp in binned],
        "retention_days": settings.BIN_RETENTION_DAYS,
    })


@require_POST
def bin_restore(request, pk):
    exp = _binned(request, pk)
    bin_service.restore(exp, by=request.user)
    messages.success(request, _("Restored %(name)s.") % {"name": exp.title})
    return redirect("ui:experiment_detail", pk=exp.pk)


def bin_download(request, pk):
    """The same file an export makes, asked about the same way — deleting it
    did not change what it would be safe to hand over."""
    return _export(request, _binned(request, pk),
                   cancel_url=reverse("ui:experiment_bin"))


def bin_dismiss(request, pk):
    """Confirm (GET) then take it out of *my* bin (POST).

    Only mine: everyone else's bin keeps it, and any of them can still restore
    it — without me, since my access goes too. When the last of its people has
    done this it moves to the site admins' bin rather than being destroyed.

    Without accounts there is one bin and nobody else to keep it for, so there
    this is what it says: deleting it for good.
    """
    exp = _binned(request, pk)
    if request.method == "POST":
        if settings.REQUIRE_LOGIN:
            bin_service.dismiss(exp, request.user)
        else:
            bin_service.purge(exp)
        return redirect("ui:experiment_bin")
    # Who will still have it — the bin, not the grants, since that is what
    # decides whether it moves to the site's.
    others = [entry.user for entry in exp.bin_entries.select_related("user")
              .exclude(user_id=request.user.pk).order_by("user__username")]
    return render(request, "ui/bin_dismiss_confirm.html",
                  {"experiment": exp, "others": others})


def import_experiment(request):
    """Upload an .ihpo file to create a saved experiment.

    A dataset and — when custom models are enabled — a model .py may be
    attached to make the imported experiment runnable; without them it loads
    read-only.
    """
    may_upload = policy().may_upload_models(request)
    context = {"allow_custom_models": may_upload,
               "groups": groups_to_choose_from(request.user)}
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

        group, error = _chosen_group(request, context)
        if error:
            context["error"] = error
            return render(request, "ui/import.html", context)

        model_upload = request.FILES.get("model") if may_upload else None
        if model_upload is not None:
            model_error = model_upload_error(model_upload)
            if model_error:
                context["error"] = model_error
                return render(request, "ui/import.html", context)
        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, dataset_file=dataset_upload, model_file=model_upload,
            owner=_owner(request), group=group,
        )
        if not _settle_storage(request, exp):
            return render(request, "ui/import.html", context)
        history.record(exp, "imported", user=request.user, source="ihpo",
                       filename=upload.name, dataset_attached=exp.data.has_dataset)
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

    group, error = _chosen_group(request, context)
    if error:
        context["error"] = error
        return render(request, "ui/import.html", context)
    exp = snapshot_adapter.experiment_from_snapshot(snapshot, owner=_owner(request),
                                                    group=group)
    history.record(exp, "imported", user=request.user, source="smac",
                   filename=snapshot.get("name") or "", dataset_attached=False)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _local_ablation_data(built, metric, idx, baseline=None):
    """The local-ablation figure + warning for one (metric, trial index) —
    HyperSHAP's `ablation` game, explaining that one trial's configuration
    against the trial at index *baseline*, or against the config space's
    default when that is None, rather than a global share like the other three
    games. `built` is `rebuild_experiment`'s full dict, since this needs the
    search space and the optimizer instance, neither of which the stored result
    carries.

    A trial explained against itself is not computed: every effect would be
    zero by construction, and a waterfall of zeros reads as a finding.

    Returns (figure_json_or_None, warning_or_None, raw_effects). The effects
    are the signed per-hyperparameter values the figure is drawn from, in the
    metric's own units, which `_tuning_progress` needs and which cannot be read
    back out of the figure. The figure is None (with a warning) when there is no
    search space to be had — see `config_space_for` for the two places one can
    come from.
    """
    config_space = config_space_for(built)
    if config_space is None:
        return None, _("Local explanation needs the search space, which this "
                       "experiment does not carry and has no model to ask."), {}
    if baseline == idx:
        return None, _("This trial is the baseline."), {}
    result = built["result"]
    ablation, warning = built["optimizer"].compute_hp_ablation(
        config_space, result.trials, metric, result.trials[idx].config,
        seed=built["seed"],
        # From the result, not the registry: an imported run's objective is
        # declared inside its own file, and the explainer needs its direction
        # (see `_pair_trials_with_oriented_scores`).
        metric=result.metric(metric),
        baseline=None if baseline is None else result.trials[baseline].config)
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
    config_space = config_space_for(built)
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
    return [figure for figure in FIGURES if shown[figure.setting_key]
            and figure.applies(exp.data)]


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


def _local_effects_data(built, metric, max_trials=0, baseline=None):
    """The beeswarm figure + warning for one metric — every sampled trial's
    local ablation, from `compute_local_effects`, against the trial at index
    *baseline* or against the config space's default when that is None.

    Same shape and same reasoning as `_partial_dependence_data`: it needs the
    search space, not just the stored result, and it is the one figure whose
    cost is per *trial*, which is what `max_trials` bounds.
    """
    config_space = config_space_for(built)
    if config_space is None:
        return None, _("Local effects need the search space, which this "
                       "experiment does not carry and has no model to ask.")
    result = built["result"]
    hp_names, rows, warning = built["optimizer"].compute_local_effects(
        config_space, result.trials, metric,
        seed=built["seed"], max_trials=max_trials, metric=result.metric(metric),
        baseline=None if baseline is None else result.trials[baseline].config)
    fig = local_effects_plot(hp_names, rows)
    return (json.loads(fig.to_json()) if fig is not None else None), warning


def _importance_rows(importance) -> list:
    """One game's importance as the table rendering lists it, largest first."""
    return sorted(importance.items(), key=lambda kv: kv[1], reverse=True)


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


def _prior_only_context(exp, built):
    """What the prior figure needs when there is nothing to acquire yet.

    The prior and no more: no acquisition curve, no candidates, no incumbent,
    because all three are the surrogate's and there are no trials to fit one
    to. The figure renders its own controls and asks the server for the
    density, the same `prior_only` request it already makes when the
    acquisition figure is waiting behind its Compute prompt.

    Empty when the figure is switched off or the space cannot be rebuilt, and
    the page then looks exactly as it did before — an empty dict updates
    nothing, so the caller needs no second branch.
    """
    figure = next((f for f in _shown_figures(exp) if f.key == "prior"), None)
    config_space = config_space_for(built)
    if figure is None or config_space is None:
        return {}
    return {
        "prior_only_figure": figure,
        # From the space rather than from a trial's configuration, which is the
        # whole point: there are no trials.
        "hp_names": list(config_space.keys()),
        "beta_min": prior_math.BETA_RATIO_MIN,
        "beta_max": prior_math.BETA_RATIO_MAX,
        **_prior_judging_context(exp),
    }


def _demo_time_columns():
    """{demo dataset path: the columns it could be ordered by in time}, for the
    create form to suggest. An uploaded file's columns are read in the page."""
    out = {}
    for path in io.demo_datasets().values():
        try:
            out[path] = io.orderable_columns(io.read_dataset(Path(path)))
        except Exception:  # noqa: BLE001 — an unreadable demo just suggests nothing
            out[path] = []
    return out


def _model_tasks():
    """What the form's script needs to offer only the models that fit: each
    built-in's tasks, and whether it only forecasts."""
    return {name: {"tasks": list(tasks.supported(model)), "forecaster": tasks.forecaster(model)}
            for name, model in MODELS.items()}


#: What each tab is called on the page.
TAB_LABELS = {
    "overview": gettext_lazy("Overview"),
    "hypershap": gettext_lazy("HyperSHAP"),
    "acquisition": gettext_lazy("Acquisition"),
    "misc": gettext_lazy("Misc"),
}


def _computed_version(exp, result):
    """The state of the results a computed figure's answer came from.

    Every figure behind a Compute button fits a model to this experiment's
    trials under its settings, so an answer is good for exactly as long as the
    trials, the run they came from and the settings stay as they were. Anything
    else — the reader's prior, the selected trial, the hyperparameter shown — is
    part of the question each figure asks, and is in its own key. Except the
    priors drawn in the configuration cube: they weigh whole configurations, so
    the acquisition figure's answer carries their weight, and is only good while
    they stand as they were.
    """
    import hashlib
    last = exp.runs.order_by("-id").values_list("pk", "status").first()
    digest = hashlib.sha1(json.dumps([resolve_settings(exp), exp.data.projected_priors or []],
                                     sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {"experiment": exp.pk,
            "version": f"{len(result.trials)}:{last}:{digest}"}


def _figure_tabs(layout, figures):
    """The page's tabs, arranged per *layout* — see ui/layout.py."""
    from .layout import placed
    return [dict(tab, label=TAB_LABELS[tab["tab"]]) for tab in placed(layout, figures)]


@experiment_view(VIEW)
@require_POST
def save_layout(request, exp):
    """Keep how this reader has arranged this experiment's page.

    `VIEW`, because arranging what you can see is not changing the experiment:
    it writes this reader's own row and nobody else's, and a viewer has as much
    use for a tidy page as its owner. From the page's script, a JSON body:

        {"layout": {...}}    keep this as mine
        {"reset": true}      forget mine, use my default

    answered with the arrangement now in force, normalised, so the page can
    redraw from what was actually kept. And from the settings page's three
    buttons — reset this one, make this my default, reset my default — ordinary
    forms, answered by going back to the experiment.
    """
    from . import layout as layouts
    # The settings page's three buttons: ordinary forms, each answered by going
    # back to the experiment, where the result can be seen.
    if request.POST.get("reset"):
        layouts.reset(exp, request.user)
        return redirect("ui:experiment_detail", pk=exp.pk)
    if request.POST.get("make_default"):
        layouts.save_user_default(request.user, layouts.layout_for(exp, request.user))
        return redirect("ui:experiment_detail", pk=exp.pk)
    if request.POST.get("reset_default"):
        layouts.reset_user_default(request.user)
        return redirect("ui:experiment_detail", pk=exp.pk)
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("not JSON")
    if not isinstance(body, dict):
        return HttpResponseBadRequest("not an object")
    if body.get("reset"):
        return JsonResponse({"layout": layouts.reset(exp, request.user)})
    if not isinstance(body.get("layout"), dict):
        return HttpResponseBadRequest("no layout")
    return JsonResponse({"layout": layouts.save(exp, request.user, body["layout"])})


#: `experiment_header`'s "rebuild it yourself", told apart from a result of None.
_UNBUILT = object()


def _prepared_summary(exp) -> str:
    """A short line about the environment, for the experiment page."""
    meta = exp.env_meta or {}
    parts = []
    deps = meta.get("dependencies") or []
    if deps:
        parts.append(", ".join(deps))
    if meta.get("python"):
        parts.append(f"Python {meta['python']}")
    return " · ".join(parts)


def experiment_header(request, exp, result=_UNBUILT):
    """What the top of every view of an experiment shows — its dashboard, its
    Data Handling and its timeline: what it is, what may be done with it and
    the buttons for that, whose it is, and anything wrong with it.

    *result* is the rebuilt OptimizationResult when the caller has one;
    otherwise it is rebuilt here.
    """
    if result is _UNBUILT:
        result = rebuild_result(exp)
    last_run = exp.runs.order_by("-id").first()

    run_summary = None
    if last_run and last_run.status in ("done", "cancelled") and last_run.duration is not None:
        total = last_run.duration
        trials = last_run.trial_seconds or 0.0
        run_summary = {"total": total, "trials": trials, "overhead": max(0.0, total - trials),
                       "count": last_run.trial_count or 0,
                       "stopped_by": STOPPED_BY_LABELS.get(last_run.stopped_by)}

    has_result = result is not None and bool(result.trials)
    kept = _kept_trial_indices(exp, result)
    # The trial the export buttons name before anything is selected: the best
    # on the metric being viewed. Whether its fitted model was kept decides
    # whether Export trial parameters starts as a link.
    current = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    best = (result.best_index(current)
            if has_result and current in exp.data.metric_names else None)

    return {
        "experiment": exp,  # the _run_status.html include reverses URLs from experiment.pk
        "summary": {
            "pk": exp.pk,
            "name": exp.title,
            "given_name": exp.name,
            "identifier": exp.identifier,
            "model_name": exp.data.model_name,
            "optimizer_name": exp.data.optimizer_name,
            "current_metric": exp.data.current_metric,
            "metric_label": metric_label(exp.data.current_metric, exp.data.original_metric),
            "seed": exp.data.seed,
            "evaluation": model_export.evaluation_label(exp),
        },
        # A result with no trials counts as no result: it is what the dashboard
        # stops building panels and plots for, so its template must take its
        # "No results yet" branch rather than render the figure script with
        # nothing for it to read.
        "has_result": has_result,
        # What this viewer may do, for the buttons. Read access got them here;
        # the rest depends on whose experiment it is.
        "may": {action: policy().may(request, exp, action)
                for action in (RUN, EDIT, DELETE, EXPORT, SHARE, RENAME)},
        "ownership": _ownership(request, exp),
        "model_export_unavailable": model_export.unavailable(exp),
        # Which trials' fitted models were kept, by their place in the result —
        # the button for their parameters follows the selected trial, and is
        # only a link for one of these.
        "kept_trial_indices": kept,
        "trial_models_reason": trial_models_reason(exp),
        "export_default_kept": best is not None and best in kept,
        "run_error": last_run.error if (last_run and last_run.status == "error") else None,
        # Why this viewer may not run this experiment's custom model, if they
        # may not. Shown rather than silently disabling the form.
        "model_refusal": policy().custom_model_refusal(exp, request.user),
        # The model's environment, for the notices under the heading.
        "env": {
            "status": exp.env_status,
            "pending": exp.env_pending,
            "in_process": exp.env_in_process,
            "ready": exp.env_status == Experiment.ENV_READY,
            "failed": exp.env_status == Experiment.ENV_FAILED,
            "error": exp.env_error,
            "summary": _prepared_summary(exp),
        },
        "run_summary": run_summary,
    }


def _detail_context(request, exp):
    """Detail-page context: identity, run state, and the figures to draw.

    Rebuilds the OptimizationResult from the stored snapshot (read-only) and,
    when present, builds the panels and per-metric figures for whichever figures
    the settings have switched on; the browser switches metrics client-side.
    """
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    metric_names = list(exp.data.metric_names)
    active_run = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    header = experiment_header(request, exp, result)
    model_refusal = header["model_refusal"]

    context = {
        **header,
        "metric_names": metric_names,
        "active_run": active_run,
        "can_run": (exp.data.has_dataset and _model_available(exp)
                    and policy().may(request, exp, RUN)
                    and not model_refusal),
        # Only an optimizer that fits a model of the objective can answer the
        # confidence criterion, so only then is it offered.
        "supports_confidence": getattr(
            _optimizer_for(exp.data.optimizer_name), "supports_confidence_stopping", False),
        # And only one that can be asked what it would run next can offer the
        # configuration cube's Requery — see `next_trials`.
        "can_requery": hasattr(_optimizer_for(exp.data.optimizer_name), "slice_challengers"),
        # Drawing a prior in the cube: for an optimizer a prior can steer, and
        # a reader who may change what the next run searches under.
        "can_draw_priors": (bool(getattr(_optimizer_for(exp.data.optimizer_name),
                                         "fits_surrogate", False))
                            and policy().may(request, exp, EDIT)),
        **_prior_judging_context(exp),
        "run_default_metric": exp.data.current_metric or (metric_names[0] if metric_names else None),
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
    #
    # One figure goes out through this return rather than being stopped by it.
    # A prior is a statement about the *search space*, which exists before the
    # search does — and before the first run is exactly when somebody has
    # something to say about where the optimum might be. Everything else on
    # that card needs a surrogate and is left out; see `_prior_only_context`.
    if result is None or not result.trials:
        context.update(_prior_only_context(exp, built))
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
    from .layout import layout_for
    figure_layout = layout_for(exp, request.user)
    shown = resolve_settings(exp)
    hp_names = list(result.trials[0].config.keys()) if result.trials else []

    panels = []
    for m in metric_names:
        best_idx = result.best_index(m)
        # No trial carries a score for this metric, so there is no best trial to
        # describe and no selection to open on — `best_index` documents None as
        # the answer a caller handles. The panel is left out rather than built
        # around a trial that isn't there, which is the same decline the
        # per-metric figures make through `has_every_score`. The page script
        # keys its maps by metric and the Run form's incumbent target is built
        # from these panels, so both simply have nothing for this one.
        if best_idx is None:
            continue
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
                    # A row per hyperparameter even with nothing stored: the
                    # table also has the incumbent's and the selected trial's
                    # columns, which the page fills whether or not there is a
                    # default one.
                    "table": (_importance_rows(getattr(result, fields["importance"]).get(m, {}))
                              or [(hp, None) for hp in hp_names]),
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
    config_space = config_space_for(built)
    # The priors drawn in the configuration cube, as their card lists them.
    context["projected_priors"] = [projected.summary(record, config_space)
                                   for record in exp.data.projected_priors or []]
    metric_plots = {
        opening_metric: _figure_plots(
            result, [f for f in figures if f.per_metric], opening_metric,
            config_space=config_space),
    } if opening_metric else {}
    static_plots = _figure_plots(result, [f for f in figures if not f.per_metric],
                                 config_space=config_space)
    static_plots = {key: plot for key, plot in static_plots.items() if plot is not None}

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
        # The tabs, each an ordered list of (figure, columns) — see
        # ui/layout.py — and the sidebar's, which are in none of them
        # (Figure.in_sidebar).
        figures=figures,
        figure_tabs=_figure_tabs(figure_layout, figures),
        # What the computed figures' answers are labelled with in the browser
        # — see ui/static/ui/computed.js.
        computed_version=_computed_version(exp, result),
        # The whole arrangement, figures switched off included, for layout.js
        # to edit and send back.
        figure_layout=figure_layout,
        sidebar_figures=[f for f in figures if f.in_sidebar],
        # Which of them take a click naming a trial and show which one is
        # selected (Figure.selects_trials), so the page's selection bus iterates
        # a declaration instead of naming figures.
        selectable_figures=[f.key for f in figures if f.selects_trials],
        # And which of them a running run can redraw from its trials alone
        # (Figure.live) — the same declaration `run_status` builds payloads for,
        # so the page and the poll cannot disagree about which figures move.
        live_figures=[f.key for f in figures if f.live],
        # And which of them draw the HyperSHAP games (Figure.explains_games),
        # the ones the sidebar's baseline measures from somewhere other than
        # the stored default.
        game_figures=[f.key for f in figures if f.explains_games],
        # The two colours the highlight is drawn in, from the builders that draw
        # everything else in them — the script used to restate them, which is
        # one place for the page and the plots to disagree about what "selected"
        # looks like.
        # `failure` for the one place the page has to draw a failure mark
        # itself: a 3D scene draws no marker outline, so the red cannot ride on
        # the glyph the way it does everywhere else. Sent rather than repeated
        # in the script, so the page restates no colour — see NEGATIVE_COLOR.
        selection_colors={"base": MARKER_COLOR, "selected": SELECTION_COLOR,
                          "comparison": ACCENT_COLOR,
                          "failure": NEGATIVE_COLOR, "proposal": PROPOSAL_COLOR,
                          # The priors drawn in the cube, in the prior figure's
                          # own colour.
                          "prior": PRIOR_COLOR},
        # The one game selector, and what each game asks — resolved here rather
        # than in the template because the labels are lazy translations and
        # `json_script` cannot serialize those.
        explanation_games=[(game, str(label)) for game, label in HP_GAME_LABELS.items()],
        explanation_game_help={game: str(text) for game, text in HP_GAME_HELP.items()},
        hp_names=hp_names,
        # Bounds for the acquisition figure's β field. Constants rather than
        # anything this experiment decided, so they are declarations.
        beta_min=prior_math.BETA_RATIO_MIN,
        beta_max=prior_math.BETA_RATIO_MAX,
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
