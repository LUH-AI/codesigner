import copy
import json
import tempfile
from importlib.metadata import version as dist_version
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_not_required
from django.core.exceptions import PermissionDenied
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from core import io

from .figures import FIGURES, FULL, HALF
from .forms import DefaultExperimentSettingsForm, ExperimentSettingsForm, NewExperimentForm
from .models import Experiment, GlobalSettings
from . import permissions
from .permissions import DELETE, EDIT, EXPORT, RUN, VIEW, experiment_view
from .registry import METRICS, MODELS, OPTIMIZERS
from .services import run as run_service
from .services import modelenv
from .services import snapshot as snapshot_adapter
from .services.run import resolve_seed
from .services.run_logic import decide_run, resolve_metric_change
from .services.settings import SETTING_DEFAULTS, global_defaults, resolve_settings

_ACTIVE = ["pending", "running"]


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
    "target_score": (float, 0.0, 1.0),
    "max_seconds": (float, 1.0, None),
    "max_trial_seconds": (float, 1.0, None),
    "no_improvement_trials": (int, 1, None),
    "incumbent_confidence": (float, 0.0, 1.0),
}


#: What ended a run, for the summary line. Every criterion is named: with no
#: privileged default, "it stopped" no longer implies the trial count.
STOPPED_BY_LABELS = {
    "max_trials": _("all the requested trials ran"),
    "target_score": _("the target score was reached"),
    "max_seconds": _("the time limit was reached"),
    "max_trial_seconds": _("the compute budget was used up"),
    "no_improvement_trials": _("the score had stopped improving"),
    "incumbent_confidence": _("the search was confident nothing better remained"),
    "cancelled": _("it was interrupted"),
    "all_failing": _("every trial was failing"),
}


def _posted_stopping(request) -> dict:
    """The stopping criteria as submitted, ignoring the blanks.

    A criterion left empty is absent rather than zero: zero would mean "stop
    immediately", which is never what an empty box asks for. Unparseable input
    is dropped the same way — and if that leaves nothing at all, the caller
    refuses the run rather than starting one that cannot end.
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
        if value < low:
            continue
        stopping[key] = min(value, high) if high is not None else value
    return stopping


def metric_label(primary_metric, original_metric):
    """The label shown for an experiment's metric.

    "~" when no metric has been committed by a run yet, "Inconsistent" when
    the current primary metric no longer matches the one the results were
    produced with, otherwise the metric name itself.
    """
    if original_metric is None:
        return "~"
    if primary_metric != original_metric:
        return _("Inconsistent")
    return primary_metric


def home(request):
    return render(request, "ui/home.html")


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


def new_experiment(request):
    """Set up (but do not run) an experiment, then go to its detail page.

    Persists the experiment with no result and no committed metric; the dataset
    is copied into MEDIA. Running is a separate action on the detail page.
    """
    may_upload = permissions.policy().may_upload_models(request)
    if request.method != "POST":
        return render(request, "ui/new_experiment.html",
                      {"form": NewExperimentForm(may_upload_models=may_upload)})

    form = NewExperimentForm(request.POST, request.FILES,
                             may_upload_models=may_upload)
    if not form.is_valid():
        return render(request, "ui/new_experiment.html", {"form": form})

    cleaned = form.cleaned_data
    seed = resolve_seed(cleaned["seed"])
    tmp_paths = []
    try:
        dataset_path = _dataset_path_from(form, tmp_paths)
        optimizer = type(OPTIMIZERS[cleaned["optimizer_name"]])()
        # A mounted model is adopted from its server-side path (unless an upload
        # was given, which takes precedence); the adapter copies it into MEDIA.
        mounted = cleaned.get("mounted_model") or ""
        snapshot = {
            "version": dist_version("codesigner"),
            "name": cleaned["name"],
            "model_name": cleaned["model_name"],
            "model_path": mounted if (mounted and not cleaned.get("model_file")) else "",
            "optimizer_name": optimizer.name,
            "optimizer_params": optimizer.get_params(),
            "primary_metric": None,
            "original_metric": None,
            "metric_names": list(METRICS),
            "seed": seed,
            "cv_folds": int(cleaned.get("cv_folds") or 0),
            "dataset_path": dataset_path,
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
                  {"sel": _selected_panel_data(result, metric, idx)})


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

    stopping = _posted_stopping(request)
    chosen = request.POST.get("optimize_metric")
    decision = request.POST.get("decision")

    if not stopping:
        # A run has to be able to end. Back to the page with the reason rather
        # than a started run that never finishes.
        context = _detail_context(request, exp)
        context["run_error"] = _("Set at least one stopping criterion, so the "
                                 "run has something to end on.")
        return render(request, "ui/experiment_detail.html", context)

    if decision:
        optimize_metric = resolve_metric_change(decision, exp.primary_metric, chosen)
        if optimize_metric is None:
            return redirect("ui:experiment_detail", pk=exp.pk)
    else:
        action, optimize_metric = decide_run(exp.original_metric, exp.primary_metric, chosen)
        if action == "warn":
            return render(request, "ui/metric_change.html", {
                "experiment": exp, "chosen": chosen,
                # Carried through the confirmation, or answering it would
                # silently drop the limits the run was set up with.
                "stopping": stopping,
            })

    run = run_service.create_run(exp, stopping, optimize_metric,
                                 started_by=_owner(request))
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


@experiment_view(VIEW)
def run_status(request, exp):
    """HTMX poll target: the current run's status, or a refresh when finished."""
    active = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    if active is None:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response
    return render(request, "ui/_run_status.html", {"experiment": exp, "run": active})


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

    POST only: it changes something, and on GET a prefetcher or an <img> tag
    pointing here would cancel someone's run without CSRF ever being consulted.
    """
    exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
    return redirect("ui:experiment_detail", pk=exp.pk)


@experiment_view(EXPORT)
def experiment_export(request, exp):
    """Download a saved experiment as a Streamlit-loadable .ihpo file.

    Server paths are always blanked, and when the experiment's
    `export_absolute_times` setting is off, per-trial `starttime`/`endtime` are
    scrubbed too, so a shared file reveals no run times (durations are kept).
    Done here, not in the adapter, so the detail page's own reconstruction and
    the run engine are unaffected; deserialize ignores the keys on re-import.
    """
    snapshot = snapshot_adapter.snapshot_from_experiment(exp)
    # The paths name files on this server, which is of no use to whoever opens
    # the file and tells them how the instance is laid out.
    snapshot["dataset_path"] = ""
    snapshot["model_path"] = ""
    if not resolve_settings(exp)["export_absolute_times"] and snapshot.get("result"):
        snapshot["result"] = copy.deepcopy(snapshot["result"])
        for entry in snapshot["result"].get("data", []):
            entry.pop("starttime", None)
            entry.pop("endtime", None)
    body = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    response = HttpResponse(body, content_type="application/octet-stream")
    response["Content-Disposition"] = f'attachment; filename="{exp.name}.ihpo"'
    return response


def _posted_settings(request):
    """The settings as submitted. Every one is a checkbox, so absent means off."""
    return {key: bool(request.POST.get(key)) for key in SETTING_DEFAULTS}


@experiment_view(EDIT)
def experiment_settings(request, exp):
    """One experiment's settings: inherit the defaults, override, reset, or
    promote its own settings to be the defaults (which asks first)."""
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
    return render(request, "ui/experiment_settings.html", {"experiment": exp, "form": form})


def appearance(request):
    """Appearance settings (display/theme options; currently a placeholder)."""
    return render(request, "ui/appearance.html", {})


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
    if request.method == "POST" and upload is not None:
        try:
            snapshot = io.parse(upload.read())
        except ValueError as exc:
            context["error"] = _("Invalid or unreadable experiment file.") + f" ({exc})"
            return render(request, "ui/import.html", context)

        model_upload = request.FILES.get("model") if may_upload else None
        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, dataset_file=request.FILES.get("dataset"), model_file=model_upload,
            owner=_owner(request),
        )
        # An imported model is re-locked here rather than trusting pins chosen by
        # whoever exported it.
        modelenv.start_preparation(exp)
        return redirect("ui:experiment_detail", pk=exp.pk)

    return render(request, "ui/import.html", context)


def _rebuild_result(exp):
    """Rebuild the OptimizationResult from a saved experiment's snapshot
    (read-only — no dataset or live model needed), or None if it can't be
    rebuilt (e.g. it names a metric/optimizer no longer available)."""
    try:
        _, built = io.build_experiment(
            snapshot_adapter.snapshot_from_experiment(exp),
            METRICS, MODELS, OPTIMIZERS, read_only=True,
        )
    except ValueError:
        return None
    return built["result"]


def _selected_panel_data(result, metric, idx):
    """The selected-config panel's data for one (metric, trial index).

    Mirrors app/analytics/selected_config.py: the trial's score for *metric*,
    and its delta against the metric's best score — omitted when the selected
    trial IS the best (so the panel matches best-config with no delta shown).
    """
    trials = result.trials
    best_idx = max(range(len(trials)), key=lambda i: trials[i].scores[metric])
    trial = trials[idx]
    is_best = idx == best_idx
    delta = None if is_best else trial.scores[metric] - trials[best_idx].scores[metric]
    return {
        "metric": metric,
        "trial_n": trial.trial,
        "score": trial.scores[metric],
        "delta": delta,
        "config": list(trial.config.items()),
    }


def _detail_context(request, exp):
    """Detail-page context: identity, run state, and the figures to draw.

    Rebuilds the OptimizationResult from the stored snapshot (read-only) and,
    when present, builds the panels and per-metric figures for whichever figures
    the settings have switched on; the browser switches metrics client-side.
    """
    result = _rebuild_result(exp)
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
            "primary_metric": exp.primary_metric,
            "metric_label": metric_label(exp.primary_metric, exp.original_metric),
            "seed": exp.seed,
            "scoring": (_("%(k)s-fold CV") % {"k": exp.cv_folds}
                        if exp.cv_folds >= 2 else _("holdout")),
        },
        "metric_names": metric_names,
        "has_result": result is not None,
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
            OPTIMIZERS.get(exp.optimizer_name)
            or next((o for o in OPTIMIZERS.values() if o.name == exp.optimizer_name), None),
            "supports_confidence_stopping", False),
        "run_default_metric": exp.primary_metric or (metric_names[0] if metric_names else None),
    }
    if result is None:
        return context

    # Which figures to draw. A figure that is switched off is not rendered and
    # its plot is not built, so nothing is computed or shipped to go unused.
    shown = resolve_settings(exp)
    figures = [figure for figure in FIGURES if shown[figure.setting_key]]

    def plot_json(figure, metric=None):
        """A figure's plot as JSON, or None when it draws no plot / has no data."""
        plot = figure.plot(result, metric)
        return json.loads(plot.to_json()) if plot is not None else None

    panels = []
    for m in metric_names:
        best_idx = max(range(len(result.trials)),
                       key=lambda i: result.trials[i].scores[m])
        best = result.trials[best_idx]
        panels.append({
            "metric": m,
            "best_n": best.trial,
            "best_score": best.scores[m],
            "best_config": list(best.config.items()),
            "warning": result.hyperparameter_importance_warning.get(m),
            # No selection has been clicked yet, so it defaults to the best trial.
            "selected": _selected_panel_data(result, m, best_idx),
        })

    # Plots, keyed by figure, built straight off the catalog — per-metric ones
    # for every metric (the browser switches between them), the rest once.
    metric_plots = {
        m: {f.key: plot_json(f, m) for f in figures if f.per_metric}
        for m in metric_names
    }
    static_plots = {f.key: plot_json(f) for f in figures if not f.per_metric}
    static_plots = {key: plot for key, plot in static_plots.items() if plot is not None}

    hp_names = list(result.trials[0].config.keys()) if result.trials else []
    context.update(
        result=result,
        panels=panels,
        metric_plots=metric_plots,
        static_plots=static_plots,
        # Each figure's declared display behavior, for the page script.
        figure_options={f.key: {"absoluteScale": f.absolute_scale} for f in figures},
        half_figures=[f for f in figures if f.width == HALF],
        full_figures=[f for f in figures if f.width == FULL],
        hp_names=hp_names,
        trial_rows=[
            {
                "n": t.trial,
                "config_values": [t.config.get(h) for h in hp_names],
                "scores": [t.scores[m] for m in metric_names],
                "duration": t.duration,
            }
            for t in result.trials
        ],
        trials_exhausted=(
            result.trials_limit is not None
            and len(result.trials) >= result.trials_limit
        ),
    )
    return context
