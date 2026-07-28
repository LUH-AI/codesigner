import copy
import json
import tempfile
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _

from core import io
from core.version import VERSION

from .charts import (
    duration_figure,
    error_reduction_spikes_figure,
    importance_figure,
    performance_figure,
    regret_convergence_figure,
    return_on_compute_figure,
)
from .forms import DefaultExperimentSettingsForm, ExperimentSettingsForm, NewExperimentForm
from .models import Experiment, GlobalSettings
from .registry import METRICS, MODELS, OPTIMIZERS
from .services import run as run_service
from .services import snapshot as snapshot_adapter
from .services.run import resolve_seed
from .services.run_logic import decide_run, resolve_metric_change
from .services.settings import global_defaults, resolve_settings

_ACTIVE = ["pending", "running"]


def _model_available(exp):
    """Whether *exp*'s model can be built to run it.

    Registry models are always available; a custom model needs its uploaded
    .py present and the ALLOW_CUSTOM_MODELS feature enabled.
    """
    if exp.model_name in MODELS:
        return True
    return bool(exp.model_file) and settings.ALLOW_CUSTOM_MODELS


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
    return render(request, "web/home.html")


def healthz(request):
    """Liveness probe for the container healthcheck — no DB, no template."""
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
    if request.method != "POST":
        return render(request, "web/new_experiment.html", {"form": NewExperimentForm()})

    form = NewExperimentForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "web/new_experiment.html", {"form": form})

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
            "version": VERSION,
            "name": cleaned["name"],
            "model_name": cleaned["model_name"],
            "model_path": mounted if (mounted and not cleaned.get("model_file")) else "",
            "optimizer_name": optimizer.name,
            "optimizer_params": optimizer.get_params(),
            "primary_metric": None,
            "original_metric": None,
            "metric_names": list(METRICS),
            "seed": seed,
            "dataset_path": dataset_path,
            "result": None,
        }
        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, model_file=cleaned.get("model_file"),
        )
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)

    return redirect("web:experiment_detail", pk=exp.pk)


def experiment_detail(request, pk):
    """Show a saved experiment: its config, results/charts, run state, Run form."""
    exp = get_object_or_404(Experiment, pk=pk)
    return render(request, "web/experiment_detail.html", _detail_context(exp))


def trial_panel(request, pk):
    """Render the selected-config panel for one (metric, trial index).

    Backs click-to-select on the performance chart (curve 0 only, point index
    into result.trials) — the browser fetches this fragment and swaps it into
    the panel for the metric currently being viewed.
    """
    exp = get_object_or_404(Experiment, pk=pk)
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

    return render(request, "web/_selected_config_inner.html",
                  {"sel": _selected_panel_data(result, metric, idx)})


def experiment_run(request, pk):
    """Launch a background run of an experiment (or confirm a metric change).

    The Run form supplies n_trials and the metric to optimize. A run that would
    change the optimized metric first shows a confirmation; the confirmation
    posts back with a `decision` of "new" or "old".
    """
    exp = get_object_or_404(Experiment, pk=pk)
    if (request.method != "POST" or not exp.dataset or exp.is_running
            or not _model_available(exp)):
        return redirect("web:experiment_detail", pk=pk)

    n_trials = max(1, min(1000, int(request.POST.get("n_trials") or 30)))
    chosen = request.POST.get("optimize_metric")
    decision = request.POST.get("decision")

    if decision:
        optimize_metric = resolve_metric_change(decision, exp.original_metric, chosen)
        if optimize_metric is None:
            return redirect("web:experiment_detail", pk=pk)
    else:
        action, optimize_metric = decide_run(exp.original_metric, exp.primary_metric, chosen)
        if action == "warn":
            return render(request, "web/metric_change.html", {
                "experiment": exp, "chosen": chosen, "n_trials": n_trials,
            })

    run = run_service.create_run(exp, n_trials, optimize_metric)
    run_service.start_background_run(run.id)
    return redirect("web:experiment_detail", pk=pk)


def run_status(request, pk):
    """HTMX poll target: the current run's status, or a refresh when finished."""
    exp = get_object_or_404(Experiment, pk=pk)
    active = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    if active is None:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response
    return render(request, "web/_run_status.html", {"experiment": exp, "run": active})


def run_cancel(request, pk):
    """Request cancellation of the experiment's active run."""
    exp = get_object_or_404(Experiment, pk=pk)
    exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
    return redirect("web:experiment_detail", pk=pk)


def experiment_export(request, pk):
    """Download a saved experiment as a Streamlit-loadable .ihpo file.

    When the experiment's `export_absolute_times` setting is off, per-trial
    `starttime`/`endtime` are scrubbed so a shared file reveals no run times
    (durations are kept). Done here, not in the adapter, so the detail page's
    own reconstruction is unaffected; deserialize ignores the keys on re-import.
    """
    exp = get_object_or_404(Experiment, pk=pk)
    snapshot = snapshot_adapter.snapshot_from_experiment(exp)
    if not resolve_settings(exp)["export_absolute_times"] and snapshot.get("result"):
        snapshot["result"] = copy.deepcopy(snapshot["result"])
        for entry in snapshot["result"].get("data", []):
            entry.pop("starttime", None)
            entry.pop("endtime", None)
    body = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    response = HttpResponse(body, content_type="application/octet-stream")
    response["Content-Disposition"] = f'attachment; filename="{exp.name}.ihpo"'
    return response


def experiment_settings(request, pk):
    """Per-experiment settings: inherit the global defaults, override, or reset."""
    exp = get_object_or_404(Experiment, pk=pk)
    if request.method == "POST":
        if "reset" in request.POST or request.POST.get("use_default_settings"):
            exp.use_default_settings = True
            exp.settings = {}
        else:
            exp.use_default_settings = False
            exp.settings = {"export_absolute_times": bool(request.POST.get("export_absolute_times"))}
        exp.save(update_fields=["use_default_settings", "settings"])
        return redirect("web:experiment_settings", pk=pk)

    effective = resolve_settings(exp)
    form = ExperimentSettingsForm(initial={
        "use_default_settings": exp.use_default_settings,
        "export_absolute_times": effective["export_absolute_times"],
    })
    return render(request, "web/experiment_settings.html", {"experiment": exp, "form": form})


def global_settings(request):
    """Global settings landing page (currently just links to the defaults)."""
    return render(request, "web/global_settings.html", {})


def default_experiment_settings(request):
    """Edit the default experiment settings that inheriting experiments use."""
    gs = GlobalSettings.get_solo()
    if request.method == "POST":
        gs.default_experiment_settings = {
            "export_absolute_times": bool(request.POST.get("export_absolute_times")),
        }
        gs.save(update_fields=["default_experiment_settings"])
        return redirect("web:default_experiment_settings")

    form = DefaultExperimentSettingsForm(initial={
        "export_absolute_times": global_defaults()["export_absolute_times"],
    })
    return render(request, "web/default_experiment_settings.html", {"form": form})


def experiment_delete(request, pk):
    """Confirm (GET) then delete (POST) a saved experiment, stopping any run."""
    exp = get_object_or_404(Experiment, pk=pk)
    if request.method == "POST":
        exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
        exp.delete()
        return redirect("web:home")
    return render(request, "web/delete_confirm.html", {"experiment": exp})


def import_experiment(request):
    """Upload an .ihpo file to create a saved experiment.

    A dataset and — when custom models are enabled — a model .py may be
    attached to make the imported experiment runnable; without them it loads
    read-only.
    """
    context = {"allow_custom_models": settings.ALLOW_CUSTOM_MODELS}
    upload = request.FILES.get("file")
    if request.method == "POST" and upload is not None:
        try:
            snapshot = io.parse(upload.read())
        except ValueError as exc:
            context["error"] = _("Invalid or unreadable experiment file.") + f" ({exc})"
            return render(request, "web/import.html", context)

        model_upload = request.FILES.get("model") if settings.ALLOW_CUSTOM_MODELS else None
        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, dataset_file=request.FILES.get("dataset"), model_file=model_upload,
        )
        return redirect("web:experiment_detail", pk=exp.pk)

    return render(request, "web/import.html", context)


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


def _detail_context(exp):
    """Detail-page context: identity, run state, per-metric panels and figures.

    Rebuilds the OptimizationResult from the stored snapshot (read-only) and,
    when present, builds each metric's best-config panel and performance /
    importance figures; the browser switches metrics client-side.
    """
    result = _rebuild_result(exp)
    metric_names = list(exp.metric_names)
    active_run = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    last_run = exp.runs.order_by("-id").first()

    run_summary = None
    if last_run and last_run.status in ("done", "cancelled") and last_run.duration is not None:
        total = last_run.duration
        trials = last_run.trial_seconds or 0.0
        run_summary = {"total": total, "trials": trials, "overhead": max(0.0, total - trials),
                       "count": last_run.trial_count or 0}

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
        },
        "metric_names": metric_names,
        "has_result": result is not None,
        "active_run": active_run,
        "can_run": bool(exp.dataset) and _model_available(exp),
        "run_error": last_run.error if (last_run and last_run.status == "error") else None,
        "run_summary": run_summary,
        "run_default_metric": exp.primary_metric or (metric_names[0] if metric_names else None),
    }
    if result is None:
        return context

    panels, figures = [], {}
    for m in metric_names:
        best_idx = max(range(len(result.trials)),
                       key=lambda i: result.trials[i].scores[m])
        best = result.trials[best_idx]
        perf = performance_figure(result, m, selected_idx=best_idx)
        imp = importance_figure(result, m)
        panels.append({
            "metric": m,
            "best_n": best.trial,
            "best_score": best.scores[m],
            "best_config": list(best.config.items()),
            "warning": result.hyperparameter_importance_warning.get(m),
            "has_importance": imp is not None,
            # No selection has been clicked yet, so it defaults to the best trial.
            "selected": _selected_panel_data(result, m, best_idx),
        })
        def _fig(f):  # per-metric figure → embedded JSON, or None
            fig = f(result, m)
            return json.loads(fig.to_json()) if fig is not None else None

        figures[m] = {
            "performance": json.loads(perf.to_json()),
            "importance": json.loads(imp.to_json()) if imp is not None else None,
            # Three candidate "efficiency" views to compare, then cull to one.
            "variant_a": _fig(error_reduction_spikes_figure),
            "variant_b": _fig(regret_convergence_figure),
            "variant_c": _fig(return_on_compute_figure),
        }

    hp_names = list(result.trials[0].config.keys()) if result.trials else []
    dfig = duration_figure(result)
    context.update(
        result=result,
        panels=panels,
        figures=figures,
        duration_figure=json.loads(dfig.to_json()) if dfig is not None else None,
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
