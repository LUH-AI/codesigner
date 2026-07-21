import json
import tempfile
from pathlib import Path

from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from core import io
from core.version import VERSION

from .charts import importance_figure, performance_figure
from .forms import NewExperimentForm
from .models import Experiment
from .registry import METRICS, MODELS, OPTIMIZERS
from .services import run as run_service
from .services import snapshot as snapshot_adapter
from .services.run import resolve_seed
from .services.run_logic import decide_run, resolve_metric_change

_ACTIVE = ["pending", "running"]


def metric_label(primary_metric, original_metric):
    """The label shown for an experiment's metric.

    "~" when no metric has been committed by a run yet, "Inconsistent" when
    the current primary metric no longer matches the one the results were
    produced with, otherwise the metric name itself.
    """
    if original_metric is None:
        return "~"
    if primary_metric != original_metric:
        return "Inconsistent"
    return primary_metric


def home(request):
    return render(request, "web/home.html")


def inspect(request):
    """Upload an .ihpo file and view its contents read-only, no database."""
    context = {}
    upload = request.FILES.get("file")
    if request.method == "POST" and upload is not None:
        try:
            snapshot = io.parse(upload.read())
            name, exp = io.build_experiment(
                snapshot, METRICS, MODELS, OPTIMIZERS, read_only=True,
            )
        except ValueError as exc:
            context["error"] = str(exc)
            return render(request, "web/inspect.html", context)

        result = exp["result"]
        context["summary"] = {
            "name": name,
            "model_name": exp["model_name"],
            "optimizer_name": exp["optimizer"].name,
            "seed": exp["seed"],
            "metric_label": metric_label(exp["primary_metric"], exp["original_metric"]),
            "n_trials": len(result.trials) if result else 0,
            "best_score": result.best_score if result else None,
        }

    return render(request, "web/inspect.html", context)


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
        snapshot = {
            "version": VERSION,
            "name": cleaned["name"],
            "model_name": cleaned["model_name"],
            "model_path": "",
            "optimizer_name": optimizer.name,
            "optimizer_params": optimizer.get_params(),
            "primary_metric": None,
            "original_metric": None,
            "metric_names": list(METRICS),
            "seed": seed,
            "dataset_path": dataset_path,
            "result": None,
        }
        exp = snapshot_adapter.experiment_from_snapshot(snapshot)
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)

    return redirect("web:experiment_detail", pk=exp.pk)


def experiment_detail(request, pk):
    """Show a saved experiment: its config, results/charts, run state, Run form."""
    exp = get_object_or_404(Experiment, pk=pk)
    return render(request, "web/experiment_detail.html", _detail_context(exp))


def experiment_run(request, pk):
    """Launch a background run of an experiment (or confirm a metric change).

    The Run form supplies n_trials and the metric to optimize. A run that would
    change the optimized metric first shows a confirmation; the confirmation
    posts back with a `decision` of "new" or "old".
    """
    exp = get_object_or_404(Experiment, pk=pk)
    if request.method != "POST" or not exp.dataset or exp.is_running:
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
    """Download a saved experiment as a Streamlit-loadable .ihpo file."""
    exp = get_object_or_404(Experiment, pk=pk)
    snapshot = snapshot_adapter.snapshot_from_experiment(exp)
    body = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    response = HttpResponse(body, content_type="application/octet-stream")
    response["Content-Disposition"] = f'attachment; filename="{exp.name}.ihpo"'
    return response


def experiment_delete(request, pk):
    """Confirm (GET) then delete (POST) a saved experiment, stopping any run."""
    exp = get_object_or_404(Experiment, pk=pk)
    if request.method == "POST":
        exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
        exp.delete()
        return redirect("web:home")
    return render(request, "web/delete_confirm.html", {"experiment": exp})


def import_experiment(request):
    """Upload an .ihpo file to create a saved experiment."""
    context = {}
    upload = request.FILES.get("file")
    if request.method == "POST" and upload is not None:
        try:
            snapshot = io.parse(upload.read())
        except ValueError as exc:
            context["error"] = f"Invalid or unreadable experiment file. ({exc})"
            return render(request, "web/import.html", context)

        if Experiment.objects.filter(name=snapshot["name"]).exists():
            context["error"] = f"An experiment named '{snapshot['name']}' already exists."
            return render(request, "web/import.html", context)

        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, dataset_file=request.FILES.get("dataset"),
        )
        return redirect("web:experiment_detail", pk=exp.pk)

    return render(request, "web/import.html", context)


def _detail_context(exp):
    """Detail-page context: identity, run state, per-metric panels and figures.

    Rebuilds the OptimizationResult from the stored snapshot (read-only) and,
    when present, builds each metric's best-config panel and performance /
    importance figures; the browser switches metrics client-side.
    """
    try:
        _, built = io.build_experiment(
            snapshot_adapter.snapshot_from_experiment(exp),
            METRICS, MODELS, OPTIMIZERS, read_only=True,
        )
        result = built["result"]
    except ValueError:
        result = None

    metric_names = list(exp.metric_names)
    active_run = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    last_run = exp.runs.order_by("-id").first()

    context = {
        "summary": {
            "pk": exp.pk,
            "name": exp.name,
            "model_name": exp.model_name,
            "optimizer_name": exp.optimizer_name,
            "primary_metric": exp.primary_metric,
            "metric_label": metric_label(exp.primary_metric, exp.original_metric),
            "seed": exp.seed,
        },
        "metric_names": metric_names,
        "has_result": result is not None,
        "active_run": active_run,
        "can_run": bool(exp.dataset),
        "run_error": last_run.error if (last_run and last_run.status == "error") else None,
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
        })
        figures[m] = {
            "performance": json.loads(perf.to_json()),
            "importance": json.loads(imp.to_json()) if imp is not None else None,
        }

    context.update(
        result=result,
        panels=panels,
        figures=figures,
        trial_rows=[
            {"n": t.trial, "scores": [t.scores[m] for m in metric_names]}
            for t in result.trials
        ],
        trials_exhausted=(
            result.trials_limit is not None
            and len(result.trials) >= result.trials_limit
        ),
    )
    return context
