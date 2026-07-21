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
from .services import snapshot as snapshot_adapter
from .services.run import resolve_seed, run_experiment


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
    """Upload an .ihpo file and view its contents read-only, no database.

    Exercises the core engine end to end in the browser: parse the file, then
    build it read-only (which validates the model/optimizer/metrics and
    deserializes the stored result) without needing the referenced dataset.
    """
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
    """Resolve the chosen dataset to a filesystem path.

    A demo dataset is already a path; an uploaded file is written to a temp
    file whose name is recorded in *tmp_paths* for the caller to clean up.
    """
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
    """Set up an experiment, run it synchronously, and save it.

    On a valid POST: build the split, run the optimizer to completion (the
    request blocks — fine for manual testing), persist the experiment with its
    result and a stored copy of the dataset, and render its detail page.
    """
    if request.method != "POST":
        return render(request, "web/new_experiment.html", {"form": NewExperimentForm()})

    form = NewExperimentForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "web/new_experiment.html", {"form": form})

    cleaned = form.cleaned_data
    seed = resolve_seed(cleaned["seed"])
    primary = cleaned["primary_metric"]
    tmp_paths = []
    try:
        dataset_path = _dataset_path_from(form, tmp_paths)
        result = run_experiment(
            model_name=cleaned["model_name"],
            optimizer_name=cleaned["optimizer_name"],
            dataset_path=dataset_path,
            seed=seed,
            primary_metric=primary,
            n_trials=cleaned["n_trials"],
        )
        optimizer = type(OPTIMIZERS[cleaned["optimizer_name"]])()
        snapshot = {
            "version": VERSION,
            "name": cleaned["name"],
            "model_name": cleaned["model_name"],
            "model_path": "",
            "optimizer_name": optimizer.name,
            "optimizer_params": optimizer.get_params(),
            "primary_metric": primary,
            "original_metric": primary,
            "metric_names": list(METRICS),
            "seed": seed,
            "dataset_path": dataset_path,
            "result": optimizer.serialize_result(result),
        }
        exp = snapshot_adapter.experiment_from_snapshot(snapshot)
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)

    return render(request, "web/experiment_detail.html", _detail_context(exp))


def experiment_detail(request, pk):
    """Show a saved experiment: its config, results, and charts."""
    exp = get_object_or_404(Experiment, pk=pk)
    return render(request, "web/experiment_detail.html", _detail_context(exp))


def experiment_export(request, pk):
    """Download a saved experiment as a Streamlit-loadable .ihpo file."""
    exp = get_object_or_404(Experiment, pk=pk)
    snapshot = snapshot_adapter.snapshot_from_experiment(exp)
    body = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    response = HttpResponse(body, content_type="application/octet-stream")
    response["Content-Disposition"] = f'attachment; filename="{exp.name}.ihpo"'
    return response


def experiment_delete(request, pk):
    """Confirm (GET) then delete (POST) a saved experiment."""
    exp = get_object_or_404(Experiment, pk=pk)
    if request.method == "POST":
        exp.delete()
        return redirect("web:home")
    return render(request, "web/delete_confirm.html", {"experiment": exp})


def import_experiment(request):
    """Upload an .ihpo file to create a saved experiment.

    Reuses core.io.parse for validation (same message as elsewhere). If a
    dataset file is supplied it is adopted; otherwise the experiment is saved
    without one (browsable, re-runnable once a dataset is attached later).
    """
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
    """Assemble the detail-page context: identity, per-metric panels, figures.

    Rebuilds the OptimizationResult from the stored snapshot (read-only) and,
    when present, builds each metric's best-config panel and performance /
    importance figures; the browser switches metrics client-side.
    """
    snapshot = snapshot_adapter.snapshot_from_experiment(exp)
    try:
        _, built = io.build_experiment(
            snapshot, METRICS, MODELS, OPTIMIZERS, read_only=True,
        )
        result = built["result"]
    except ValueError:
        result = None

    summary = {
        "pk": exp.pk,
        "name": exp.name,
        "model_name": exp.model_name,
        "optimizer_name": exp.optimizer_name,
        "primary_metric": exp.primary_metric,
        "metric_label": metric_label(exp.primary_metric, exp.original_metric),
        "seed": exp.seed,
    }
    context = {
        "summary": summary,
        "metric_names": list(exp.metric_names),
        "has_result": result is not None,
    }
    if result is None:
        return context

    metric_names = list(exp.metric_names)
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
