import tempfile
from pathlib import Path

from django.shortcuts import render

from core import io

from .forms import NewExperimentForm
from .registry import METRICS, MODELS, OPTIMIZERS
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
    """Set up an experiment, run it synchronously, and show the results.

    The whole MVP in one page: on GET show the form; on a valid POST build the
    split, run the optimizer to completion (the request blocks — fine for
    manual testing), and render the result. Nothing is persisted.
    """
    if request.method != "POST":
        return render(request, "web/new_experiment.html", {"form": NewExperimentForm()})

    form = NewExperimentForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "web/new_experiment.html", {"form": form})

    seed = resolve_seed(form.cleaned_data["seed"])
    primary_metric = form.cleaned_data["primary_metric"]
    tmp_paths = []
    try:
        dataset_path = _dataset_path_from(form, tmp_paths)
        result = run_experiment(
            model_name=form.cleaned_data["model_name"],
            optimizer_name=form.cleaned_data["optimizer_name"],
            dataset_path=dataset_path,
            seed=seed,
            primary_metric=primary_metric,
            n_trials=form.cleaned_data["n_trials"],
        )
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)

    best_idx = max(
        range(len(result.trials)),
        key=lambda i: result.trials[i].scores[primary_metric],
    )
    metric_names = list(METRICS)
    trial_rows = [
        {"n": t.trial, "scores": [t.scores[m] for m in metric_names]}
        for t in result.trials
    ]
    context = {
        "summary": {
            "name": form.cleaned_data["name"],
            "model_name": form.cleaned_data["model_name"],
            "optimizer_name": form.cleaned_data["optimizer_name"],
            "primary_metric": primary_metric,
            "seed": seed,
        },
        "result": result,
        "metric_names": metric_names,
        "trial_rows": trial_rows,
        "best_config": result.trials[best_idx].config,
        "trials_exhausted": (
            result.trials_limit is not None
            and len(result.trials) >= result.trials_limit
        ),
    }
    return render(request, "web/results.html", context)
