from django.shortcuts import render

from core import io

from .registry import METRICS, MODELS, OPTIMIZERS


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
