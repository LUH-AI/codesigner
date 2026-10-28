import tempfile
from pathlib import Path

from django.contrib.auth.decorators import login_not_required  # pyright: ignore[reportAttributeAccessIssue]
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import redirect, render

from access.policy import visible_experiments
from core import datasets, tasks
from core.registry import MODELS

from .. import drafts
from ..forms import EVALUATION_SCHEMES


def home(request):
    return render(request, "ui/home.html")


#: Experiments per page; the sidebar shows only `SIDEBAR_LIMIT`.
EXPERIMENT_LIST_PAGE_SIZE = 50


def experiment_list(request):
    """`experiment_list` paginates `visible_experiments`, the sidebar's own
    queryset.
    """
    paginator = Paginator(visible_experiments(request), EXPERIMENT_LIST_PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "ui/experiment_list.html", {"page": page})


@login_not_required
def healthz(request):
    """`healthz` skips the login wall because the container runtime has no
    session and a 302 would report the instance down.
    """
    return HttpResponse("ok", content_type="text/plain")


def _dataset_path_from(form, tmp_paths):
    """`_dataset_path_from` returns the demo or a temp upload copy."""
    demo = form.cleaned_data.get("demo_dataset")
    if demo:
        return demo
    upload = form.cleaned_data["dataset_file"]
    # The original filename carries into the stored copy and exports.
    folder = Path(tempfile.mkdtemp(prefix="codesigner-upload-"))
    path = folder / (Path(upload.name).name or "dataset.csv")
    with open(path, "wb") as out:
        for chunk in upload.chunks():
            out.write(chunk)
    tmp_paths.append(str(path))
    return str(path)


def _evaluation_schemes():
    """`_evaluation_schemes` sends `EVALUATION_SCHEMES` to the form's script,
    resolving lazy strings that `json_script` cannot serialize.
    """
    return {name: {**spec, "label": str(spec["label"]), "help": str(spec["help"])}
            for name, spec in EVALUATION_SCHEMES.items()}


def new_experiment(request):
    """`new_experiment` creates the draft at once so the draft can be left and
    resumed. A GET renders only the button.
    """
    if request.method != "POST":
        return render(request, "ui/new_experiment.html")
    return redirect("ui:setup", pk=drafts.new(request).pk)


def _demo_columns():
    """`_demo_columns` maps each demo dataset to its columns, which the create
    form offers as targets."""
    out = {}
    for path in datasets.demo_datasets().values():
        try:
            out[path] = [str(c) for c in datasets.read_dataset(Path(path)).columns]
        except Exception:  # noqa: BLE001 — an unreadable demo just suggests nothing
            out[path] = []
    return out


def _demo_time_columns():
    out = {}
    for path in datasets.demo_datasets().values():
        try:
            out[path] = datasets.orderable_columns(datasets.read_dataset(Path(path)))
        except Exception:  # noqa: BLE001 — an unreadable demo just suggests nothing
            out[path] = []
    return out


def _model_tasks():
    """`_model_tasks` maps each built-in model to its tasks and forecaster flag."""
    return {name: {"tasks": list(tasks.supported(model)), "forecaster": tasks.forecaster(model)}
            for name, model in MODELS.items()}
