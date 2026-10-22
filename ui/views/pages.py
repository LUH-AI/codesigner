import tempfile
from pathlib import Path

from django.contrib.auth.decorators import login_not_required  # pyright: ignore[reportAttributeAccessIssue]
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import redirect, render

from access.policy import visible_experiments
from core import datasets, tasks
from core.registry import MODELS
from experiments.models import Experiment

from ..forms import EVALUATION_SCHEMES
from .common import _owner


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


def _demo_time_columns():
    """{demo dataset path: the columns it could be ordered by in time}, for the
    create form to suggest. An uploaded file's columns are read in the page."""
    out = {}
    for path in datasets.demo_datasets().values():
        try:
            out[path] = datasets.orderable_columns(datasets.read_dataset(Path(path)))
        except Exception:  # noqa: BLE001 — an unreadable demo just suggests nothing
            out[path] = []
    return out


def _model_tasks():
    """What the form's script needs to offer only the models that fit: each
    built-in's tasks, and whether it only forecasts."""
    return {name: {"tasks": list(tasks.supported(model)), "forecaster": tasks.forecaster(model)}
            for name, model in MODELS.items()}
