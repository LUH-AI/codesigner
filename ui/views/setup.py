"""Every view here reaches only drafts (`drafts=ONLY`); `ui/setup_steps.py`
defines the four steps.
"""

from importlib.metadata import version as dist_version
from pathlib import Path

from django.contrib import messages
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import EDIT, groups_to_choose_from, policy
from core import io, provenance, tasks
from core.datasets import demo_datasets
from core.metrics import metrics_for
from core.optimizers.trial import offers_probabilities
from core.registry import MODELS, OPTIMIZERS
from experiments.services import history, modelenv, processing, storage
from experiments.services.run import resolve_seed
from experiments.services import snapshot as snapshot_adapter

from .. import optimizer_form
from ..figures.data_handling import SECTIONS
from ..forms import (
    EVALUATION_BACKTEST, EVALUATION_HOLDOUT, EVALUATION_KFOLD, NewExperimentForm,
)
from ..permissions import ONLY, experiment_view
from .. import setup_steps as steps
from .common import _group_by_pk, _settle_storage
from .dashboard import _prior_only_context, experiment_header
from .data_handling import tabs_for
from .pages import _dataset_path_from, _demo_time_columns, _evaluation_schemes, _model_tasks


def _page(request, exp, step, template, **context):
    return render(request, template, {
        **experiment_header(request, exp, None),
        "steps": steps.bar(exp, step),
        "step": step,
        **context,
    })


@experiment_view(EDIT, drafts=ONLY)
def setup_resume(request, exp):
    """`setup_resume` opens the first unsaved step."""
    return redirect(steps.resume(exp))


@experiment_view(EDIT, drafts=ONLY)
def setup(request, exp):
    """`setup` shows step 1, what the experiment is."""
    if request.method == "POST":
        form, saved = save_setup(request, exp)
        if saved is None:
            return redirect("ui:home")
        if saved:
            return redirect(steps.url(exp, steps.OPTIMIZER))
    else:
        form = None
    return _page(request, exp, steps.SETUP, "ui/setup_step_setup.html",
                 **setup_form_context(request, exp, form))


@experiment_view(EDIT, drafts=ONLY)
def setup_optimizer(request, exp):
    """`setup_optimizer` shows step 2, the optimizer and its settings."""
    if not steps.reachable(exp, steps.OPTIMIZER):
        return redirect(steps.url(exp, steps.SETUP))
    if request.method == "POST":
        refusal = optimizer_form.save(request, exp)
        if refusal:
            messages.error(request, refusal)
            return redirect(steps.url(exp, steps.OPTIMIZER))
        steps.mark(exp, steps.OPTIMIZER)
        return redirect(steps.url(exp, steps.DATA))
    return _page(request, exp, steps.OPTIMIZER, "ui/setup_step_optimizer.html",
                 **optimizer_form.context(exp))


@experiment_view(EDIT, drafts=ONLY)
def setup_data(request, exp):
    """`setup_data` shows step 3, the Data Handling panels settable on Overview."""
    if not steps.reachable(exp, steps.DATA):
        return redirect(steps.url(exp, steps.SETUP))
    if request.method == "POST":
        refusal = processing.save(exp, request.POST, request.user)
        if refusal:
            messages.error(request, refusal)
            return redirect(steps.url(exp, steps.DATA))
        steps.mark(exp, steps.DATA)
        return redirect(steps.url(exp, steps.PRIORS))
    here = steps.url(exp, steps.DATA)
    return _page(request, exp, steps.DATA, "ui/setup_step_data.html",
                 tabs=tabs_for(exp, lambda s: here + ("" if s is SECTIONS[0] else f"#{s.slug}"),
                               setup=True),
                 section=SECTIONS[0], locked=False, save_url=here,
                 save_label=_("Save and continue"),
                 uploaded_model=bool(exp.data.model_file))


@experiment_view(EDIT, drafts=ONLY)
def setup_priors(request, exp):
    """`setup_priors` shows step 4, the prior figure and the Create button."""
    if not steps.reachable(exp, steps.PRIORS):
        return redirect(steps.url(exp, steps.SETUP))
    built = snapshot_adapter.rebuild_experiment(exp)
    return _page(request, exp, steps.PRIORS, "ui/setup_step_priors.html",
                 not_ready=steps.not_ready(exp),
                 run_default_metric=(exp.data.metric_names or [None])[0],
                 **_prior_only_context(exp, built))


@require_POST
@experiment_view(EDIT, drafts=ONLY)
def setup_create(request, exp):
    """`setup_create` turns the draft into an experiment once steps 1 to 3 are
    saved."""
    refusal = steps.not_ready(exp)
    if refusal:
        messages.error(request, refusal)
        return redirect(steps.resume(exp))
    exp.draft = False
    exp.save(update_fields=["draft"])
    exp.data.created_at = timezone.now()
    exp.data.save(update_fields=["created_at"])
    history.record(exp, "created", user=request.user, configuration=history.configuration(
        snapshot_adapter.snapshot_from_experiment(exp)))
    for hp, prior in (exp.data.priors or {}).items():
        history.record(exp, "prior_stated", user=request.user,
                       hyperparameter=hp, old=None, new=prior)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(EDIT, drafts=ONLY)
def setup_discard(request, exp):
    """`setup_discard` deletes the draft and its files, skipping the bin."""
    name = exp.title
    storage.delete_experiment_files(exp)
    messages.success(request, _("Discarded the draft “%(name)s”.") % {"name": name})
    return redirect("ui:home")


# Step 1

def _form(request, exp, *args):

    data = exp.data
    current_model = None
    if data.model_file:
        try:
            with data.model_file.open("rb") as f:
                current_model = f.read()
        except OSError:
            current_model = None
    return NewExperimentForm(
        *args, may_upload_models=policy().may_upload_models(request),
        groups=groups_to_choose_from(request.user),
        current_dataset=data.dataset_path() if data.has_dataset else None,
        current_model=current_model)


def _initial(exp) -> dict:
    """Form values read back from a saved draft, else its name."""
    data = exp.data
    if not steps.saved(exp, steps.SETUP):
        return {"name": data.name}
    if data.forecasts:
        scheme, value = EVALUATION_BACKTEST, data.cv_folds
    elif data.cv_folds >= 2:
        scheme, value = EVALUATION_KFOLD, data.cv_folds
    else:
        scheme, value = EVALUATION_HOLDOUT, data.test_size
    demo = None
    if data.demo_dataset:
        demo = demo_datasets().get(data.demo_dataset)
    return {
        "name": data.name, "task": data.task, "seed": data.seed,
        "model_name": "" if data.model_file else data.model_name,
        "demo_dataset": demo or "",
        "evaluation_scheme": scheme, "evaluation_value": value,
        "time_column": data.time_column, "time_gap": data.time_gap,
        "horizon": data.horizon, "series_column": data.series_column, "season": data.season,
        "group": exp.group_id,
    }


def setup_form_context(request, exp, form=None) -> dict:
    """`setup_form_context` builds step 1's template context with the draft's
    current files."""
    data = exp.data
    if form is None:
        form = _form(request, exp)
        form.initial.update(_initial(exp))
    return {
        "form": form,
        "evaluation_schemes": _evaluation_schemes(),
        "model_tasks": _model_tasks(),
        "demo_time_columns": _demo_time_columns(),
        "current_dataset": (data.demo_dataset or (Path(data.dataset.name).name if data.dataset else "")),
        "current_model": Path(data.model_file.name).name if data.model_file else "",
    }


def save_setup(request, exp):
    """`save_setup` rebuilds the draft's data from the form and returns (form,
    saved), with saved None when the group lacked room and deleted the draft. A
    changed dataset, task or model unmarks step 3."""

    form = _form(request, exp, request.POST, request.FILES)
    if not form.is_valid():
        return form, False

    cleaned = form.cleaned_data
    seed = resolve_seed(cleaned["seed"])
    old = exp.data
    tmp_paths = []
    try:
        dataset_path = (str(cleaned["keep_dataset"]) if cleaned.get("keep_dataset")
                        else _dataset_path_from(form, tmp_paths))
        # The first optimizer stands in until step 2 picks one.
        chosen = OPTIMIZERS.get(old.optimizer_name) or next(iter(OPTIMIZERS.values()))
        optimizer = chosen(**chosen.known_params(old.optimizer_params or {}))
        # An upload overrides a mounted model's server path.
        mounted = cleaned.get("mounted_model") or ""
        model_path = (mounted if (mounted and not cleaned.get("model_file"))
                      else old.model_file.path if cleaned.get("keep_model") else "")
        folds = int(cleaned.get("cv_folds") or 0)
        snapshot = io.normalize({
            "format": io.SNAPSHOT_FORMAT,
            "version": dist_version("codesigner"),
            "name": cleaned["name"],
            "seed": seed,
            "dataset": {"filename": Path(dataset_path).name, "path": dataset_path,
                        "processing": old.processing or {}},
            "model": {"kind": "file" if model_path else "registry",
                      "name": cleaned["model_name"], "path": model_path},
            "evaluation": provenance.evaluation(folds, test_size=cleaned.get("test_size"),
                                                task=cleaned["task"],
                                                time_column=cleaned.get("time_column", ""),
                                                gap=cleaned.get("time_gap", 0),
                                                horizon=cleaned.get("horizon", 0),
                                                series_column=cleaned.get("series_column", ""),
                                                season=cleaned.get("season", 0)),
            # Each trial computes every task metric, plus probability metrics
            # when `_free_probabilities` allows.
            "metrics": {"names": metrics_for(cleaned["task"],
                                             probabilities=_free_probabilities(cleaned),
                                             forecast=bool(cleaned.get("horizon"))),
                        "current": None, "original": None},
            "optimizer": {"name": optimizer.name, "params": optimizer.get_params()},
            "result": None,
        })
        # Every path here comes from a validated choice or a local temp file, so
        # `adopt_paths` is safe.
        new = snapshot_adapter.data_from_snapshot(
            snapshot, model_file=cleaned.get("model_file"), adopt_paths=True)
        if "group" in form.fields:
            exp.group = _group_by_pk(groups_to_choose_from(request.user), cleaned.get("group"))
        changed = snapshot_adapter.replace_data(exp, new)
        if not _settle_storage(request, exp):
            return form, None
        if changed & {"dataset", "task", "model"}:
            steps.unmark(exp, steps.DATA)
        steps.mark(exp, steps.SETUP)
        if "model" in changed:
            # The environment build takes minutes, so it starts now.
            modelenv.start_preparation(exp, cleaned.get("model_source"))
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)
            if Path(path).parent.name.startswith("codesigner-upload-"):
                Path(path).parent.rmdir()
    return form, True


def _free_probabilities(cleaned) -> bool:
    """`_free_probabilities` tracks ROC AUC and log loss when the classifier's
    own fit yields probabilities at no extra cost. An uploaded model's claim is
    trusted.
    """

    if cleaned["task"] != tasks.CLASSIFICATION:
        return False
    info = cleaned.get("model_source")
    if info is not None:
        return bool(info.has_proba)
    model = MODELS.get(cleaned["model_name"])
    return (model is not None and offers_probabilities(model)
            and not getattr(model, "costly_probabilities", False))
