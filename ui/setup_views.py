"""A draft's setup: its four steps, creating it, and throwing it away.

See ui/services/setup.py for what the steps are and what saving each means.
Every view here reaches only drafts (`experiment_view(..., drafts=ONLY)`); an
experiment that has been created is sent to its dashboard instead.
"""

from pathlib import Path

from django.contrib import messages
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST


from .permissions import EDIT, ONLY, experiment_view
from .services import history, optimizer_form
from .services import setup as steps
from .services import snapshot as snapshot_adapter


def _page(request, exp, step, template, **context):
    from .views import experiment_header

    return render(request, template, {
        **experiment_header(request, exp, None),
        "steps": steps.bar(exp, step),
        "step": step,
        **context,
    })


@experiment_view(EDIT, drafts=ONLY)
def setup_resume(request, exp):
    """A draft opened from the sidebar: the first step not yet saved."""
    return redirect(steps.resume(exp))


@experiment_view(EDIT, drafts=ONLY)
def setup(request, exp):
    """Step 1: what the experiment is, filled from the draft; Save keeps it
    and moves on."""
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
    """Step 2: the optimizer and its settings; Save keeps them and moves on."""
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
    """Step 3: the Data Handling page, its settable panels on Overview; Save
    keeps their choices and moves on."""
    from datahandling.sections import SECTIONS
    from datahandling.views import save_processing, tabs_for

    if not steps.reachable(exp, steps.DATA):
        return redirect(steps.url(exp, steps.SETUP))
    if request.method == "POST":
        refusal = save_processing(request, exp)
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
    """Step 4: what is believed about where the optimum is — the prior figure,
    as an experiment with no trials shows it — and Create."""
    from .views import _prior_only_context, _rebuild_experiment

    if not steps.reachable(exp, steps.PRIORS):
        return redirect(steps.url(exp, steps.SETUP))
    built = _rebuild_experiment(exp)
    return _page(request, exp, steps.PRIORS, "ui/setup_step_priors.html",
                 not_ready=steps.not_ready(exp),
                 run_default_metric=(exp.data.metric_names or [None])[0],
                 **_prior_only_context(exp, built))


@require_POST
@experiment_view(EDIT, drafts=ONLY)
def setup_create(request, exp):
    """Make the draft an experiment: refused until steps 1 to 3 are saved.
    Its history begins here, with how it was set up — Data Handling's choices
    among it, which the timeline lists as an entry of their own — and each
    prior its steps stated."""
    refusal = steps.not_ready(exp)
    if refusal:
        messages.error(request, refusal)
        return redirect(steps.resume(exp))
    exp.draft = False
    exp.save(update_fields=["draft"])
    exp.data.created_at = timezone.now()
    exp.data.save(update_fields=["created_at"])
    # Its history begins with how it was set up — the whole configuration, so
    # it can be rebuilt, the processing chosen included — then each prior
    # stated, as an entry of its own. The processing is not recorded as a
    # change: nothing ran under anything else.
    history.record(exp, "created", user=request.user, configuration=history.configuration(
        snapshot_adapter.snapshot_from_experiment(exp)))
    for hp, prior in (exp.data.priors or {}).items():
        history.record(exp, "prior_stated", user=request.user,
                       hyperparameter=hp, old=None, new=prior)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(EDIT, drafts=ONLY)
def setup_discard(request, exp):
    """Throw the draft away, files and all. Nothing was run, so there is
    nothing for a bin to keep."""
    from .services import storage

    name = exp.title
    storage.delete_experiment_files(exp)
    messages.success(request, _("Discarded the draft “%(name)s”.") % {"name": name})
    return redirect("ui:home")


# ── step 1 ───────────────────────────────────────────────────────────────────

def _form(request, exp, *args):
    from access.policy import groups_to_choose_from

    from .forms import NewExperimentForm
    from .permissions import policy

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
    """The form's values for a draft that has been saved before, read back
    from what it is; just its name for one that has not."""
    from .forms import EVALUATION_BACKTEST, EVALUATION_HOLDOUT, EVALUATION_KFOLD

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
        from core.io import demo_datasets

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
    """What step 1's template needs: the form (bound after a failed Save, else
    filled from the draft), and what the draft already has, so keeping it is
    visible."""
    from .views import _demo_time_columns, _evaluation_schemes, _model_tasks

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
    """Step 1's Save: build what the form describes and put it in place of the
    draft's portable half. Returns (form, saved) — saved None when the group
    had no room for it, and the draft is gone.

    Built the way creating an experiment always built one — a snapshot from the
    form, adopted by `data_from_snapshot` — and kept what the later steps set:
    the processing, and the priors unless the model changed. A different
    dataset, task or model asks for step 2 again (ui/services/setup.py)."""
    from django.db import transaction

    from core import io
    from core.metrics import metrics_for
    from core import provenance
    from importlib.metadata import version as dist_version

    from .registry import OPTIMIZERS
    from .services import modelenv
    from .services.run import resolve_seed
    from .views import (
        _dataset_path_from, _group_by_pk, _settle_storage,
    )

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
        # Step 2's to choose; until it has been, the first offered on its
        # defaults, so the draft is whole.
        chosen = OPTIMIZERS.get(old.optimizer_name) or next(iter(OPTIMIZERS.values()))
        optimizer = type(chosen)(**type(chosen).known_params(old.optimizer_params or {}))
        # A mounted model is adopted from its server-side path (unless an upload
        # was given, which takes precedence); a kept one from where it is.
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
            # Every metric for the task is computed on every trial; which one is
            # optimized is chosen per run. The probability ones only when asked
            # for, which the form has checked the model can answer.
            "metrics": {"names": metrics_for(cleaned["task"],
                                             probabilities=_free_probabilities(cleaned),
                                             forecast=bool(cleaned.get("horizon"))),
                        "current": None, "original": None},
            "optimizer": {"name": optimizer.name, "params": optimizer.get_params()},
            "result": None,
        })
        # This snapshot's paths were built right here: a validated demo or
        # mounted choice, a temp file written above, or the draft's own file —
        # so they are ours to adopt.
        new = snapshot_adapter.data_from_snapshot(
            snapshot, model_file=cleaned.get("model_file"), adopt_paths=True)
        changed = steps.what_changed(old, new)
        new.processing = old.processing
        new.priors = {} if "model" in changed else (old.priors or {})
        new.config_space = None if "model" in changed else old.config_space
        with transaction.atomic():
            new.save()
            exp.data = new
            if "group" in form.fields:
                from access.policy import groups_to_choose_from

                exp.group = _group_by_pk(groups_to_choose_from(request.user), cleaned.get("group"))
            exp.save()
            for field in (old.dataset, old.model_file):
                if field:
                    field.delete(save=False)
            old.delete()
        if not _settle_storage(request, exp):
            # Refused for want of room: the draft has been deleted, files and
            # all, and the reason is in the messages.
            return form, None
        if changed & {"dataset", "task", "model"}:
            steps.unmark(exp, steps.DATA)
        steps.mark(exp, steps.SETUP)
        if "model" in changed:
            # A custom model needs an environment before it can run; resolving
            # it takes minutes, so it starts now, in the background.
            modelenv.start_preparation(exp, cleaned.get("model_source"))
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)
            if Path(path).parent.name.startswith("codesigner-upload-"):
                Path(path).parent.rmdir()
    return form, True


def _free_probabilities(cleaned) -> bool:
    """Whether the model step 1 chose gives class probabilities from its own
    fit, so ROC AUC and log loss are tracked with every other metric.

    Not for regression, which has no classes; not for a model with no
    probabilities; and not for one whose cost more than its fit (the SVM's
    calibration), which every trial would pay. An uploaded model that offers
    them is taken at its word.
    """
    from core import tasks
    from core.optimizers.trial import offers_probabilities

    from .registry import MODELS

    if cleaned["task"] != tasks.CLASSIFICATION:
        return False
    info = cleaned.get("model_source")
    if info is not None:
        return bool(info.has_proba)
    model = MODELS.get(cleaned["model_name"])
    return (model is not None and offers_probabilities(model)
            and not getattr(model, "costly_probabilities", False))
