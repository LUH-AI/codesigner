"""A draft from start to Create: making one, saving each of its steps, and
creating the experiment.

What the setup pages do, as functions of the draft and plain values, so the
pages and anything else that sets a draft up go through the same code. Each
takes the *request* of the person acting, for who they are and what the policy
lets them do, and nothing else from it: the values come as mappings, and what
there is to tell the person comes back as notices rather than messages.

See ui/services/setup.py for what the steps are and what saving each means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from django.utils import timezone
from django.utils.translation import gettext as _

from . import history
from . import optimizer_form
from . import setup as steps
from . import snapshot as snapshot_adapter


@dataclass
class Saved:
    """What saving a step came to.

    `saved` whether it was; `form` step 1's bound form, whose errors say why it
    was not; `refusal` why any other step was not; `notices` what to tell the
    person, each `(level, text)`; `gone` when the draft was deleted for want of
    room in its group.
    """
    saved: bool
    form: object = None
    refusal: str = ""
    notices: list = field(default_factory=list)
    gone: bool = False


def new(request):
    """A new, empty draft, owned by *request*'s person."""
    from ..models import Experiment
    from ..views import _owner

    return Experiment.objects.create(
        name="", model_name="", optimizer_name="",
        metric_names=[], seed=0, owner=_owner(request), draft=True)


def stage_dataset(request, exp, *, demo="", upload=None, group=None) -> Saved:
    """Put a dataset on the draft *exp* before its first step is saved, as
    *request*'s person: a bundled demo by name (or path), or an *upload*.

    So the data can be looked at, and talked about, before the rest of step 1
    is decided; step 1 then keeps it as the draft's dataset unless another is
    chosen there. Checked as step 1 checks one — its extension, its size, that
    it reads — and stored as step 1 stores one, the group's room settled the
    same way. *group* is the pk of the group the experiment goes in, asked for
    only when the person has more than one to choose from.

    Not once step 1 is saved: from then on the dataset is changed there, with
    everything it checks against the rest of the step.
    """
    from . import storage

    if steps.saved(exp, steps.SETUP):
        return Saved(False, refusal=_("This draft's first step is saved: change its dataset there."))

    stem, refusal = check_dataset(demo, upload)
    if refusal:
        return Saved(False, refusal=refusal)
    chosen, refusal = check_group(request, group)
    if refusal:
        return Saved(False, refusal=refusal)
    if chosen is not None:
        exp.group = chosen
        exp.save(update_fields=["group"])

    data = exp.data
    if data.dataset:
        data.dataset.delete(save=False)
    data.demo_dataset = stem
    if upload is not None and not stem:
        data.dataset.save(Path(upload.name).name, upload, save=False)
    # A target named for the old data names nothing in the new.
    data.target_column = ""
    data.save()
    kept, notices = storage.settle(exp)
    if not kept:
        return Saved(False, notices=notices, gone=True)
    return Saved(True, notices=notices)


def check_dataset(demo="", upload=None) -> tuple[str, str]:
    """A dataset as step 1 would take it: `(stem, "")` for a bundled demo
    named by its stem or path, `("", "")` for an upload that is a CSV within
    the size limit and reads, or `("", why)` for anything else."""
    from core import io

    from ..validators import dataset_upload_error

    if demo:
        demos = io.demo_datasets()
        stem = demo if demo in demos else next(
            (name for name, path in demos.items() if path == str(demo)), "")
        if not stem:
            return "", _("There is no demo dataset called “%(name)s”.") % {"name": demo}
        return stem, ""
    if upload is None:
        return "", _("Choose a demo dataset or upload a CSV file.")
    problem = dataset_upload_error(upload)
    if problem:
        return "", problem
    try:
        io.read_dataset(upload.read())
    except Exception as exc:  # noqa: BLE001 — any unreadable file is the same problem
        return "", _("The dataset could not be read: %(error)s") % {"error": exc}
    finally:
        upload.seek(0)
    return "", ""


def check_group(request, group):
    """The group *group* (a pk) names, for a person with more than one to
    choose from: `(group, "")`, or `(None, why)` when it is not one of theirs.
    `(None, "")` when there is nothing to choose."""
    from access.policy import groups_to_choose_from

    from ..views import _group_by_pk

    groups = groups_to_choose_from(request.user)
    if not groups:
        return None, ""
    chosen = _group_by_pk(groups, group)
    if chosen is None:
        return None, _("Choose which group this experiment is in.")
    return chosen, ""


def save_step(request, exp, step, data, files=None, *, partial=False) -> Saved:
    """Save *step* of the draft *exp* from *data* (and step 1's *files*), as
    *request*'s person, and mark it saved.

    *data* is the step's form as the page posts it. With *partial*, the
    Optimizer step's *data* says only what changes and the rest keeps its value
    (`optimizer_form.save`); the Data step always works that way, and step 1
    is always whole. The Priors step has nothing to save; its beliefs are
    stated one at a time (ui/services/priors.py).
    """
    from datahandling.views import save_processing

    if step == steps.SETUP:
        return _save_setup(request, exp, data, files or {})
    if not steps.reachable(exp, step):
        return Saved(False, refusal=_("Save the first step before this one."))
    if step == steps.OPTIMIZER:
        refusal = optimizer_form.save(request, exp, data, partial=partial)
    elif step == steps.DATA:
        refusal = save_processing(request, exp, data)
    else:
        raise ValueError(f"step {step!r} has nothing to save")
    if refusal:
        return Saved(False, refusal=refusal)
    steps.mark(exp, step)
    return Saved(True)


def create(request, exp) -> str | None:
    """Make the draft *exp* an experiment, as *request*'s person. Returns why
    it could not — steps 1 to 3 not all saved — or None.

    Its history begins here, with how it was set up — the whole configuration,
    so it can be rebuilt, the processing chosen included — then each prior
    stated, as an entry of its own. The processing is not recorded as a change:
    nothing ran under anything else.
    """
    refusal = steps.not_ready(exp)
    if refusal:
        return refusal
    exp.draft = False
    exp.save(update_fields=["draft"])
    exp.data.created_at = timezone.now()
    exp.data.save(update_fields=["created_at"])
    history.record(exp, "created", user=request.user, configuration=history.configuration(
        snapshot_adapter.snapshot_from_experiment(exp)))
    for hp, prior in (exp.data.priors or {}).items():
        history.record(exp, "prior_stated", user=request.user,
                       hyperparameter=hp, old=None, new=prior)
    return None


# ── step 1 ───────────────────────────────────────────────────────────────────

def setup_form(request, exp, data=None, files=None):
    """Step 1's form for *exp*, as *request*'s person may fill it — bound to
    *data* and *files* when given, else unbound for the page to fill."""
    stored = exp.data
    current_model = None
    if stored.model_file:
        try:
            with stored.model_file.open("rb") as f:
                current_model = f.read()
        except OSError:
            current_model = None
    return setup_form_for(request, data, files,
                          current_dataset=stored.dataset_path() if stored.has_dataset else None,
                          current_model=current_model)


def setup_form_for(request, data=None, files=None, *, current_dataset=None, current_model=None):
    """Step 1's form around a dataset already in hand at *current_dataset* —
    for checking what a draft would be before there is one."""
    from access.policy import groups_to_choose_from

    from ..forms import NewExperimentForm
    from ..permissions import policy

    args = () if data is None else (data, files)
    return NewExperimentForm(
        *args, may_upload_models=policy().may_upload_models(request),
        groups=groups_to_choose_from(request.user),
        current_dataset=current_dataset, current_model=current_model)


def setup_initial(exp) -> dict:
    """The form's values for a draft that has been saved before, read back
    from what it is; for one that has not, just its name and the demo dataset
    already put on it, if any."""
    from core.io import demo_datasets

    from ..forms import EVALUATION_BACKTEST, EVALUATION_HOLDOUT, EVALUATION_KFOLD

    data = exp.data
    demo = demo_datasets().get(data.demo_dataset) if data.demo_dataset else None
    if not steps.saved(exp, steps.SETUP):
        return {"name": data.name, **({"demo_dataset": demo} if demo else {})}
    if data.forecasts:
        scheme, value = EVALUATION_BACKTEST, data.cv_folds
    elif data.cv_folds >= 2:
        scheme, value = EVALUATION_KFOLD, data.cv_folds
    else:
        scheme, value = EVALUATION_HOLDOUT, data.test_size
    return {
        "name": data.name, "task": data.task, "seed": data.seed,
        "model_name": "" if data.model_file else data.model_name,
        "target_column": data.target_column,
        "demo_dataset": demo or "",
        "evaluation_scheme": scheme, "evaluation_value": value,
        "time_column": data.time_column, "time_gap": data.time_gap,
        "horizon": data.horizon, "series_column": data.series_column, "season": data.season,
        "group": exp.group_id,
    }


def _save_setup(request, exp, data, files) -> Saved:
    """Step 1's Save: build what *data* and *files* describe and put it in
    place of the draft's portable half.

    Built the way creating an experiment always built one — a snapshot from the
    form, adopted by `data_from_snapshot` — and kept what the later steps set:
    the processing, and the priors unless the model changed. A different
    dataset, task or model asks for step 2 again (ui/services/setup.py)."""
    from django.db import transaction

    from core import io
    from core.metrics import metrics_for
    from core import provenance
    from importlib.metadata import version as dist_version

    from ..registry import OPTIMIZERS
    from ..views import _dataset_path_from, _group_by_pk
    from . import modelenv
    from . import storage
    from .run import resolve_seed

    form = setup_form(request, exp, data, files)
    if not form.is_valid():
        return Saved(False, form=form)

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
                        "target_column": cleaned.get("target_column") or None,
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
        kept, notices = storage.settle(exp)
        if not kept:
            # Refused for want of room: the draft has been deleted, files and
            # all, and the notice says why.
            return Saved(False, form=form, notices=notices, gone=True)
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
    return Saved(True, form=form, notices=notices)


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

    from ..registry import MODELS

    if cleaned["task"] != tasks.CLASSIFICATION:
        return False
    info = cleaned.get("model_source")
    if info is not None:
        return bool(info.has_proba)
    model = MODELS.get(cleaned["model_name"])
    return (model is not None and offers_probabilities(model)
            and not getattr(model, "costly_probabilities", False))
