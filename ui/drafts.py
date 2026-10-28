"""`drafts` sets a draft up from plain values, so the setup pages and the
assistant run the same code. Each function takes the acting person's request
for who they are, and returns what to tell them as notices instead of
writing messages. `ui/setup_steps.py` defines the steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version as dist_version
from pathlib import Path

from django.core.exceptions import ValidationError
from django.utils import timezone
from django.utils.translation import gettext as _
from django.utils.translation import ngettext

from access.policy import groups_to_choose_from, policy
from core import io, provenance, tasks
from core.datasets import demo_datasets, read_dataset
from core.metrics import metrics_for
from core.optimizers.trial import offers_probabilities
from core.registry import MODELS, OPTIMIZERS
from experiments.models import Experiment, account
from experiments.services import history, modelenv, processing, storage
from experiments.services import snapshot as snapshot_adapter
from experiments.services.run import resolve_seed

from . import optimizer_form
from . import setup_steps as steps
from .formatting import human
from .forms import EVALUATION_BACKTEST, EVALUATION_HOLDOUT, EVALUATION_KFOLD, NewExperimentForm
from .validators import validate_dataset_upload


@dataclass
class Saved:
    """`Saved` reports one step's save. `form` is step 1's bound form, whose
    errors explain a failure there; `refusal` explains a failure elsewhere.
    Each notice is a `(level, text)` pair, and `gone` means the draft was
    deleted for want of room in its group.
    """
    saved: bool
    form: object = None
    refusal: str = ""
    notices: list = field(default_factory=list)
    gone: bool = False


def new(request):
    """`new` makes an empty draft owned by *request*'s person."""
    return Experiment.objects.create(
        name="", model_name="", optimizer_name="",
        metric_names=[], seed=0, owner=account(request.user), draft=True)


def settle_storage(exp) -> tuple[bool, list[tuple[str, str]]]:
    """`settle_storage` deletes kept trial models, stalest first, until *exp*'s
    group fits its limit, and returns `(kept, notices)`. The experiment is
    refused and its files deleted only when datasets and model files alone
    exceed the limit.
    """
    group = exp.group
    if group is None or (not exp.data.dataset and not exp.data.model_file):
        return True, []
    within, deleted = storage.make_room(group)
    now = storage.usage(group)
    numbers = {"group": group.name, "used": human(now["total"]),
               "limit": human(now["limit"])}
    if not within:
        storage.delete_experiment_files(exp)
        return False, [("error", _(
            "%(group)s has no room for this: its datasets and model files alone would take "
            "%(used)s of its %(limit)s. Delete experiments it no longer needs, or ask a site "
            "administrator for more.") % numbers)]
    if deleted:
        return True, [("warning", ngettext(
            "To make room in %(group)s, %(n)s kept trial model was deleted, the stalest "
            "first. It now uses %(used)s of %(limit)s.",
            "To make room in %(group)s, %(n)s kept trial models were deleted, the stalest "
            "first. It now uses %(used)s of %(limit)s.", deleted) % dict(numbers, n=deleted))]
    return True, []


def stage_dataset(request, exp, *, demo="", upload=None, group=None) -> Saved:
    """`stage_dataset` puts a dataset on the draft before step 1 is saved, so
    the data can be explored first. Step 1 keeps the staged dataset unless the
    person chooses another, and checks it as step 1 checks any dataset. The
    *group* pk is required only when the person has several groups.
    """
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
    # A target column of the old data names nothing in the new.
    data.target_column = ""
    data.save()
    kept, notices = settle_storage(exp)
    return Saved(kept, notices=notices, gone=not kept)


def check_dataset(demo="", upload=None) -> tuple[str, str]:
    """`check_dataset` returns `(stem, "")` for a bundled demo named by stem or
    path, `("", "")` for an upload that passes step 1's checks, and `("", why)`
    for anything else.
    """
    if demo:
        demos = demo_datasets()
        stem = demo if demo in demos else next(
            (name for name, path in demos.items() if path == str(demo)), "")
        if not stem:
            return "", _("There is no demo dataset called “%(name)s”.") % {"name": demo}
        return stem, ""
    if upload is None:
        return "", _("Choose a demo dataset or upload a CSV file.")
    try:
        validate_dataset_upload(upload)
    except ValidationError as error:
        return "", error.messages[0]
    try:
        read_dataset(upload.read())
    except Exception as exc:  # noqa: BLE001 — any unreadable file is the same problem
        return "", _("The dataset could not be read: %(error)s") % {"error": exc}
    finally:
        upload.seek(0)
    return "", ""


def check_group(request, group):
    """`check_group` returns `(group, "")` for one of the person's groups, by
    pk, and `(None, why)` for any other. A person with no group to choose from
    gets `(None, "")`.
    """
    groups = groups_to_choose_from(request.user)
    if not groups:
        return None, ""
    chosen = next((g for g in groups if str(g.pk) == str(group)), None)
    if chosen is None:
        return None, _("Choose which group this experiment is in.")
    return chosen, ""


def save_step(request, exp, step, data, files=None, *, partial=False) -> Saved:
    """`save_step` saves *step* of the draft from *data*, the step's posted
    form, and marks the step saved. With *partial*, the Optimizer step's
    *data* holds only the changes (`optimizer_form.save`). The Data step always
    reads that way, and step 1 always takes a whole form. The Priors step saves
    nothing, since each belief is stated alone (`experiments.services.priors`).
    """
    if step == steps.SETUP:
        return _save_setup(request, exp, data, files or {})
    if not steps.reachable(exp, step):
        return Saved(False, refusal=_("Save the first step before this one."))
    if step == steps.OPTIMIZER:
        refusal = optimizer_form.save(request, exp, data, partial=partial)
    elif step == steps.DATA:
        refusal = processing.save(exp, data, request.user)
    else:
        raise ValueError(f"step {step!r} has nothing to save")
    if refusal:
        return Saved(False, refusal=refusal)
    steps.mark(exp, step)
    return Saved(True)


def create(request, exp) -> str | None:
    """`create` turns the draft into an experiment once steps 1 to 3 are saved,
    and returns a refusal reason or None. The history starts with the whole
    configuration and then each stated prior.
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


# Step 1

def setup_form(request, exp, data=None, files=None):
    """`setup_form` builds step 1's form for the draft, bound when *data* is
    given."""
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
    """`setup_form_for` builds step 1's form around a dataset already in hand,
    which lets a caller check a setup before any draft exists."""
    args = () if data is None else (data, files)
    return NewExperimentForm(
        *args, may_upload_models=policy().may_upload_models(request),
        groups=groups_to_choose_from(request.user),
        current_dataset=current_dataset, current_model=current_model)


def setup_initial(exp) -> dict:
    """`setup_initial` reads the form's values back from a saved draft. An
    unsaved draft yields its name and any staged demo dataset.
    """
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
    """`_save_setup` rebuilds the draft's data from step 1's form. A changed
    dataset, task or model unmarks step 3.
    """
    from .views.pages import _dataset_path_from

    form = setup_form(request, exp, data, files)
    if not form.is_valid():
        return Saved(False, form=form)

    cleaned = form.cleaned_data
    seed = resolve_seed(cleaned["seed"])
    old = exp.data
    tmp_paths: list = []
    notices: list = []
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
        new_data = snapshot_adapter.data_from_snapshot(
            snapshot, model_file=cleaned.get("model_file"), adopt_paths=True)
        if "group" in form.fields:
            exp.group, _refusal = check_group(request, cleaned.get("group"))
        changed = snapshot_adapter.replace_data(exp, new_data)
        kept, notices = settle_storage(exp)
        if not kept:
            return Saved(False, form=form, notices=notices, gone=True)
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
    return Saved(True, form=form, notices=notices)


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
