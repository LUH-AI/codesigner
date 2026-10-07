"""A new experiment's setup: four steps, and what saving each one means.

**New** makes a draft at once (`Experiment.draft`), listed in Experiment
Selection, and it is set up in four steps before it becomes an experiment:

1. **Setup** — what the experiment is: its dataset, task, model, how it is
   evaluated.
2. **Optimizer** — what searches its hyperparameters, and with what settings
   (`ui/services/optimizer_form.py`, also on the page before its first run).
3. **Data** — how its columns are processed before a model sees them: the
   Data Handling panels that can be set now.
4. **Priors & constraints** — what is believed about where the optimum is.

Steps 1 to 3 have to be saved, each explicitly, before the draft can be
created; step 4 is optional. Saving step 1 again with a different dataset,
task or model asks for the Data step again — what the model needs, and so
what the processing comes to, depends on all three — and a different model
withdraws the priors, which were statements about the old model's search
space.

A draft is its owner's alone, and nothing about it is history yet: the
experiment's history starts with its creation, which records how it was set
up, every step included.
"""

from __future__ import annotations

from django.urls import reverse
from django.utils import timezone

SETUP, OPTIMIZER, DATA, PRIORS = "setup", "optimizer", "data", "priors"
#: In order.
STEPS = (SETUP, OPTIMIZER, DATA, PRIORS)
#: The steps that must be saved before the draft can be created.
REQUIRED = (SETUP, OPTIMIZER, DATA)

_URLS = {SETUP: "ui:setup", OPTIMIZER: "ui:setup_optimizer", DATA: "ui:setup_data",
         PRIORS: "ui:setup_priors"}


def url(exp, step) -> str:
    return reverse(_URLS[step], args=[exp.pk])


def saved(exp, step) -> bool:
    return bool((exp.setup_saved or {}).get(step))


def mark(exp, step) -> None:
    """Record that *step* was saved, now."""
    exp.setup_saved = {**(exp.setup_saved or {}), step: timezone.now().isoformat()}
    exp.save(update_fields=["setup_saved"])


def unmark(exp, step) -> None:
    """Ask for *step* to be saved again."""
    if saved(exp, step):
        exp.setup_saved = {k: v for k, v in exp.setup_saved.items() if k != step}
        exp.save(update_fields=["setup_saved"])


def reachable(exp, step) -> bool:
    """Whether *step* can be opened yet: the first always, the rest once the
    first has been saved — there is nothing to search, process or believe
    about before there is a dataset and a model."""
    return step == SETUP or saved(exp, SETUP)


def resume(exp) -> str:
    """Where to pick the setup up again: the first step not yet saved, or the
    last."""
    step = next((s for s in REQUIRED if not saved(exp, s)), PRIORS)
    return url(exp, step)


def not_ready(exp) -> str | None:
    """Why *exp* cannot be created yet, or None."""
    from django.utils.translation import gettext as _

    if not saved(exp, SETUP):
        return _("Save step 1, Setup, first.")
    if not saved(exp, OPTIMIZER):
        return _("Save step 2, Optimizer, first.")
    if not saved(exp, DATA):
        return _("Save step 3, Data, first.")
    data = exp.data
    if not (data.has_dataset and data.model_name):
        return _("Step 1 is missing a dataset or a model.")
    if not data.optimizer_name:
        return _("Step 2 is missing an optimizer.")
    return None


def bar(exp, current) -> list[dict]:
    """The step bar: each step's number, label, address and state."""
    from django.utils.translation import gettext_lazy as _

    labels = {SETUP: _("Setup"), OPTIMIZER: _("Optimizer"), DATA: _("Data"),
              PRIORS: _("Priors & constraints")}
    return [{"number": i, "step": step, "label": labels[step], "url": url(exp, step),
             "current": step == current, "saved": saved(exp, step),
             "required": step in REQUIRED, "reachable": reachable(exp, step)}
            for i, step in enumerate(STEPS, start=1)]


def what_changed(before, after) -> set[str]:
    """Which of dataset, task and model differ between two `ExperimentData`."""
    from core.provenance import sha256

    def dataset(data):
        if data.demo_dataset:
            return ("demo", data.demo_dataset)
        path = data.dataset_path()
        return ("file", sha256(path)) if path and path.is_file() else None

    def model(data):
        if data.model_file:
            try:
                with data.model_file.open("rb") as f:
                    return ("file", f.read())
            except OSError:
                return ("file", data.model_file.name)
        return ("registry", data.model_name)

    changed = set()
    if dataset(before) != dataset(after):
        changed.add("dataset")
    if before.task != after.task:
        changed.add("task")
    if model(before) != model(after):
        changed.add("model")
    return changed
