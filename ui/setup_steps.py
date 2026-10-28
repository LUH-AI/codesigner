"""`setup_steps` defines the four steps (Setup, Optimizer, Data, Priors &
constraints) that a draft from **New** goes through before creation. Steps 1 to
3 must each be saved, and step 4 is optional. A new dataset, task or model in
step 1 asks for the Data step again.
"""

from __future__ import annotations

from django.urls import reverse
from django.utils import timezone

SETUP, OPTIMIZER, DATA, PRIORS = "setup", "optimizer", "data", "priors"
STEPS = (SETUP, OPTIMIZER, DATA, PRIORS)
#: Steps required before creation.
REQUIRED = (SETUP, OPTIMIZER, DATA)

_URLS = {SETUP: "ui:setup", OPTIMIZER: "ui:setup_optimizer", DATA: "ui:setup_data",
         PRIORS: "ui:setup_priors"}


def url(exp, step) -> str:
    return reverse(_URLS[step], args=[exp.pk])


def saved(exp, step) -> bool:
    return bool((exp.setup_saved or {}).get(step))


def mark(exp, step) -> None:
    """`mark` saves *step*'s timestamp."""
    exp.setup_saved = {**(exp.setup_saved or {}), step: timezone.now().isoformat()}
    exp.save(update_fields=["setup_saved"])


def unmark(exp, step) -> None:
    """`unmark` clears *step*'s saved mark."""
    if saved(exp, step):
        exp.setup_saved = {k: v for k, v in exp.setup_saved.items() if k != step}
        exp.save(update_fields=["setup_saved"])


def reachable(exp, step) -> bool:
    """`reachable` allows later steps once step 1 is saved, since they need its
    dataset and model."""
    return step == SETUP or saved(exp, SETUP)


def resume(exp) -> str:
    """`resume` opens the first unsaved required step, else the last."""
    step = next((s for s in REQUIRED if not saved(exp, s)), PRIORS)
    return url(exp, step)


def not_ready(exp) -> str | None:
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
    """`bar` lists every step with its state."""
    from django.utils.translation import gettext_lazy as _

    labels = {SETUP: _("Setup"), OPTIMIZER: _("Optimizer"), DATA: _("Data"),
              PRIORS: _("Priors & constraints")}
    return [{"number": i, "step": step, "label": labels[step], "url": url(exp, step),
             "current": step == current, "saved": saved(exp, step),
             "required": step in REQUIRED, "reachable": reachable(exp, step)}
            for i, step in enumerate(STEPS, start=1)]
