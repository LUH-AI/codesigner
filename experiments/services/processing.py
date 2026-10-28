"""`processing` saves the `core.processing` choices and guards their changes."""

from django.utils.translation import gettext as _

from core.processing import CHOICES, choices_of
from core.registry import MODELS

from . import history


def _has_trials(exp):
    result = exp.data.result or {}
    return bool(result.get("data") or result.get("trials"))


def why_fixed(exp):
    if _has_trials(exp):
        return _("This experiment has trials, and they were scored on its columns "
                 "processed as they are now; changing that would make the next ones "
                 "incomparable.")
    if exp.is_running:
        return _("A run is under way, and it is using the columns processed as they "
                 "are now.")
    return None


def save(exp, posted, user):
    """`save` returns the refusal or None, and history records each change."""
    refusal = why_fixed(exp)
    if refusal:
        return refusal
    before = choices_of(exp.data.processing)
    processing = dict(before)
    for step, options in CHOICES.items():
        value = posted.get(step)
        if value in options:
            processing[step] = value
    exp.data.processing = processing
    exp.data.save(update_fields=["processing"])
    for step in CHOICES:
        if processing[step] != before[step]:
            history.record(exp, "processing_changed", user=user,
                           step=step, old=before[step], new=processing[step])
    return None


def model_of(exp):
    """`model_of` returns None for an uploaded model, which receives raw
    columns."""
    model = MODELS.get(exp.data.model_name)
    if model is None or exp.data.model_file:
        return None
    model = model.__class__()
    model.task = exp.data.task
    return model
