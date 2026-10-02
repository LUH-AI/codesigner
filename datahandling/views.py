from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from core.processing import AUTO, CHOICES, choices_of, resolve
from ui.permissions import EDIT, VIEW, experiment_view, policy
from ui.registry import MODELS

from .sections import BY_SLUG, SECTIONS

#: What each processing choice is called on the page.
CHOICE_LABELS = {
    "impute": gettext_lazy("Fill in"),
    "keep": gettext_lazy("Leave for the model"),
    "standardize": gettext_lazy("Standardize"),
    "none": gettext_lazy("Leave as they are"),
    "one_hot": gettext_lazy("One column per label"),
    "categories": gettext_lazy("As categories"),
}

#: What a resolved plan value means, for "As the model needs (…)".
PLAN_LABELS = {
    **CHOICE_LABELS,
    "codes": gettext_lazy("as categories"),
    "text": gettext_lazy("as categories, in words"),
    "numbers": gettext_lazy("as numbers"),
}


def _has_trials(exp):
    result = exp.data.result or {}
    return bool(result.get("data") or result.get("trials"))


def _steps(exp, section):
    """The processing controls *section* shows for *exp*: each with its
    choices, the one made, and what "as the model needs" comes to here."""
    model = MODELS.get(exp.data.model_name)
    chosen = choices_of(exp.data.processing)
    needs = None
    if model is not None:
        model = model.__class__()
        model.task = exp.data.task
        needs, _overruled = resolve({}, model)
    steps = []
    for capability in section.planned:
        if not capability.step:
            continue
        auto = (_("As the model needs (%(model)s: %(what)s)") % {
            "model": model.name, "what": str(PLAN_LABELS[needs[capability.step]]).lower()}
            if needs else _("As the model needs"))
        steps.append({
            "capability": capability,
            "name": capability.step,
            "options": [(value, auto if value == AUTO else CHOICE_LABELS[value])
                        for value in CHOICES[capability.step]],
            "value": chosen[capability.step],
        })
    return steps


def _page(request, exp, section):
    has_trials = _has_trials(exp)
    may_edit = policy().may(request, exp, EDIT)
    if request.method == "POST" and section.slug in ("cleaning", "features"):
        if not may_edit:
            raise Http404
        if has_trials:
            messages.error(request, _("This experiment has trials, and they were scored on "
                                      "its columns processed as they are now; changing that "
                                      "would make the next ones incomparable."))
        else:
            processing = choices_of(exp.data.processing)
            for step, options in CHOICES.items():
                value = request.POST.get(step)
                if value in options:
                    processing[step] = value
            exp.data.processing = processing
            exp.data.save(update_fields=["processing"])
            messages.success(request, _("Saved."))
        return redirect(request.path)
    return render(request, "datahandling/section.html", {
        "experiment": exp,
        "sections": SECTIONS,
        "section": section,
        "steps": _steps(exp, section),
        "locked": has_trials or not may_edit,
        "uploaded_model": bool(exp.data.model_file),
    })


@experiment_view(VIEW)
def data_overview(request, exp):
    """The first of the sections, at the section's own root."""
    return _page(request, exp, SECTIONS[0])


@experiment_view(VIEW)
def data_section(request, exp, section):
    """Any other section, by its slug. The Cleaning and Features pages also
    save the processing choices they show."""
    found = BY_SLUG.get(section)
    if found is None or found is SECTIONS[0]:
        raise Http404
    return _page(request, exp, found)
