from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _

from core.processing import CHOICES, choices_of
from ui.layout import data_tabs
from ui.permissions import EDIT, VIEW, experiment_view, policy
from ui.services import history

from .panels import ProcessingStep
from .sections import BY_SLUG, SECTIONS


def _has_trials(exp):
    result = exp.data.result or {}
    return bool(result.get("data") or result.get("trials"))


def why_fixed(exp):
    """Why *exp*'s processing can no longer change, or None if it can."""
    if _has_trials(exp):
        return _("This experiment has trials, and they were scored on its columns "
                 "processed as they are now; changing that would make the next ones "
                 "incomparable.")
    if exp.is_running:
        return _("A run is under way, and it is using the columns processed as they "
                 "are now.")
    return None


def save_processing(request, exp):
    """Store the processing choices posted by *request* on *exp*. Returns why
    it could not, or None.

    Each choice that changed is an entry in the experiment's history (a draft
    has none; its choices are part of how it was set up)."""
    refusal = why_fixed(exp)
    if refusal:
        return refusal
    before = choices_of(exp.data.processing)
    processing = dict(before)
    for step, options in CHOICES.items():
        value = request.POST.get(step)
        if value in options:
            processing[step] = value
    exp.data.processing = processing
    exp.data.save(update_fields=["processing"])
    for step in CHOICES:
        if processing[step] != before[step]:
            history.record(exp, "processing_changed", user=request.user,
                           step=step, old=before[step], new=processing[step])
    return None


def tabs_for(exp, url_for, setup=False):
    """The page's tabs for *exp*: each section, its address (`url_for`), and its
    slots — every panel with what it needs to render — and whether the tab
    holds controls to save. With *setup*, a draft's, whose Overview is its own
    (`ui.layout.data_tabs`)."""
    tabs = []
    for tab in data_tabs(exp, setup=setup):
        slots = []
        for slot in tab["slots"]:
            figure = slot["figure"]
            control = figure.control(exp) if issubclass(figure, ProcessingStep) else None
            slots.append({**slot, "control": control})
        tabs.append({**tab, "url": url_for(tab["section"]), "slots": slots,
                     "has_controls": any(s["control"] for s in slots)})
    return tabs


def _url(exp, section):
    if section is SECTIONS[0]:
        return reverse("datahandling:data_overview", args=[exp.pk])
    return reverse("datahandling:data_section", args=[exp.pk, section.slug])


def _page(request, exp, section):
    may_edit = policy().may(request, exp, EDIT)
    if request.method == "POST":
        if not may_edit:
            raise Http404
        refusal = save_processing(request, exp)
        if refusal:
            messages.error(request, refusal)
        else:
            messages.success(request, _("Saved."))
        return redirect(request.path)

    # One page, every section a tab of it: the experiment's own heading above
    # (as on its dashboard and timeline), the section asked for showing.
    from ui.views import experiment_header

    fixed = why_fixed(exp)
    return render(request, "datahandling/section.html", {
        **experiment_header(request, exp),
        "tabs": tabs_for(exp, lambda s: _url(exp, s)),
        "section": section,
        "locked": bool(fixed) or not may_edit,
        "locked_reason": fixed,
        "uploaded_model": bool(exp.data.model_file),
    })


@experiment_view(VIEW)
def data_overview(request, exp):
    """The first of the sections, at the section's own root."""
    return _page(request, exp, SECTIONS[0])


@experiment_view(VIEW)
def data_section(request, exp, section):
    """Any other section, by its slug."""
    found = BY_SLUG.get(section)
    if found is None or found is SECTIONS[0]:
        raise Http404
    return _page(request, exp, found)
