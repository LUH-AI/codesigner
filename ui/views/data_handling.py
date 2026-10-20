from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _

from access.policy import EDIT, VIEW, policy
from experiments.services import processing

from ..figures.data_handling import BY_SLUG, SECTIONS, ProcessingStep
from ..layout import data_tabs
from ..permissions import experiment_view
from .dashboard import experiment_header


def tabs_for(exp, url_for, setup=False):
    """The Data Handling page's tabs for *exp*: each section, its address
    (`url_for`), and its slots — every panel with what it needs to render — and
    whether the tab holds controls to save. With *setup*, a draft's, whose
    Overview is its own (`ui.layout.data_tabs`)."""
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


def _data_url(exp, section):
    if section is SECTIONS[0]:
        return reverse("ui:data_overview", args=[exp.pk])
    return reverse("ui:data_section", args=[exp.pk, section.slug])


def _data_page(request, exp, section):
    may_edit = policy().may(request, exp, EDIT)
    if request.method == "POST":
        if not may_edit:
            raise Http404
        refusal = processing.save(exp, request.POST, request.user)
        if refusal:
            messages.error(request, refusal)
        else:
            messages.success(request, _("Saved."))
        return redirect(request.path)

    # One page, every section a tab of it: the experiment's own heading above
    # (as on its dashboard and timeline), the section asked for showing.
    fixed = processing.why_fixed(exp)
    return render(request, "ui/data_handling/section.html", {
        **experiment_header(request, exp),
        "tabs": tabs_for(exp, lambda s: _data_url(exp, s)),
        "section": section,
        "locked": bool(fixed) or not may_edit,
        "locked_reason": fixed,
        "uploaded_model": bool(exp.data.model_file),
    })


@experiment_view(VIEW)
def data_overview(request, exp):
    """The first of the sections, at the section's own root."""
    return _data_page(request, exp, SECTIONS[0])


@experiment_view(VIEW)
def data_section(request, exp, section):
    """Any other section, by its slug."""
    found = BY_SLUG.get(section)
    if found is None or found is SECTIONS[0]:
        raise Http404
    return _data_page(request, exp, found)
