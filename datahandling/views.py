from django.http import Http404
from django.shortcuts import render

from ui.permissions import VIEW, experiment_view

from .sections import BY_SLUG, SECTIONS


def _page(request, exp, section):
    return render(request, "datahandling/section.html", {
        "experiment": exp,
        "sections": SECTIONS,
        "section": section,
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
