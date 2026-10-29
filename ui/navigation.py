"""Breadcrumbs give a page's place in the app and come from its resolved URL.
The back link gives where the user came from, carried in a validated `?next=`,
for pages reached from outside their hierarchy.
"""

from urllib.parse import urlparse

from django.urls import Resolver404, resolve, reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext_lazy as _

NEXT_PARAM = "next"

#: `_EXPERIMENT_PAGES` defines a crumb label for each per-experiment route, and
#: None stands for the experiment's own page.
_EXPERIMENT_PAGES = {
    "experiment_detail": None,
    "experiment_settings": _("Settings"),
    "experiment_delete": _("Delete"),
    # Only rendered for a metric change.
    "experiment_run": _("Change metric"),
    "experiment_timeline": _("Timeline"),
}


def _experiment(request, pk):
    """`_experiment` filters through the policy, so crumbs never reveal hidden
    experiments.
    """
    from access.policy import visible_experiments

    return visible_experiments(request).filter(pk=pk).first()


def _crumbs_for(request, url_name, kwargs):
    """`_crumbs_for` builds label and url pairs from the root; unknown routes
    get none.
    """
    experiments = (_("Experiments"), reverse("ui:home"))
    settings_root = (_("Settings"), reverse("ui:appearance"))

    if url_name == "home":
        return [experiments]
    if url_name == "experiment_list":
        return [experiments, (_("All experiments"), None)]
    if url_name == "experiment_bin":
        return [experiments, (_("Bin"), None)]
    if url_name == "new_experiment":
        return [experiments, (_("New experiment"), None)]
    if url_name == "import_experiment":
        return [experiments, (_("Import experiment"), None)]
    if url_name == "appearance":
        return [settings_root, (_("Appearance"), None)]
    if url_name == "export_preferences":
        return [settings_root, (_("Export"), None)]
    if url_name == "default_experiment_settings":
        return [settings_root, (_("Default experiment settings"), None)]

    if url_name in _EXPERIMENT_PAGES:
        exp = _experiment(request, kwargs.get("pk"))
        if exp is None:
            return [experiments]
        itself = (exp.title, reverse("ui:experiment_detail", args=[exp.pk]))
        page = _EXPERIMENT_PAGES[url_name]
        return [experiments, itself] if page is None else [experiments, itself, (page, None)]

    if url_name in ("setup", "setup_optimizer", "setup_data", "setup_priors"):
        # Drafts have no dashboard, so the trail ends at setup.
        exp = _experiment(request, kwargs.get("pk"))
        if exp is None:
            return [experiments]
        return [experiments, (exp.title, None), (_("Setup"), None)]

    if url_name in ("data_overview", "data_section"):
        # Sections are tabs, so the trail ignores them.
        exp = _experiment(request, kwargs.get("pk"))
        if exp is None:
            return [experiments]
        itself = (exp.title, reverse("ui:experiment_detail", args=[exp.pk]))
        return [experiments, itself, (_("Data Handling"), None)]

    return []


def breadcrumbs(request):
    match = getattr(request, "resolver_match", None)
    if match is None:
        return []
    return [{"label": label, "url": url}
            for label, url in _crumbs_for(request, match.url_name, match.kwargs)]


def safe_next(request):
    """`safe_next` keeps `?next=` only on the same host, blocking open
    redirects.
    """
    target = request.GET.get(NEXT_PARAM) or request.POST.get(NEXT_PARAM)
    if target and url_has_allowed_host_and_scheme(
            target, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
        return target
    return None


def back_link(request):
    """`back_link` reuses the breadcrumb label, so the two always agree.
    """
    target = safe_next(request)
    if not target:
        return None
    try:
        match = resolve(urlparse(target).path)
    except Resolver404:
        return None
    crumbs = _crumbs_for(request, match.url_name, match.kwargs)
    if not crumbs:
        return None
    return {"label": crumbs[-1][0], "url": target}
