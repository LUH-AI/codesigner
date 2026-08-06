"""Where a page sits, and where the user came from.

Two separate things, deliberately kept separate:

**Breadcrumbs** say where a page *sits* — its place in the app, which is the
same however you arrived. They are derived from the resolved URL here, so a
page's trail follows from its route instead of being declared in its template.

**The back link** says where the user *came from*, and only exists for links
that jump out of the hierarchy: the global default-settings page reached from
one experiment has no parent experiment, so the origin has to be carried. That
is what `?next=` is for — Django's own idiom (`django.contrib.auth` uses it for
login redirects) — and it is validated as a local URL before being trusted.

What this deliberately is *not*: a back stack in the session (two tabs would
fight over one stack, and it would drift out of step with the browser's own back
button) or anything based on the Referer header (routinely stripped, absent on a
fresh load, and untrusted input). The browser already handles history; these
express structure, which survives a bookmark or a shared link.
"""

from urllib.parse import urlparse

from django.urls import Resolver404, resolve, reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext_lazy as _

#: The query parameter carrying where to return to.
NEXT_PARAM = "next"

#: Pages that live under one experiment, and what to call them. The experiment
#: itself is the parent crumb; None means the page *is* the experiment.
_EXPERIMENT_PAGES = {
    "experiment_detail": None,
    "experiment_settings": _("Settings"),
    "experiment_delete": _("Delete"),
    # experiment_run only renders a page when it asks about a metric change.
    "experiment_run": _("Change metric"),
}


def _experiment(request, pk):
    """The experiment named by a route, if this request may see it.

    Through the policy, so a crumb never names an experiment its own link would
    404 on — the trail would otherwise be a directory of what exists.
    """
    from .permissions import visible_experiments

    return visible_experiments(request).filter(pk=pk).first()


def _crumbs_for(request, url_name, kwargs):
    """(label, url) pairs for a route: root first, the page itself last.

    The last crumb has no url — it is the page you are on. An unknown route
    gets no trail rather than a guess.
    """
    experiments = (_("Experiments"), reverse("ui:home"))
    settings_root = (_("Settings"), reverse("ui:appearance"))

    if url_name == "home":
        return [experiments]
    if url_name == "new_experiment":
        return [experiments, (_("New experiment"), None)]
    if url_name == "import_experiment":
        return [experiments, (_("Import experiment"), None)]
    if url_name == "appearance":
        return [settings_root, (_("Appearance"), None)]
    if url_name == "default_experiment_settings":
        return [settings_root, (_("Default experiment settings"), None)]

    if url_name in _EXPERIMENT_PAGES:
        exp = _experiment(request, kwargs.get("pk"))
        if exp is None:
            return [experiments]
        itself = (exp.name, reverse("ui:experiment_detail", args=[exp.pk]))
        page = _EXPERIMENT_PAGES[url_name]
        return [experiments, itself] if page is None else [experiments, itself, (page, None)]

    return []


def breadcrumbs(request):
    """The trail for the page being rendered."""
    match = getattr(request, "resolver_match", None)
    if match is None:
        return []
    return [{"label": label, "url": url}
            for label, url in _crumbs_for(request, match.url_name, match.kwargs)]


def safe_next(request):
    """The `?next=` target, if it is a URL on this site. None otherwise.

    Unvalidated, this would be an open redirect — anyone could hand out a link
    that sends a user off-site from a page that looks like ours.
    """
    target = request.GET.get(NEXT_PARAM) or request.POST.get(NEXT_PARAM)
    if target and url_has_allowed_host_and_scheme(
            target, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
        return target
    return None


def back_link(request):
    """Where the user came from, labelled by resolving it. None if not carried.

    The label comes from the same trail definition as the breadcrumbs, so the
    two can never disagree about what a page is called.
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
