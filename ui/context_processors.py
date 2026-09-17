from django.conf import settings

from . import navigation as nav
from .permissions import policy, visible_experiments

#: The sidebar is rendered on every page, so it stays cheap and recent rather
#: than listing every experiment an instance has ever run — the full,
#: paginated list lives at `ui:experiment_list`, linked from the sidebar.
SIDEBAR_LIMIT = 20


def sidebar_experiments(request):
    """Expose the most recent saved experiments to every template, for the
    sidebar list — see `ui:experiment_list` for the rest.

    Through the policy, so the sidebar never lists an experiment a page would
    then refuse to open. `Experiment.Meta.ordering` (`-created_at`) already
    puts the most recent first, so slicing is all a "most recent N" needs.
    """
    return {"sidebar_experiments": visible_experiments(request)[:SIDEBAR_LIMIT]}


_SETTINGS_URL_NAMES = {"appearance", "account", "default_experiment_settings"}
_GROUP_URL_NAMES = {"group_people", "group_add_person", "group_remove_person",
                    "group_work"}
_SITE_URL_NAMES = {"site_groups", "site_group_save", "site_usage", "site_jobs",
                   "site_job_stop"}


def active_tab(request):
    """Which icon-rail tab is selected, driving which inner sidebar renders."""
    url_name = getattr(request.resolver_match, "url_name", None)
    if url_name in _SETTINGS_URL_NAMES:
        tab = "settings"
    elif url_name in _GROUP_URL_NAMES:
        tab = "group"
    elif url_name in _SITE_URL_NAMES:
        tab = "site"
    else:
        tab = "experiments"
    return {"active_tab": tab}


def navigation(request):
    """Where this page sits (breadcrumbs) and where the user came from (?next=).

    Both are derived from the request, so pages get them without passing
    anything — see ui/navigation.py for why the two are separate.
    """
    return {
        "breadcrumbs": nav.breadcrumbs(request),
        "back_link": nav.back_link(request),
    }


def offered_languages(request):
    """The languages the rail's selector shows.

    Not `LANGUAGES`, which is what Django will actually serve and is English
    alone until the German and Spanish drafts have been reviewed. The selector
    shows what the interface is going to offer so that its place on the page is
    settled before the catalogs are; picking one does nothing yet. See
    `OFFERED_LANGUAGES` in config/settings.py.
    """
    return {"offered_languages": settings.OFFERED_LANGUAGES}


def capabilities(request):
    """Instance-wide permissions the layout itself branches on.

    Only what a template outside one experiment's pages needs — the sidebar's
    link to the default experiment settings, and the button that promotes an
    experiment's settings to be those defaults.
    """
    from access.policy import is_lead, is_site_admin

    user = getattr(request, "user", None)
    return {
        "may_change_defaults": policy().may_change_defaults(request),
        # Which rail tabs exist for this person. Computed here rather than in
        # the template so the layout asks one question rather than reaching
        # into the policy through three.
        "is_group_lead": is_lead(user),
        "is_site_admin": is_site_admin(user),
    }
