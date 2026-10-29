from django.conf import settings

from . import navigation as nav
from access.policy import policy, visible_experiments

#: The sidebar renders on every page, so it only lists recent experiments;
#: `ui:experiment_list` has the rest.
SIDEBAR_LIMIT = 20


#: Each route maps to an experiment view, and pages reached from the dashboard's
#: header count as the dashboard.
_EXPERIMENT_VIEWS: dict[str | None, str] = {
    "experiment_detail": "dashboard",
    "experiment_settings": "dashboard",
    "experiment_delete": "dashboard",
    "experiment_run": "dashboard",
    "experiment_optimizer": "dashboard",
    "experiment_rename": "dashboard",
    "data_overview": "data",
    "data_section": "data",
    "experiment_timeline": "timeline",
    # An open draft has no views yet.
    "setup": "setup",
    "setup_optimizer": "setup",
    "setup_data": "setup",
    "setup_priors": "setup",
}


def sidebar_experiments(request):
    """`sidebar_experiments` gives templates the recent visible experiments plus
    the open one, which joins the list when missing.
    """
    experiments = list(visible_experiments(request)[:SIDEBAR_LIMIT])
    match = getattr(request, "resolver_match", None)
    view = _EXPERIMENT_VIEWS.get(getattr(match, "url_name", None))
    open_pk = match.kwargs.get("pk") if view and match is not None else None

    if open_pk is not None and all(e.pk != open_pk for e in experiments):
        opened = policy().experiments(request).filter(pk=open_pk).first()
        if opened is None:
            open_pk = view = None
        else:
            experiments.append(opened)

    return {"sidebar_experiments": experiments,
            "sidebar_open": open_pk, "sidebar_view": view}


_SETTINGS_URL_NAMES = {"appearance", "export_preferences", "account",
                       "default_experiment_settings"}
_GROUP_URL_NAMES = {"group_index", "group_detail", "group_add_person",
                    "group_remove_person", "group_set_role",
                    "group_transfer_primary"}
_SITE_URL_NAMES = {"site_groups", "site_group_save", "site_group_delete",
                   "site_usage", "site_jobs", "site_job_stop",
                   "site_trash", "site_trash_rehome", "site_trash_delete"}


def active_tab(request):
    """`active_tab`'s rail tab chooses the inner sidebar."""
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
    """`navigation` derives breadcrumbs and the `?next=` back link from the
    request.
    """
    return {
        "breadcrumbs": nav.breadcrumbs(request),
        "back_link": nav.back_link(request),
    }


def offered_languages(request):
    """The rail's language selector offers `OFFERED_LANGUAGES`, which a test
    keeps equal to `LANGUAGES`.
    """
    return {"offered_languages": settings.OFFERED_LANGUAGES}


def capabilities(request):
    """`capabilities` gives the layout instance-wide permissions for pages
    outside an experiment.
    """
    from access.policy import is_lead, is_site_admin, memberships_of

    user = getattr(request, "user", None)
    return {
        "may_change_defaults": policy().may_change_defaults(request),
        # The template reads the signed-in user's rail tabs from here, computed
        # once.
        "is_group_lead": is_lead(user),
        "is_site_admin": is_site_admin(user),
        # Groups for the Group tab's sidebar, from cached memberships.
        "my_groups": sorted((m.group for m in memberships_of(user)),
                            key=lambda g: g.name.lower()),
    }


#: A cookie keeps the theme per browser and works without accounts, before the
#: page draws.
THEMES = ("light", "dark", "system")
THEME_COOKIE = "codesigner-theme"


def theme(request):
    """`theme` returns the chosen theme, or the system's."""
    chosen = request.COOKIES.get(THEME_COOKIE)
    return {"theme": chosen if chosen in THEMES else "system"}


def storage(request):
    """`storage` lists the signed-in person's nearly full groups for the banner."""
    from experiments.services.storage import warnings_for

    from .formatting import human

    try:
        nearly_full = warnings_for(getattr(request, "user", None))
    except Exception:  # noqa: BLE001 — a page must never fail over its banner
        nearly_full = []
    return {"storage_warnings": [
        {"group": w["group"], "used": human(w["total"]), "limit": human(w["limit"]),
         "percent": round(100 * w["share"]), "full": w["share"] >= 1}
        for w in nearly_full]}
