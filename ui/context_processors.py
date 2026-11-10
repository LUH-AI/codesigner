from django.conf import settings

from . import navigation as nav
from access.policy import policy, visible_experiments

#: Every page renders the sidebar; `ui:experiment_list` has the rest.
SIDEBAR_LIMIT = 20


#: Dashboard-header pages count as the dashboard.
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
    "experiment_notes": "notes",
    # Drafts lack views.
    "setup": "setup",
    "setup_optimizer": "setup",
    "setup_data": "setup",
    "setup_priors": "setup",
}


def sidebar_experiments(request):
    """`sidebar_experiments` appends the open experiment when it is missing.
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


def active_tab(request):
    url_name = getattr(request.resolver_match, "url_name", None)
    if url_name in _SETTINGS_URL_NAMES:
        tab = "settings"
    elif url_name in _GROUP_URL_NAMES:
        tab = "group"
    else:
        tab = "experiments"
    return {"active_tab": tab}


def navigation(request):
    return {
        "breadcrumbs": nav.breadcrumbs(request),
        "back_link": nav.back_link(request),
    }


def offered_languages(request):
    """A test keeps `OFFERED_LANGUAGES` equal to `LANGUAGES`.
    """
    return {"offered_languages": settings.OFFERED_LANGUAGES}


def legal(request):
    """`legal` tells the footer which of the deployment's legal pages exist,
    and gives an empty mapping while both hooks are unset."""
    links = {"imprint": bool(settings.LEGAL_PAGES), "privacy": bool(settings.LEGAL_PAGES),
             "terms": bool(settings.ACCOUNT_AGREEMENT)}
    return {"legal_links": links if any(links.values()) else {}}


def capabilities(request):
    """`capabilities` exposes instance-wide permissions to the layout.
    """
    from access.policy import is_lead, memberships_of

    user = getattr(request, "user", None)
    return {
        "may_change_defaults": policy().may_change_defaults(request),
        # Computed once for the rail tabs.
        "is_group_lead": is_lead(user),
        # Group tab sidebar, from cached memberships.
        "my_groups": sorted((m.group for m in memberships_of(user)),
                            key=lambda g: g.name.lower()),
    }


#: A per-browser cookie sets the theme before the page draws.
THEMES = ("light", "dark", "system")
THEME_COOKIE = "codesigner-theme"


def theme(request):
    """`theme` falls back to "system"."""
    chosen = request.COOKIES.get(THEME_COOKIE)
    return {"theme": chosen if chosen in THEMES else "system"}


def storage(request):
    """`storage` feeds the nearly-full-group banner."""
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
