from . import navigation as nav
from .permissions import visible_experiments


def sidebar_experiments(request):
    """Expose the saved experiments to every template, for the sidebar list.

    Through the policy, so the sidebar never lists an experiment a page would
    then refuse to open.
    """
    return {"sidebar_experiments": visible_experiments(request)}


_SETTINGS_URL_NAMES = {"appearance", "default_experiment_settings"}


def active_tab(request):
    """Which icon-rail tab is selected, driving which inner sidebar renders."""
    url_name = getattr(request.resolver_match, "url_name", None)
    return {"active_tab": "settings" if url_name in _SETTINGS_URL_NAMES else "experiments"}


def navigation(request):
    """Where this page sits (breadcrumbs) and where the user came from (?next=).

    Both are derived from the request, so pages get them without passing
    anything — see ui/navigation.py for why the two are separate.
    """
    return {
        "breadcrumbs": nav.breadcrumbs(request),
        "back_link": nav.back_link(request),
    }
