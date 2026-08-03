from .models import Experiment


def sidebar_experiments(request):
    """Expose the saved experiments to every template, for the sidebar list."""
    return {"sidebar_experiments": Experiment.objects.all()}


_SETTINGS_URL_NAMES = {"appearance", "default_experiment_settings"}


def active_tab(request):
    """Which icon-rail tab is selected, driving which inner sidebar renders."""
    url_name = getattr(request.resolver_match, "url_name", None)
    return {"active_tab": "settings" if url_name in _SETTINGS_URL_NAMES else "experiments"}
