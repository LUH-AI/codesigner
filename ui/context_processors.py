from .models import Experiment


def sidebar_experiments(request):
    """Expose the saved experiments to every template, for the sidebar list."""
    return {"sidebar_experiments": Experiment.objects.all()}
