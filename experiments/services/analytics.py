"""Computing an experiment's stored analytics after the fact."""

from django.conf import settings

from core import analytics
from core.optimizers.base import game_fields
from experiments.models import ExperimentData

from .snapshot import config_space_for, rebuild_experiment


def compute_games(exp) -> None:
    """Compute *exp*'s explanation games and write them onto its stored result,
    as a completed run would have. Nothing to do without trials."""
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return
    assert built is not None

    games = analytics.compute_hp_games(
        config_space_for(built), result.trials, built["metrics"], seed=built["seed"],
        budget=settings.ANALYTICS_EAGER_MAX_COALITIONS or None)

    for name, value in game_fields(games).items():
        setattr(result, name, value)

    ExperimentData.objects.filter(experiment__pk=exp.pk).update(
        result=built["optimizer"].serialize_result(result))
