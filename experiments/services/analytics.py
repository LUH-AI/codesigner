"""Computing an experiment's stored analytics after the fact."""

from django.conf import settings

from core.optimizers.base import HP_GAME_FIELDS
from experiments.models import ExperimentData

from .snapshot import config_space_for, rebuild_experiment


def compute_games(exp) -> None:
    """Compute *exp*'s explanation games and write them onto its stored result,
    as a completed run would have. Nothing to do without trials."""
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return

    optimizer = built["optimizer"]
    optimizer.analytics_max_coalitions = settings.ANALYTICS_EAGER_MAX_COALITIONS or None
    games = optimizer.compute_hp_games(
        config_space_for(built), result.trials, built["metrics"], seed=built["seed"])

    for game, fields in HP_GAME_FIELDS.items():
        importance, warning, interactions, moebius, total = games[game]
        setattr(result, fields["importance"], importance)
        setattr(result, fields["warning"], warning)
        setattr(result, fields["interactions"], interactions)
        setattr(result, fields["moebius"], moebius)
    result.hyperparameter_interactions_warning = games["tunability"][1]
    result.hyperparameter_tunability_total = games["tunability"][4]

    ExperimentData.objects.filter(experiment__pk=exp.pk).update(
        result=optimizer.serialize_result(result))
