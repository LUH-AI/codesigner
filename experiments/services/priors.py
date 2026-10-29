"""`priors` supplies a stated prior's run budget and metric scale; `core.priors`
does the arithmetic."""

import logging

import numpy as np

from core import io
from core import priors as prior_math
from core.metrics import scale as metric_scale_of
from core.registry import METRICS, MODELS, OPTIMIZERS

from .snapshot import snapshot_from_experiment

logger = logging.getLogger(__name__)


def run_budget(exp, offset):
    """`run_budget` takes the trial cap of the latest run, since an experiment
    has no budget of its own.

    *offset* is the anchor trial where the prior was stated; passing the live
    count instead stops the prior from fading.
    """
    from core.optimizers.smac.facade import scenario_budget

    last = exp.runs.order_by("-pk").first()
    return scenario_budget(offset, (last.stopping or {}).get("max_trials") if last else None)


def decayed_prior(exp, hp_name, trials, inside_design=False):
    """`decayed_prior` computes the decay exponent at the current trial count.
    The server computes the exponent because the six decay schedules are SMAC's.
    """
    stated = (exp.data.priors or {}).get(hp_name)
    if not stated:
        return None
    budget = run_budget(exp, int(stated.get("at_trial") or 0))
    return prior_math.with_decay(stated, trials, budget, inside_design)


def metric_scale(exp, metric_name, result):
    """`metric_scale` uses a bounded metric's range, else the distance from its
    best end to the null score. The spread of trial scores is the last fallback.
    """
    from core.optimizers.trial import null_scores

    metric = result.metric(metric_name)
    found = metric_scale_of(metric)
    if found is not None:
        return found
    try:
        built = io.build_experiment(snapshot_from_experiment(exp),
                                    METRICS, MODELS, OPTIMIZERS, load_model=False)[1]
        null = null_scores({metric_name: metric}, built["splits"])[metric_name]
        found = metric_scale_of(metric, null)
    except Exception as error:  # noqa: BLE001 — the fallback below still answers
        logger.info("No null score for %s: %s", metric_name, error)
    if found is not None:
        return found
    scores = [t.scores.get(metric_name) for t in result.trials]
    scores = [float(v) for v in scores if v is not None and np.isfinite(v)]
    spread = max(scores) - min(scores) if scores else 0.0
    return spread if spread > 0 else None
