"""A stated prior against this experiment's runs: the budget its decay is
measured in, and the scale its acceptance tolerance is a share of. The
arithmetic is `core.priors`'s."""

import logging

import numpy as np

from core import io
from core import priors as prior_math
from core.metrics import scale as metric_scale_of
from core.registry import METRICS, MODELS, OPTIMIZERS

from .snapshot import snapshot_from_experiment

logger = logging.getLogger(__name__)


def run_budget(exp, offset):
    """The budget a run starting at *offset* trials would be given.

    Taken from the most recent run's own trial cap, because that is where the
    cap lives — an experiment has no budget of its own, each run is started with
    one. A run bounded by a deadline or a target score instead has no cap, and
    `scenario_budget` substitutes the same stand-in the run itself would use.

    *offset* is where the prior was stated, not how many trials have since
    finished. That distinction is the whole of the decay: β is fixed at the
    anchor and only the step count moves, so the exponent falls. Passing the
    live count instead grows β and the denominator together, and the prior
    stops fading — which is what this did before, and it drew a decaying prior
    getting stronger.
    """
    from core.optimizers.smac_optimizer import scenario_budget

    last = exp.runs.order_by("-pk").first()
    return scenario_budget(offset, (last.stopping or {}).get("max_trials") if last else None)


def decayed_prior(exp, hp_name, trials, inside_design=False):
    """What was stated, and how strongly it still counts.

    A prior does not weight the acquisition by its own density but by that
    density raised to a decay exponent, which shrinks as the run proceeds. The
    figure draws what the search would actually weight by *now*, so the exponent
    is computed here against the current trial count rather than pinned at one.

    Computed server-side because the schedules are SMAC's — reimplementing six
    curves in the browser would be a second definition of a decay to keep in
    step with the first. One number crosses instead.
    """
    stated = (exp.data.priors or {}).get(hp_name)
    if not stated:
        return None
    budget = run_budget(exp, int(stated.get("at_trial") or 0))
    return prior_math.with_decay(stated, trials, budget, inside_design)


def metric_scale(exp, metric_name, result):
    """How far apart a perfect and a worthless score of *metric_name* are, in
    its units, for the prior tolerance to be a share of — or None.

    A metric with two ends is its range, which for an accuracy or an error rate
    is the [0, 1] DynaBO's threshold was tuned on. One without is measured from
    its best end to what a model that knows nothing scores on this experiment's
    validation rows, the same score a failed trial is given. An experiment
    whose data is not here falls back to the spread of the scores its trials
    reached.
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
