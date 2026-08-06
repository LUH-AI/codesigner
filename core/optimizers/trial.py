"""One trial, measured: predict, score, and record what happened.

Every optimizer's inner loop is this function. It exists because a trial can now
fail in ways that are not the run's fault — a model that raises on a particular
configuration, one that stops answering — and all three optimizers must treat
those identically, or a run's history would depend on which one produced it.

It is also where scoring happens, now that models return predictions rather than
scores: the validation labels stay here and never reach the model.

A trial is evaluated over the folds of `core.splits.Splits` — one for a holdout,
k for cross-validation — and the fold scores are averaged. Whichever it is, the
trial still yields one score per metric, so nothing downstream (the figures, the
incumbent, resume, `.ihpo`) has to know which happened.
"""

import numpy as np

from ..metrics import score_all
from ..modelhost.errors import ModelTrialError, TrialTimeout
from .timing import STATUS_CRASHED, STATUS_TIMEOUT, timed_evaluation


def evaluate_trial(model, config, splits, metrics, seed=0):
    """Run *config* through *model* over every fold of *splits*, and score it.

    Returns ``(scores, run_info)``. A trial that failed — the model raised, was
    stopped for exceeding its deadline, or returned something unscoreable —
    scores 0.0 on every metric, with ``run_info["status"]`` set and the reason
    under ``additional_info["error"]``. That is a data point (this configuration
    is unusable), not the end of the run; whether too many failures in a row
    should stop the search is the collector's decision, not this function's.

    A fold that fails fails the whole trial. A configuration that works on four
    fifths of the data and dies on the rest is not one to hand the search as a
    partial success.

    Two things are deliberately *not* caught. `TrialCancelled` means the loop
    should stop and hand back what it has, and only the loop can do that.
    `ModelProcessError` means the conversation with the model is broken rather
    than the model disagreeing with a configuration, so continuing would just
    produce a run of identical failures.
    """
    failure = None
    status = STATUS_CRASHED
    fold_scores = []
    cpu_reported = 0.0

    # A model in another process cannot be handed a slice — its copy of the
    # dataset is over there — so it is told which fold instead. A local model
    # needs nothing: it gets the arrays. Asked for rather than required, so the
    # contract user models implement does not grow a method none of them wants.
    select_fold = getattr(model, "use_fold", None)

    with timed_evaluation(seed=seed) as run_info:
        for index, (train_idx, val_idx) in enumerate(splits.folds):
            if select_fold is not None:
                select_fold(index)
            try:
                y_pred = model.fit_predict(
                    config, splits.X[train_idx], splits.y[train_idx],
                    splits.X[val_idx], seed=seed)
            except TrialTimeout as exc:
                failure, status = str(exc), STATUS_TIMEOUT
                break
            except ModelTrialError as exc:
                failure = str(exc)
                break
            except Exception as exc:  # noqa: BLE001 — a local model may fail any way it likes
                failure = f"{type(exc).__name__}: {exc}"
                break

            # A model running in its own process timed itself: this thread's CPU
            # clock measured none of the work. Summed, because each fold is a
            # separate round trip. See core.modelhost.
            reported = getattr(model, "last_cpu_time", None)
            if reported is not None:
                cpu_reported += reported

            try:
                fold_scores.append(score_all(splits.y[val_idx], y_pred, metrics))
            except Exception as exc:  # noqa: BLE001 — wrong label type, wrong length
                failure = f"predictions could not be scored: {type(exc).__name__}: {exc}"
                break

    if cpu_reported:
        run_info["cpu_time"] = cpu_reported

    if failure is None:
        scores = {name: float(np.mean([s[name] for s in fold_scores])) for name in metrics}
    else:
        scores = {name: 0.0 for name in metrics}
        run_info["status"] = status
        run_info["additional_info"] = {"error": failure}

    return scores, run_info
