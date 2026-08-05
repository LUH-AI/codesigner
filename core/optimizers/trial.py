"""One trial, measured: predict, score, and record what happened.

Every optimizer's inner loop is this function. It exists because a trial can now
fail in ways that are not the run's fault — a model that raises on a particular
configuration, one that stops answering — and all three optimizers must treat
those identically, or a run's history would depend on which one produced it.

It is also where scoring happens, now that models return predictions rather than
scores: the validation labels stay here and never reach the model.
"""

from ..metrics import score_all
from ..modelhost.errors import ModelTrialError, TrialTimeout
from .timing import STATUS_CRASHED, STATUS_TIMEOUT, timed_evaluation


def evaluate_trial(model, config, X_train, y_train, X_val, y_val, metrics, seed=0):
    """Run *config* through *model* and score its predictions against *y_val*.

    Returns ``(scores, run_info)``. A trial that failed — the model raised, was
    stopped for exceeding its deadline, or returned something unscoreable —
    scores 0.0 on every metric, with ``run_info["status"]`` set and the reason
    under ``additional_info["error"]``. That is a data point (this configuration
    is unusable), not the end of the run; whether too many failures in a row
    should stop the search is the collector's decision, not this function's.

    Two things are deliberately *not* caught. `TrialCancelled` means the loop
    should stop and hand back what it has, and only the loop can do that.
    `ModelProcessError` means the conversation with the model is broken rather
    than the model disagreeing with a configuration, so continuing would just
    produce a run of identical failures.
    """
    failure = None
    status = STATUS_CRASHED
    y_pred = None

    with timed_evaluation(seed=seed) as run_info:
        try:
            y_pred = model.fit_predict(config, X_train, y_train, X_val, seed=seed)
        except TrialTimeout as exc:
            failure, status = str(exc), STATUS_TIMEOUT
        except ModelTrialError as exc:
            failure = str(exc)
        except Exception as exc:  # noqa: BLE001 — a local model may fail any way it likes
            failure = f"{type(exc).__name__}: {exc}"

    # A model running in its own process timed itself: this thread's CPU clock
    # measured none of the work. See core.modelhost.
    reported = getattr(model, "last_cpu_time", None)
    if reported is not None:
        run_info["cpu_time"] = reported

    if failure is None:
        try:
            scores = score_all(y_val, y_pred, metrics)
        except Exception as exc:  # noqa: BLE001 — wrong label type, wrong length
            failure = f"predictions could not be scored: {type(exc).__name__}: {exc}"

    if failure is not None:
        scores = {name: 0.0 for name in metrics}
        run_info["status"] = status
        run_info["additional_info"] = {"error": failure}

    return scores, run_info
