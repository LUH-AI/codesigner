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

import traceback

import numpy as np

from ..metrics import PROBABILITIES, score_all
from ..modelhost.errors import ModelTrialError, TrialCancelled, TrialTimeout
from .timing import CANCELLED, STATUS_CRASHED, STATUS_TIMEOUT, timed_evaluation


def _offers_proba(model) -> bool:
    """Whether a model *in this process* actually implements probabilities.

    Every model inherits the SDK's stub, so presence says nothing — the stub
    marks itself. See `core.modelhost.harness._capabilities`, which asks the
    same question of a model in its own process and has to survive a cached
    environment holding an older SDK.
    """
    fn = getattr(model, "fit_predict_proba", None)
    return callable(fn) and not getattr(fn, "_is_stub", False)


def _null_scores(metrics, splits) -> dict:
    """What a trial that produced nothing scores, per metric.

    Not zero. Zero is what the four metrics this app shipped with happen to give
    a model that knows nothing, and it was written down as though it were a
    property of failure itself — so on an RMSE, where 0.0 is a *perfect* score, a
    crashed trial would have been the best result in the run.

    Each metric says instead what a model that knows nothing gets: still 0.0 for
    accuracy and friends, a coin flip for ROC AUC, the null predictor's error for
    an RMSE. Scored against the labels the trial would have been measured on —
    every fold's validation rows, which under cross-validation is each row once
    and under a holdout is the held-out slice.

    A metric whose null score cannot be computed falls back to the worst end of
    its range, or to 0.0 when it has no worst end. Defensive rather than
    expected: this runs on the path that is already handling a failure, and
    failing there would turn a recorded bad trial into a dead run.
    """
    import numpy as np

    y_true = np.concatenate([splits.y[val] for _, val in splits.folds])
    scores = {}
    for name, metric in metrics.items():
        try:
            scores[name] = float(metric.null_score(y_true))
        except Exception:  # noqa: BLE001 — see above
            worst = getattr(metric, "worst_bound", 0.0)
            scores[name] = float(worst) if worst is not None else 0.0
    return scores


def evaluate_trial(model, config, splits, metrics, seed=0):
    """Run *config* through *model* over every fold of *splits*, and score it.

    Returns ``(scores, run_info)``. A trial that failed — the model raised, was
    stopped for exceeding its deadline, or returned something unscoreable —
    scores what a model that knows nothing would score (see `_null_scores`;
    still 0.0 for every metric this app shipped with), with
    ``run_info["status"]`` set, the reason under
    ``additional_info["error"]`` and, where there is one worth keeping, the
    traceback under ``additional_info["traceback"]``. That is a data point (this
    configuration is unusable), not the end of the run; whether too many
    failures should stop the search is the collector's decision, not this
    function's.

    The reason is one line, for a reader who wants to know what happened; the
    traceback is for one who needs to know where. A timeout has no traceback
    worth storing, since nothing over here is what went wrong.

    A fold that fails fails the whole trial. A configuration that works on four
    fifths of the data and dies on the rest is not one to hand the search as a
    partial success.

    A trial that ran out of time is recorded and the run goes on. Enforcing a
    deadline means killing the child that missed it, so the next trial is
    answered by a replacement — see `RemoteModel._live_process`. Whether a run
    of timeouts is worth continuing is the collector's decision through
    `max_failures`, the same as for any other failure.

    `ModelProcessError` — the conversation itself breaking rather than the model
    disagreeing with a configuration — is caught with everything else and
    recorded as a crashed trial. It reads as one: the failure limits stop a run
    that keeps producing them, and a broken child is replaced before the next
    attempt rather than poisoning every trial after it.

    `TrialCancelled` is the one outcome that is neither a success nor a failure.
    The run was stopped while this trial was in flight, so nothing was measured
    and there is nothing to say about the configuration — scoring it 0.0 would
    put a trial that never happened in the history, at the worst possible score,
    where every figure and every surrogate would read it as a measurement. It is
    marked with `CANCELLED` instead and `TrialCollector.record` drops it, which
    is where that decision belongs: it is the same for all three optimizers, and
    all three already call `record`.
    """
    failure = None
    detail = ""
    cancelled = False
    status = STATUS_CRASHED
    fold_scores = []
    cpu_reported = 0.0

    # A model in another process cannot be handed a slice — its copy of the
    # dataset is over there — so it is told which fold instead. A local model
    # needs nothing: it gets the arrays. Asked for rather than required, so the
    # contract user models implement does not grow a method none of them wants.
    select_fold = getattr(model, "use_fold", None)

    # Whether anything being computed needs more than a label. Asked once, not
    # per fold: it is a property of the metrics, and an SVM fits an internal
    # calibration to answer it — several times the cost of the plain fit — so a
    # run scoring nothing that needs probabilities must never take this path.
    #
    # `supports_proba` is False for a model in another process that did not say
    # it could; a local model is asked directly. Which metrics can be scored at
    # all is settled before the run starts (`core.metrics.usable`), so by here
    # the two agree and this is a route, not a fallback.
    wants_proba = any(m.needs == PROBABILITIES for m in metrics.values())
    can_proba = (model.supports_proba if hasattr(model, "supports_proba")
                 else _offers_proba(model))
    use_proba = wants_proba and can_proba

    with timed_evaluation(seed=seed) as run_info:
        for index, (train_idx, val_idx) in enumerate(splits.folds):
            if select_fold is not None:
                select_fold(index)
            y_proba = classes = None
            try:
                if use_proba:
                    y_pred, y_proba, classes = model.fit_predict_proba(
                        config, splits.X[train_idx], splits.y[train_idx],
                        splits.X[val_idx], seed=seed)
                else:
                    y_pred = model.fit_predict(
                        config, splits.X[train_idx], splits.y[train_idx],
                        splits.X[val_idx], seed=seed)
            except TrialCancelled as exc:
                # Not a failure, and not a measurement: the run was stopped
                # while this trial was in flight, so nothing is known about this
                # configuration. Marked rather than scored, and `record` drops
                # it — see the module docstring.
                failure, cancelled = str(exc), True
                break
            except TrialTimeout as exc:
                failure, status = str(exc), STATUS_TIMEOUT
                # No traceback worth keeping: the exception is raised here, by
                # the deadline expiring, so a traceback would show this loop
                # rather than whatever the model was doing when time ran out.
                break
            except ModelTrialError as exc:
                # The model's own traceback, from over in its process. A
                # `format_exc()` here would show this loop receiving the reply.
                failure, detail = str(exc), getattr(exc, "detail", "")
                break
            except Exception as exc:  # noqa: BLE001 — a local model may fail any way it likes
                failure, detail = f"{type(exc).__name__}: {exc}", traceback.format_exc()
                break

            # A model running in its own process timed itself: this thread's CPU
            # clock measured none of the work. Summed, because each fold is a
            # separate round trip. See core.modelhost.
            reported = getattr(model, "last_cpu_time", None)
            if reported is not None:
                cpu_reported += reported

            try:
                fold_scores.append(score_all(splits.y[val_idx], y_pred, metrics,
                                             y_proba=y_proba, classes=classes))
            except Exception as exc:  # noqa: BLE001 — wrong label type, wrong length
                failure = f"predictions could not be scored: {type(exc).__name__}: {exc}"
                detail = traceback.format_exc()
                break

    if cpu_reported:
        run_info["cpu_time"] = cpu_reported

    if failure is None:
        scores = {name: float(np.mean([s[name] for s in fold_scores])) for name in metrics}
    else:
        scores = _null_scores(metrics, splits)
        run_info["status"] = status
        run_info["additional_info"] = {"error": failure}
        if detail:
            run_info["additional_info"]["traceback"] = detail
        if cancelled:
            # The zeros above are returned because the shape has to be the same
            # either way — every loop reads `all_scores[primary_metric]` — but
            # they are never recorded. `status` stays CRASHED so that SMAC's
            # `tell` still gets a value its own enum accepts on the way past.
            run_info[CANCELLED] = True

    return scores, run_info
