"""`evaluate_trial` is the inner loop of all three optimizers, so each records a
failing model the same way. Scoring happens here, and validation labels never
reach the model. Scores are averaged over the folds of `core.splits.Splits`, so
holdout and cross-validation both yield one score per metric.
"""

import traceback

import numpy as np

from ..metrics import HISTORY, PROBABILITIES, score_all, usable
from ..modelhost.errors import ModelTrialError, TrialCancelled, TrialTimeout
from .timing import CANCELLED, FITTED, STATUS_CRASHED, STATUS_TIMEOUT, timed_evaluation


def offers_probabilities(model) -> bool:
    """`offers_probabilities` looks for a method other than the SDK's
    self-marking stub, then reads the model's `probabilities` flag.
    `core.modelhost.harness._capabilities` asks the same of a model in its own
    process.
    """
    fn = getattr(model, "fit_predict_proba", None)
    return (callable(fn) and not getattr(fn, "_is_stub", False)
            and bool(getattr(model, "probabilities", True)))


class NoProbabilities(ValueError):

    def __init__(self, metric: str):
        super().__init__(f"{metric} is scored on class probabilities, and this model does "
                         "not give them. Optimize another metric.")
        self.metric = metric


def scoreable(metrics, model, primary_metric):
    """A model file can change after the create form filtered the metrics, so
    `scoreable` drops metrics that need missing probabilities. `scoreable`
    raises `NoProbabilities` when the optimized metric is among them, since the
    run would be impossible.
    """
    can = (model.supports_proba if hasattr(model, "supports_proba")
           else offers_probabilities(model))
    keep = usable(metrics, probabilities=can)
    if primary_metric not in keep:
        raise NoProbabilities(primary_metric)
    return {name: metrics[name] for name in keep}


def _history(splits, train_idx, metrics):
    if not any(m.needs == HISTORY for m in metrics.values()):
        return None
    series = getattr(splits, "series", None)
    return {"y": splits.y[train_idx],
            "series": None if series is None else series[train_idx],
            "season": getattr(splits, "season", 1)}


def null_scores(metrics, splits) -> dict:
    """`null_scores` gives each metric the score of a model that knows nothing,
    such as a coin flip for ROC AUC or the null predictor's error for RMSE. A
    zero would make a crashed trial the best RMSE in the run. A metric whose
    null score fails falls back to the worst end of its range or 0.0, because
    raising here would kill the run.
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


def keeps_its_fit(model) -> bool:
    """Only an in-process built-in can hand over its fit, since a remote model's
    fitted state stays in the child."""
    from codesigner_model import OptimizableBaseModel, OutputBaseModel

    return isinstance(model, OutputBaseModel) and isinstance(model, OptimizableBaseModel)


def evaluate_trial(model, config, splits, metrics, seed=0, keep=False):
    """`evaluate_trial` returns ``(scores, run_info)``, and with *keep* a model
    passing `keeps_its_fit` fits the last fold through its own `fit` and returns
    that partial model as ``run_info[FITTED]``.

    A failed trial (an exception, a timeout, unscoreable output, or a broken
    child process) scores `null_scores` and records the status,
    ``additional_info["error"]`` and any traceback. One failing fold fails the
    whole trial, and the collector's failure limits decide whether the run
    stops. A cancelled trial gets `CANCELLED` instead, and
    `TrialCollector.record` drops the trial because nothing was measured.
    """
    failure = None
    detail = ""
    cancelled = False
    status = STATUS_CRASHED
    fold_scores = []
    cpu_reported = 0.0

    # A model in another process has its own copy of the dataset, so
    # `use_fold` names the fold instead of slicing.
    select_fold = getattr(model, "use_fold", None)

    # `wants_proba` is computed once from the metrics, because an SVM fits a
    # costly calibration for probabilities. `supports_proba` is False for a
    # remote model that never claimed them.
    wants_proba = any(m.needs == PROBABILITIES for m in metrics.values())
    can_proba = (model.supports_proba if hasattr(model, "supports_proba")
                 else offers_probabilities(model))
    use_proba = wants_proba and can_proba

    with timed_evaluation(seed=seed) as run_info:
        for index, (train_idx, val_idx) in enumerate(splits.folds):
            if select_fold is not None:
                select_fold(index)
            y_proba = classes = None
            last = index == len(splits.folds) - 1
            try:
                if keep and last and keeps_its_fit(model):
                    fitted = model.fit(config, splits.X[train_idx], splits.y[train_idx],
                                       seed, probabilities=use_proba)
                    y_pred = model.predict(fitted, splits.X[val_idx])
                    if use_proba:
                        y_proba = model.predict_proba(fitted, splits.X[val_idx])
                        classes = fitted.classes_
                    run_info[FITTED] = fitted
                elif use_proba:
                    y_pred, y_proba, classes = model.fit_predict_proba(
                        config, splits.X[train_idx], splits.y[train_idx],
                        splits.X[val_idx], seed=seed)
                else:
                    y_pred = model.fit_predict(
                        config, splits.X[train_idx], splits.y[train_idx],
                        splits.X[val_idx], seed=seed)
            except TrialCancelled as exc:
                # A cancelled trial is marked, then dropped by `record`.
                failure, cancelled = str(exc), True
                break
            except TrialTimeout as exc:
                failure, status = str(exc), STATUS_TIMEOUT
                # A traceback would only show the fold loop, where the deadline
                # raised `TrialTimeout`.
                break
            except ModelTrialError as exc:
                # The model's traceback comes from its own process.
                failure, detail = str(exc), getattr(exc, "detail", "")
                break
            except Exception as exc:  # noqa: BLE001 — a local model may fail any way it likes
                failure, detail = f"{type(exc).__name__}: {exc}", traceback.format_exc()
                break

            # An out-of-process model reports its own CPU time, summed over
            # folds.
            reported = getattr(model, "last_cpu_time", None)
            if reported is not None:
                cpu_reported += reported

            try:
                fold_scores.append(score_all(splits.y[val_idx], y_pred, metrics,
                                             y_proba=y_proba, classes=classes,
                                             history=_history(splits, train_idx, metrics)))
            except Exception as exc:  # noqa: BLE001 — wrong label type, wrong length
                failure = f"predictions could not be scored: {type(exc).__name__}: {exc}"
                detail = traceback.format_exc()
                break

    if cpu_reported:
        run_info["cpu_time"] = cpu_reported

    if failure is None:
        scores = {name: float(np.mean([s[name] for s in fold_scores])) for name in metrics}
    else:
        scores = null_scores(metrics, splits)
        run_info.pop(FITTED, None)
        run_info["status"] = status
        run_info["additional_info"] = {"error": failure}
        if detail:
            run_info["additional_info"]["traceback"] = detail
        if cancelled:
            # The null scores keep the return shape, and `record` discards them.
            # `status` stays CRASHED so SMAC's `tell` gets a valid `StatusType`.
            run_info[CANCELLED] = True

    return scores, run_info
