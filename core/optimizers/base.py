import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional
from hypershap import ExplanationTask, HyperSHAP

from .timing import RUN_INFO_KEYS


@dataclass
class OptimizerParam:
    """Describes one user-configurable parameter of an optimizer, for form rendering."""
    name: str                               # kwarg name passed to __init__
    label: str                              # human-readable label shown in the form
    type: str                               # "int", "float", or "select"
    default: Any
    min: Any = None                         # lower bound for int / float
    max: Any = None                         # upper bound for int / float
    choices: List[Any] = field(default_factory=list)  # options for select


@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    scores: Dict[str, float]    # score for every metric
    score: float                # primary metric score (used internally)
    incumbent_score: float      # running incumbent score
    incumbent_config: Dict[str, Any]
    run_info: Dict[str, Any] = field(default_factory=dict)  # SMAC-native per-trial fields (timing, seed, status, …)

    @property
    def duration(self) -> float:
        """Wall-clock seconds the trial's evaluation took (0.0 if unrecorded)."""
        return self.run_info.get("time") or 0.0


@dataclass
class OptimizationResult:
    """Accumulated results of a completed or partial optimization run."""

    trials: List[TrialResult]
    primary_metric: str
    best_config: Dict[str, Any]
    best_score: float
    hyperparameter_importance: Dict[str, Dict[str, float]]          # metric → {hp: importance}
    hyperparameter_importance_warning: Dict[str, Optional[str]]     # metric → warning or None
    trials_limit: Optional[int] = None    # None = unlimited; set by optimizers with a finite search space
    metadata: Dict[str, Any] = field(default_factory=dict)


def rebase_history(previous_result, primary_metric: str):
    """Re-read an earlier run's trials under *primary_metric*.

    Returns ``(result, stale)``. *stale* is True when the metric has changed,
    which means any optimizer state carried alongside the trials — a fitted
    surrogate above all — was built against a different objective and must be
    discarded rather than resumed.

    An experiment that changes the metric it optimizes has, until now, carried
    its history forward unchanged: `TrialResult.score` and the incumbent
    trajectory still meant the *old* metric, so the incumbent figure drew a
    curve for an objective nobody was optimizing any more, and SMAC fitted its
    surrogate across two different cost functions at once.

    Nothing has to be thrown away to fix that, because every trial records a
    score for *every* metric. The history is simply re-read: each trial's score
    becomes its score under the new metric, and the incumbent trajectory is
    recomputed from those. The trials themselves are untouched — the same
    configurations were evaluated on the same data.

    The one case that cannot be re-read is a very old `.ihpo` that stored only
    the optimized metric's score. Then the trials stay as they are and only
    *stale* is reported, so the optimizer still discards its state instead of
    resuming against the wrong objective.
    """
    if previous_result is None:
        return None, False

    stored = previous_result.primary_metric
    if not stored or stored == primary_metric:
        return previous_result, False

    if not all(primary_metric in t.scores for t in previous_result.trials):
        return previous_result, True

    trials: List[TrialResult] = []
    best_score = float("-inf")
    best_config: Optional[Dict[str, Any]] = None
    for t in previous_result.trials:
        score = t.scores[primary_metric]
        if score > best_score:
            best_score, best_config = score, t.config
        trials.append(replace(
            t, score=score, incumbent_score=best_score, incumbent_config=best_config or t.config,
        ))

    return replace(
        previous_result,
        trials=trials,
        primary_metric=primary_metric,
        best_score=best_score if trials else 0.0,
        best_config=best_config or {},
    ), True


#: Every stopping criterion, on equal footing. A run needs at least one and may
#: have any combination; the first to fire ends it.
STOPPING_CRITERIA = ("max_trials", "max_seconds", "max_trial_seconds",
                     "target_score", "no_improvement_trials",
                     "incumbent_confidence")

#: Not a criterion — nothing in the collector can decide it — but it ends runs
#: and so belongs in the same vocabulary. Stored in `Run.stopped_by` by whoever
#: noticed the interruption, so "why did this stop?" has one answer to read
#: rather than a reason for the criteria and a status field for everything else.
STOPPED_BY_CANCELLED = "cancelled"

#: Criteria that may never fire. A run set up with only these has no guaranteed
#: end — a target score the search never reaches, or a surrogate that stays
#: unsure — so a caller should pair them with something bounded.
UNBOUNDED_CRITERIA = ("target_score", "incumbent_confidence")


class NoStoppingCriterion(ValueError):
    """A run was set up with nothing that could ever end it."""


def merge_stopping(n_trials, stopping) -> Dict[str, Any]:
    """`n_trials` as sugar for `stopping["max_trials"]`.

    Kept on `optimize()` because "run this many more trials" is the natural way
    to ask for a search from code, and every caller in the tests says it that
    way. It is only spelling: the collector sees one set of criteria with no
    privileged member, and an explicit `max_trials` wins.
    """
    merged = dict(stopping or {})
    if n_trials is not None:
        merged.setdefault("max_trials", n_trials)
    return merged


class TrialCollector:
    """Reusable bookkeeper for optimizer trial results, and what ends a run.

    Tracks the per-trial results list and the running incumbent as trials
    complete.  Optimizer-specific callbacks should inherit from (or delegate
    to) this class and call ``record()`` once per evaluated trial.

    Every optimizer loop is ``while not collector.done``, so this is the one
    place a stopping rule has to be written to apply to all of them.

    Parameters
    ----------
    trial_offset:
        Number of trials already recorded in a previous run; used to produce
        globally-sequential trial numbers when resuming.
    initial_best_score:
        Best primary-metric score seen before this run (``-inf`` for a fresh
        run).  Ensures the incumbent is correct relative to full history.
    initial_best_config:
        Config that produced ``initial_best_score``.
    stopping:
        The criteria, any of which may end the run — see ``STOPPING_CRITERIA``.
        At least one is required; an absent or None key means that criterion
        does not apply. No criterion is privileged: a trial cap is a limit like
        any other, not a mandatory backstop the rest hang off.

        ``max_trials``           this many new trials
        ``max_seconds``          wall-clock for this run
        ``max_trial_seconds``    cumulative time spent inside trials, which is
                                 the compute actually consumed rather than how
                                 long the run has been open
        ``target_score``         the incumbent reaches this
        ``no_improvement_trials``  this many trials in a row did not improve
                                 the incumbent
        ``incumbent_confidence`` the optimizer's surrogate is at least this sure
                                 nothing left will beat the incumbent. Only an
                                 optimizer that fits a surrogate can answer, and
                                 it does so through `note_confidence`; for one
                                 that cannot, this never fires.
    """

    def __init__(
        self,
        trial_offset: int = 0,
        initial_best_score: float = float("-inf"),
        initial_best_config: Optional[Dict[str, Any]] = None,
        stopping: Optional[Dict[str, Any]] = None,
    ):
        self.results: List[TrialResult] = []
        self._trial_offset = trial_offset
        self._incumbent_score = initial_best_score
        self._incumbent_config = initial_best_config

        self._stopping = {k: v for k, v in (stopping or {}).items()
                          if k in STOPPING_CRITERIA and v is not None}
        if not self._stopping:
            raise NoStoppingCriterion(
                "a run needs at least one stopping criterion; got "
                f"{sorted(stopping or {})}")

        self._started = time.monotonic()
        self._trial_seconds = 0.0
        self._since_improvement = 0
        self._confidence: Optional[float] = None
        #: Which criterion ended the run, or None while it is still going.
        self.stopped_by: Optional[str] = None

    @property
    def incumbent_score(self) -> float:
        """The best primary-metric score so far, over this run and any it
        resumed from. Read by an optimizer that needs the incumbent to ask its
        surrogate about."""
        return self._incumbent_score

    def note_confidence(self, probability: Optional[float]) -> None:
        """How sure the optimizer's surrogate is that nothing left is better.

        Reported rather than computed here: it needs a fitted model of the
        objective, which only the optimizer has. None means the optimizer could
        not say — no surrogate, or too little data to have trained one yet — and
        leaves the criterion unfired rather than guessing.
        """
        self._confidence = probability

    def _fired(self) -> Optional[str]:
        """The first criterion that says to stop, or None to keep going.

        Checked in the order a user would find least surprising to be told
        about: the two that mean "we are done" first, then the budgets that mean
        "we ran out", then stagnation, which is a judgement call.
        """
        if self._incumbent_score >= self._stopping.get("target_score", float("inf")):
            return "target_score"
        wanted = self._stopping.get("incumbent_confidence")
        if wanted is not None and self._confidence is not None and self._confidence >= wanted:
            return "incumbent_confidence"
        if len(self.results) >= self._stopping.get("max_trials", float("inf")):
            return "max_trials"
        if time.monotonic() - self._started >= self._stopping.get("max_seconds", float("inf")):
            return "max_seconds"
        if self._trial_seconds >= self._stopping.get("max_trial_seconds", float("inf")):
            return "max_trial_seconds"
        stagnant = self._stopping.get("no_improvement_trials")
        if stagnant is not None and self._since_improvement >= stagnant:
            return "no_improvement_trials"
        return None

    @property
    def done(self) -> bool:
        """True once any stopping criterion has fired.

        Latched: the first criterion to fire is the one reported, so a run does
        not get relabelled by a later check (wall-clock keeps advancing after
        the trial count is reached).
        """
        if self.stopped_by is None:
            self.stopped_by = self._fired()
        return self.stopped_by is not None

    def record(
        self,
        config: Dict[str, Any],
        score: float,
        all_scores: Dict[str, float],
        run_info: Optional[Dict[str, Any]] = None,
    ) -> TrialResult:
        """Record one completed trial, update the incumbent, return the TrialResult."""
        if score > self._incumbent_score:
            self._incumbent_score = score
            self._incumbent_config = config
            self._since_improvement = 0
        else:
            self._since_improvement += 1

        self._trial_seconds += (run_info or {}).get("time") or 0.0

        trial = TrialResult(
            trial=self._trial_offset + len(self.results) + 1,
            config=config,
            scores=all_scores,
            score=score,
            incumbent_score=self._incumbent_score,
            incumbent_config=self._incumbent_config or config,
            run_info=run_info or {},
        )
        self.results.append(trial)
        return trial


class BaseOptimizer(ABC):
    """Base class for all hyperparameter optimizers."""

    params_schema: List[OptimizerParam] = []

    @property
    @abstractmethod
    def name(self) -> str: ...

    #: Whether this optimizer fits a model of the objective it can be asked how
    #: sure it is. Only such an optimizer can answer `incumbent_confidence`, and
    #: the Run form uses this to decide whether to offer the criterion at all.
    supports_confidence_stopping: bool = False

    @abstractmethod
    def optimize(
        self,
        model,
        X_train, y_train,
        X_val, y_val,
        metrics: dict,
        primary_metric: str,
        n_trials: Optional[int] = None,
        previous_result: Optional["OptimizationResult"] = None,
        seed: int = 0,
        cancel_event=None,
        stopping: Optional[Dict[str, Any]] = None,
        splits=None,
    ) -> OptimizationResult: ...

    def serialize_result(self, result: "OptimizationResult") -> dict:
        """Serialize result to a runhistory-mirrored dict with .ihpo extensions.

        Returns the dict stored under the ``result`` key in the .ihpo file.
        The structure mirrors SMAC's runhistory.json at the top level, with
        .ihpo-specific fields (``scores``, ``incumbent_score``,
        ``incumbent_config_id``) added to each data entry.

        Optimizers with internal state to preserve (e.g. a surrogate model)
        should override this and embed that state under ``optimizer_state``.
        """
        configs: Dict[str, Any] = {}
        config_id_by_key: Dict[tuple, str] = {}

        for t in result.trials:
            cid = str(t.trial)
            configs[cid] = t.config
            config_id_by_key[tuple(sorted(t.config.items()))] = cid

        data = []
        for t in result.trials:
            incumbent_key = tuple(sorted(t.incumbent_config.items()))
            incumbent_cid = config_id_by_key.get(incumbent_key, str(t.trial))
            data.append({
                **t.run_info,  # SMAC-native fields (timing, seed, status, …); disjoint from the keys below
                "config_id": t.trial,
                "cost": 1.0 - t.score,
                "scores": t.scores,
                "incumbent_score": t.incumbent_score,
                "incumbent_config_id": int(incumbent_cid),
            })

        best_key = tuple(sorted(result.best_config.items())) if result.best_config else ()
        best_config_id = config_id_by_key.get(best_key) or (
            str(result.trials[-1].trial) if result.trials else "0"
        )

        return {
            "stats": {"submitted": len(result.trials), "finished": len(result.trials), "running": 0},
            "data": data,
            "configs": configs,
            "config_origins": {str(t.trial): self.name for t in result.trials},
            "optimizer_state": {},
            "primary_metric": result.primary_metric,
            "best_score": result.best_score,
            "best_config_id": best_config_id,
            "hyperparameter_importance": result.hyperparameter_importance,
            "hyperparameter_importance_warning": result.hyperparameter_importance_warning,
            "trials_limit": result.trials_limit,
        }

    def deserialize_result(self, d: dict) -> "OptimizationResult":
        """Reconstruct an OptimizationResult from a runhistory-mirrored result dict.

        Inverse of serialize_result.  Optimizers that override serialize_result
        to embed extra state should override this to restore it.
        """
        configs = d.get("configs", {})
        data = d.get("data", [])
        primary_metric = d.get("primary_metric", "")

        trials = []
        for entry in data:
            cid = str(entry["config_id"])
            incumbent_cid = str(entry.get("incumbent_config_id", entry["config_id"]))
            trials.append(TrialResult(
                trial=entry["config_id"],
                config=configs[cid],
                scores=entry.get("scores", {primary_metric: 1.0 - entry["cost"]}),
                score=1.0 - entry["cost"],
                incumbent_score=entry.get("incumbent_score", 1.0 - entry["cost"]),
                incumbent_config=configs.get(incumbent_cid, configs.get(cid, {})),
                run_info={k: entry[k] for k in RUN_INFO_KEYS if k in entry},
            ))

        best_config_id = str(d.get("best_config_id") or (str(trials[-1].trial) if trials else "0"))
        return OptimizationResult(
            trials=trials,
            primary_metric=primary_metric,
            best_config=configs.get(best_config_id, {}),
            best_score=d.get("best_score", 0.0),
            hyperparameter_importance=d.get("hyperparameter_importance", {}),
            hyperparameter_importance_warning=d.get("hyperparameter_importance_warning", {}),
            trials_limit=d.get("trials_limit"),
            metadata={},
        )

    def get_params(self) -> dict:
        """Return current optimizer parameters as ``{name: value}`` for each schema entry.

        Uses the naming convention that a schema param named ``foo`` is stored
        on the instance as ``self._foo``.  Optimizers with no ``params_schema``
        return an empty dict.
        """
        return {p.name: getattr(self, f"_{p.name}") for p in self.params_schema}

    def compute_hp_importance(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """Estimate hyperparameter importance from a completed list of trials.

        Returns (importance_dict, warning_message).  warning_message is None
        when HyperSHAP succeeds.
        """
        import numpy as np
        from ConfigSpace import Configuration

        params = list(config_space.keys())

        data: list[tuple] = []
        for t in trials:
            try:
                cfg = Configuration(config_space, values=t.config)
                data.append((cfg, t.scores[metric_name]))
            except Exception:
                continue

        if len(data) < 2:
            uniform = 1.0 / len(params) if params else 0.0
            return (
                {p: uniform for p in params},
                "Not enough trials for importance estimation; showing uniform weights.",
            )

        try:
            task = ExplanationTask.from_data(config_space, data)
            hs = HyperSHAP(task)
            iv = hs.tunability()
            order1 = iv.get_n_order(order=1).dict_values
            raw = {params[idx]: abs(val) for (idx,), val in order1.items()}
            total = sum(raw.values()) or 1.0
            return {k: v / total for k, v in raw.items()}, None
        except Exception as e:
            warning = f"HyperSHAP failed ({e}) — falling back to surrogate feature importances."

        try:
            from sklearn.ensemble import RandomForestRegressor
            X = np.array([cfg.get_array() for cfg, _ in data])
            y = np.array([score for _, score in data])
            rf = RandomForestRegressor(n_estimators=100, random_state=seed)
            rf.fit(X, y)
            importances = rf.feature_importances_
            total = importances.sum() or 1.0
            return {p: float(importances[i] / total) for i, p in enumerate(params)}, warning
        except Exception as e2:
            warning += f" Surrogate fallback also failed ({e2}); showing uniform weights."
            uniform = 1.0 / len(params) if params else 0.0
            return {p: uniform for p in params}, warning
