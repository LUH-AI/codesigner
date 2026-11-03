import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional

from ..analytics import EAGER_MAX_COALITIONS, compute_hp_games, fit_surrogate
from ..metrics import METRICS, declarations, describe, from_cost, metric_for, to_cost
from ..splits import holdout
from .timing import CANCELLED, FITTED, RUN_INFO_KEYS, STATUS_SUCCESS
from .trial import evaluate_trial


@dataclass
class OptimizerParam:
    """`label` is an English fallback, since `core` has no Django; `ui`
    translates labels keyed by `name`. `depends_on` hides a setting unless
    another parameter has a given value, and `enabled_by` greys it out until its
    boolean parameters are on.
    """
    name: str                               # `__init__` kwarg
    label: str
    type: str                               # `int`/`float`/`bool`/`select`
    default: Any
    min: Any = None                         # lower bound
    max: Any = None                         # upper bound
    choices: List[Any] = field(default_factory=list)
    advanced: bool = False                  # collapsed initially
    depends_on: Optional[tuple] = None      # (parameter, value)
    enabled_by: tuple = ()                  # boolean parameters
    group: str = ""                         # form block


@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    scores: Dict[str, float]
    score: float                # primary metric
    incumbent_score: float
    incumbent_config: Dict[str, Any]
    run_info: Dict[str, Any] = field(default_factory=dict)  # runhistory fields
    #: `origin` stays outside `run_info`, matching SMAC's separate origin map.
    origin: str = ""

    @property
    def duration(self) -> float:
        """`duration` is wall-clock seconds."""
        return self.run_info.get("time") or 0.0

    @property
    def failed(self) -> bool:
        """A failed trial carries `null_scores` that figures read as scores, so
        `failed` marks it.
        """
        return self.run_info.get("status", STATUS_SUCCESS) != STATUS_SUCCESS

    @property
    def failure(self) -> str:
        return (self.run_info.get("additional_info") or {}).get("error", "")

    @property
    def traceback(self) -> str:
        """`traceback` is empty for successes, older failures and stripped
        exports.
        """
        return (self.run_info.get("additional_info") or {}).get("traceback", "")


#: Tunability's field names omit the game, as every `.ihpo` stores them.
HP_GAME_FIELDS = {
    "tunability": {
        "importance": "hyperparameter_importance",
        "warning": "hyperparameter_importance_warning",
        "interactions": "hyperparameter_interactions",
        "moebius": "hyperparameter_moebius",
    },
    "mistunability": {
        "importance": "hyperparameter_mistunability",
        "warning": "hyperparameter_mistunability_warning",
        "interactions": "hyperparameter_mistunability_interactions",
        "moebius": "hyperparameter_mistunability_moebius",
    },
    "sensitivity": {
        "importance": "hyperparameter_sensitivity",
        "warning": "hyperparameter_sensitivity_warning",
        "interactions": "hyperparameter_sensitivity_interactions",
        "moebius": "hyperparameter_sensitivity_moebius",
    },
}


@dataclass
class OptimizationResult:

    trials: List[TrialResult]
    primary_metric: str
    best_config: Dict[str, Any]
    best_score: Optional[float]
    hyperparameter_importance: Dict[str, Dict[str, float]]
    hyperparameter_importance_warning: Dict[str, Optional[str]]
    trials_limit: Optional[int] = None    # None means unlimited
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Empty defaults let older results deserialize.
    hyperparameter_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hyperparameter_sensitivity_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    hyperparameter_mistunability: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hyperparameter_mistunability_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    #: `hyperparameter_interactions` stores symmetric order-2 values with
    #: order-1 values on the diagonal.
    hyperparameter_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_interactions_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    #: `hyperparameter_moebius` stores one Möbius term per coalition, for views
    #: needing terms above order 2.
    hyperparameter_moebius: Dict[str, list] = field(default_factory=dict)
    hyperparameter_sensitivity_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_sensitivity_moebius: Dict[str, list] = field(default_factory=dict)
    hyperparameter_mistunability_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_mistunability_moebius: Dict[str, list] = field(default_factory=dict)
    #: `hyperparameter_tunability_total` sums the raw order-1 magnitudes, and a
    #: share times it gives metric units.
    hyperparameter_tunability_total: Dict[str, float] = field(default_factory=dict)
    #: `declared_metrics` stores the metrics that an imported run declares and
    #: the registry lacks. Two experiments can then define "cost" differently.
    declared_metrics: Dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str):
        """Unknown metrics fall back to higher-is-better on [0, 1].
        """
        return self.declared_metrics.get(name) or metric_for(name)

    def _derived(self) -> Dict[tuple, Any]:
        """`dataclasses.replace` skips the plain-attribute memo, so stale values
        stay behind.
        """
        memo = getattr(self, "_derived_memo", None)
        if memo is None:
            memo = self._derived_memo = {}
        return memo

    def best_index(self, metric: str) -> Optional[int]:
        """`best_index` skips trials without a score for *metric*. Every panel
        highlights the same trial because the lowest index wins a tie.
        """
        memo, key = self._derived(), ("best_index", metric)
        if key not in memo:
            scored = [i for i, t in enumerate(self.trials) if metric in t.scores]
            picker = max if self.metric(metric).higher_is_better else min
            memo[key] = (picker(scored, key=lambda i: self.trials[i].scores[metric])
                         if scored else None)
        return memo[key]

    def has_every_score(self, metric: str) -> bool:
        """Figure builders call `has_every_score` before drawing, since Plotly
        mishandles a missing y-value.
        """
        memo, key = self._derived(), ("has_every_score", metric)
        if key not in memo:
            memo[key] = all(metric in t.scores for t in self.trials)
        return memo[key]

    def incumbent_scores(self, metric: str) -> List[float]:
        """`incumbent_scores` returns a memoized list that callers must leave
        unchanged. Leading entries stay None until a trial has a score.
        """
        memo, key = self._derived(), ("incumbent_scores", metric)
        if key not in memo:
            improves = self.metric(metric).better
            best = None
            running = []
            for t in self.trials:
                score = t.scores.get(metric)
                if score is not None and (best is None or improves(score, best)):
                    best = score
                running.append(best)
            memo[key] = running
        return memo[key]


GAME_RESULT_FIELDS = tuple(f.name for f in fields(OptimizationResult)
                           if f.name.startswith("hyperparameter_"))


def game_fields(games) -> dict:
    out = {}
    for game, names in HP_GAME_FIELDS.items():
        importance, warning, interactions, moebius, _total = games[game]
        out.update({names["importance"]: importance, names["warning"]: warning,
                    names["interactions"]: interactions, names["moebius"]: moebius})
    out["hyperparameter_interactions_warning"] = games["tunability"][1]
    out["hyperparameter_tunability_total"] = games["tunability"][4]
    return out


def storable_score(score) -> Optional[float]:
    """`storable_score` returns None for infinity, since `json.dumps` writes
    `-Infinity`, invalid JSON. `TrialCollector` reads None as nothing to beat.
    """
    if score is None:
        return None
    try:
        return float(score) if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def rebase_history(previous_result, primary_metric: str):
    """`rebase_history` returns ``(result, stale)`` with trials rescored under
    the new metric. *stale* tells the optimizer to discard state such as a
    fitted surrogate. Without new-metric scores, the result stays unchanged with
    *stale* set.
    """
    if previous_result is None:
        return None, False

    stored = previous_result.primary_metric
    if not stored or stored == primary_metric:
        return previous_result, False

    if not all(primary_metric in t.scores for t in previous_result.trials):
        return previous_result, True

    trials: List[TrialResult] = []
    metric = previous_result.metric(primary_metric)
    best_score = metric.worst_sentinel
    best_config: Optional[Dict[str, Any]] = None
    for t in previous_result.trials:
        score = t.scores[primary_metric]
        if metric.better(score, best_score):
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


#: The first criterion to fire ends the run.
STOPPING_CRITERIA = ("max_trials", "max_seconds", "max_trial_seconds",
                     "target_score", "no_improvement_trials",
                     "incumbent_confidence", "max_failures",
                     "max_consecutive_failures")

#: A run needs a stopping criterion outside the failure limits, because a run
#: only bounded by failures never ends if nothing breaks.
FAILURE_CRITERIA = ("max_failures", "max_consecutive_failures")

#: `DEFAULT_MAX_CONSECUTIVE_FAILURES` stops a model that cannot fit from burning
#: the whole budget.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 15

#: `DEFAULT_MAX_FAILURES` ends a run at its first failure, and a search of
#: invalid regions needs more.
DEFAULT_MAX_FAILURES = 1

STOPPED_BY_ALL_FAILING = "all_failing"

#: Whoever notices the interruption stores `STOPPED_BY_CANCELLED` in
#: `.Run.stopped_by`.
STOPPED_BY_CANCELLED = "cancelled"

#: `STOPPED_BY_EXHAUSTED` marks a fully tried space.
STOPPED_BY_EXHAUSTED = "exhausted"

#: Default origin of requested configurations.
REQUESTED_ORIGIN = "Requested"


class NoStoppingCriterion(ValueError):
    """`NoStoppingCriterion` flags unbounded runs."""


def merge_stopping(n_trials, stopping) -> Dict[str, Any]:
    """`merge_stopping` prefers `max_trials`.
    """
    merged = dict(stopping or {})
    if n_trials is not None:
        merged.setdefault("max_trials", n_trials)
    return merged


class PolledFlag:
    """`PolledFlag` caches *check* for *ttl* seconds across trial polls."""

    def __init__(self, check, ttl: float = 1.0):
        self._check = check
        self._ttl = ttl
        self._value = False
        self._checked = None

    def is_set(self) -> bool:
        now = time.monotonic()
        if self._checked is None or (now - self._checked) >= self._ttl:
            self._value = bool(self._check())
            self._checked = now
        return self._value


class TrialCollector:
    """`TrialCollector` records trials and tracks the incumbent, and every
    optimizer loop checks its `done`. `target_score` must be beaten strictly,
    and `incumbent_confidence` only fires on a probability from
    `note_confidence`. Missing failure limits take the defaults.
    """

    def __init__(
        self,
        trial_offset: int = 0,
        initial_best_score: Optional[float] = None,
        initial_best_config: Optional[Dict[str, Any]] = None,
        stopping: Optional[Dict[str, Any]] = None,
        on_record=None,
        metric=None,
        on_keep=None,
    ):
        #: `_on_keep` receives each trial and its last fold's model.
        self._on_keep = on_keep
        #: `_on_record` receives the collector after each recorded trial.
        self._on_record = on_record
        self._metric = metric or metric_for("")
        self.results: List[TrialResult] = []
        self._trial_offset = trial_offset
        #: A missing best score means the worst end, +inf for RMSE.
        self._incumbent_score = (self._metric.worst_sentinel
                                 if initial_best_score is None else initial_best_score)
        self._incumbent_config = initial_best_config

        self._stopping = {k: v for k, v in (stopping or {}).items()
                          if k in STOPPING_CRITERIA and v is not None}
        if not any(k not in FAILURE_CRITERIA for k in self._stopping):
            raise NoStoppingCriterion(
                "a run needs at least one stopping criterion that is not a "
                f"failure limit; got {sorted(stopping or {})}")
        self._stopping.setdefault("max_failures", DEFAULT_MAX_FAILURES)
        self._stopping.setdefault("max_consecutive_failures",
                                  DEFAULT_MAX_CONSECUTIVE_FAILURES)

        self._started = time.monotonic()
        self._trial_seconds = 0.0
        self._since_improvement = 0
        self._consecutive_failures = 0
        self._failures = 0
        self._confidence: Optional[float] = None
        self.stopped_by: Optional[str] = None

    @property
    def incumbent_score(self) -> float:
        return self._incumbent_score

    @property
    def incumbent_config(self) -> Optional[Dict[str, Any]]:
        return self._incumbent_config

    def note_confidence(self, probability: Optional[float]) -> None:
        """A None probability leaves `incumbent_confidence` unfired.
        """
        self._confidence = probability

    @property
    def metric(self):
        return self._metric

    def _fired(self) -> Optional[str]:
        # `.Metric.better` is strict, so a target at the incumbent's score needs
        # a better trial.
        target = self._stopping.get("target_score")
        if target is not None and self._metric.better(self._incumbent_score, target):
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
        if (self._failures >= self._stopping["max_failures"]
                or self._consecutive_failures
                >= self._stopping["max_consecutive_failures"]):
            return STOPPED_BY_ALL_FAILING
        stagnant = self._stopping.get("no_improvement_trials")
        if stagnant is not None and self._since_improvement >= stagnant:
            return "no_improvement_trials"
        return None

    @property
    def done(self) -> bool:
        if self.stopped_by is None:
            self.stopped_by = self._fired()
        return self.stopped_by is not None

    def record(
        self,
        config: Dict[str, Any],
        score: float,
        all_scores: Dict[str, float],
        run_info: Optional[Dict[str, Any]] = None,
        origin: str = "",
    ) -> Optional[TrialResult]:
        """*origin* comes from proposal time, since SMAC's local search relabels
        runhistory configurations in place. `record` returns None for a
        cancelled attempt, whose null scores would count toward `max_failures`.
        """
        if (run_info or {}).get(CANCELLED):
            # Latching ends every loop and stores the interruption.
            self.stopped_by = STOPPED_BY_CANCELLED
            return None

        # Failed trials never become the incumbent, since their null scores can
        # beat a bad RMSE.
        failed = (run_info or {}).get("status", STATUS_SUCCESS) != STATUS_SUCCESS
        if not failed and self._metric.better(score, self._incumbent_score):
            self._incumbent_score = score
            self._incumbent_config = config
            self._since_improvement = 0
        else:
            self._since_improvement += 1

        self._trial_seconds += (run_info or {}).get("time") or 0.0
        if (run_info or {}).get("status", STATUS_SUCCESS) == STATUS_SUCCESS:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            # Never resets.
            self._failures += 1

        trial = TrialResult(
            trial=self._trial_offset + len(self.results) + 1,
            config=config,
            scores=all_scores,
            score=score,
            incumbent_score=self._incumbent_score,
            incumbent_config=self._incumbent_config or config,
            run_info=run_info or {},
            origin=origin or "",
        )
        fitted = (run_info or {}).pop(FITTED, None)
        self.results.append(trial)
        if fitted is not None and self._on_keep is not None:
            # `_on_keep` errors never propagate.
            try:
                self._on_keep(trial, fitted)
            except Exception:  # noqa: BLE001 — see above
                pass
        if self._on_record is not None:
            # `_on_record` runs after the append, so `results` includes the
            # trial.
            try:
                self._on_record(self)
            except Exception:  # noqa: BLE001 — see above
                pass
        return trial


class BaseOptimizer(ABC):

    params_schema: List[OptimizerParam] = []

    #: `run_safe` names the settings a later run may change once trials exist.
    run_safe: frozenset = frozenset()

    #: `aliases` keeps old names that `.ihpo` files record.
    aliases: tuple = ()

    name: str

    def refit_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None) -> tuple[Any, Optional[str]]:
        """`refit_surrogate` returns ``(surrogate, warning)`` or ``(None,
        None)``. A stored binary would be version-bound, so the surrogate is
        refit from settings and trials.
        """
        return None, None

    def slice_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None):
        """`slice_surrogate` only depends on *metric_name* and *values*, so one
        fit serves all slices of a run.
        """
        rf, warning = self.refit_surrogate(config_space, trials, metric_name, seed,
                                           values=values)
        if rf is None:
            rf, warning = fit_surrogate(config_space, trials, metric_name, seed,
                                        values=values)
        return rf, warning

    #: The Run form only offers `incumbent_confidence` when
    #: `supports_confidence_stopping` is True.
    supports_confidence_stopping: bool = False

    #: `fits_surrogate` tells whether a rebase discards a fitted model.
    fits_surrogate: bool = False

    @classmethod
    def judges_priors(cls, params: Mapping[str, Any]) -> bool:
        """The page only offers "Evaluate Prior" when True.
        """
        return False

    def acquisition_function(self) -> Optional[dict]:
        """`acquisition_function` returns `kind` ("ei" or "pi") and `xi`, or
        None."""
        return None

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
        # Model-free optimizers ignore `priors`.
        priors=None,
        # Evaluated first.
        configurations=None,
    ) -> OptimizationResult: ...

    def serialize_result(self, result: "OptimizationResult") -> dict:
        """The stored dict follows SMAC's runhistory.json plus `scores`,
        `incumbent_score` and `incumbent_config_id` per entry. Subclasses store
        internal state under ``optimizer_state``.
        """
        configs: Dict[str, Any] = {}
        config_id_by_key: Dict[tuple, str] = {}

        for t in result.trials:
            cid = str(t.trial)
            configs[cid] = t.config
            config_id_by_key[tuple(sorted(t.config.items()))] = cid

        # Cost metrics store themselves, and older metrics store `1.0 - score`
        # (see `.to_cost`).
        cost_of = result.metric(result.primary_metric)

        data = []
        for t in result.trials:
            incumbent_key = tuple(sorted(t.incumbent_config.items()))
            incumbent_cid = config_id_by_key.get(incumbent_key, str(t.trial))
            data.append({
                **t.run_info,  # disjoint from keys below
                "config_id": t.trial,
                "cost": to_cost(cost_of, t.score),
                "scores": t.scores,
                "incumbent_score": storable_score(t.incumbent_score),
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
            # An empty origin means unrecorded, never model-chosen.
            "config_origins": {str(t.trial): t.origin for t in result.trials},
            "optimizer_state": {},
            "primary_metric": result.primary_metric,
            # Registry metrics stay out, so later corrections win.
            "declared_metrics": {
                name: describe(metric)
                for name, metric in result.declared_metrics.items()
                if name not in METRICS
            },
            "best_score": storable_score(result.best_score),
            "best_config_id": best_config_id,
            **{name: getattr(result, name) for name in GAME_RESULT_FIELDS},
            "trials_limit": result.trials_limit,
        }

    def deserialize_result(self, d: dict) -> "OptimizationResult":
        configs = d.get("configs", {})
        data = d.get("data", [])
        origins = d.get("config_origins") or {}
        primary_metric = d.get("primary_metric", "")
        declared = declarations(d.get("declared_metrics"))

        cost_of = declared.get(primary_metric) or metric_for(primary_metric)

        trials = []
        for entry in data:
            cid = str(entry["config_id"])
            incumbent_cid = str(entry.get("incumbent_config_id", entry["config_id"]))
            from_stored = from_cost(cost_of, entry["cost"])
            trials.append(TrialResult(
                trial=entry["config_id"],
                config=configs[cid],
                scores=entry.get("scores", {primary_metric: from_stored}),
                score=from_stored,
                incumbent_score=entry.get("incumbent_score", from_stored),
                incumbent_config=configs.get(incumbent_cid, configs.get(cid, {})),
                run_info={k: entry[k] for k in RUN_INFO_KEYS if k in entry},
                origin=str(origins.get(cid) or ""),
            ))

        best_config_id = str(d.get("best_config_id") or (str(trials[-1].trial) if trials else "0"))
        return OptimizationResult(
            trials=trials,
            declared_metrics=declared,
            primary_metric=primary_metric,
            best_config=configs.get(best_config_id, {}),
            best_score=storable_score(d.get("best_score", 0.0)),
            **{name: d.get(name, {}) for name in GAME_RESULT_FIELDS},
            trials_limit=d.get("trials_limit"),
            metadata={},
        )

    def get_params(self) -> dict:
        """Parameter ``foo`` lives on ``self._foo``.
        """
        return {p.name: getattr(self, f"_{p.name}") for p in self.params_schema}

    def resolved_params(self) -> dict:
        """Subclasses fill unset settings with the defaults in force;
        reconstruction uses `.Run.optimizer_params` instead.
        """
        return dict(self.get_params())

    @classmethod
    def effective_defaults(cls) -> dict:
        """`effective_defaults` maps each setting to the value a blank one runs
        with.
        """
        return {p.name: p.default for p in cls.params_schema}

    @classmethod
    def known_params(cls, stored: Mapping[str, Any]) -> Dict[str, Any]:
        """`known_params` drops renamed or withdrawn parameters, which would
        raise `TypeError` in `__init__`.
        """
        known = {p.name for p in cls.params_schema}
        stored = cls.migrate_params(stored or {})
        return {k: v for k, v in stored.items() if k in known}

    @classmethod
    def migrate_params(cls, stored: Mapping[str, Any]) -> Mapping[str, Any]:
        """An optimizer that reshapes its settings translates stored ones here.
        """
        return stored

    #: `experiments/services/run.py` copies `ANALYTICS_EAGER_MAX_COALITIONS`
    #: here, keeping `core/` Django-free.
    analytics_max_coalitions: Optional[int] = EAGER_MAX_COALITIONS

    #: False makes `.compute_hp_games` skip work for a page without importance
    #: or interaction figures.
    analytics_wanted: bool = True

    #: `experiments/services/run.py` sets `progress` to a throttled
    #: partial-result write.
    progress = None

    #: `experiments/services/run.py` sets `keep_model` to keep fitted models for
    #: export.
    keep_model = None

    def new_collector(self, metric_name: str = "", previous_result=None,
                      **kwargs) -> TrialCollector:
        """`new_collector` takes the metric from the resumed run first, since an
        imported run only declares its objective in its file.
        """
        declared = getattr(previous_result, "declared_metrics", None) or {}
        return TrialCollector(
            on_record=self.progress,
            on_keep=self.keep_model,
            metric=declared.get(metric_name) or metric_for(metric_name),
            **kwargs)

    def _begin(self, previous_result, primary_metric: str, n_trials, stopping, splits,
               X_train, y_train, X_val, y_val):
        """`_begin` returns splits (a holdout by default), the history rebased
        to the metric, and a collector continuing its numbering."""
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)
        previous_result, _ = rebase_history(previous_result, primary_metric)
        collector = self.new_collector(
            metric_name=primary_metric,
            previous_result=previous_result,
            trial_offset=len(previous_result.trials) if previous_result else 0,
            initial_best_score=previous_result.best_score if previous_result else None,
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=merge_stopping(n_trials, stopping),
        )
        return splits, previous_result, collector

    def _finish(self, collector, previous_result, config_space, metrics, primary_metric: str,
                seed: int, cancel_event, **result_fields) -> "OptimizationResult":
        all_trials = (previous_result.trials if previous_result else []) + collector.results
        metric = collector.metric
        best = metric.best(all_trials, key=lambda t: t.scores[primary_metric])
        games = compute_hp_games(config_space, all_trials, metrics, seed=seed,
                                 cancel_event=cancel_event, budget=self.analytics_max_coalitions,
                                 wanted=self.analytics_wanted)
        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=best.config if best is not None else {},
            best_score=metric.best((t.scores[primary_metric] for t in all_trials), default=0.0),
            **game_fields(games),
            metadata={"stopped_by": collector.stopped_by},
            **result_fields,
        )

    def evaluate_requested(self, configurations, *, model, splits, metrics, primary_metric,
                           seed, collector, cancel_event=None, told=None) -> set:
        """`evaluate_requested` runs *configurations* in order while criteria
        allow and returns their sorted-item keys. The *told* callback receives
        `(config, all_scores, run_info)` after each trial.
        """
        done = set()
        for requested in configurations or ():
            if collector.done or (cancel_event and cancel_event.is_set()):
                break
            config = dict(requested["config"])
            all_scores, run_info = evaluate_trial(model, config, splits, metrics, seed=seed,
                                                   keep=self.keep_model is not None)
            if told is not None:
                told(config, all_scores, run_info)
            collector.record(config, all_scores[primary_metric], all_scores,
                             run_info=run_info,
                             origin=requested.get("origin") or REQUESTED_ORIGIN)
            done.add(tuple(sorted(config.items())))
        return done
