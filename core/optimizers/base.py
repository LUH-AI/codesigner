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
    """`label` is an English fallback because `core` has no Django for
    translation, and `ui` supplies translated labels keyed by `name`. A
    `default` of None keeps the component's own default, and the form shows an
    empty field. `depends_on` hides a setting unless another parameter has a
    given value, keeping the stored value, while `enabled_by` greys a setting
    out until its boolean parameters are on. `group` places related settings in
    one block the form lays out.
    """
    name: str                               # `__init__` kwarg
    label: str
    type: str                               # `int`/`float`/`bool`/`select`
    default: Any
    min: Any = None                         # int/float lower bound
    max: Any = None                         # int/float upper bound
    choices: List[Any] = field(default_factory=list)  # select options
    advanced: bool = False                  # hidden by default
    depends_on: Optional[tuple] = None      # (parameter, required value)
    enabled_by: tuple = ()                  # required boolean parameters
    group: str = ""                         # named form block


@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    scores: Dict[str, float]    # per-metric scores
    score: float                # primary metric score
    incumbent_score: float
    incumbent_config: Dict[str, Any]
    run_info: Dict[str, Any] = field(default_factory=dict)  # SMAC runhistory fields
    #: `origin` records how the configuration arose, kept outside `run_info`
    #: because SMAC stores origins in a separate map beside its runhistory
    #: entries. An empty string means the origin went unrecorded.
    origin: str = ""

    @property
    def duration(self) -> float:
        """`duration` is wall-clock seconds, or 0.0."""
        return self.run_info.get("time") or 0.0

    @property
    def failed(self) -> bool:
        """`failed` tells a bad trial from a failed one, since a failed trial
        still carries `null_scores` that figures read as a score. A missing
        status means success.
        """
        return self.run_info.get("status", STATUS_SUCCESS) != STATUS_SUCCESS

    @property
    def failure(self) -> str:
        """`failure` gives the one-line reason, if any."""
        return (self.run_info.get("additional_info") or {}).get("error", "")

    @property
    def traceback(self) -> str:
        """`traceback` is empty for a success, an older failure, or one stripped
        on export.
        """
        return (self.run_info.get("additional_info") or {}).get("traceback", "")


#: `HP_GAME_FIELDS` maps each HyperSHAP game (see `core.analytics.HP_GAMES`) to
#: its four result fields, in the page's selector order. Tunability's field
#: names omit the game because every `.ihpo` already stores them.
HP_GAME_FIELDS = {
    "tunability": {
        "importance": "hyperparameter_importance",
        "warning": "hyperparameter_importance_warning",
        "interactions": "hyperparameter_interactions",
        "moebius": "hyperparameter_moebius",
    },
    # Mistunability mirrors tunability, so the two read as a pair.
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
    """`OptimizationResult` stores a run's results."""

    trials: List[TrialResult]
    primary_metric: str
    best_config: Dict[str, Any]
    best_score: Optional[float]
    hyperparameter_importance: Dict[str, Dict[str, float]]
    hyperparameter_importance_warning: Dict[str, Optional[str]]
    trials_limit: Optional[int] = None    # None means unlimited (finite spaces set it)
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Older results still deserialize because the sensitivity and mistunability
    #: fields have empty defaults.
    hyperparameter_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hyperparameter_sensitivity_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    hyperparameter_mistunability: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hyperparameter_mistunability_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    #: `hyperparameter_interactions` stores symmetric order-2 values as `metric →
    #: {hp_a: {hp_b: value}}`, with each order-1 value on the diagonal.
    hyperparameter_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_interactions_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    #: `hyperparameter_moebius` stores the full Möbius decomposition, one term per
    #: coalition, for views that need terms above order 2. The FSII grid above
    #: stays separate (see `core.analytics.compute_hp_game`).
    hyperparameter_moebius: Dict[str, list] = field(default_factory=dict)
    #: The other games' interactions and Möbius terms come from the same call,
    #: named per game like the importance fields.
    hyperparameter_sensitivity_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_sensitivity_moebius: Dict[str, list] = field(default_factory=dict)
    hyperparameter_mistunability_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_mistunability_moebius: Dict[str, list] = field(default_factory=dict)
    #: `hyperparameter_tunability_total` is the sum of raw order-1 magnitudes per
    #: metric, which gives a share its scale in metric units. One float per
    #: metric is enough, because each value equals its share times the total.
    hyperparameter_tunability_total: Dict[str, float] = field(default_factory=dict)
    #: `declared_metrics` stores metrics that an imported run declares and the
    #: running build's registry lacks, and it stays empty for local runs. Two
    #: open experiments can define "cost" differently because the metrics live
    #: in a field instead of a global.
    declared_metrics: Dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str):
        """Unknown metrics fall back to higher-is-better on [0, 1], which older
        files assumed.
        """
        return self.declared_metrics.get(name) or metric_for(name)

    def _derived(self) -> Dict[tuple, Any]:
        """The memo is a plain attribute, so `dataclasses.replace` in
        `rebase_history` leaves it behind instead of copying stale values.
        """
        memo = getattr(self, "_derived_memo", None)
        if memo is None:
            memo = self._derived_memo = {}
        return memo

    def best_index(self, metric: str) -> Optional[int]:
        """`best_index` follows the metric's direction and breaks ties toward
        the lowest index. Therefore every panel highlights the same trial.
        Trials without a score for *metric* are skipped, guarding database rows
        older than `_check_trial_scores`.
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
        mishandles a missing y-value. The check only fails for rows stored
        before `_check_trial_scores` existed.
        """
        memo, key = self._derived(), ("has_every_score", metric)
        if key not in memo:
            memo[key] = all(metric in t.scores for t in self.trials)
        return memo[key]

    def incumbent_scores(self, metric: str) -> List[float]:
        """`incumbent_scores` returns the memoized running best, monotone in the
        metric's direction, and callers must leave the list unchanged. A trial
        without a score carries the best forward, and leading entries stay None
        until a score appears.
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


#: Fields the games fill.
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
    """An all-failing start leaves the incumbent at an infinite worst end, which
    `json.dumps` writes as `-Infinity`, invalid JSON. None is valid, and
    `TrialCollector` reads None as nothing to beat.
    """
    if score is None:
        return None
    try:
        return float(score) if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def rebase_history(previous_result, primary_metric: str):
    """`rebase_history` returns ``(result, stale)``, rescoring each trial and
    the incumbent trajectory under the new metric from the scores every trial
    records. *stale* tells the optimizer to discard state such as a surrogate
    fitted to the old objective. An old `.ihpo` only storing the optimized score
    stays unchanged, with *stale* still set.
    """
    if previous_result is None:
        return None, False

    stored = previous_result.primary_metric
    if not stored or stored == primary_metric:
        return previous_result, False

    if not all(primary_metric in t.scores for t in previous_result.trials):
        return previous_result, True

    trials: List[TrialResult] = []
    # The incumbent follows the new metric's direction.
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


#: A run needs at least one criterion, and the first to fire ends it.
STOPPING_CRITERIA = ("max_trials", "max_seconds", "max_trial_seconds",
                     "target_score", "no_improvement_trials",
                     "incumbent_confidence", "max_failures",
                     "max_consecutive_failures")

#: Failure limits end a run but fail to satisfy the one-criterion rule, because
#: a run only bounded by failures never ends if nothing breaks. Both have
#: defaults, so counting them would make the rule always pass.
FAILURE_CRITERIA = ("max_failures", "max_consecutive_failures")

#: Without `DEFAULT_MAX_CONSECUTIVE_FAILURES`, a model that cannot fit would
#: fail instantly on every trial and burn the whole budget.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 15

#: `DEFAULT_MAX_FAILURES` is one, because the first failure is the one worth
#: seeing. A search probing a region that cannot work raises the limit.
DEFAULT_MAX_FAILURES = 1

#: One stop reason covers both failure counts, since the finding is that trials
#: were failing.
STOPPED_BY_ALL_FAILING = "all_failing"

#: `STOPPED_BY_CANCELLED` goes into `Run.stopped_by` from whoever noticed the
#: interruption, so one field answers why a run stopped.
STOPPED_BY_CANCELLED = "cancelled"

#: `STOPPED_BY_EXHAUSTED` marks a small discrete space with every configuration
#: tried.
STOPPED_BY_EXHAUSTED = "exhausted"

#: The origin of a requested configuration without its own.
REQUESTED_ORIGIN = "Requested"


class NoStoppingCriterion(ValueError):
    """Raised for a run nothing could end."""


def merge_stopping(n_trials, stopping) -> Dict[str, Any]:
    """An explicit `max_trials` wins over *n_trials*.
    """
    merged = dict(stopping or {})
    if n_trials is not None:
        merged.setdefault("max_trials", n_trials)
    return merged


class PolledFlag:
    """`PolledFlag` caches *check* for *ttl* seconds, since the optimizer polls
    once per trial."""

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
    """`TrialCollector` records each trial, tracks the incumbent and decides
    when a run stops, and every optimizer loop checks `done`. *trial_offset*,
    *initial_best_score* and *initial_best_config* continue the numbering and
    the incumbent of a resumed run. *stopping* contains `STOPPING_CRITERIA`, and
    any criterion can end the run, but `target_score` must be beaten strictly
    and only `note_confidence` can trigger `incumbent_confidence`. Only the
    failure limits default when absent (`DEFAULT_MAX_FAILURES`,
    `DEFAULT_MAX_CONSECUTIVE_FAILURES`), so a failing run always stops, and a
    large number permits failures.
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
        #: `_on_keep` receives each trial and its last fold's model, and the
        #: collector keeps neither.
        self._on_keep = on_keep
        #: `_on_record` sees the collector after each recorded trial (see
        #: `BaseOptimizer.new_collector`), and a cancelled attempt never
        #: triggers it.
        self._on_record = on_record
        #: The default metric reads higher-is-better on [0, 1], as older callers
        #: assumed.
        self._metric = metric or metric_for("")
        self.results: List[TrialResult] = []
        self._trial_offset = trial_offset
        #: Without a best score, the incumbent starts at the metric's own worst
        #: end, which is +inf for RMSE.
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
        #: The criterion that ended the run.
        self.stopped_by: Optional[str] = None

    @property
    def incumbent_score(self) -> float:
        """`incumbent_score` is the best score so far, resumed runs included."""
        return self._incumbent_score

    @property
    def incumbent_config(self) -> Optional[Dict[str, Any]]:
        """`incumbent_config` serves surrogate queries with the best
        configuration."""
        return self._incumbent_config

    def note_confidence(self, probability: Optional[float]) -> None:
        """The optimizer reports the probability from its surrogate, and None
        leaves `incumbent_confidence` unfired.
        """
        self._confidence = probability

    @property
    def metric(self):
        """All three optimizers read the metric here instead of resolving it
        again.
        """
        return self._metric

    def _fired(self) -> Optional[str]:
        """`_fired` checks the target and confidence first, then budgets,
        failures and stagnation.
        """
        # The target must be surpassed in the metric's direction, so filling it
        # with the incumbent's score runs until something beats it.
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
        """`done` keeps the first stop reason found.
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
        origin: str = "",
    ) -> Optional[TrialResult]:
        """*origin* is taken at proposal time because SMAC's local search
        relabels runhistory configurations in place. A cancelled attempt returns
        None and stays out of the history, because its null scores would read as
        a real failed measurement and count toward `max_failures`.
        `SMACOptimizer.serialize_result` only keeps runhistory entries that
        match a recorded trial.
        """
        if (run_info or {}).get(CANCELLED):
            # Latching here ends every loop on the collector's terms and stores
            # the interruption.
            self.stopped_by = STOPPED_BY_CANCELLED
            return None

        # A failed trial can never become the incumbent, because its null score
        # is a real value that can beat a bad RMSE.
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
            # The total never resets.
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
            # Keeping a model never fails the run.
            try:
                self._on_keep(trial, fitted)
            except Exception:  # noqa: BLE001 — see above
                pass
        if self._on_record is not None:
            # `_on_record` runs after the append so `results` includes the
            # trial, and its failures never fail the run.
            try:
                self._on_record(self)
            except Exception:  # noqa: BLE001 — see above
                pass
        return trial


class BaseOptimizer(ABC):
    """`BaseOptimizer` underlies every optimizer."""

    params_schema: List[OptimizerParam] = []

    #: `.ihpo` files record the optimizer by name, so renames keep old ones
    #: here.
    aliases: tuple = ()

    name: str

    def refit_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None) -> tuple[Any, Optional[str]]:
        """`refit_surrogate` returns ``(surrogate, warning)``, or ``(None,
        None)`` for random and grid search, which fit nothing. The surrogate is
        refit from the recorded class, settings and trials, since a stored
        binary would be versioned and cross hosts.
        """
        return None, None

    def slice_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None):
        """`slice_surrogate` depends on *metric_name* and *values* only, so a
        caller fits once for all slices of a run. Refitting per hyperparameter
        cost about 300 ms for a GP and 750 ms for a forest over fifteen trials.
        """
        rf, warning = self.refit_surrogate(config_space, trials, metric_name, seed,
                                           values=values)
        if rf is None:
            rf, warning = fit_surrogate(config_space, trials, metric_name, seed,
                                        values=values)
        return rf, warning

    #: Only such an optimizer can answer `incumbent_confidence`, and the Run
    #: form offers the criterion only then.
    supports_confidence_stopping: bool = False

    #: `fits_surrogate` decides whether a rebased run lost a fitted model or
    #: only rescored its trials.
    fits_surrogate: bool = False

    @classmethod
    def judges_priors(cls, params: Mapping[str, Any]) -> bool:
        """The page offers "Evaluate Prior" only when `judges_priors` is True
        for *params*.
        """
        return False

    def acquisition_function(self) -> Optional[dict]:
        """`acquisition_function` returns `kind` ("ei" or "pi") and `xi`, or
        None, since the page computes the acquisition itself."""
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
        # Optimizers without a model of the objective ignore `priors`.
        priors=None,
        # Evaluated first; see `evaluate_requested`.
        configurations=None,
    ) -> OptimizationResult: ...

    def serialize_result(self, result: "OptimizationResult") -> dict:
        """The stored dict follows SMAC's runhistory.json and adds `scores`,
        `incumbent_score` and `incumbent_config_id` to each entry. An optimizer
        with internal state overrides `serialize_result` and stores the state
        under ``optimizer_state``.
        """
        configs: Dict[str, Any] = {}
        config_id_by_key: Dict[tuple, str] = {}

        for t in result.trials:
            cid = str(t.trial)
            configs[cid] = t.config
            config_id_by_key[tuple(sorted(t.config.items()))] = cid

        # The cost derives from the metric, so a cost metric stores itself and
        # older metrics keep `1.0 - score` (see `core.metrics.to_cost`).
        cost_of = result.metric(result.primary_metric)

        data = []
        for t in result.trials:
            incumbent_key = tuple(sorted(t.incumbent_config.items()))
            incumbent_cid = config_id_by_key.get(incumbent_key, str(t.trial))
            data.append({
                **t.run_info,  # SMAC fields, disjoint from the keys below
                "config_id": t.trial,
                "cost": to_cost(cost_of, t.score),
                "scores": t.scores,
                # The incumbent stays None until a trial succeeds, since
                # infinity is invalid JSON.
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
            # An empty origin stays empty, since "unrecorded" and "chosen by the
            # model" differ downstream.
            "config_origins": {str(t.trial): t.origin for t in result.trials},
            "optimizer_state": {},
            "primary_metric": result.primary_metric,
            # Registry metrics stay out of the file, so a later build's
            # correction wins.
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
        """`deserialize_result` inverts `serialize_result`.
        """
        configs = d.get("configs", {})
        data = d.get("data", [])
        origins = d.get("config_origins") or {}
        primary_metric = d.get("primary_metric", "")
        declared = declarations(d.get("declared_metrics"))

        # For older metrics the inverse is `1.0 - cost`, and `declared` reads an
        # imported run's objective.
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
            # Fields missing from older files read as empty.
            **{name: d.get(name, {}) for name in GAME_RESULT_FIELDS},
            trials_limit=d.get("trials_limit"),
            metadata={},
        )

    def get_params(self) -> dict:
        """Schema parameter ``foo`` lives on the instance as ``self._foo``.
        """
        return {p.name: getattr(self, f"_{p.name}") for p in self.params_schema}

    def resolved_params(self) -> dict:
        """`resolved_params` fills unset settings with the defaults in force,
        for the record, while reconstruction uses `optimizer_params`. The base
        class leaves unknown defaults visibly unanswered.
        """
        return dict(self.get_params())

    @classmethod
    def known_params(cls, stored: Mapping[str, Any]) -> Dict[str, Any]:
        """A renamed or withdrawn parameter would raise `TypeError` in
        `__init__`, which the result page fails to catch, so `known_params`
        drops it.
        """
        known = {p.name for p in cls.params_schema}
        stored = cls.migrate_params(stored or {})
        return {k: v for k, v in stored.items() if k in known}

    @classmethod
    def migrate_params(cls, stored: Mapping[str, Any]) -> Mapping[str, Any]:
        """An optimizer that reshapes its settings translates old ones here, so
        a stored run returns with the configuration it ran with.
        """
        return stored

    #: `experiments/services/run.py` copies the `ANALYTICS_EAGER_MAX_COALITIONS`
    #: setting here, keeping `core/` free of Django (see
    #: `eager_analytics_budget_exceeded`).
    analytics_max_coalitions: Optional[int] = EAGER_MAX_COALITIONS

    #: False makes `compute_hp_games` skip work when the page shows no
    #: importance or interaction figure. `experiments/services/run.py` sets the
    #: flag per instance.
    analytics_wanted: bool = True

    #: `experiments/services/run.py` sets `progress` to a throttled
    #: partial-result write, so the page can draw a running experiment.
    progress = None

    #: The caller sets `keep_model` to keep fitted models for export (see
    #: experiments/services/trial_models.py).
    keep_model = None

    def new_collector(self, metric_name: str = "", previous_result=None,
                      **kwargs) -> TrialCollector:
        """`new_collector` passes `progress`, `keep_model` and the metric to the
        collector of every optimizer. The metric comes from the resumed run
        first, since an imported run declares its objective only in its file.
        """
        declared = getattr(previous_result, "declared_metrics", None) or {}
        return TrialCollector(
            on_record=self.progress,
            on_keep=self.keep_model,
            metric=declared.get(metric_name) or metric_for(metric_name),
            **kwargs)

    def _begin(self, previous_result, primary_metric: str, n_trials, stopping, splits,
               X_train, y_train, X_val, y_val):
        """`_begin` returns the evaluation splits, a holdout unless *splits* is
        given, then the history rebased to the current metric, then a collector
        that continues the run's trial numbering."""
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)
        previous_result, _ = rebase_history(previous_result, primary_metric)
        collector = self.new_collector(
            metric_name=primary_metric,
            previous_result=previous_result,
            trial_offset=len(previous_result.trials) if previous_result else 0,
            # None means the metric's own worst end, +inf for lower-is-better.
            initial_best_score=previous_result.best_score if previous_result else None,
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=merge_stopping(n_trials, stopping),
        )
        return splits, previous_result, collector

    def _finish(self, collector, previous_result, config_space, metrics, primary_metric: str,
                seed: int, cancel_event, **result_fields) -> "OptimizationResult":
        """`_finish` returns every trial, the best one and the explanation
        games."""
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
        """*configurations* contains `{"config", "origin"}` entries, evaluated in
        order while the criteria allow and recorded as trials. The *told*
        callback receives `(config, all_scores, run_info)` after each, and the
        returned set contains sorted-item keys for skipping repeats.
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
