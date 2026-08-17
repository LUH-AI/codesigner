import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional
from hypershap import ExplanationTask, HyperSHAP

from .timing import RUN_INFO_KEYS, STATUS_SUCCESS


@dataclass
class OptimizerParam:
    """Describes one user-configurable parameter of an optimizer, for form rendering.

    `label` is a plain English fallback, not the string shown to a user: this
    layer has no Django and so cannot mark anything for translation. The
    interface supplies translated labels and help keyed by `name`, the same
    division the stopping criteria use — the vocabulary lives here, what to call
    it lives in `ui`.

    A `default` of None means "whatever the thing being configured already
    does". The form renders an empty field, and the optimizer is expected to
    leave that component alone rather than substitute a number of its own.

    `depends_on` names another parameter and the value that makes this one
    apply — `("search_strategy", "rf")` for a setting that only means anything
    under the random forest. It is declared rather than described so the form
    can hide what does not apply without knowing what any of it is. Hidden, not
    dropped: the value stays in the experiment and comes back if the other
    setting changes back.

    `enabled_by` names the boolean parameters that have to be *on* for this one
    to mean anything. The difference from `depends_on` is what the form does
    about it: a setting that belongs to the other search strategy is not there
    at all, while a setting switched off by its own checkbox is greyed out
    beside it — the reader has to see what turning it back on would offer.

    `group` puts the setting in a named block the form lays out itself, for the
    few that only make sense read together.
    """
    name: str                               # kwarg name passed to __init__
    label: str                              # plain-English fallback label
    type: str                               # "int", "float", "bool", or "select"
    default: Any
    min: Any = None                         # lower bound for int / float
    max: Any = None                         # upper bound for int / float
    choices: List[Any] = field(default_factory=list)  # (value, label) for select
    advanced: bool = False                  # folded away unless asked for
    depends_on: Optional[tuple] = None      # (other parameter, value it must have)
    enabled_by: tuple = ()                  # boolean parameters that must be on
    group: str = ""                         # a block the form lays out itself


@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    scores: Dict[str, float]    # score for every metric
    score: float                # primary metric score (used internally)
    incumbent_score: float      # running incumbent score
    incumbent_config: Dict[str, Any]
    run_info: Dict[str, Any] = field(default_factory=dict)  # SMAC-native per-trial fields (timing, seed, status, …)
    #: How this configuration was arrived at — sampled from the initial design,
    #: chosen by the model, drawn at random. Its own field rather than part of
    #: `run_info` because that mirrors SMAC's runhistory entry, and SMAC keeps
    #: origins in a separate top-level map beside the entries. Empty means
    #: unrecorded, which every trial from before this existed is.
    origin: str = ""

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
    # The other two global HyperSHAP games (see BaseOptimizer.HP_GAMES) —
    # same shape as hyperparameter_importance/_warning above, defaulted to
    # empty so a result from before these existed still deserializes.
    hyperparameter_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hyperparameter_sensitivity_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    hyperparameter_mistunability: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hyperparameter_mistunability_warning: Dict[str, Optional[str]] = field(default_factory=dict)
    # Pairwise (order-2) tunability interactions — a free byproduct of the
    # HyperSHAP call `hyperparameter_importance` already makes, since it asks
    # for order=2 and previously discarded everything past order 1.
    # metric → {hp_a: {hp_b: interaction_value}}, symmetric, diagonal is that
    # hyperparameter's own (signed, unnormalized) order-1 value.
    hyperparameter_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_interactions_warning: Dict[str, Optional[str]] = field(default_factory=dict)


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

#: Consecutive failed trials before a run gives up. Not something to configure:
#: it is not a budget anyone would choose, it is the difference between "this
#: search is exploring a bad region" and "nothing here can work". A model that
#: cannot fit the dataset at all fails instantly and identically every time, and
#: without this it burns the whole budget and reports a tidy run of zeros.
MAX_CONSECUTIVE_FAILURES = 15

#: Every trial failing is a reason a run ended, like being interrupted: nothing
#: the caller asked for, but something the page has to be able to say.
STOPPED_BY_ALL_FAILING = "all_failing"

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
        ``target_score``         the incumbent surpasses this — strictly, so a
                                 target equal to the score already in hand is
                                 not met until something beats it
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
        self._consecutive_failures = 0
        self._confidence: Optional[float] = None
        #: Which criterion ended the run, or None while it is still going.
        self.stopped_by: Optional[str] = None

    @property
    def incumbent_score(self) -> float:
        """The best primary-metric score so far, over this run and any it
        resumed from. Read by an optimizer that needs the incumbent to ask its
        surrogate about."""
        return self._incumbent_score

    @property
    def incumbent_config(self) -> Optional[Dict[str, Any]]:
        """The best configuration so far, for an optimizer that needs to ask its
        surrogate about it rather than about a bare number."""
        return self._incumbent_config

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
        # Strictly greater. The target is a score to *surpass*, which is what
        # makes filling it with the incumbent's own score mean "run until
        # something does better" rather than "stop immediately".
        if self._incumbent_score > self._stopping.get("target_score", float("inf")):
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
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return STOPPED_BY_ALL_FAILING
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
        origin: str = "",
    ) -> TrialResult:
        """Record one completed trial, update the incumbent, return the TrialResult.

        *origin* is where the configuration came from, as the optimizer knew it
        at the moment it was proposed. Taken here rather than read back off the
        optimizer afterwards, because by then it may not say the same thing —
        SMAC's local-search maximizer relabels configurations it takes out of
        the runhistory, in place.
        """
        if score > self._incumbent_score:
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
        self.results.append(trial)
        return trial


def _pair_trials_with_scores(config_space, trials: List[TrialResult], metric_name: str):
    """Every trial's config, as a `ConfigSpace.Configuration`, paired with its
    *metric_name* score — the input shape both HyperSHAP and a plain
    surrogate fit need, and the one place that pairing happens, so
    `_compute_hp_game`'s own explainer and `fit_surrogate` never drift apart
    on how a trial becomes training data. A trial whose config the current
    config_space rejects (e.g. a leftover from before a model's search space
    changed) is skipped rather than raised on — one bad trial should not
    blank out every other one's contribution.
    """
    from ConfigSpace import Configuration

    data = []
    for t in trials:
        try:
            cfg = Configuration(config_space, values=t.config)
            data.append((cfg, t.scores[metric_name]))
        except Exception:
            continue
    return data


def fit_surrogate(config_space, trials: List[TrialResult], metric_name: str, seed: int = 0):
    """Fit a cheap `RandomForestRegressor` surrogate over *trials*' recorded
    *metric_name* scores.

    Factored out of `_compute_hp_game`'s own fallback rung (used when
    HyperSHAP itself fails) so a future surrogate-consuming feature — Partial
    Dependencies (`docs/PLAN-analytics.md` Phase 6) is the first one planned —
    does not have to duplicate it. `optimize()` does not keep the real model
    around after it returns (SMAC's facade, in particular, is a local
    variable, discarded at the end of the call), so anything needing a
    stand-in model afterward needs one fit fresh, the same way this one is.

    Returns (surrogate, warning). surrogate is `None` (with a warning
    explaining why) when there are fewer than two usable trials or the fit
    itself raises — what "no surrogate" means for the caller's own output
    (a further fallback rung, an empty plot, ...) is the caller's call, not
    this function's.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    data = _pair_trials_with_scores(config_space, trials, metric_name)
    if len(data) < 2:
        return None, "Not enough trials to fit a surrogate."

    try:
        X = np.array([cfg.get_array() for cfg, _ in data])
        y = np.array([score for _, score in data])
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(X, y)
        return rf, None
    except Exception as e:
        return None, f"Surrogate fit failed ({e})."


def _hp_grid(hp, n_points: int) -> list:
    """*n_points* values spanning one hyperparameter's domain — its full set
    of choices for a categorical hyperparameter (there is no "spanning" a
    finite, unordered set); sorted-unique rounded integers for an integer
    one, so the grid never suggests a value it couldn't actually take;
    otherwise *n_points* evenly spaced floats across its bounds.
    """
    import numpy as np
    from ConfigSpace import UniformIntegerHyperparameter

    if hasattr(hp, "choices"):
        return list(hp.choices)
    raw = np.linspace(hp.lower, hp.upper, n_points)
    if isinstance(hp, UniformIntegerHyperparameter):
        return sorted({int(round(v)) for v in raw})
    return [float(v) for v in raw]


class BaseOptimizer(ABC):
    """Base class for all hyperparameter optimizers."""

    params_schema: List[OptimizerParam] = []

    #: Names this optimizer used to be called. An `.ihpo` records the optimizer
    #: by name, so a rename would otherwise orphan every file and row written
    #: before it.
    aliases: tuple = ()

    @property
    @abstractmethod
    def name(self) -> str: ...

    #: Whether this optimizer fits a model of the objective it can be asked how
    #: sure it is. Only such an optimizer can answer `incumbent_confidence`, and
    #: the Run form uses this to decide whether to offer the criterion at all.
    supports_confidence_stopping: bool = False

    #: Whether this optimizer carries fitted state from one run into the next.
    #: Read when recording *why* a run rebased its history: for an optimizer
    #: that fits nothing, a changed metric is a rescoring and no more; for one
    #: that does, the model it had was fitted to the old objective and had to be
    #: thrown away.
    fits_surrogate: bool = False

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
            # Keyed by trial number, and written from what each trial recorded
            # at the moment it was proposed. An empty origin is written through
            # rather than filled with this optimizer's name: "we did not record
            # where this came from" and "the model chose it" are different
            # facts, and something downstream reads them apart.
            "config_origins": {str(t.trial): t.origin for t in result.trials},
            "optimizer_state": {},
            "primary_metric": result.primary_metric,
            "best_score": result.best_score,
            "best_config_id": best_config_id,
            "hyperparameter_importance": result.hyperparameter_importance,
            "hyperparameter_importance_warning": result.hyperparameter_importance_warning,
            "hyperparameter_sensitivity": result.hyperparameter_sensitivity,
            "hyperparameter_sensitivity_warning": result.hyperparameter_sensitivity_warning,
            "hyperparameter_mistunability": result.hyperparameter_mistunability,
            "hyperparameter_mistunability_warning": result.hyperparameter_mistunability_warning,
            "hyperparameter_interactions": result.hyperparameter_interactions,
            "hyperparameter_interactions_warning": result.hyperparameter_interactions_warning,
            "trials_limit": result.trials_limit,
        }

    def deserialize_result(self, d: dict) -> "OptimizationResult":
        """Reconstruct an OptimizationResult from a runhistory-mirrored result dict.

        Inverse of serialize_result.  Optimizers that override serialize_result
        to embed extra state should override this to restore it.
        """
        configs = d.get("configs", {})
        data = d.get("data", [])
        origins = d.get("config_origins") or {}
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
                origin=str(origins.get(cid) or ""),
            ))

        best_config_id = str(d.get("best_config_id") or (str(trials[-1].trial) if trials else "0"))
        return OptimizationResult(
            trials=trials,
            primary_metric=primary_metric,
            best_config=configs.get(best_config_id, {}),
            best_score=d.get("best_score", 0.0),
            hyperparameter_importance=d.get("hyperparameter_importance", {}),
            hyperparameter_importance_warning=d.get("hyperparameter_importance_warning", {}),
            hyperparameter_sensitivity=d.get("hyperparameter_sensitivity", {}),
            hyperparameter_sensitivity_warning=d.get("hyperparameter_sensitivity_warning", {}),
            hyperparameter_mistunability=d.get("hyperparameter_mistunability", {}),
            hyperparameter_mistunability_warning=d.get("hyperparameter_mistunability_warning", {}),
            hyperparameter_interactions=d.get("hyperparameter_interactions", {}),
            hyperparameter_interactions_warning=d.get("hyperparameter_interactions_warning", {}),
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

    def resolved_params(self) -> dict:
        """`get_params`, with the defaults that were actually in force filled in.

        A setting left unset means "whatever this component already does", which
        is the right thing to store and the wrong thing to read: nobody opening
        a file six months later knows what SMAC's random forest uses for its
        leaf size. This is the same dict with those blanks answered, for the
        record rather than for reconstruction — reconstruction uses
        `optimizer_params`, which is what was asked for.

        Base implementation has nothing to add, since a default it does not know
        about is better left visibly unanswered than guessed at.
        """
        return dict(self.get_params())

    @classmethod
    def known_params(cls, stored: Dict[str, Any]) -> Dict[str, Any]:
        """*stored* narrowed to parameters this optimizer still has.

        A parameter that was renamed or withdrawn would otherwise reach
        `__init__` as an unexpected keyword and raise `TypeError` — from the
        page that rebuilds a result to display it, which catches `ValueError`
        and nothing else. Dropped rather than fatal, the same way the collector
        drops a stopping key it does not recognise.
        """
        known = {p.name for p in cls.params_schema}
        stored = cls.migrate_params(stored or {})
        return {k: v for k, v in stored.items() if k in known}

    @classmethod
    def migrate_params(cls, stored: Dict[str, Any]) -> Dict[str, Any]:
        """*stored* with settings from an earlier shape read onto today's.

        Dropping a renamed setting is safe but silent, and silent is the wrong
        answer here: a stored experiment or an `.ihpo` written last month would
        come back configured differently from how it ran, with nothing said. An
        optimizer that reshapes its settings translates them here instead.

        Base implementation has nothing to translate.
        """
        return stored

    #: HyperSHAP's global "explanation games" this app surfaces, each answering
    #: a different question about the same trial history: tunability = how
    #: much upside does tuning this hyperparameter offer (HyperSHAP's MAX
    #: aggregation), sensitivity = how much does performance vary as it moves
    #: (VAR), mistunability = how much downside risk does getting it wrong
    #: carry (MIN). Same call shape, same fallback ladder — see
    #: `_compute_hp_game`. `optimizer_bias` exists too but needs a live
    #: ensemble of optimizers to compare against, not just one run's trial
    #: history, so it doesn't fit this app's model and isn't offered.
    HP_GAMES = ("tunability", "sensitivity", "mistunability")

    def compute_hp_importance(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """Estimate hyperparameter importance (HyperSHAP's "tunability" game)
        from a completed list of trials.

        Returns (importance_dict, warning_message).  warning_message is None
        when HyperSHAP succeeds. Kept as its own method, rather than folded
        into `compute_hp_games`, because it predates the other games and
        existing callers (tests, every optimizer's serialize path before this)
        name it directly.
        """
        importance, warning, _interactions = self._compute_hp_game(
            config_space, trials, metric_name, "tunability", seed)
        return importance, warning

    def compute_hp_sensitivity(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """HyperSHAP's "sensitivity" game — how much performance varies as
        each hyperparameter moves, holding the others at random draws. Same
        contract as `compute_hp_importance`."""
        importance, warning, _interactions = self._compute_hp_game(
            config_space, trials, metric_name, "sensitivity", seed)
        return importance, warning

    def compute_hp_mistunability(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """HyperSHAP's "mistunability" game — how much downside a
        hyperparameter risks if it ends up wrong. Same contract as
        `compute_hp_importance`."""
        importance, warning, _interactions = self._compute_hp_game(
            config_space, trials, metric_name, "mistunability", seed)
        return importance, warning

    def compute_hp_interactions(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
    ) -> tuple[Dict[str, Dict[str, float]], Optional[str]]:
        """Pairwise (order-2) tunability interactions between hyperparameters —
        not a separate HyperSHAP call: `compute_hp_importance`'s own call
        already asks for order 2 by default and previously discarded
        everything past order 1, so this is a free re-extraction of the same
        computation, not additional cost.

        Returns (interactions_dict, warning_message), where interactions_dict
        is `{hp_a: {hp_b: value}}` — symmetric, and the diagonal holds that
        hyperparameter's own (signed, unnormalized) order-1 value, so the
        result reads as one square hyperparameter × hyperparameter grid rather
        than an off-diagonal-only matrix. On any fallback (no HyperSHAP `iv` to
        re-extract from), interactions_dict is `{}` — a surrogate's
        `feature_importances_` and uniform weights have no interaction
        structure to report.
        """
        _importance, warning, interactions = self._compute_hp_game(
            config_space, trials, metric_name, "tunability", seed)
        return interactions, warning

    def compute_hp_ablation(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        config_of_interest: Dict[str, Any],
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """HyperSHAP's "ablation" game: how much each hyperparameter's value in
        *config_of_interest* helped or hurt *metric_name*, versus the config
        space's default — a *local* explanation of one specific trial, unlike
        `HP_GAMES`' global ones.

        Signed, deliberately: positive means that hyperparameter's value in
        this trial beat the default, negative means it lost to it. Zeroing
        that out with `abs()` (as the three global games do, on purpose,
        since they answer "how much does this matter" rather than "which
        direction") would throw away the one thing this view exists to show.

        Returns (values_dict, warning_message). Unlike `_compute_hp_game`,
        there is no RandomForest-fallback rung: a surrogate's plain
        `feature_importances_` has no sign and does not answer the same
        question, so a failure here is reported rather than answered with
        something that resembles an answer but is not one.
        """
        from ConfigSpace import Configuration

        params = list(config_space.keys())
        data = _pair_trials_with_scores(config_space, trials, metric_name)

        if len(data) < 2:
            return {}, "Not enough trials for a local explanation."

        try:
            task = ExplanationTask.from_data(config_space, data)
            hs = HyperSHAP(task)
            config = Configuration(config_space, values=config_of_interest)
            baseline = config_space.get_default_configuration()
            iv = hs.ablation(config_of_interest=config, baseline_config=baseline)
            order1 = iv.get_n_order(order=1).dict_values
            return {params[idx]: val for (idx,), val in order1.items()}, None
        except Exception as e:
            return {}, f"HyperSHAP (ablation) failed ({e})."

    def compute_partial_dependence(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        hp_name: str,
        seed: int = 0,
        n_points: int = 20,
    ) -> tuple[list, List[List[Optional[float]]], List[Optional[float]], Optional[str]]:
        """Partial dependence (and per-trial ICE) of *metric_name* on
        *hp_name*, from a `fit_surrogate` fit over *trials* — DeepCave's
        PDP/ICE plugin, minus a second surrogate-fitting code path: the same
        stand-in every global HyperSHAP game's own fallback rung already
        builds does this job too, so Phase 5 factored it out for exactly this
        reuse.

        For each of *hp_name*'s grid values (see `_hp_grid`), every trial's
        own *other* hyperparameter values are held fixed and only *hp_name*
        is swapped to that grid value — the surrogate's prediction for that
        synthetic configuration is one point on that trial's Individual
        Conditional Expectation (ICE) curve. The Partial Dependence curve is
        the grid-wise mean across every trial's ICE curve — DeepCave's own
        definition, and the standard one. A trial whose config plus the
        swapped-in grid value the config space rejects (never happens with
        this app's current registry models, none of which have conditional
        or forbidden-clause hyperparameters, but a future model's might)
        contributes `None` at that grid point rather than raising.

        Returns (grid, ice_lines, pdp, warning). `ice_lines` is one list per
        trial, each the same length as `grid` (possibly holding `None`s);
        `pdp` is `grid`'s own length, each entry the mean of the non-`None`
        values across `ice_lines` at that index (`None` if every trial's
        was). All three are empty (with a warning) when there are too few
        trials to fit a surrogate, or *hp_name* has no valid configuration
        anywhere on its grid.
        """
        from ConfigSpace import Configuration

        rf, warning = fit_surrogate(config_space, trials, metric_name, seed)
        if rf is None:
            return [], [], [], warning

        grid = _hp_grid(config_space[hp_name], n_points)

        ice_lines = []
        for t in trials:
            row = []
            for value in grid:
                try:
                    values = dict(t.config)
                    values[hp_name] = value
                    cfg = Configuration(config_space, values=values)
                    row.append(float(rf.predict([cfg.get_array()])[0]))
                except Exception:
                    row.append(None)
            if any(v is not None for v in row):
                ice_lines.append(row)

        if not ice_lines:
            return [], [], [], "No valid configurations on this hyperparameter's grid."

        pdp = []
        for i in range(len(grid)):
            column = [row[i] for row in ice_lines if row[i] is not None]
            pdp.append(sum(column) / len(column) if column else None)

        return grid, ice_lines, pdp, None

    def compute_hp_games(
        self,
        config_space,
        trials: List[TrialResult],
        metrics,
        seed: int = 0,
    ) -> Dict[str, tuple[Dict[str, Dict[str, float]], Dict[str, Optional[str]], Dict[str, Dict[str, Dict[str, float]]]]]:
        """Every metric's per-game hyperparameter importance, for all of
        `HP_GAMES` — what each optimizer's `optimize()` calls once, instead of
        looping `compute_hp_importance`/`_sensitivity`/`_mistunability`
        separately over every metric three times.

        Returns `{game: (importance_by_metric, warning_by_metric,
        interactions_by_metric)}`, the first pair shaped exactly like an
        `OptimizationResult`'s `hyperparameter_<game>`/`hyperparameter_<game>_warning`
        fields expect. `interactions_by_metric` is a free byproduct of the
        same call (see `compute_hp_interactions`) for every game, though only
        the `"tunability"` entry currently gets stored anywhere
        (`OptimizationResult.hyperparameter_interactions`) — sensitivity's/
        mistunability's interaction grids are computed here at zero extra
        cost but not yet wired to a field or a view.

        Builds the HyperSHAP explainer once per metric, not once per (game,
        metric) pair: `tunability`/`sensitivity`/`mistunability` all explain
        the *same* trial history for a given metric, so refitting the
        surrogate underneath them 3 times over is pure waste. See
        `_build_explainer`/`_compute_hp_game`'s `explainer` parameter.
        """
        by_game = {game: ({}, {}, {}) for game in self.HP_GAMES}
        for metric_name in metrics:
            explainer = self._build_explainer(config_space, trials, metric_name)
            for game in self.HP_GAMES:
                importance, warning, interactions = self._compute_hp_game(
                    config_space, trials, metric_name, game, seed, explainer=explainer)
                by_game[game][0][metric_name] = importance
                by_game[game][1][metric_name] = warning
                by_game[game][2][metric_name] = interactions
        return by_game

    def _build_explainer(self, config_space, trials: List[TrialResult], metric_name: str):
        """Pair *trials* with *metric_name* scores and fit the HyperSHAP
        explainer every global game (`tunability`/`sensitivity`/
        `mistunability`) shares for a given metric — the one expensive step
        (constructing an `ExplanationTask` fits a surrogate internally)
        that's identical across all three; only each game's own aggregation
        (MAX/VAR/MIN) differs once it's built. `compute_hp_games` builds this
        once per metric and reuses it across `HP_GAMES`; `_compute_hp_game`
        builds its own when called standalone (e.g. `compute_hp_importance`)
        and no `explainer` was handed to it.

        Returns (hs, warning, insufficient) — pass straight through as
        `_compute_hp_game`'s `explainer`. `insufficient` is True only for
        "fewer than two usable trials," a condition `_compute_hp_game`
        answers with uniform weights directly, skipping the RandomForest
        rung entirely (there's nothing to fit it from either); it's False
        (with `hs` None) when HyperSHAP itself failed to build one, which
        does still warrant trying that rung.
        """
        data = _pair_trials_with_scores(config_space, trials, metric_name)
        if len(data) < 2:
            return None, "Not enough trials for importance estimation; showing uniform weights.", True
        try:
            task = ExplanationTask.from_data(config_space, data)
            return HyperSHAP(task), None, False
        except Exception as e:
            return None, f"HyperSHAP failed to build a surrogate ({e})", False

    def _compute_hp_game(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        game: str,
        seed: int = 0,
        explainer=None,
    ) -> tuple[Dict[str, float], Optional[str], Dict[str, Dict[str, float]]]:
        """Shared scaffolding behind `compute_hp_importance`/`_sensitivity`/
        `_mistunability`/`_interactions`: ask HyperSHAP's *game* for order-1
        (and, as a free byproduct, order-2) values from the built explainer,
        and fall back to a surrogate's feature_importances_ and then uniform
        weights if that fails. *game* is a `HyperSHAP` instance method name
        taking no required arguments (`tunability`, `sensitivity`,
        `mistunability` all qualify; `ablation` does not — it needs a
        specific configuration to explain, not just trial history, so it is
        computed separately, on demand, for one selected trial).

        *explainer*, when given, is `_build_explainer`'s own
        `(hs, warning, insufficient)` return for this (config_space, trials,
        metric_name) — see `compute_hp_games`, which builds it once and
        reuses it across every game — passed straight through instead of
        rebuilding (and re-fitting) an identical surrogate per game. `None`
        (the default) builds its own, for a standalone call like
        `compute_hp_importance`.

        Returns (values_dict, warning_message, interactions_dict).
        warning_message is None when HyperSHAP succeeds. interactions_dict is
        `{}` on either fallback rung — a plain feature_importances_ or
        uniform weights carry no pairwise structure to report.
        """
        params = list(config_space.keys())

        if explainer is None:
            explainer = self._build_explainer(config_space, trials, metric_name)
        hs, build_warning, insufficient = explainer

        if insufficient:
            uniform = 1.0 / len(params) if params else 0.0
            return {p: uniform for p in params}, build_warning, {}

        warning = None
        if hs is not None:
            try:
                iv = getattr(hs, game)()
                order1 = iv.get_n_order(order=1).dict_values
                raw = {params[idx]: abs(val) for (idx,), val in order1.items()}
                total = sum(raw.values()) or 1.0
                importance = {k: v / total for k, v in raw.items()}
                interactions = self._extract_pairwise(iv, params)
                return importance, None, interactions
            except Exception as e:
                warning = f"HyperSHAP ({game}) failed ({e}) — falling back to surrogate feature importances."
        else:
            warning = f"HyperSHAP ({game}) {build_warning} — falling back to surrogate feature importances."

        rf, fit_warning = fit_surrogate(config_space, trials, metric_name, seed)
        if rf is not None:
            importances = rf.feature_importances_
            total = importances.sum() or 1.0
            return {p: float(importances[i] / total) for i, p in enumerate(params)}, warning, {}
        warning += f" Surrogate fallback also failed ({fit_warning}); showing uniform weights."
        uniform = 1.0 / len(params) if params else 0.0
        return {p: uniform for p in params}, warning, {}

    @staticmethod
    def _extract_pairwise(iv, params: List[str]) -> Dict[str, Dict[str, float]]:
        """Reshape a HyperSHAP `InteractionValues` object's order-1 and
        order-2 terms into one square hyperparameter × hyperparameter grid —
        `{hp_a: {hp_b: value}}`, symmetric, diagonal = that hyperparameter's
        own signed order-1 value. Raw values, not `abs()`-ed or normalized
        like `_compute_hp_game`'s `importance` return: a heatmap comparing
        interaction strength needs sign and a shared scale between the
        diagonal and off-diagonal, not a share-of-100%.
        """
        order1 = iv.get_n_order(order=1).dict_values
        order2 = iv.get_n_order(order=2).dict_values
        grid: Dict[str, Dict[str, float]] = {p: {} for p in params}
        for (idx,), val in order1.items():
            grid[params[idx]][params[idx]] = float(val)
        for (i, j), val in order2.items():
            a, b = params[i], params[j]
            grid[a][b] = grid[b][a] = float(val)
        return grid
