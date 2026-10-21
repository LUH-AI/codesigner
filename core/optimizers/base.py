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

    @property
    def failed(self) -> bool:
        """Whether this trial did not produce a measurement.

        The model raised, the trial ran past its deadline, or its predictions
        could not be scored — see `evaluate_trial`, which is what sets the
        status. Such a trial still carries a score, 0.0 on every metric, so
        every figure and every downstream computation reads it as a trial that
        did badly; this is how something that wants to say otherwise asks.

        Absent status means success, which is what every trial recorded before
        failures were tracked has.
        """
        return self.run_info.get("status", STATUS_SUCCESS) != STATUS_SUCCESS

    @property
    def failure(self) -> str:
        """Why it failed, in one line; empty for a trial that did not."""
        return (self.run_info.get("additional_info") or {}).get("error", "")

    @property
    def traceback(self) -> str:
        """The full traceback, for a trial that failed with one.

        Empty for a trial that succeeded, one that failed before tracebacks were
        recorded, and one whose traceback was stripped on export.
        """
        return (self.run_info.get("additional_info") or {}).get("traceback", "")


#: Which OptimizationResult fields back each HyperSHAP "explanation game" —
#: see core.analytics.HP_GAMES for what each one answers.
#: Order here is the games' order in the page's one game selector.
#:
#: Four fields each, because a game is not only an importance: the same call
#: produces the order-2 interaction grid and the full Möbius decomposition, and
#: every figure that reads any of them reads the game the sidebar names. The
#: tunability entry's names have no game in them — they were the only ones
#: stored back when only tunability's had anywhere to go, and they are in every
#: .ihpo ever written.
HP_GAME_FIELDS = {
    "tunability": {
        "importance": "hyperparameter_importance",
        "warning": "hyperparameter_importance_warning",
        "interactions": "hyperparameter_interactions",
        "moebius": "hyperparameter_moebius",
    },
    # Mistunability before sensitivity: it is tunability's mirror — what there
    # is to lose against what there is to gain — and the two read as a pair.
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
    """Accumulated results of a completed or partial optimization run."""

    trials: List[TrialResult]
    primary_metric: str
    best_config: Dict[str, Any]
    best_score: Optional[float]
    hyperparameter_importance: Dict[str, Dict[str, float]]          # metric → {hp: importance}
    hyperparameter_importance_warning: Dict[str, Optional[str]]     # metric → warning or None
    trials_limit: Optional[int] = None    # None = unlimited; set by optimizers with a finite search space
    metadata: Dict[str, Any] = field(default_factory=dict)
    # The other two global HyperSHAP games (see core.analytics.HP_GAMES) —
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
    # The full Möbius (Harsanyi) decomposition of the same tunability call —
    # one value per coalition, of every size, rather than the order-2 grid
    # above. Read by the interaction figure's graph/upset/by-order views, which
    # need terms above order 2; the heatmap and top-pairs views keep reading
    # `hyperparameter_interactions`, whose FSII numbers this deliberately does
    # not replace (see `core.analytics.compute_hp_game` for why the two are not
    # interchangeable). metric → [{"members": [...], "value": float}, ...].
    hyperparameter_moebius: Dict[str, list] = field(default_factory=dict)
    # The other two games' interactions and Möbius terms. Same shape as
    # tunability's two fields above, and produced by the same call at no extra
    # cost — `compute_hp_games` has always computed all three and kept one, back
    # when only tunability's had anywhere to go. Named after their game rather
    # than folded into one field keyed by game, because that is how the
    # importance fields already read and because the shape of
    # `hyperparameter_interactions` is in every .ihpo ever written.
    hyperparameter_sensitivity_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_sensitivity_moebius: Dict[str, list] = field(default_factory=dict)
    hyperparameter_mistunability_interactions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hyperparameter_mistunability_moebius: Dict[str, list] = field(default_factory=dict)
    # What the tunability shares are shares *of*, per metric: the sum of the raw
    # order-1 magnitudes, in the metric's own units. A share cannot say whether
    # the whole is worth eight accuracy points or eight thousandths of one, and
    # it cannot be compared against the local ablation, which is raw and signed.
    # Both of those are wanted: "98% of the achievable gain, and 89% of it
    # already banked" is a complete sentence where "98%" alone is a misleading
    # one. One float per metric rather than a second copy of every value, since
    # `share x total` recovers the rest.
    hyperparameter_tunability_total: Dict[str, float] = field(default_factory=dict)
    #: Metrics this result carries that the running build does not have in its
    #: own registry — an imported run's own objective, which exists only inside
    #: the file it came from. Empty for everything produced here.
    #:
    #: A field rather than a global the importer mutates: two experiments open at
    #: once must be able to disagree about what "cost" means, and a module-level
    #: registry could not hold both. Carried through `dataclasses.replace`, which
    #: is what `rebase_history` needs.
    declared_metrics: Dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str):
        """What *name* means here: declared by the file, or known to this build.

        Falls back to a higher-is-better [0, 1] reading when neither knows it —
        which is what every version of this code assumed before metrics could
        describe themselves, so a file from then is read exactly as it was
        written.
        """
        return self.declared_metrics.get(name) or metric_for(name)

    def _derived(self) -> Dict[tuple, Any]:
        """Memo for values derived from `trials`, keyed by (what, metric).

        Deliberately *not* a dataclass field. `rebase_history` rebuilds a result
        with `dataclasses.replace`, which passes declared fields through to
        `__init__` — so a field would carry a memo computed against the old
        metric into the rebased copy, which is precisely the bug rebasing
        exists to fix. As a plain attribute it is simply absent on the new
        object and gets recomputed.
        """
        memo = getattr(self, "_derived_memo", None)
        if memo is None:
            memo = self._derived_memo = {}
        return memo

    def best_index(self, metric: str) -> Optional[int]:
        """Index of the best-scoring trial by *metric*; None with no trials.

        "Best" is the metric's own direction — the largest accuracy, the
        smallest RMSE — so this is an argmax or an argmin depending on what is
        being asked about.

        One implementation of this argmax, because there were six: the detail
        page's panels, `_selected_panel_data`, and `PerformanceOverTime.plot`
        once per each of its four views. The duplicated cost was trivial — the
        point is that six copies of a tie-break rule are five chances to
        disagree about which trial is "best", and the highlight the page draws
        has to be the same trial the panel below it describes.

        Ties go to the lowest index, which is what `max` already did.

        A trial with no score for *metric* is skipped rather than raised on,
        and the answer is None when no trial has one at all. `io.parse` refuses
        such a file (see `_check_trial_scores`), so this is the second line of
        defence, for a row already in the database from before that check
        existed: the metric then has no best trial, which every caller already
        handles, instead of a KeyError five frames down and a 500 on the page.
        """
        memo, key = self._derived(), ("best_index", metric)
        if key not in memo:
            scored = [i for i, t in enumerate(self.trials) if metric in t.scores]
            picker = max if self.metric(metric).higher_is_better else min
            memo[key] = (picker(scored, key=lambda i: self.trials[i].scores[metric])
                         if scored else None)
        return memo[key]

    def has_every_score(self, metric: str) -> bool:
        """True when every trial carries a score for *metric*.

        What the per-metric figure builders check before drawing anything. A
        figure that needs one y-value per trial (the performance curve, the
        cube's colour array, a parallel-coordinates axis) has no honest way to
        draw a trial with no score — Plotly's own null handling ranges from a
        gap to a silently mis-scaled axis depending on the trace type — so they
        decline as a whole and show the "no data" caption they already have.

        `io.parse` refuses such a file outright (`_check_trial_scores`), so this
        only ever fires for a row stored before that check existed. Kept as one
        memoized predicate rather than repeated in each builder so they cannot
        disagree about what "complete" means.
        """
        memo, key = self._derived(), ("has_every_score", metric)
        if key not in memo:
            memo[key] = all(metric in t.scores for t in self.trials)
        return memo[key]

    def incumbent_scores(self, metric: str) -> List[float]:
        """The running best score by *metric*, per trial.

        Monotone in the metric's own improving direction: non-decreasing for a
        higher-is-better metric, non-increasing for one where lower wins.

        Recomputed once per view before this — four times per metric for
        `performance_over_time`'s four axis combinations, all identical.

        Returns the memo itself, not a copy: callers read it into a plot trace
        and none mutate it.

        A trial with no score for *metric* carries the running best forward
        unchanged rather than raising — same reasoning as `best_index`, and it
        keeps the list one entry per trial, which every caller indexes by trial
        position. The leading entries are None if the first trials have no such
        score, since there is no running best yet to carry.
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


#: Every OptimizationResult field the explanation games fill.
GAME_RESULT_FIELDS = tuple(f.name for f in fields(OptimizationResult)
                           if f.name.startswith("hyperparameter_"))


def game_fields(games) -> dict:
    """*games*, `compute_hp_games`'s answer, as the OptimizationResult fields
    that hold it."""
    out = {}
    for game, names in HP_GAME_FIELDS.items():
        importance, warning, interactions, moebius, _total = games[game]
        out.update({names["importance"]: importance, names["warning"]: warning,
                    names["interactions"]: interactions, names["moebius"]: moebius})
    out["hyperparameter_interactions_warning"] = games["tunability"][1]
    out["hyperparameter_tunability_total"] = games["tunability"][4]
    return out


def storable_score(score) -> Optional[float]:
    """*score* as something a file can hold, or None.

    The incumbent starts at the metric's worst end — an infinity — and stays
    there until a trial succeeds. A run whose opening trials all fail therefore
    has a best score of `-inf`, and `experiments/services/run.py` writes a partial result
    while that is still true.

    `json.dumps` will emit that as the bare token `-Infinity`, which Python reads
    back and nothing else does: it is not JSON, so an `.ihpo` carrying one cannot
    be opened by any strict parser. `None` is both valid and more honest — there
    is no best yet — and it flows back through `TrialCollector`, which already
    reads `None` as "nothing to beat".
    """
    if score is None:
        return None
    try:
        return float(score) if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


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
    # The new metric's own direction: re-reading an accuracy history under an
    # RMSE means the incumbent trajectory is a running minimum, not a maximum.
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


#: Every stopping criterion, on equal footing. A run needs at least one and may
#: have any combination; the first to fire ends it.
STOPPING_CRITERIA = ("max_trials", "max_seconds", "max_trial_seconds",
                     "target_score", "no_improvement_trials",
                     "incumbent_confidence", "max_failures",
                     "max_consecutive_failures")

#: The two of those that answer "when has this gone wrong?" rather than "when is
#: this done?". They end a run like any other criterion, and they do not count
#: towards the requirement that a run have one: a run whose only limit is how
#: many trials may fail has no end anyone chose — it stops when it breaks, and
#: if nothing breaks it never stops. Both default (see below), so they are also
#: the two that are set whether or not a caller mentions them, which would make
#: the requirement unsatisfiable to check if they counted.
FAILURE_CRITERIA = ("max_failures", "max_consecutive_failures")

#: Consecutive failed trials before a run gives up, when the caller named no
#: `max_consecutive_failures` of their own. A model that cannot fit the dataset
#: at all fails instantly and identically every time, and without a limit it
#: burns the whole budget and reports a tidy run of zeros.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 15

#: Failed trials in total before a run gives up, when the caller named no
#: `max_failures`. One, because the first failure is the one worth seeing: a
#: model still being written fails for a reason its author can fix, and there is
#: nothing to learn from watching it fail forty more times. A search deliberately
#: probing a region that cannot work is the other case, and raising this is how
#: someone says that is what they are doing.
#:
#: The two count different things and both are criteria. In a row means "nothing
#: here can work"; in total means "something is wrong and I want to know now".
DEFAULT_MAX_FAILURES = 1

#: Too many trials failing is a reason a run ended, like being interrupted:
#: nothing the caller asked for in the sense that a budget is, but something the
#: page has to be able to say. One name for both counts — which of them fired is
#: a detail of the arithmetic, and "the trials were failing" is the finding.
STOPPED_BY_ALL_FAILING = "all_failing"

#: Not a criterion — nothing in the collector can decide it — but it ends runs
#: and so belongs in the same vocabulary. Stored in `Run.stopped_by` by whoever
#: noticed the interruption, so "why did this stop?" has one answer to read
#: rather than a reason for the criteria and a status field for everything else.
STOPPED_BY_CANCELLED = "cancelled"

#: Not a criterion either: a search over a small discrete space that has tried
#: every configuration it can find has nothing left to run. Named, because the
#: page otherwise has to say a run with trials to spare stopped for no reason.
STOPPED_BY_EXHAUSTED = "exhausted"

#: Where a trial asked for by name came from, when the caller gives no origin of
#: its own: a configuration somebody chose, not one the search produced.
REQUESTED_ORIGIN = "Requested"


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
        ``max_failures``         this many trials failed, anywhere in the run
        ``max_consecutive_failures``  this many failed in a row

        The two failure counts are the one pair here that defaults rather than
        not applying when absent — to `DEFAULT_MAX_FAILURES` and
        `DEFAULT_MAX_CONSECUTIVE_FAILURES`. Every other criterion absent means
        "do not stop for this"; a failing run has to stop for *something*, or a
        model that cannot fit the dataset spends the whole budget proving it.
        Pass a large number to say that failures are expected here.
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
        #: Called with each recorded trial and the model its last fold fitted,
        #: when the run was asked to keep fitted models (see
        #: `core.optimizers.trial.FITTED`) — to keep it or not. Never stored here.
        self._on_keep = on_keep
        #: Called with this collector after each trial is appended, for a caller
        #: that wants to see a run's results while it is still running. Every
        #: optimizer gets it without knowing about it — see
        #: `BaseOptimizer.new_collector`. Never called for a trial that was not
        #: recorded, so a cancelled attempt does not trigger a write of a result
        #: it is not in.
        self._on_record = on_record
        #: What the score being collected means. Defaults to the reading every
        #: version of this before metrics described themselves assumed —
        #: higher-is-better on [0, 1] — so a caller that says nothing (the tests,
        #: a script) gets exactly the behaviour it always had.
        self._metric = metric or metric_for("")
        self.results: List[TrialResult] = []
        self._trial_offset = trial_offset
        #: None means "nothing to beat yet", which is the metric's own worst
        #: end rather than a hardcoded -inf: for an RMSE every real score is
        #: *smaller* than the incumbent, not larger.
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

    @property
    def metric(self):
        """What the score being collected means — direction and range.

        Read by the optimizers when they assemble the result and when they tell
        SMAC a cost, so all three agree with the collector rather than each
        resolving the metric again.
        """
        return self._metric

    def _fired(self) -> Optional[str]:
        """The first criterion that says to stop, or None to keep going.

        Checked in the order a user would find least surprising to be told
        about: the two that mean "we are done" first, then the budgets that mean
        "we ran out", then stagnation, which is a judgement call.
        """
        # Strictly better, in the metric's own direction. The target is a score
        # to *surpass*, which is what makes filling it with the incumbent's own
        # score mean "run until something does better" rather than "stop
        # immediately" — and for a lower-is-better metric surpassing it means
        # getting under it.
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
    ) -> Optional[TrialResult]:
        """Record one completed trial, update the incumbent, return the TrialResult.

        *origin* is where the configuration came from, as the optimizer knew it
        at the moment it was proposed. Taken here rather than read back off the
        optimizer afterwards, because by then it may not say the same thing —
        SMAC's local-search maximizer relabels configurations it takes out of
        the runhistory, in place.

        **A cancelled attempt is not recorded at all**, and `None` comes back
        instead of a trial. The run was stopped while that trial was in flight,
        so it was never measured: it has no score, and the 0.0 `evaluate_trial`
        returns for shape's sake would otherwise enter the history as the worst
        result there is — a trial that never happened, marked failed on four
        figures, dragged through every surrogate, and counted against
        `max_failures` so the run reports that it gave up on failures when in
        fact it was interrupted.

        Here rather than in the three optimizer loops because it is the same
        decision for all three and all three already call this. Nothing else has
        to change: SMAC tells its own runhistory before calling this, and
        `SmacOptimizer.serialize_result` already keeps only the entries that
        match a recorded trial — for exactly this hazard, "a trial asked for and
        not told because the run stopped".
        """
        if (run_info or {}).get(CANCELLED):
            # Latched here too, so the loop ends on the collector's own terms
            # rather than relying on each optimizer to re-read the cancel flag,
            # and so the stored result says it was interrupted.
            self.stopped_by = STOPPED_BY_CANCELLED
            return None

        # A trial that produced no measurement can never be the best one. It
        # scores the metric's null score — what a model that knows nothing gets
        # — and that is a real number on the scale, not a sentinel: for an RMSE
        # it beats a genuinely-bad trial outright.
        #
        # This was wrong before any of that, and visibly so: with every failure
        # scoring 0.0 and the incumbent seeded at -inf, the first crash of an
        # all-failing run passed `0.0 > -inf` and the page reported a crashed
        # configuration as the best one. Being <= every real score is what kept
        # it from being noticed, not anything that made it right.
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
            # Never reset. A run's total is a total, which is what makes it a
            # different question from how many failed in a row.
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
            # A kept model is a convenience, like a partial write: failing to
            # keep one never fails the run.
            try:
                self._on_keep(trial, fitted)
            except Exception:  # noqa: BLE001 — see above
                pass
        if self._on_record is not None:
            # After the append, so a caller reading `results` sees this trial.
            # Failures are the caller's problem and never the run's: a partial
            # write is a convenience, and a run that died because a snapshot
            # could not be saved would be a bad trade.
            try:
                self._on_record(self)
            except Exception:  # noqa: BLE001 — see above
                pass
        return trial


class BaseOptimizer(ABC):
    """Base class for all hyperparameter optimizers."""

    params_schema: List[OptimizerParam] = []

    #: Names this optimizer used to be called. An `.ihpo` records the optimizer
    #: by name, so a rename would otherwise orphan every file and row written
    #: before it.
    aliases: tuple = ()

    name: str

    def refit_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None) -> tuple[Any, Optional[str]]:
        """The model *this optimizer* would have fitted, refit on *trials*.

        Returns `(surrogate, warning)`, or `(None, None)` for an optimizer that
        has no surrogate to speak of — random and grid search fit nothing, and
        a caller wanting a picture of some belief should fall back to a generic
        one rather than be told a search that models nothing modelled this.

        Refit rather than retained deliberately. The fitted model does not
        outlive the run that made it, and keeping one would mean a versioned
        binary in a JSON snapshot that also has to cross the boundary to
        whichever host ran the search. Its class and settings do survive, and
        so do the trials, which is enough to build the same belief again — and
        unlike a stored blob it is reproducible from what is already recorded.
        """
        return None, None

    def slice_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None):
        """The model a slice reads: this run's own, or a generic forest.

        Split out from `compute_incumbent_slice` because it depends on
        *metric_name* and *values* and **not** on which hyperparameter is being
        sliced. A caller drawing several slices of one run therefore fits once
        and hands the result over, instead of refitting per hyperparameter —
        measured at roughly 300ms for a Gaussian process and 750ms for a random
        forest over fifteen trials, so a six-hyperparameter model was paying
        that six times as the reader clicked through.

        Returns `(surrogate, warning)`, the pair `refit_surrogate` returns.
        """
        rf, warning = self.refit_surrogate(config_space, trials, metric_name, seed,
                                           values=values)
        if rf is None:
            rf, warning = fit_surrogate(config_space, trials, metric_name, seed,
                                        values=values)
        return rf, warning

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

    def judges_priors(self, params: Mapping[str, Any]) -> bool:
        """Whether this optimizer, configured with *params*, can tell a prior
        worth acting on from one that is not. The page offers "Evaluate Prior"
        only when it can. A method rather than a flag like the two above,
        because for some optimizers it depends on how they are configured.
        """
        return False

    def acquisition_function(self) -> Optional[dict]:
        """Which acquisition function this optimizer maximizes, as `kind`
        ("ei" or "pi") and its improvement margin `xi`, or None for one that
        fits no model of the objective. The page computes the acquisition from
        the surrogate's mean and spread itself, so it needs to know which."""
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
        # Accepted and ignored: a search that fits no model of the objective has
        # no acquisition function for a prior to weight. The caller passes what
        # the reader stated without having to know which optimizer will get it.
        priors=None,
        # Configurations to evaluate before searching for any — see
        # `evaluate_requested`. None searches from the first trial.
        configurations=None,
        # Beliefs stated in a view of the space rather than on one axis — see
        # `core.projected_prior`. Ignored, like `priors`, by a search with no
        # acquisition function to weight.
        projected_priors=None,
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

        # What the optimizer was asked to minimise. Derived from the metric
        # rather than hardcoded, so that a metric that is already a cost stores
        # itself unchanged — and so that every metric that predates this stores
        # exactly the `1.0 - score` it always did. See `core.metrics.to_cost`.
        cost_of = result.metric(result.primary_metric)

        data = []
        for t in result.trials:
            incumbent_key = tuple(sorted(t.incumbent_config.items()))
            incumbent_cid = config_id_by_key.get(incumbent_key, str(t.trial))
            data.append({
                **t.run_info,  # SMAC-native fields (timing, seed, status, …); disjoint from the keys below
                "config_id": t.trial,
                "cost": to_cost(cost_of, t.score),
                "scores": t.scores,
                # None until something has succeeded: a run whose opening
                # trials all fail has no incumbent to record, and an infinity is
                # not JSON. Provenance only — every figure recomputes the
                # trajectory from the scores (`incumbent_scores`).
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
            # Keyed by trial number, and written from what each trial recorded
            # at the moment it was proposed. An empty origin is written through
            # rather than filled with this optimizer's name: "we did not record
            # where this came from" and "the model chose it" are different
            # facts, and something downstream reads them apart.
            "config_origins": {str(t.trial): t.origin for t in result.trials},
            "optimizer_state": {},
            "primary_metric": result.primary_metric,
            # Only the ones this build does not already know. A file should not
            # freeze the registry's own definitions into itself — those are code,
            # and a later build correcting one must not be overruled by every
            # file written before the correction.
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
        """Reconstruct an OptimizationResult from a runhistory-mirrored result dict.

        Inverse of serialize_result.  Optimizers that override serialize_result
        to embed extra state should override this to restore it.
        """
        configs = d.get("configs", {})
        data = d.get("data", [])
        origins = d.get("config_origins") or {}
        primary_metric = d.get("primary_metric", "")
        declared = declarations(d.get("declared_metrics"))

        # The inverse of what `serialize_result` wrote — see `core.metrics`. For
        # every metric that predates metrics describing themselves this is
        # exactly `1.0 - cost`, which is what makes every file ever written read
        # back unchanged. `declared` is what an imported run's own objective is
        # read through; empty for anything produced here.
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
            # A field absent from an older file reads as empty: no game, no
            # Möbius view, no scale to read a share in.
            **{name: d.get(name, {}) for name in GAME_RESULT_FIELDS},
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
    def known_params(cls, stored: Mapping[str, Any]) -> Dict[str, Any]:
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
    def migrate_params(cls, stored: Mapping[str, Any]) -> Mapping[str, Any]:
        """*stored* with settings from an earlier shape read onto today's.

        Dropping a renamed setting is safe but silent, and silent is the wrong
        answer here: a stored experiment or an `.ihpo` written last month would
        come back configured differently from how it ran, with nothing said. An
        optimizer that reshapes its settings translates them here instead.

        Base implementation has nothing to translate.
        """
        return stored

    #: How many coalition evaluations the eager, at-run-completion analytics may
    #: spend before they are skipped instead. Overridden per instance by the
    #: `ANALYTICS_EAGER_MAX_COALITIONS` Django setting — see
    #: `experiments/services/run.py`, which sets it before calling `optimize()`. It lives
    #: here as a plain attribute so `core/` stays Django-free.
    #:
    #: See `eager_analytics_budget_exceeded` for what the number means and why
    #: it is counted in coalitions rather than seconds.
    analytics_max_coalitions: Optional[int] = EAGER_MAX_COALITIONS

    #: Whether anything on the experiment page will actually display these
    #: numbers. False makes `compute_hp_games` decline outright — an experiment
    #: whose importance and interactions figures are both switched off was
    #: paying 2^n_hp coalition evaluations per game per metric to fill fields no
    #: page would read.
    #:
    #: Set per instance by `experiments/services/run.py`, the same way
    #: `analytics_max_coalitions` is, so `core/` stays Django-free. Defaults to
    #: True: a caller that says nothing gets the analytics, which is what every
    #: direct caller (the tests, a script) means.
    analytics_wanted: bool = True

    #: Called with the run's `TrialCollector` after every recorded trial, or
    #: None. How the application sees a run's trials while it is still running:
    #: `experiments/services/run.py` sets it to a throttled write of the partial result,
    #: so the experiment page has something newer than "nothing until it
    #: finishes" to draw. Set per instance, like the two above, so `core/` stays
    #: Django-free and a direct caller pays nothing for a feature it is not
    #: using.
    progress = None

    #: Called with each recorded trial and the model its last fold fitted, to
    #: keep for export — or None, and no trial keeps anything (see
    #: experiments/services/trial_models.py). Set by the caller, like `progress`.
    keep_model = None

    def new_collector(self, metric_name: str = "", previous_result=None,
                      **kwargs) -> TrialCollector:
        """The `TrialCollector` for one run of this optimizer.

        A factory rather than three constructor calls, so anything every run
        needs is wired once here instead of being remembered separately in
        `RandomOptimizer`, `GridOptimizer` and `SmacOptimizer` — which is how
        `progress` reaches all three without one line about it in any of them,
        and now what the score *means* as well.

        The metric is resolved from the run being resumed first, because a run
        imported with its own objective declares that objective in its own file
        and nowhere else.
        """
        declared = getattr(previous_result, "declared_metrics", None) or {}
        return TrialCollector(
            on_record=self.progress,
            on_keep=self.keep_model,
            metric=declared.get(metric_name) or metric_for(metric_name),
            **kwargs)

    def _begin(self, previous_result, primary_metric: str, n_trials, stopping, splits,
               X_train, y_train, X_val, y_val):
        """`(splits, previous_result, collector)` for a search that fits no model of
        the objective: one fold unless the caller divided the data itself (see
        core.splits), the history re-read under the current metric so the
        incumbent trajectory means what the page says it means (no surrogate
        here, so nothing else is stale), and a collector continuing it."""
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)
        previous_result, _ = rebase_history(previous_result, primary_metric)
        collector = self.new_collector(
            metric_name=primary_metric,
            previous_result=previous_result,
            trial_offset=len(previous_result.trials) if previous_result else 0,
            # None, not -inf: "nothing to beat yet" is the metric's own worst
            # end, which for a lower-is-better metric is +inf.
            initial_best_score=previous_result.best_score if previous_result else None,
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=merge_stopping(n_trials, stopping),
        )
        return splits, previous_result, collector

    def _finish(self, collector, previous_result, config_space, metrics, primary_metric: str,
                seed: int, cancel_event, **result_fields) -> "OptimizationResult":
        """The result of a search that fits no model: every trial so far, the
        best of them by the metric's own direction, and the explanation games."""
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
        """Evaluate the configurations a run was asked for by name, before it
        searches for any.

        *configurations* is a list of `{"config": {...}, "origin": "..."}`,
        evaluated in order while the run's criteria allow, each recorded as a
        trial like any the search produced. *told*, when given, is called with
        `(config, all_scores, run_info)` after each, for an optimizer that keeps
        a model of what it has seen. Returns the configurations evaluated, as
        the sorted-item keys the optimizers already use to skip repeats.
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
