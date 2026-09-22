import math
import time
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional
from hypershap import ExplanationTask, HyperSHAP

from ..metrics import (
    METRICS, declarations, describe, from_cost, metric_for, to_cost)
from .timing import CANCELLED, RUN_INFO_KEYS, STATUS_SUCCESS


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
    # The full Möbius (Harsanyi) decomposition of the same tunability call —
    # one value per coalition, of every size, rather than the order-2 grid
    # above. Read by the interaction figure's graph/upset/by-order views, which
    # need terms above order 2; the heatmap and top-pairs views keep reading
    # `hyperparameter_interactions`, whose FSII numbers this deliberately does
    # not replace (see `_shared_exact_computer` for why the two are not
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


def storable_score(score) -> Optional[float]:
    """*score* as something a file can hold, or None.

    The incumbent starts at the metric's worst end — an infinity — and stays
    there until a trial succeeds. A run whose opening trials all fail therefore
    has a best score of `-inf`, and `ui/services/run.py` writes a partial result
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

#: Default coalition budget for the eager, at-run-completion analytics. See
#: `BaseOptimizer.eager_analytics_budget_exceeded` for what a coalition costs and
#: why the budget is counted in them. 1024 admits both registry models (4 and 6
#: hyperparameters, 192 and 768 coalitions over 4 metrics — 3.6s and 12.7s) and
#: turns away the custom-upload pathology (10 hyperparameters would be 12,288
#: coalitions, around 3 minutes appended to every run). `None` means no limit.
#:
#: A default rather than a hard rule: deployments override it through
#: `ANALYTICS_EAGER_MAX_COALITIONS`, which is a statement about the machine's
#: capacity, not about any one experiment's taste.
EAGER_MAX_COALITIONS = 1024

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
    ):
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
        self.results.append(trial)
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


def _pair_trials_with_scores(config_space, trials: List[TrialResult], metric_name: str,
                             values=None):
    """Every trial's config, as a `ConfigSpace.Configuration`, paired with its
    *metric_name* score — the input shape both HyperSHAP and a plain
    surrogate fit need, and the one place that pairing happens, so
    `_compute_hp_game`'s own explainer and `fit_surrogate` never drift apart
    on how a trial becomes training data. A trial whose config the current
    config_space rejects (e.g. a leftover from before a model's search space
    changed) is skipped rather than raised on — one bad trial should not
    blank out every other one's contribution.

    *values* names what to learn, for the callers that want a surrogate over
    something other than a metric — `lambda t: t.duration` to model how long a
    configuration takes, which is a measured output of every trial that no
    metric reports. It is a callable rather than a second field name because
    what is modellable lives on `TrialResult` in more than one shape (`scores`
    is a mapping, `duration` a property), and because a caller that raises on
    a trial it cannot read should lose that trial the same way a rejected
    config does, not the whole fit. Default reads *metric_name* out of
    `scores`, which is what every existing caller means.
    """
    from ConfigSpace import Configuration

    if values is None:
        def values(t):
            return t.scores[metric_name]

    data = []
    for t in trials:
        try:
            cfg = Configuration(config_space, values=t.config)
            data.append((cfg, values(t)))
        except Exception:
            continue
    return data


def _pair_trials_with_oriented_scores(config_space, trials: List[TrialResult],
                                      metric_name: str, metric=None):
    """As `_pair_trials_with_scores`, with every score turned so bigger is better.

    `HP_GAMES` map MAX/VAR/MIN onto tunability/sensitivity/mistunability, and
    those names are only true of a score where bigger wins. Handed an optimizer
    cost, MAX finds the configuration that performed *worst* and the page calls
    it the most tunable — a wrong answer presented exactly like a right one.

    The training data is turned rather than the games, so no stored field
    changes meaning: `hyperparameter_mistunability` is a key in every `.ihpo`
    ever written, and a build that quietly redefined it would make old files and
    new ones disagree with nothing to say which is which.

    Negation, not `metric.high - score`, because it needs no bounds and an
    imported objective has none. Both preserve every difference between two
    scores, which is what the games are built out of.

    Deliberately separate from `fit_surrogate` rather than a flag on it. That
    one feeds partial dependence, which labels its axis with the metric and must
    stay in the metric's own units, and the uncertainty field, whose spread
    across trees is the same either way. One function with a flag would put the
    decision at every call site instead of in the two that need it.
    """
    metric = metric or metric_for(metric_name)
    data = _pair_trials_with_scores(config_space, trials, metric_name)
    if metric.higher_is_better:
        return data
    return [(cfg, -score) for cfg, score in data]


def fit_surrogate(config_space, trials: List[TrialResult], metric_name: str, seed: int = 0,
                  values=None):
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

    *values* is passed through to `_pair_trials_with_scores`: a callable
    naming what to learn, for a surrogate over a measured output that is not a
    metric (trial duration, say). *metric_name* is then only what the warning
    text talks about.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    data = _pair_trials_with_scores(config_space, trials, metric_name, values=values)
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


def predict_mean_std(surrogate, rows):
    """`(mean, std)` over *rows* from whichever surrogate this is.

    A SMAC model answers through `predict_marginalized`, which is the belief
    the search itself consulted. Anything else is a scikit-learn forest, whose
    band is tree disagreement — see `_predict_with_spread` for what that does
    and does not license.
    """
    import numpy as np

    predict = getattr(surrogate, "predict_marginalized", None)
    if predict is None:
        return _predict_with_spread(surrogate, rows)

    mean, var = predict(np.asarray(rows))
    return (np.asarray(mean, dtype=float).ravel(),
            np.sqrt(np.clip(np.asarray(var, dtype=float), 0.0, None)).ravel())


def _predict_with_spread(rf, rows):
    """The forest's own mean prediction over *rows*, and how much its trees
    disagree there — `(mean, std)`, each as long as *rows*.

    Every tree over every row in one pass each, rather than a call per row:
    sklearn's per-call overhead dwarfs the traversal at these sizes, so 200
    points through 100 trees is 100 calls, not 200.

    The mean is what `rf.predict` would return; computing it here from the same
    stack the spread needs costs nothing and keeps the pair consistent by
    construction — a caller drawing a band around a mean should never be able to
    get them from two different passes.

    **The spread is tree disagreement, not a posterior**, and it fails in a
    specific direction: far outside the region the trials cover, every tree
    falls into the same extreme leaf and agrees completely, so it reports
    confidence exactly where there is no data. `compute_surrogate_uncertainty`
    says more about what that does and does not license.
    """
    import numpy as np

    per_tree = np.stack([tree.predict(rows) for tree in rf.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


class _MetricExplainer:
    """Everything the three global games for one metric can share.

    `tunability`/`sensitivity`/`mistunability` all explain the *same* trial
    history for a given metric — only the aggregation (MAX/VAR/MIN) differs
    once the surrogate is fitted. So both expensive pieces are built at most
    once here and reused across the games: the HyperSHAP explainer eagerly, and
    the RandomForest stand-in lazily, since most runs never reach the fallback
    rung at all.

    Replaces the `(hs, warning, insufficient)` tuple this used to be, purely so
    the fallback surrogate has somewhere to live.
    """

    __slots__ = ("hs", "warning", "insufficient", "_fallback")

    def __init__(self, hs, warning: Optional[str], insufficient: bool):
        #: The built HyperSHAP explainer, or None if one couldn't be built.
        self.hs = hs
        #: Why there isn't one, when there isn't one.
        self.warning = warning
        #: True only for "fewer than two usable trials" — a condition answered
        #: with uniform weights directly, since there is nothing to fit the
        #: RandomForest rung from either.
        self.insufficient = insufficient
        self._fallback: Optional[tuple] = None

    def fallback_surrogate(self, config_space, trials, metric_name: str, seed: int) -> tuple:
        """`fit_surrogate`'s `(surrogate, warning)`, fitted at most once.

        The explainer above was already shared across games; this rung was not,
        so a metric whose HyperSHAP call raised refit an identical forest once
        per game. Memoized including the failure case: a fit that failed will
        fail the same way on the next game, and reporting it three times is not
        three pieces of information.
        """
        if self._fallback is None:
            self._fallback = fit_surrogate(config_space, trials, metric_name, seed)
        return self._fallback


#: What `HyperSHAP.tunability()`/`.sensitivity()`/`.mistunability()` build for
#: themselves, named here so `_shared_exact_computer` can build the same thing.
#: The aggregation is the only difference between the three games once the
#: surrogate exists: MAX = how much upside is there, VAR = how much does
#: performance move, MIN = how much downside does getting it wrong carry.
_GAME_PIECES = {
    "tunability": ("TunabilityExplanationTask", "TunabilityGame", "MAX"),
    "sensitivity": ("SensitivityExplanationTask", "SensitivityGame", "VAR"),
    "mistunability": ("MistunabilityExplanationTask", "MistunabilityGame", "MIN"),
}

#: How many configurations each game's searcher draws — HyperSHAP's own default
#: for `HyperSHAP.tunability` and its siblings, which `_shared_exact_computer`
#: has to match or it would be computing a different game from the one the
#: facade computes. The equality test in `tests/core/test_hp_importance.py` is
#: what catches it drifting.
#:
#: The searcher's *seed* is not here: it is the experiment's, passed down from
#: `optimize()` like every other stochastic thing. HyperSHAP defaults it to 0,
#: which meant two experiments with different seeds drew the same 10,000
#: configurations to explain themselves with — deterministic, but not the
#: experiment's determinism. The facade fallback in `_game_values` is given the
#: same seed, so the two paths stay comparable.
_HPO_SIMULATION_SAMPLES = 10_000


def _shared_exact_computer(explainer, game: str, seed: int = 0):
    """One `shapiq.ExactComputer` for *game*, read twice: FSII and Möbius.

    Returns `(fsii_order_2, moebius)` — the interaction values today's
    importance and interactions figures are built from, plus the full Möbius
    (Harsanyi) decomposition the graph, upset and by-order views need.

    **Why this reaches past HyperSHAP's facade**, which is a real cost and
    wants justifying:

    - *Why not call the facade twice?* `HyperSHAP.tunability()` builds a fresh
      `ExactComputer` per call, and an `ExactComputer` evaluates 2^n_hp
      coalitions. Two calls means paying that twice — measured at 6
      hyperparameters, 1110 ms each. One computer caches its game evaluations,
      so the second index is free: **1060 ms for FSII, then 1 ms for Möbius**.
      Doubling the most expensive thing a run does, to avoid the twenty lines
      below, is the wrong trade.

    - *Why not just ask the facade for a higher-order FSII?* Because FSII is
      the *faithful* least-squares k-order approximation — its terms are fitted
      jointly, so the order-1 values of an order-3 fit are not the order-1
      values of an order-2 fit. Measured: order-1 moves by 5% of the largest
      term, order-2 by 24%. Raising the order in place would silently rewrite
      `hyperparameter_importance` for every new run and make it incomparable
      with every stored one.

    - *Why Möbius?* It is the unique decomposition assigning one value per
      coalition, and unlike FSII it is truncation-independent — measured
      identical at order 2 and at order n. So it can be stored once and read at
      any order, which is exactly what a plot showing 2-way and 5-way
      interactions in the same picture needs. It is also the transform the
      graph is named after.

    **What protects this.** `hypershap` is at 0.0.6 and its internals may move.
    Two things guard that: a test asserting this function's FSII output is
    bit-identical to `HyperSHAP.<game>()`'s for all three games, which fails
    loudly on any drift; and the caller's `except`, which falls back to the
    facade and simply does without the Möbius-based views. Neither the numbers
    on the page nor a stored result can go quietly wrong.
    """
    from shapiq import ExactComputer
    import hypershap.games as hs_games
    import hypershap.task as hs_task
    from hypershap.utils import Aggregation, RandomConfigSpaceSearcher

    task_name, game_name, aggregation = _GAME_PIECES[game]
    source = explainer.hs.explanation_task
    surrogate = source.surrogate_model
    if isinstance(surrogate, list):
        surrogate = surrogate[0]

    config_space = source.config_space
    game_task = getattr(hs_task, task_name)(
        config_space=config_space, surrogate_model=surrogate,
        baseline_config=config_space.get_default_configuration())
    built = getattr(hs_games, game_name)(
        explanation_task=game_task,
        cs_searcher=RandomConfigSpaceSearcher(
            explanation_task=game_task, n_samples=_HPO_SIMULATION_SAMPLES,
            mode=getattr(Aggregation, aggregation), seed=seed))

    n_players = built.get_num_hyperparameters()
    computer = ExactComputer(n_players=n_players, game=built)
    # Order matters only for cost: FSII pays for the coalitions, Möbius reads
    # the cached game afterwards.
    return computer(index="FSII", order=2), computer(index="Moebius", order=n_players)


def _moebius_terms(iv, params: List[str]) -> list:
    """A Möbius `InteractionValues` as JSON-storable rows, strongest first.

    `[{"members": ["max_depth", "n_estimators"], "value": 0.013}, ...]` — a list
    rather than the nested dict `_extract_pairwise` produces, because these
    terms are of every size from 1 to n_hp and a dict keyed by member would
    need a separator convention no hyperparameter name is guaranteed to avoid.

    The empty coalition is dropped: it is the game's value with nothing tuned,
    a constant offset rather than anything attributable to a hyperparameter.
    """
    rows = [{"members": [params[i] for i in key], "value": float(value)}
            for key, value in iv.dict_values.items() if key]
    rows.sort(key=lambda row: abs(row["value"]), reverse=True)
    return rows


def _hp_grid(hp, n_points: int) -> list:
    """*n_points* values spanning one hyperparameter's domain — its full set
    of choices for a categorical hyperparameter (there is no "spanning" a
    finite, unordered set), otherwise *n_points* points spaced evenly across
    the hyperparameter's *own* scale and converted back to the values it
    actually takes (sorted-unique for an integer one, so the grid never
    suggests a value it couldn't hold).

    "Its own scale" is what `to_value` supplies, and it is the whole point of
    routing through ConfigSpace rather than reaching for `hp.lower`/`hp.upper`
    directly: for a log-scaled hyperparameter, evenly spaced in native units
    is not evenly spaced in the units it is *searched* in. `C` on the SVM
    model spans 0.01 to 100 logarithmically, so a native linspace puts 19 of
    20 points above 5.27 — while 69% of the configurations ConfigSpace
    actually samples fall below that second point. The grid would then explore
    almost entirely the tail the surrogate has the least to say about.

    Evenly spacing the *normalized* [0, 1] vector and mapping it back through
    `to_value` gives a log-spaced grid for a log hyperparameter and an
    unchanged, identical one for every linear hyperparameter. This is also
    what DeepCAVE's PDP does (`pyPDP`'s ICE grids on `linspace(0, 1)` in the
    same normalized representation).
    """
    import numpy as np
    from ConfigSpace import UniformIntegerHyperparameter

    if hasattr(hp, "choices"):
        return list(hp.choices)
    raw = hp.to_value(np.linspace(0.0, 1.0, n_points))
    if isinstance(hp, UniformIntegerHyperparameter):
        return sorted({int(v) for v in raw})
    return [float(v) for v in raw]


def _sample_evenly(items: list, limit: int) -> list:
    """At most *limit* of *items*, spread evenly across it. 0 means all.

    Evenly spaced rather than the first *limit*, because a trial history is
    ordered and its two ends do not look alike: the front is exploration and
    the back is exploitation, so a prefix would show only half the story. The
    first and last items are always included.

    Deterministic, so the same run draws the same picture on every request —
    a random sample would make the figure flicker between reloads for no
    benefit.
    """
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[-1]]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]


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

    def refit_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None):
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
            "hyperparameter_importance": result.hyperparameter_importance,
            "hyperparameter_importance_warning": result.hyperparameter_importance_warning,
            "hyperparameter_sensitivity": result.hyperparameter_sensitivity,
            "hyperparameter_sensitivity_warning": result.hyperparameter_sensitivity_warning,
            "hyperparameter_mistunability": result.hyperparameter_mistunability,
            "hyperparameter_mistunability_warning": result.hyperparameter_mistunability_warning,
            "hyperparameter_interactions": result.hyperparameter_interactions,
            "hyperparameter_interactions_warning": result.hyperparameter_interactions_warning,
            "hyperparameter_moebius": result.hyperparameter_moebius,
            "hyperparameter_sensitivity_interactions": result.hyperparameter_sensitivity_interactions,
            "hyperparameter_sensitivity_moebius": result.hyperparameter_sensitivity_moebius,
            "hyperparameter_mistunability_interactions": result.hyperparameter_mistunability_interactions,
            "hyperparameter_mistunability_moebius": result.hyperparameter_mistunability_moebius,
            "hyperparameter_tunability_total": result.hyperparameter_tunability_total,
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
            hyperparameter_importance=d.get("hyperparameter_importance", {}),
            hyperparameter_importance_warning=d.get("hyperparameter_importance_warning", {}),
            hyperparameter_sensitivity=d.get("hyperparameter_sensitivity", {}),
            hyperparameter_sensitivity_warning=d.get("hyperparameter_sensitivity_warning", {}),
            hyperparameter_mistunability=d.get("hyperparameter_mistunability", {}),
            hyperparameter_mistunability_warning=d.get("hyperparameter_mistunability_warning", {}),
            hyperparameter_interactions=d.get("hyperparameter_interactions", {}),
            hyperparameter_interactions_warning=d.get("hyperparameter_interactions_warning", {}),
            # Absent from every .ihpo written before this field existed, which
            # reads as "no Möbius views for this run" — the same empty state a
            # failed game already produces.
            hyperparameter_moebius=d.get("hyperparameter_moebius", {}),
            # Likewise absent from anything written before the interaction
            # figures could show a game other than tunability.
            hyperparameter_sensitivity_interactions=d.get("hyperparameter_sensitivity_interactions", {}),
            hyperparameter_sensitivity_moebius=d.get("hyperparameter_sensitivity_moebius", {}),
            hyperparameter_mistunability_interactions=d.get("hyperparameter_mistunability_interactions", {}),
            hyperparameter_mistunability_moebius=d.get("hyperparameter_mistunability_moebius", {}),
            # Absent from anything written before the shares could be read in
            # the metric's own units, which reads as "no scale for these" — the
            # figure then offers the achievable view alone.
            hyperparameter_tunability_total=d.get("hyperparameter_tunability_total", {}),
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

    #: How many coalition evaluations the eager, at-run-completion analytics may
    #: spend before they are skipped instead. Overridden per instance by the
    #: `ANALYTICS_EAGER_MAX_COALITIONS` Django setting — see
    #: `ui/services/run.py`, which sets it before calling `optimize()`. It lives
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
    #: Set per instance by `ui/services/run.py`, the same way
    #: `analytics_max_coalitions` is, so `core/` stays Django-free. Defaults to
    #: True: a caller that says nothing gets the analytics, which is what every
    #: direct caller (the tests, a script) means.
    analytics_wanted: bool = True

    #: Called with the run's `TrialCollector` after every recorded trial, or
    #: None. How the application sees a run's trials while it is still running:
    #: `ui/services/run.py` sets it to a throttled write of the partial result,
    #: so the experiment page has something newer than "nothing until it
    #: finishes" to draw. Set per instance, like the two above, so `core/` stays
    #: Django-free and a direct caller pays nothing for a feature it is not
    #: using.
    progress = None

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
            metric=declared.get(metric_name) or metric_for(metric_name),
            **kwargs)

    def eager_analytics_budget_exceeded(self, config_space, metrics) -> Optional[str]:
        """Why the global games shouldn't be computed for this run, or None.

        A shapiq `ExactComputer` evaluates **2^n_hp coalitions** per game, at
        roughly 15-19ms each (the constant drifts down as the count grows, as
        each game's fixed overhead amortizes). Crucially that is exponential in
        *hyperparameter count* and independent of trial count, so the expensive
        case is a short run of a wide model — the opposite of most people's
        intuition about what makes analytics slow.

        The budget is counted in coalitions rather than seconds on purpose: a
        seconds budget would bake a machine-calibrated constant into the code,
        and an HP-count budget would ignore how many metrics multiply it.

            4 HPs x 4 metrics =   192   compute   3.6s  measured
            6 HPs x 4 metrics =   768   compute  12.7s  measured
            8 HPs x 4 metrics =  3072   skip     45.6s  measured
           10 HPs x 4 metrics = 12288   skip     ~3min  extrapolated from the above

        Returns a message intended for the per-metric `hyperparameter_*_warning`
        fields — the page already renders those, so a skipped run explains itself
        with no extra UI.
        """
        budget = self.analytics_max_coalitions
        if not budget:
            return None
        # Materialized once: read three times below, and a generator would be
        # empty after the first — silently making `cost` zero and the guard a
        # no-op, which is the one failure mode a cost guard must not have.
        metrics = list(metrics)
        n_hp = len(list(config_space.keys()))
        cost = (2 ** n_hp) * len(self.HP_GAMES) * len(metrics)
        if cost <= budget:
            return None
        return (
            f"Importance analytics were skipped: {n_hp} hyperparameters over "
            f"{len(metrics)} metrics would need {cost:,} coalition "
            f"evaluations, past the {budget:,} this deployment allows "
            f"(ANALYTICS_EAGER_MAX_COALITIONS). Cost grows as 2^hyperparameters."
        )

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
        importance, warning, _interactions, _moebius, _total = self._compute_hp_game(
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
        importance, warning, _interactions, _moebius, _total = self._compute_hp_game(
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
        importance, warning, _interactions, _moebius, _total = self._compute_hp_game(
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
        _importance, warning, interactions, _moebius, _total = self._compute_hp_game(
            config_space, trials, metric_name, "tunability", seed)
        return interactions, warning

    def compute_hp_ablation(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        config_of_interest: Dict[str, Any],
        seed: int = 0,
        explainer=None,
        metric=None,
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

        *seed* reaches the surrogate the same way `_build_explainer` does,
        and for the same reason — it was declared but unused here until the
        compute-policy pass, so a local explanation silently ignored the
        experiment's seed.

        *explainer*, when given, is a `_MetricExplainer` for this
        (config_space, trials, metric_name) whose surrogate is reused instead of
        a fresh one being fitted — exactly what `_compute_hp_game` already
        accepts, and for the same reason. Measured: 79 ms per call standalone
        against 35 ms once plus 43 ms each shared. One trial's explanation
        barely notices; explaining every trial (see `compute_local_effects`)
        would otherwise refit an identical forest once per trial.
        """
        from ConfigSpace import Configuration
        from sklearn.ensemble import RandomForestRegressor

        params = list(config_space.keys())

        if explainer is not None:
            if explainer.hs is None:
                return {}, explainer.warning or "No surrogate for a local explanation."
            return self._ablate(explainer.hs, config_space, config_of_interest, params)

        data = _pair_trials_with_oriented_scores(config_space, trials, metric_name, metric)
        if len(data) < 2:
            return {}, "Not enough trials for a local explanation."

        try:
            task = ExplanationTask.from_data(
                config_space, data,
                base_model=RandomForestRegressor(n_estimators=100, random_state=seed))
            hs = HyperSHAP(task)
            config = Configuration(config_space, values=config_of_interest)
            baseline = config_space.get_default_configuration()
            iv = hs.ablation(config_of_interest=config, baseline_config=baseline)
            order1 = iv.get_n_order(order=1).dict_values
            return {params[idx]: val for (idx,), val in order1.items()}, None
        except Exception as e:
            return {}, f"HyperSHAP (ablation) failed ({e})."

    @staticmethod
    def _ablate(hs, config_space, config_of_interest, params):
        """One ablation game against the config space's default, from a
        HyperSHAP that already exists. The half of `compute_hp_ablation` that
        does not depend on how the explainer was obtained."""
        from ConfigSpace import Configuration

        try:
            iv = hs.ablation(
                config_of_interest=Configuration(config_space, values=config_of_interest),
                baseline_config=config_space.get_default_configuration())
            order1 = iv.get_n_order(order=1).dict_values
            return {params[idx]: val for (idx,), val in order1.items()}, None
        except Exception as e:
            return {}, f"HyperSHAP (ablation) failed ({e})."

    def compute_local_effects(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
        max_trials: int = 0,
        metric=None,
    ) -> tuple[List[str], List[dict], Optional[str]]:
        """Every sampled trial's local ablation, for the beeswarm figure.

        `compute_hp_ablation` answers "what did each hyperparameter's value do
        for *this* trial". This runs that question across the run, so the figure
        can show the *spread* of each hyperparameter's effect rather than one
        trial's or an average. A hyperparameter whose points cluster tightly
        behaves the same wherever you are in the space; one whose points fan
        across zero helps in some regions and hurts in others, and no averaged
        view can tell you that.

        Returns (hyperparameter_names, rows, warning), where each row is
        `{"index": i, "trial": n, "effects": {hp: signed_value}}` — `index` its
        position in *trials* and `trial` its number, which differ because the
        rows are a sample.

        **The expensive one, and the only new figure that is.** One ablation
        game per trial, each 2^n_hp coalitions. The explainer is built once and
        shared (see `compute_hp_ablation`'s `explainer` parameter, which exists
        for this), which takes it from 79 ms per trial to 35 ms once plus 43 ms
        each — but 43 ms per trial still adds up, so *max_trials* caps how many
        are asked for, 0 meaning all. Sampled evenly across the run by
        `_sample_evenly` rather than as a prefix, for the same reason partial
        dependence does: the front of a run is exploration and the back is
        exploitation, and a beeswarm of only the exploration half would describe
        a search that never happened.
        """
        explainer = self._build_explainer(config_space, trials, metric_name, seed,
                                          metric=metric)
        if explainer.hs is None:
            return [], [], (explainer.warning
                            or "Not enough trials for local explanations.")

        params = list(config_space.keys())
        rows = []
        # Positions, not just the trials: the sample skips trials, so a row's
        # place in this list says nothing about which trial it explains, and the
        # page's selection is a position in `trials`. See `_selection_meta`.
        for index, trial in _sample_evenly(list(enumerate(trials)), max_trials):
            effects, warning = self.compute_hp_ablation(
                config_space, trials, metric_name, trial.config, seed,
                explainer=explainer)
            if warning:
                return [], [], warning
            rows.append({"index": index, "trial": trial.trial, "effects": effects})

        if not rows:
            return [], [], "No trials to explain."
        return params, rows, None

    def compute_partial_dependence(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        hp_name: str,
        seed: int = 0,
        n_points: int = 20,
        max_ice_curves: int = 0,
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
        the grid-wise mean across every trial's ICE curve, which is the
        standard definition of the average.

        What it averages *over* is this app's own choice, and is not
        DeepCave's: the curves here are one per trial, so the mean is weighted
        by where the search actually went, while DeepCave draws its base
        configurations at random from the whole config space
        (`PDP.from_random_points(..., num_samples=10 x n_hp)`, capped at
        10,000). So this reads as "what the optimizer saw as it explored"
        rather than "what the space looks like on average" — deliberate, since
        every other figure on this page describes the run rather than the
        space, but not interchangeable with DeepCave's.

        A trial whose config plus the
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

        *max_ice_curves* caps how many trials contribute, 0 meaning all of
        them. Both the work and the answer scale with it: this is a cap on the
        trials that get predicted, not a cap on what is drawn afterwards, so
        the partial-dependence curve is the mean over the sampled trials rather
        than over every one. That is the only version of the cap that is worth
        having — the cost being capped is `n_trials x n_points` predictions and
        an equally large payload, and trimming after the fact would save
        neither.

        The sample is evenly spaced through the run rather than its first *k*
        trials, so the band still spans early exploration and late exploitation
        instead of showing only the beginning. It is deterministic, so the same
        run gives the same picture every time.

        Sampling is in keeping with what this plot means elsewhere: DeepCave's
        own PDP averages over configurations drawn at random from the config
        space, and caps what it draws too (`MAX_SHOWN_SAMPLES = 100`). A mean
        over a hundred trials spread across a run is a perfectly good estimate
        of a mean over ten thousand; six seconds and four megabytes to compute
        the exact one is not a good trade.

        Every synthetic configuration is predicted in **one** `rf.predict`
        call rather than one call each. That is not a micro-optimization:
        sklearn's per-call overhead dwarfs the actual forest traversal at this
        size, so the one-at-a-time version this replaced spent 1.544s where
        the batched one spends 0.0035s on the same 600 rows (30 trials × a
        20-point grid), and the gap grows with both. End to end that took this
        function from 1553ms to 52ms; what's left is mostly `fit_surrogate`.

        The `Configuration` construction stays per-point — it was only 10ms of
        the 1.55s, and it is what enforces conditionals and forbidden clauses,
        which is where the `None` holes come from.
        """
        import numpy as np
        from ConfigSpace import Configuration

        rf, warning = fit_surrogate(config_space, trials, metric_name, seed)
        if rf is None:
            return [], [], [], warning

        grid = _hp_grid(config_space[hp_name], n_points)

        trials = _sample_evenly(trials, max_ice_curves)

        # Build every (trial, grid value) row first, remembering where each one
        # belongs, and leave a hole where the config space rejected it.
        rows, slots = [], []
        for i, t in enumerate(trials):
            for j, value in enumerate(grid):
                try:
                    values = dict(t.config)
                    values[hp_name] = value
                    rows.append(Configuration(config_space, values=values).get_array())
                except Exception:
                    continue
                slots.append((i, j))

        ice_by_trial: List[List[Optional[float]]] = [
            [None] * len(grid) for _ in trials]
        if rows:
            predictions = rf.predict(np.array(rows))
            for (i, j), value in zip(slots, predictions):
                ice_by_trial[i][j] = float(value)

        ice_lines = [row for row in ice_by_trial if any(v is not None for v in row)]

        if not ice_lines:
            return [], [], [], "No valid configurations on this hyperparameter's grid."

        pdp = []
        for i in range(len(grid)):
            column = [row[i] for row in ice_lines if row[i] is not None]
            pdp.append(sum(column) / len(column) if column else None)

        return grid, ice_lines, pdp, None

    def compute_surrogate_uncertainty(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        x_hp: str,
        y_hp: str,
        seed: int = 0,
        n_points: int = 20,
    ) -> tuple[list, list, List[List[Optional[float]]], Optional[str]]:
        """How unsure the surrogate is across the plane spanned by two
        hyperparameters — the field drawn under the configuration cube's axes
        view.

        Returns `(x_grid, y_grid, z_rows, warning)`. `z_rows` is one row per
        *y_grid* value, each as long as *x_grid*, holding `None` where the
        config space rejected that combination.

        **The estimate is the spread across the forest's own trees.**
        `fit_surrogate` fits a `RandomForestRegressor`; each of its trees
        predicts the same point separately, and the standard deviation of those
        predictions is how much they disagree. No new fit and no second model —
        the uncertainty was already sitting inside the one PDP and the games'
        fallback rung already build.

        Standard deviation rather than variance, which is what makes it
        readable: it is in the metric's own units, so "the trees disagree by
        0.04 accuracy here" is a sentence. Squared accuracy is not.

        **What it is not.** Tree disagreement is not a calibrated posterior, and
        it fails in a specific direction worth knowing about: far outside the
        region the trials cover, every tree falls into the same extreme leaf and
        agrees *completely*, so the field goes confident exactly where there is
        no data. It reads as "where do the trees disagree", which is a real
        thing to see over a search's own footprint, and not as "where might the
        truth be".

        **A slice, not an average.** The other hyperparameters are held at the
        best trial's configuration rather than averaged over every trial the way
        `compute_partial_dependence` averages its ICE curves. Two reasons: the
        average would cost `n_trials x n_points^2` rows through every tree
        rather than `n_points^2`, and — the deciding one — a plane through the
        incumbent passes through a configuration that was actually evaluated,
        while a plane through the config-space default may sit somewhere the
        search never went and report uniform ignorance. The default is what
        tunability is measured from; this is a different question and takes a
        different reference.
        """
        import numpy as np
        from ConfigSpace import Configuration

        for name in (x_hp, y_hp):
            if name not in config_space:
                return [], [], [], f"No such hyperparameter: {name}."
        if x_hp == y_hp:
            return [], [], [], "Two different hyperparameters are needed for a plane."

        rf, warning = fit_surrogate(config_space, trials, metric_name, seed)
        if rf is None:
            return [], [], [], warning

        # The metric's own best, and only among trials that have the score: a
        # `.get(..., 0.0)` default would hand the plane to a trial that was
        # never measured under this metric, and under a lower-is-better one a
        # missing score of 0.0 would win outright.
        metric = metric_for(metric_name)
        scored = [t for t in trials if metric_name in t.scores]
        if not scored:
            return [], [], [], f"No trial carries a score for {metric_name}."
        picker = max if metric.higher_is_better else min
        best = picker(scored, key=lambda t: t.scores[metric_name])
        x_grid = _hp_grid(config_space[x_hp], n_points)
        y_grid = _hp_grid(config_space[y_hp], n_points)
        if len(x_grid) < 2 or len(y_grid) < 2:
            return [], [], [], "A plane needs both axes to take more than one value."

        rows, slots = [], []
        for j, y_value in enumerate(y_grid):
            for i, x_value in enumerate(x_grid):
                try:
                    values = dict(best.config)
                    values[x_hp], values[y_hp] = x_value, y_value
                    rows.append(Configuration(config_space, values=values).get_array())
                except Exception:  # noqa: BLE001 — conditionals, forbidden clauses
                    continue
                slots.append((j, i))

        z: List[List[Optional[float]]] = [[None] * len(x_grid) for _ in y_grid]
        if not rows:
            return [], [], [], "No valid configurations on this pair of axes."

        _, spread = _predict_with_spread(rf, np.array(rows))
        for (j, i), value in zip(slots, spread):
            z[j][i] = float(value)

        return x_grid, y_grid, z, None

    def compute_incumbent_slice(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        hp_name: str,
        seed: int = 0,
        n_points: int = 101,
        values=None,
    ) -> tuple[list, list, list, list, Optional[float], Optional[str]]:
        """The surrogate along one hyperparameter, every other one held at the
        incumbent — the 1-D restriction of `compute_surrogate_uncertainty`,
        sharing its surrogate, its grid and its choice of where to slice.

        Returns `(grid, positions, mu, sigma, eta, warning)`.

        **`positions` is the point of this function**, and the reason it does
        not simply hand back `linspace(0, 1)`. A prior over a hyperparameter is
        a density on ConfigSpace's *normalized* representation — the space
        `get_array` produces and the one SMAC's acquisition weights are
        evaluated in — so anything drawing or editing such a density needs each
        grid value's normalized coordinate, not its native one. The two are not
        the same axis: `lr` spanning 1e-4 to 1e-1 logarithmically has grid
        values 1e-4, 5.6e-4, 3.2e-3, 1.8e-2, 1e-1 sitting at a perfectly even
        0, .25, .5, .75, 1. A belief drawn against the native values would put
        its mass in a different place than the one the optimizer would read.

        Nor is `positions` evenly spaced in general, which is why it is computed
        rather than assumed: `_hp_grid` collapses an integer hyperparameter's
        grid to its distinct values, so a 5-point grid over 1..20 comes back as
        1, 6, 10, 15, 20 at 0, .263, .474, .737, 1. A categorical's positions
        are its choice indices, ConfigSpace's own encoding for them.

        *values* names what to model, and is passed through to `fit_surrogate` —
        `lambda t: t.duration` gives the same slice through a model of how long
        a configuration takes, which is what an output constraint needs to show
        the probability its bound holds. The incumbent is picked by
        *metric_name* either way, deliberately: two slices that do not pass
        through the same configuration cannot be read against each other.

        `eta` is the best *observed* value of whatever was modelled, among the
        trials that trained the surrogate — the incumbent's own score in the
        default case, which is what expected improvement improves over. `None`
        when nothing usable was observed.
        """
        import numpy as np
        from ConfigSpace import Configuration

        if hp_name not in config_space:
            return [], [], [], [], None, f"No such hyperparameter: {hp_name}."

        # The run's own model class where the optimizer can rebuild one, so a
        # Gaussian-process run is read through a Gaussian process rather than
        # through a stand-in forest that never saw the search. The object itself
        # does not survive `optimize()`, but its class and settings do, and the
        # trials it was fitted on are recorded — so it is refitted, not
        # recovered. Anything that cannot (random search, grid search, a refit
        # that raises) keeps the forest.
        rf, warning = self.refit_surrogate(config_space, trials, metric_name, seed, values=values)
        if rf is None:
            rf, warning = fit_surrogate(config_space, trials, metric_name, seed, values=values)
        if rf is None:
            return [], [], [], [], None, warning

        # The same incumbent `compute_surrogate_uncertainty` slices through, and
        # for the same reason: a line through a configuration that was actually
        # evaluated, rather than through a config-space default the search may
        # never have visited. Only trials carrying the score count — a
        # `.get(..., 0.0)` default would hand the slice to an unmeasured trial,
        # and under a lower-is-better metric a missing 0.0 would win outright.
        metric = metric_for(metric_name)
        scored = [t for t in trials if metric_name in t.scores]
        if not scored:
            return [], [], [], [], None, f"No trial carries a score for {metric_name}."
        picker = max if metric.higher_is_better else min
        best = picker(scored, key=lambda t: t.scores[metric_name])

        hp = config_space[hp_name]
        grid = _hp_grid(hp, n_points)
        if len(grid) < 2:
            return [], [], [], [], None, "A slice needs the hyperparameter to take more than one value."

        # Built together so a grid value the config space rejects drops out of
        # every one of them at once — three lists the browser indexes in
        # lockstep must not be able to fall out of step here.
        kept_values, kept_positions, rows = [], [], []
        positions = np.asarray(hp.to_vector(np.asarray(grid)), dtype=float).ravel()
        for value, position in zip(grid, positions):
            try:
                config = dict(best.config)
                config[hp_name] = value
                rows.append(Configuration(config_space, values=config).get_array())
            except Exception:  # noqa: BLE001 — conditionals, forbidden clauses
                continue
            kept_values.append(value)
            kept_positions.append(float(position))

        if len(rows) < 2:
            return [], [], [], [], None, f"No valid configurations along {hp_name}."

        mu, sigma = predict_mean_std(rf, np.array(rows))

        observed = [v for _, v in _pair_trials_with_scores(
            config_space, trials, metric_name, values=values)]
        eta = None
        if observed:
            eta = float(picker(observed) if values is None else min(observed))

        return (kept_values, kept_positions,
                [float(v) for v in mu], [float(v) for v in sigma], eta, None)

    def _skipped_games(self, metrics, reason: str) -> Dict[str, tuple]:
        """`compute_hp_games`' return shape for "these weren't computed".

        Empty values plus the reason as every metric's warning — deliberately
        the same shape a HyperSHAP failure already produces, so nothing
        downstream has to tell "couldn't" from "wouldn't". Empty importance
        degrades cleanly at both consumers: parallel coordinates falls back to
        config order, and the partial-dependence picker to the first
        hyperparameter.
        """
        return {game: ({m: {} for m in metrics},
                       {m: reason for m in metrics},
                       {m: {} for m in metrics},
                       {m: [] for m in metrics},
                       {m: 0.0 for m in metrics})
                for game in self.HP_GAMES}

    def compute_hp_games(
        self,
        config_space,
        trials: List[TrialResult],
        metrics,
        seed: int = 0,
        cancel_event=None,
    ) -> Dict[str, tuple[Dict[str, Dict[str, float]], Dict[str, Optional[str]], Dict[str, Dict[str, Dict[str, float]]]]]:
        """Every metric's per-game hyperparameter importance, for all of
        `HP_GAMES` — what each optimizer's `optimize()` calls once, instead of
        looping `compute_hp_importance`/`_sensitivity`/`_mistunability`
        separately over every metric three times.

        Returns `{game: (importance_by_metric, warning_by_metric,
        interactions_by_metric, moebius_by_metric, total_by_metric)}`, the
        first pair shaped exactly like an
        `OptimizationResult`'s `hyperparameter_<game>`/`hyperparameter_<game>_warning`
        fields expect. `interactions_by_metric` and `moebius_by_metric` are a
        free byproduct of the same call (see `compute_hp_interactions`) for
        every game — which is what lets the page's one game selector drive the
        interaction figures as well as the importance figure: all three games'
        grids were always computed here, and for a long time two of them were
        thrown away for want of a field to put them in.

        Builds the HyperSHAP explainer once per metric, not once per (game,
        metric) pair: `tunability`/`sensitivity`/`mistunability` all explain
        the *same* trial history for a given metric, so refitting the
        surrogate underneath them 3 times over is pure waste. See
        `_build_explainer`/`_compute_hp_game`'s `explainer` parameter.

        Declines in three cases, all reported through `_skipped_games`:

        - **The run was cancelled.** Pressing Cancel used to still buy you the
          full analytics bill — every optimizer breaks out of its trial loop on
          `cancel_event` and then called this unconditionally, so on a wide model
          you waited minutes for a run you had just stopped. Nothing is lost
          permanently: resuming recomputes over `previous + new` trials.
        - **Nobody is going to look.** Both figures that display these numbers
          are switched off for this experiment, so the fields would be filled
          and never read. See `analytics_wanted`.
        - **The run is too wide to afford.** See
          `eager_analytics_budget_exceeded`.

        Checked in that order: a cancelled run shouldn't be told about a budget
        it never got to spend, and neither should one nobody asked for.
        """
        if cancel_event is not None and cancel_event.is_set():
            return self._skipped_games(
                metrics,
                "Importance analytics were skipped because the run was "
                "cancelled. Resuming the run computes them over every trial.")

        if not self.analytics_wanted:
            return self._skipped_games(
                metrics,
                "Importance analytics were skipped: both figures that show "
                "them are switched off for this experiment. Switch one on and "
                "run again to compute them.")

        too_wide = self.eager_analytics_budget_exceeded(config_space, metrics)
        if too_wide:
            return self._skipped_games(metrics, too_wide)

        by_game = {game: ({}, {}, {}, {}, {}) for game in self.HP_GAMES}
        # *metrics* is a mapping of name to `Metric` from the application and a
        # bare list of names from plenty of callers that only need the names.
        # Both iterate the same way, but only a mapping can say which direction
        # a metric runs — from a list the registry answers, which is right for
        # every metric this build ships and wrong only for one a file declared.
        # A caller holding declared metrics has the mapping.
        directions = metrics if isinstance(metrics, Mapping) else {}
        for metric_name in metrics:
            explainer = self._build_explainer(config_space, trials, metric_name, seed,
                                              metric=directions.get(metric_name))
            for game in self.HP_GAMES:
                importance, warning, interactions, moebius, total = self._compute_hp_game(
                    config_space, trials, metric_name, game, seed, explainer=explainer)
                by_game[game][0][metric_name] = importance
                by_game[game][1][metric_name] = warning
                by_game[game][2][metric_name] = interactions
                by_game[game][3][metric_name] = moebius
                by_game[game][4][metric_name] = total
        return by_game

    def _build_explainer(self, config_space, trials: List[TrialResult], metric_name: str,
                         seed: int = 0, metric=None):
        """Pair *trials* with *metric_name* scores and fit the HyperSHAP
        explainer every global game (`tunability`/`sensitivity`/
        `mistunability`) shares for a given metric — the one expensive step
        (constructing an `ExplanationTask` fits a surrogate internally)
        that's identical across all three; only each game's own aggregation
        (MAX/VAR/MIN) differs once it's built. `compute_hp_games` builds this
        once per metric and reuses it across `HP_GAMES`; `_compute_hp_game`
        builds its own when called standalone (e.g. `compute_hp_importance`)
        and no `explainer` was handed to it.

        Returns a `_MetricExplainer` — pass straight through as
        `_compute_hp_game`'s `explainer`. Its `insufficient` is True only for
        "fewer than two usable trials," a condition `_compute_hp_game`
        answers with uniform weights directly, skipping the RandomForest
        rung entirely (there's nothing to fit it from either); it's False
        (with `hs` None) when HyperSHAP itself failed to build one, which
        does still warrant trying that rung.

        *seed* reaches the surrogate through an explicit `base_model`.
        HyperSHAP's own default is `RandomForestRegressor(random_state=0)`
        — a fixed 0, not whatever seed the experiment was run with — so
        leaving it out (as this did until the compute-policy pass) quietly
        made every explanation ignore the seed. At `seed=0` this builds the
        identical estimator, so the numbers only move where they were wrong.
        """
        from sklearn.ensemble import RandomForestRegressor

        data = _pair_trials_with_oriented_scores(config_space, trials, metric_name, metric)
        if len(data) < 2:
            return _MetricExplainer(
                None,
                "Not enough trials for importance estimation; showing uniform weights.",
                True)
        try:
            task = ExplanationTask.from_data(
                config_space, data,
                base_model=RandomForestRegressor(n_estimators=100, random_state=seed))
            return _MetricExplainer(HyperSHAP(task), None, False)
        except Exception as e:
            return _MetricExplainer(None, f"HyperSHAP failed to build a surrogate ({e})", False)

    def _compute_hp_game(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        game: str,
        seed: int = 0,
        explainer=None,
    ) -> tuple[Dict[str, float], Optional[str], Dict[str, Dict[str, float]], list]:
        """Shared scaffolding behind `compute_hp_importance`/`_sensitivity`/
        `_mistunability`/`_interactions`: ask HyperSHAP's *game* for order-1
        (and, as a free byproduct, order-2) values from the built explainer,
        and fall back to a surrogate's feature_importances_ and then uniform
        weights if that fails. *game* is a `HyperSHAP` instance method name
        taking no required arguments (`tunability`, `sensitivity`,
        `mistunability` all qualify; `ablation` does not — it needs a
        specific configuration to explain, not just trial history, so it is
        computed separately, on demand, for one selected trial).

        *explainer*, when given, is `_build_explainer`'s own `_MetricExplainer`
        for this (config_space, trials, metric_name) — see `compute_hp_games`,
        which builds it once and reuses it across every game — passed straight
        through instead of rebuilding (and re-fitting) an identical surrogate
        per game. `None` (the default) builds its own, for a standalone call
        like `compute_hp_importance`. The fallback surrogate is shared through
        the same object, so a metric whose HyperSHAP call fails fits one forest
        rather than one per game.

        Returns (values_dict, warning_message, interactions_dict, moebius_terms,
        total). *total* is the sum of the raw order-1 magnitudes, before
        values_dict was normalised into shares of it — so `share x total`
        recovers what a hyperparameter is worth in the metric's own units. Kept
        because a share cannot say whether the whole is worth eight accuracy
        points or eight thousandths of one, and because "how much of this has
        already been banked" is a ratio of raw values, not of shares (see
        `OptimizationResult.hyperparameter_tunability_total`). 0.0 on every rung
        that did not get a real answer out of HyperSHAP.
        warning_message is None when HyperSHAP succeeds and found something.
        interactions_dict is `{}` on either fallback rung — a plain
        feature_importances_ or uniform weights carry no pairwise structure to
        report — and values_dict is `{}` too on the one rung where HyperSHAP
        succeeds but every value it returns is zero, which is a real answer
        rather than a failure (see the branch itself).

        moebius_terms is the full Möbius decomposition (see `_moebius_terms`),
        or `[]` — on either fallback rung, and also when only
        `_shared_exact_computer` failed, in which case the FSII numbers are
        still HyperSHAP's own and only the views that read Möbius go empty.
        """
        params = list(config_space.keys())

        if explainer is None:
            explainer = self._build_explainer(config_space, trials, metric_name, seed)
        build_warning = explainer.warning

        if explainer.insufficient:
            uniform = 1.0 / len(params) if params else 0.0
            return {p: uniform for p in params}, build_warning, {}, [], 0.0

        warning = None
        if explainer.hs is not None:
            try:
                iv, moebius_iv = self._game_values(explainer, game, seed)
                moebius = _moebius_terms(moebius_iv, params) if moebius_iv is not None else []
                order1 = iv.get_n_order(order=1).dict_values
                raw = {params[idx]: abs(val) for (idx,), val in order1.items()}
                total = sum(raw.values())
                if not total:
                    # Every contribution was exactly zero — which happens on a
                    # saturated objective, where the surrogate's aggregate is
                    # the same whichever hyperparameters a coalition frees.
                    # That is an answer ("nothing here"), not a failure, and it
                    # is emphatically not the uniform-weights rung above, which
                    # means the opposite ("we could not tell"). Reported as
                    # empty because every consumer already handles empty
                    # correctly: the plot builder returns None so the figure
                    # shows its caption, parallel coordinates falls back to
                    # config order and `top_hp` to the first hyperparameter.
                    # Normalizing anyway (the `or 1.0` this replaces) produced
                    # a dict of zeros, which is truthy — so the pie was built
                    # from it and drew nothing, with no warning to say why.
                    return {}, (
                        f"HyperSHAP ({game}) found no signal in this run's trials: "
                        f"every hyperparameter's contribution was zero. This "
                        f"usually means the objective is saturated — try a "
                        f"harder dataset or a wider search space."), {}, [], 0.0
                importance = {k: v / total for k, v in raw.items()}
                interactions = self._extract_pairwise(iv, params)
                return importance, None, interactions, moebius, total
            except Exception as e:
                warning = f"HyperSHAP ({game}) failed ({e}) — falling back to surrogate feature importances."
        else:
            # `build_warning` already names HyperSHAP, so prefixing it with the
            # game alone avoids "HyperSHAP (tunability) HyperSHAP failed …".
            warning = f"{build_warning} ({game}) — falling back to surrogate feature importances."

        rf, fit_warning = explainer.fallback_surrogate(
            config_space, trials, metric_name, seed)
        if rf is not None:
            importances = rf.feature_importances_
            total = importances.sum() or 1.0
            return ({p: float(importances[i] / total) for i, p in enumerate(params)},
                    warning, {}, [], 0.0)
        warning += f" Surrogate fallback also failed ({fit_warning}); showing uniform weights."
        uniform = 1.0 / len(params) if params else 0.0
        return {p: uniform for p in params}, warning, {}, [], 0.0

    def _game_values(self, explainer, game: str, seed: int = 0):
        """*game*'s FSII order-2 values and its Möbius decomposition.

        One `shapiq.ExactComputer` gives both for the price of the first — see
        `_shared_exact_computer`, which explains at length why this doesn't go
        through `HyperSHAP.<game>()` and what guards that.

        The fallback is the point of keeping this separate: if hypershap's
        internals move under us, we still get the facade's own FSII and only
        lose the Möbius-based views, which then show the "no data" caption they
        already show when a game fails outright. A failure here must not be able
        to take the importance numbers with it. Both paths are given the
        experiment's *seed*, so falling back changes what is computed as little
        as it can.
        """
        try:
            return _shared_exact_computer(explainer, game, seed)
        except Exception:
            return getattr(explainer.hs, game)(seed=seed), None

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
