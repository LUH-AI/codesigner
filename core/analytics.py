"""What a search's trials say about its hyperparameters: surrogates fitted
to them, the HyperSHAP games, slices and samples of the space, and what the
stored analytics say when read together."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

from hypershap import ExplanationTask, HyperSHAP
from shapiq.game_theory import MoebiusConverter

from .metrics import metric_for

if TYPE_CHECKING:
    import numpy as np

    from .optimizers.base import TrialResult

#: HyperSHAP's global "explanation games" this app surfaces, each answering
#: a different question about the same trial history: tunability = how
#: much upside does tuning this hyperparameter offer (HyperSHAP's MAX
#: aggregation), sensitivity = how much does performance vary as it moves
#: (VAR), mistunability = how much downside risk does getting it wrong
#: carry (MIN). Same call shape, same fallback ladder — see
#: `compute_hp_game`. `optimizer_bias` exists too but needs a live
#: ensemble of optimizers to compare against, not just one run's trial
#: history, so it doesn't fit this app's model and isn't offered.
HP_GAMES = ("tunability", "sensitivity", "mistunability")

#: How many configurations the configuration cube's prediction layer is
#: drawn from. A power of two, which is what keeps a Sobol sequence balanced,
#: and eight to a cell of the 32 x 32 field the page bins them into.
PREDICTION_SAMPLES = 8192

#: Default coalition budget for the eager, at-run-completion analytics. See
#: `eager_analytics_budget_exceeded` for what a coalition costs and
#: why the budget is counted in them. 1024 was set against the two original
#: models (4 and 6 hyperparameters, 192 and 768 coalitions over 4 metrics — 3.6s
#: and 12.7s) and turns away the custom-upload pathology (10 hyperparameters
#: would be 12,288 coalitions, around 3 minutes appended to every run). A
#: classification run now scores five metrics, seven with the probability ones:
#: every built-in model has at most five hyperparameters and stays inside it
#: even then, except the SVM, whose six are 960 coalitions without them and
#: 1,344 with. `None` means no limit.
#:
#: A default rather than a hard rule: deployments override it through
#: `ANALYTICS_EAGER_MAX_COALITIONS`, which is a statement about the machine's
#: capacity, not about any one experiment's taste.
EAGER_MAX_COALITIONS = 1024


def pair_trials_with_scores(config_space, trials: List[TrialResult], metric_name: str,
                             values=None):
    """Every trial's config, as a `ConfigSpace.Configuration`, paired with its
    *metric_name* score — the input shape both HyperSHAP and a plain
    surrogate fit need, and the one place that pairing happens, so
    `compute_hp_game`'s own explainer and `fit_surrogate` never drift apart
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
        values = lambda t: t.scores[metric_name]

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
    """As `pair_trials_with_scores`, with every score turned so bigger is better.

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
    data = pair_trials_with_scores(config_space, trials, metric_name)
    if metric.higher_is_better:
        return data
    return [(cfg, -score) for cfg, score in data]


def fit_surrogate(config_space, trials: List[TrialResult], metric_name: str, seed: int = 0,
                  values=None):
    """Fit a cheap `RandomForestRegressor` surrogate over *trials*' recorded
    *metric_name* scores.

    Factored out of `compute_hp_game`'s own fallback rung (used when
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

    *values* is passed through to `pair_trials_with_scores`: a callable
    naming what to learn, for a surrogate over a measured output that is not a
    metric (trial duration, say). *metric_name* is then only what the warning
    text talks about.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    data = pair_trials_with_scores(config_space, trials, metric_name, values=values)
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


class IncumbentSlice(NamedTuple):
    """What `compute_incumbent_slice` answers with.

    A named type rather than a tuple because this has grown twice — five fields,
    then six with `positions`, now seven — and every caller unpacked it
    positionally, so each addition silently rewrote what every name meant. The
    panels being built on top of it will want more.

    `incumbent` is `(position, value)`, or None where there is nothing measured
    to mark: the one configuration on the slice that was actually evaluated,
    since every other hyperparameter is held at its value. `eta` is the best
    observed value of whatever was modelled, which is a different question and
    a different number whenever `values=` is in play.
    """

    grid: list
    positions: list
    mu: list
    sigma: list
    eta: Optional[float]
    incumbent: Optional[tuple]
    #: `(positions, mu, sigma)` for configurations drawn from the *whole* space,
    #: projected onto this hyperparameter's axis — the shadow the slice cannot
    #: see, because the slice holds every other hyperparameter fixed. None when
    #: no cloud was sampled.
    cloud: Optional[tuple]
    warning: Optional[str]
    #: The configurations the slice and the cloud were predicted at, in
    #: ConfigSpace's vectorized form — what a prior over whole configurations
    #: is evaluated at to weigh them.
    slice_rows: Optional["np.ndarray"] = None
    cloud_rows: Optional["np.ndarray"] = None


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


def _baseline_config(config_space, baseline: Optional[Dict[str, Any]]):
    """*baseline* as a `Configuration` of *config_space*, or the space's default
    when there is none: what every HyperSHAP game here measures its gains from.
    """
    from ConfigSpace import Configuration

    if baseline is None:
        return config_space.get_default_configuration()
    return Configuration(config_space, values=baseline)


def _moebius_terms(named: Dict[tuple, float]) -> list:
    """Möbius terms, as `HyperSHAP.get_interaction_values_with_names` keys
    them, turned into JSON-storable rows, strongest first.

    `[{"members": ["max_depth", "n_estimators"], "value": 0.013}, ...]` — a list
    rather than the nested dict `_extract_pairwise` produces, because these
    terms are of every size from 1 to n_hp and a dict keyed by member would
    need a separator convention no hyperparameter name is guaranteed to avoid.

    The empty coalition is dropped: it is the game's value with nothing tuned,
    a constant offset rather than anything attributable to a hyperparameter.
    """
    rows = [{"members": list(members), "value": float(value)}
            for members, value in named.items() if members]
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


def _from_unit(hp, u: float):
    """The value hyperparameter *hp* takes at *u* in [0, 1) along its own scale:
    logarithmic for a log one, an even share of its choices for a categorical,
    its value for a constant."""
    import numpy as np

    choices = getattr(hp, "choices", None) or getattr(hp, "sequence", None)
    if choices:
        return choices[min(int(u * len(choices)), len(choices) - 1)]
    if not hasattr(hp, "lower"):
        return getattr(hp, "value", getattr(hp, "default_value", None))
    value = hp.to_value(np.array([u]))[0]
    return int(value) if isinstance(hp.default_value, (int, np.integer)) else float(value)


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


_GameResults = tuple[Dict[str, Dict[str, float]], Dict[str, Optional[str]],
                     Dict[str, Dict[str, Dict[str, float]]], Dict[str, list], Dict[str, float]]


def eager_analytics_budget_exceeded(config_space, metrics,
                                    budget: Optional[int] = EAGER_MAX_COALITIONS) -> Optional[str]:
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
    if not budget:
        return None
    # Materialized once: read three times below, and a generator would be
    # empty after the first — silently making `cost` zero and the guard a
    # no-op, which is the one failure mode a cost guard must not have.
    metrics = list(metrics)
    n_hp = len(list(config_space.keys()))
    cost = (2 ** n_hp) * len(HP_GAMES) * len(metrics)
    if cost <= budget:
        return None
    return (
        f"Importance analytics were skipped: {n_hp} hyperparameters over "
        f"{len(metrics)} metrics would need {cost:,} coalition "
        f"evaluations, past the {budget:,} this deployment allows "
        f"(ANALYTICS_EAGER_MAX_COALITIONS). Cost grows as 2^hyperparameters."
    )


def compute_hp_ablation(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    config_of_interest: Dict[str, Any],
    seed: int = 0,
    explainer=None,
    metric=None,
    baseline: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, float], Optional[str]]:
    """HyperSHAP's "ablation" game: how much each hyperparameter's value in
    *config_of_interest* helped or hurt *metric_name*, versus *baseline* —
    a *local* explanation of one specific trial, unlike `HP_GAMES`' global
    ones. *baseline* is a configuration as a dict, the config space's
    default when left out.

    Signed, deliberately: positive means that hyperparameter's value in
    this trial beat the baseline's, negative means it lost to it. Zeroing
    that out with `abs()` (as the three global games do, on purpose,
    since they answer "how much does this matter" rather than "which
    direction") would throw away the one thing this view exists to show.

    Returns (values_dict, warning_message). Unlike `compute_hp_game`,
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
    a fresh one being fitted — exactly what `compute_hp_game` already
    accepts, and for the same reason. Measured: 79 ms per call standalone
    against 35 ms once plus 43 ms each shared. One trial's explanation
    barely notices; explaining every trial (see `compute_local_effects`)
    would otherwise refit an identical forest once per trial.
    """
    from sklearn.ensemble import RandomForestRegressor

    if explainer is not None:
        if explainer.hs is None:
            return {}, explainer.warning or "No surrogate for a local explanation."
        return _ablate(explainer.hs, config_space, config_of_interest, baseline)

    data = _pair_trials_with_oriented_scores(config_space, trials, metric_name, metric)
    if len(data) < 2:
        return {}, "Not enough trials for a local explanation."

    try:
        task = ExplanationTask.from_data(
            config_space, data,
            base_model=RandomForestRegressor(n_estimators=100, random_state=seed))
    except Exception as e:
        return {}, f"HyperSHAP (ablation) failed ({e})."
    return _ablate(HyperSHAP(task), config_space, config_of_interest, baseline)


def _ablate(hs, config_space, config_of_interest, baseline=None):
    """One ablation game against *baseline* (the config space's default when
    None), from a HyperSHAP that already exists. The half of
    `compute_hp_ablation` that does not depend on how the explainer was
    obtained."""
    from ConfigSpace import Configuration

    try:
        iv = hs.ablation(
            config_of_interest=Configuration(config_space, values=config_of_interest),
            baseline_config=_baseline_config(config_space, baseline))
        named = hs.get_interaction_values_with_names(iv)
        return {members[0]: value for members, value in named.items()
                if len(members) == 1}, None
    except Exception as e:
        return {}, f"HyperSHAP (ablation) failed ({e})."


def compute_local_effects(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    seed: int = 0,
    max_trials: int = 0,
    metric=None,
    baseline: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], List[dict], Optional[str]]:
    """Every sampled trial's local ablation, for the beeswarm figure.

    `compute_hp_ablation` answers "what did each hyperparameter's value do
    for *this* trial", against *baseline* (the config space's default when
    left out). This runs that question across the run, so the figure
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
    explainer = _build_explainer(config_space, trials, metric_name, seed,
                                      metric=metric)
    if explainer.hs is None:
        return [], [], (explainer.warning
                        or "Not enough trials for local explanations.")

    names = list(config_space.keys())
    rows = []
    # Positions, not just the trials: the sample skips trials, so a row's
    # place in this list says nothing about which trial it explains, and the
    # page's selection is a position in `trials`. See `_selection_meta`.
    for index, trial in _sample_evenly(list(enumerate(trials)), max_trials):
        effects, warning = compute_hp_ablation(
            config_space, trials, metric_name, trial.config, seed,
            explainer=explainer, baseline=baseline)
        if warning:
            return [], [], warning
        rows.append({"index": index, "trial": trial.trial, "effects": effects})

    if not rows:
        return [], [], "No trials to explain."
    return names, rows, None


def compute_partial_dependence(
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
        column = [v for row in ice_lines if (v := row[i]) is not None]
        pdp.append(sum(column) / len(column) if column else None)

    return grid, ice_lines, pdp, None


def compute_surrogate_uncertainty(
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


def sample_prediction_space(
    optimizer,
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    seed: int = 0,
    n: int = PREDICTION_SAMPLES,
    surrogate=None,
) -> dict:
    """What the surrogate predicts across the whole search space, for the
    configuration cube's prediction layer: one even sample of the space,
    predicted once, and placed in every view the cube has.

    **Even, and once.** A scrambled Sobol sequence over the space's unit
    cube, so the sample covers it evenly rather than in random clumps; each
    hyperparameter reads its coordinate on its own scale, so a log one is
    sampled logarithmically and a categorical takes each choice equally
    often. The model is this run's own where it can be rebuilt — see
    `slice_surrogate` — and it is asked once, for the mean and the spread
    both, since the view the reader picks changes where a configuration is
    drawn and not what was predicted for it.

    **Forward, never back.** A projected view places each sampled
    configuration through the same fitted PCA or PLS the trials are drawn
    with, rather than mapping points of the picture back to
    configurations: every point drawn is then a legal configuration, and a
    trial and a sampled configuration with the same values land in the same
    place. Many configurations project to one place, so the page aggregates
    what lands together. `outlines` is the edge of where anything legal can
    land, per projection and dimensionality — see
    `core.projection.Fitted.outline`.

    Returns a dict: `hp_names`, `values` (one row of hyperparameter values
    per sample, for the axes view, rounded), `configurations` (the same
    samples exactly), `vectors` (the same samples in ConfigSpace's
    vectorized form), `projections` (per method: the samples'
    `coordinates`, the components' `labels`, `outlines` keyed by
    dimensionality, and the projection itself as a linear `map`), `mu` and
    `sigma` per sample, `eta` (the best observed
    score, what expected improvement improves on), `higher_is_better`,
    `smooth` (whether the model is a smooth function, which a forest is
    not), and `warning`.

    *surrogate*, when given, is the model to ask — one the caller already
    fitted for this metric and these trials — instead of fitting another.
    """
    import numpy as np
    from ConfigSpace import Configuration
    from ConfigSpace.util import deactivate_inactive_hyperparameters
    from scipy.stats import qmc

    from .projection import METHODS, fit as fit_projection

    empty = {"hp_names": [], "values": [], "projections": {}, "mu": [],
             "sigma": [], "eta": None, "higher_is_better": True,
             "smooth": False, "warning": None}
    metric = metric_for(metric_name)
    scores = [t.scores[metric_name] for t in trials if metric_name in t.scores]
    if not scores:
        return dict(empty, warning=f"No trial carries a score for {metric_name}.")

    hps = list(config_space.values())
    names = [hp.name for hp in hps]
    unit = qmc.Sobol(d=len(hps), scramble=True, seed=seed).random(n)  # pyright: ignore[reportCallIssue]
    configurations, rows = [], []
    for point in unit:
        values = {hp.name: _from_unit(hp, u) for hp, u in zip(hps, point)}
        try:
            configuration = Configuration(config_space, values=values)
        except Exception:  # noqa: BLE001 — a child its parent switched off
            try:
                configuration = deactivate_inactive_hyperparameters(values, config_space)
            except Exception:  # noqa: BLE001 — forbidden: not part of the space
                continue
        configurations.append(dict(configuration))
        rows.append(configuration.get_array())
    if not configurations:
        return dict(empty, warning="No configuration the space allows was sampled.")

    model, warning = surrogate, None
    if model is None:
        model, warning = optimizer.slice_surrogate(config_space, trials, metric_name, seed)
    if model is None:
        return dict(empty, warning=warning)
    try:
        mu, sigma = predict_mean_std(model, np.array(rows))
    except Exception as e:  # noqa: BLE001 — a layer must not cost the figure
        return dict(empty, warning=f"The surrogate could not predict here ({e}).")

    # Five significant figures: the page bins these into a few dozen
    # cells, and the answer is twice the size at full precision.
    def number(value):
        return float(f"{value:.5g}") if np.isfinite(value) else None

    projections = {}
    if len(scores) == len(trials):
        for method in METHODS:
            fitted, _reason = fit_projection(config_space, trials, scores, method)
            if fitted is None:
                continue
            projections[method] = {
                "coordinates": [[number(v) for v in row]
                                for row in fitted.transform(configurations)],
                "map": fitted.linear_map(),
                "labels": fitted.labels,
                "outlines": {str(k): fitted.outline(k)
                             for k in range(1, len(fitted.labels) + 1)},
            }

    return {
        "hp_names": names,
        "values": [[number(v) if isinstance(v, float) else v
                    for v in (c.get(name) for name in names)]
                   for c in configurations],
        # Where each sample sits on every hyperparameter's own vectorized
        # axis, which is the axis a prior's density is stated on: what the
        # page weights the acquisition by, per sample, as a prior changes.
        "vectors": [[number(v) for v in row] for row in rows],
        # Every sample exactly, where `values` is rounded for the page: what
        # a sample is run as, when one is picked to become a trial.
        "configurations": configurations,
        "projections": projections,
        "mu": [number(v) for v in mu],
        "sigma": [number(v) for v in sigma],
        "eta": float(max(scores) if metric.higher_is_better else min(scores)),
        "higher_is_better": metric.higher_is_better,
        "smooth": bool(getattr(model, "smooth", False)),
        "warning": warning,
    }


def slice_axis(config_space, hp_name: str, n_points: int = 101):
    """The axis one hyperparameter is drawn on, without fitting anything.

    `compute_incumbent_slice` returns this too, but only after a surrogate
    exists — it is a model of the objective, and getting one costs a fit.
    A prior needs none of that. It is a statement about the search space,
    so the space is all it takes to draw one, and making the reader wait
    for a model before they can say where they think the optimum is gets
    the dependency backwards.

    Returns `(grid, positions, kind)`: the values along the hyperparameter's
    own scale, their coordinates in ConfigSpace's normalized representation
    (which is the space a prior is evaluated in), and whether it is
    categorical.
    """
    import numpy as np

    if hp_name not in config_space:
        return [], [], "continuous"
    hp = config_space[hp_name]
    grid = _hp_grid(hp, n_points)
    if len(grid) < 2:
        return [], [], "continuous"

    positions = np.asarray(hp.to_vector(np.asarray(grid)), dtype=float).ravel()
    kind = "categorical" if hasattr(hp, "choices") else "continuous"
    return list(grid), [float(p) for p in positions], kind


def compute_incumbent_slice(
    optimizer,
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    hp_name: str,
    seed: int = 0,
    n_points: int = 101,
    values=None,
    surrogate=None,
    cloud=None,
) -> "IncumbentSlice":
    """The surrogate along one hyperparameter, every other one held at the
    incumbent — the 1-D restriction of `compute_surrogate_uncertainty`,
    sharing its surrogate, its grid and its choice of where to slice.

    Returns an `IncumbentSlice`.

    **`positions` is the point of this function**, and the reason it does
    not simply hand back `linspace(0, 1)`. A prior over a hyperparameter is
    a density on ConfigSpace's *normalized* representation — the space
    `get_array` produces and the one SMAC's acquisition weights are
    evaluated in — so anything drawing or editing such a density needs each
    grid value's normalized coordinate, not its native one. The two are not
    the same axis: `lr` spanning 1e-4 to 1e-1 logarithmically has grid
    values 1e-4, 5.6e-4, 3.2e-3, 1.8e-2, 1e-1 sitting at a perfectly even
    0, .25, .5, .75, 1. A prior drawn against the native values would put
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

    *cloud* is `(X, mu, sigma)` from `sample_prediction_space` — predictions
    for configurations drawn from the *whole* space. Handed in for the same
    reason *surrogate* is: it does not depend on which hyperparameter is on
    the axis, so sampling it per slice would be waste.

    *surrogate* is an already-fitted model from `slice_surrogate`, for a
    caller slicing several hyperparameters of one run — the fit does not
    depend on which one, so refitting per hyperparameter is pure waste. Left
    out, one is fitted here.

    `incumbent` is `(position, value)` for the configuration the slice passes
    through: the only measured point on it. Under *values* that value is the
    incumbent's own modelled quantity rather than `eta`, which by then is
    the best observed value of a different question.
    """
    import numpy as np
    from ConfigSpace import Configuration

    if hp_name not in config_space:
        return IncumbentSlice([], [], [], [], None, None, None, f"No such hyperparameter: {hp_name}.")

    # The run's own model class where the optimizer can rebuild one, so a
    # Gaussian-process run is read through a Gaussian process rather than
    # through a stand-in forest that never saw the search. The object itself
    # does not survive `optimize()`, but its class and settings do, and the
    # trials it was fitted on are recorded — so it is refitted, not
    # recovered. Anything that cannot (random search, grid search, a refit
    # that raises) keeps the forest.
    rf, warning = surrogate, None
    if rf is None:
        rf, warning = optimizer.slice_surrogate(config_space, trials, metric_name, seed,
                                           values=values)
    if rf is None:
        return IncumbentSlice([], [], [], [], None, None, None, warning)

    # The same incumbent `compute_surrogate_uncertainty` slices through, and
    # for the same reason: a line through a configuration that was actually
    # evaluated, rather than through a config-space default the search may
    # never have visited. Only trials carrying the score count — a
    # `.get(..., 0.0)` default would hand the slice to an unmeasured trial,
    # and under a lower-is-better metric a missing 0.0 would win outright.
    metric = metric_for(metric_name)
    scored = [t for t in trials if metric_name in t.scores]
    if not scored:
        return IncumbentSlice([], [], [], [], None, None, None, f"No trial carries a score for {metric_name}.")
    picker = max if metric.higher_is_better else min
    best = picker(scored, key=lambda t: t.scores[metric_name])

    hp = config_space[hp_name]
    grid = _hp_grid(hp, n_points)
    if len(grid) < 2:
        return IncumbentSlice([], [], [], [], None, None, None, "A slice needs the hyperparameter to take more than one value.")

    # Built together so a grid value the config space rejects drops out of
    # every one of them at once — three lists the browser indexes in
    # lockstep must not be able to fall out of step here.
    kept_values, kept_positions, rows = [], [], []
    cloud_rows = None
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
        return IncumbentSlice([], [], [], [], None, None, None, f"No valid configurations along {hp_name}.")

    slice_rows = np.array(rows)
    mu, sigma = predict_mean_std(rf, slice_rows)

    observed = [v for _, v in pair_trials_with_scores(
        config_space, trials, metric_name, values=values)]
    eta = None
    if observed:
        eta = float(picker(observed) if values is None else min(observed))

    # The one point on this line that was measured. The slice holds every
    # other hyperparameter at the incumbent's value, so it passes exactly
    # through that configuration and nothing else on the grid is an
    # observation — which is why marking the other trials would assert they
    # lie on a line they do not.
    #
    # Its position is where the incumbent's own value falls on the
    # normalized axis. That need not be a grid point and must not be snapped
    # to one: the grid is evenly spaced in normalized space and the
    # incumbent is wherever the search put it.
    #
    # The value is read from `best`, not from `eta`. With *values* the
    # incumbent is still chosen by *metric_name* while `eta` is the best
    # observed value of whatever was modelled, so the two part company —
    # the marker belongs to the configuration the slice passes through.
    incumbent = None
    try:
        position = float(np.asarray(
            hp.to_vector(np.asarray([best.config[hp_name]]))).ravel()[0])
        measured = values(best) if values is not None else best.scores.get(metric_name)
        if measured is not None:
            incumbent = (position, float(measured))
    except Exception:  # noqa: BLE001 — a marker must not cost anyone the slice
        incumbent = None

    # The cloud projected onto this hyperparameter's axis. Only the column
    # changes per slice, which is why the sampling and the predictions are
    # done once by the caller and handed in — they do not depend on which
    # hyperparameter is on the axis.
    projected = None
    if cloud is not None:
        try:
            column = list(config_space.keys()).index(hp_name)
            sampled, cloud_mu, cloud_sigma = cloud
            # A configuration where this hyperparameter is inactive has no
            # place on its axis, and one the model could not predict has
            # nothing to say there; both are left out.
            kept = [i for i in range(len(sampled))
                    if np.isfinite(sampled[i, column]) and np.isfinite(cloud_mu[i])
                    and np.isfinite(cloud_sigma[i])]
            projected = ([float(sampled[i, column]) for i in kept],
                         [float(cloud_mu[i]) for i in kept],
                         [float(cloud_sigma[i]) for i in kept])
            cloud_rows = sampled[kept]
        except Exception:  # noqa: BLE001 — a shadow must not cost the slice
            projected = None

    return IncumbentSlice(kept_values, kept_positions,
                          [float(v) for v in mu], [float(v) for v in sigma],
                          eta, incumbent, projected, None,
                          slice_rows=slice_rows, cloud_rows=cloud_rows)


def _skipped_games(metrics, reason: str) -> Dict[str, tuple]:
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
            for game in HP_GAMES}


def compute_hp_games(
    config_space,
    trials: List[TrialResult],
    metrics,
    seed: int = 0,
    cancel_event=None,
    baseline: Optional[Dict[str, Any]] = None,
    budget: Optional[int] = EAGER_MAX_COALITIONS,
    wanted: bool = True,
) -> Dict[str, _GameResults]:
    """Every metric's per-game hyperparameter importance, for all of
    `HP_GAMES` — what each optimizer's `optimize()` calls once, instead of
    calling `compute_hp_game` per game and metric.

    Every game is a gain over *baseline*, a configuration as a dict, or over
    the config space's default when it is left out. What `optimize()`
    stores is always the latter; the page asks for the others on request.

    Returns `{game: (importance_by_metric, warning_by_metric,
    interactions_by_metric, moebius_by_metric, total_by_metric)}`, the
    first pair shaped exactly like an
    `OptimizationResult`'s `hyperparameter_<game>`/`hyperparameter_<game>_warning`
    fields expect. `interactions_by_metric` and `moebius_by_metric` are a
    free byproduct of the same call (see `compute_hp_game`) for
    every game — which is what lets the page's one game selector drive the
    interaction figures as well as the importance figure: all three games'
    grids were always computed here, and for a long time two of them were
    thrown away for want of a field to put them in.

    Builds the HyperSHAP explainer once per metric, not once per (game,
    metric) pair: `tunability`/`sensitivity`/`mistunability` all explain
    the *same* trial history for a given metric, so refitting the
    surrogate underneath them 3 times over is pure waste. See
    `_build_explainer`/`compute_hp_game`'s `explainer` parameter.

    Declines in three cases, all reported through `_skipped_games`:

    - **The run was cancelled.** Pressing Cancel used to still buy you the
      full analytics bill — every optimizer breaks out of its trial loop on
      `cancel_event` and then called this unconditionally, so on a wide model
      you waited minutes for a run you had just stopped. Nothing is lost
      permanently: resuming recomputes over `previous + new` trials.
    - **Nobody is going to look.** Both figures that display these numbers
      are switched off for this experiment, so the fields would be filled
      and never read. See `BaseOptimizer.analytics_wanted`.
    - **The run is too wide to afford.** See
      `eager_analytics_budget_exceeded`.

    Checked in that order: a cancelled run shouldn't be told about a budget
    it never got to spend, and neither should one nobody asked for.
    """
    if cancel_event is not None and cancel_event.is_set():
        return _skipped_games(
            metrics,
            "Importance analytics were skipped because the run was "
            "cancelled. Resuming the run computes them over every trial.")

    if not wanted:
        return _skipped_games(
            metrics,
            "Importance analytics were skipped: both figures that show "
            "them are switched off for this experiment. Switch one on and "
            "run again to compute them.")

    too_wide = eager_analytics_budget_exceeded(config_space, metrics, budget)
    if too_wide:
        return _skipped_games(metrics, too_wide)

    by_game: Dict[str, _GameResults] = {game: ({}, {}, {}, {}, {}) for game in HP_GAMES}
    # *metrics* is a mapping of name to `Metric` from the application and a
    # bare list of names from plenty of callers that only need the names.
    # Both iterate the same way, but only a mapping can say which direction
    # a metric runs — from a list the registry answers, which is right for
    # every metric this build ships and wrong only for one a file declared.
    # A caller holding declared metrics has the mapping.
    directions = metrics if isinstance(metrics, Mapping) else {}
    for metric_name in metrics:
        explainer = _build_explainer(config_space, trials, metric_name, seed,
                                          metric=directions.get(metric_name))
        for game in HP_GAMES:
            importance, warning, interactions, moebius, total = compute_hp_game(
                config_space, trials, metric_name, game, seed, explainer=explainer,
                baseline=baseline)
            by_game[game][0][metric_name] = importance
            by_game[game][1][metric_name] = warning
            by_game[game][2][metric_name] = interactions
            by_game[game][3][metric_name] = moebius
            by_game[game][4][metric_name] = total
    return by_game


def _build_explainer(config_space, trials: List[TrialResult], metric_name: str,
                     seed: int = 0, metric=None):
    """Pair *trials* with *metric_name* scores and fit the HyperSHAP
    explainer every global game (`tunability`/`sensitivity`/
    `mistunability`) shares for a given metric — the one expensive step
    (constructing an `ExplanationTask` fits a surrogate internally)
    that's identical across all three; only each game's own aggregation
    (MAX/VAR/MIN) differs once it's built. `compute_hp_games` builds this
    once per metric and reuses it across `HP_GAMES`; `compute_hp_game`
    builds its own when called standalone and no `explainer` was handed to it.

    Returns a `_MetricExplainer` — pass straight through as
    `compute_hp_game`'s `explainer`. Its `insufficient` is True only for
    "fewer than two usable trials," a condition `compute_hp_game`
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


def compute_hp_game(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    game: str,
    seed: int = 0,
    explainer=None,
    baseline: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, float], Optional[str], Dict[str, Dict[str, float]], list, float]:
    """One explanation game for one metric: ask HyperSHAP's *game* for order-1
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
    per game. `None` (the default) builds its own. The fallback surrogate is shared through
    the same object, so a metric whose HyperSHAP call fails fits one forest
    rather than one per game.

    *baseline* is the configuration the game's gains are measured from, the
    config space's default when None — see `compute_hp_games`.

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
    or `[]` on either fallback rung.

    Each game is played once, through HyperSHAP's own facade, and asked for
    its Möbius decomposition: the one transform that assigns a value to
    every coalition and does not change with the order it is truncated at,
    so it is stored once and read at any order by the graph, upset and
    by-order views. The FSII order-2 values behind the importance and the
    pairwise grid are derived from it by shapiq's `MoebiusConverter` rather
    than played for again, since a second facade call would evaluate all
    2^n_hp coalitions a second time. Order 2 and not higher because FSII's
    terms are fitted jointly — the order-1 terms of an order-3 fit are not
    those of an order-2 fit — so raising it would change what every stored
    importance means.
    """
    names = list(config_space.keys())

    if explainer is None:
        explainer = _build_explainer(config_space, trials, metric_name, seed)
    build_warning = explainer.warning

    if explainer.insufficient:
        uniform = 1.0 / len(names) if names else 0.0
        return {p: uniform for p in names}, build_warning, {}, [], 0.0

    warning = None
    if explainer.hs is not None:
        hs = explainer.hs
        try:
            moebius_iv = getattr(hs, game)(
                baseline_config=_baseline_config(config_space, baseline),
                index="Moebius", order=len(names), seed=seed)
            fsii = hs.get_interaction_values_with_names(
                MoebiusConverter(moebius_iv)(index="FSII", order=2))
            moebius = _moebius_terms(hs.get_interaction_values_with_names(moebius_iv))
            raw = {p: abs(fsii.get((p,), 0.0)) for p in names}
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
            interactions = _extract_pairwise(fsii, names)
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
        return ({p: float(importances[i] / total) for i, p in enumerate(names)},
                warning, {}, [], 0.0)
    warning += f" Surrogate fallback also failed ({fit_warning}); showing uniform weights."
    uniform = 1.0 / len(names) if names else 0.0
    return {p: uniform for p in names}, warning, {}, [], 0.0


def _extract_pairwise(named: Dict[tuple, float], names: List[str]) -> Dict[str, Dict[str, float]]:
    """Reshape order-1 and order-2 terms, as
    `HyperSHAP.get_interaction_values_with_names` keys them, into one
    square hyperparameter × hyperparameter grid — `{hp_a: {hp_b: value}}`,
    symmetric, diagonal = that hyperparameter's own signed order-1 value.
    Raw values, not `abs()`-ed or normalized like `compute_hp_game`'s
    `importance` return: a heatmap comparing interaction strength needs
    sign and a shared scale between the diagonal and off-diagonal, not a
    share-of-100%.
    """
    grid = {p: {p: float(named.get((p,), 0.0))} for p in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            grid[a][b] = grid[b][a] = float(named.get((a, b), 0.0))
    return grid


#: Below this share of the achievable gain, "what is left" is rounding rather
#: than headroom, and a pie of it would be a picture of noise.
SETTLED = 0.02


def tuning_progress(result, metric, ablation):
    """What each hyperparameter was worth, and how much of it has been banked.

    Tunability decomposes (best achievable − default); the ablation decomposes
    (this trial − default). Both are FSII order-1 terms of an order-2 fit,
    against the same baseline, and both games are zeroed at that baseline —
    measured, `v(none) = 0` for each — so the two are in the same units from the
    same origin and the subtraction is well posed.

    **Two things it is not**, both measured rather than assumed, and both
    reasons to read the ordering rather than the digits:

    - *Achievable is a lower bound.* The tunability game's maximum comes from
      10,000 random draws of the configuration space, and a real optimizer beats
      random search — that is what it is for. So a well-tuned hyperparameter can
      show more banked than achievable, and it is not an error: it means the
      search found something the estimate did not. `beyond` marks those, and
      they are the strongest evidence an axis is finished, not a contradiction.
    - *Order-1 is not the whole game.* Measured on a four-hyperparameter run, the
      order-1 terms account for about 70-100% of each game's own endpoint, the
      rest living in order-2 interactions that neither number carries. Both
      sides are deflated by the same mechanism, so the comparison survives it
      better than either figure alone does.

    Both halves are needed in the metric's own units, and only one of them is
    stored that way: `hyperparameter_importance` is a share of 100%, and
    `hyperparameter_tunability_total` is what it is a share *of*. A result
    written before that field existed has no scale to multiply by, and gets
    nothing rather than a ratio computed against the wrong denominator.

    Returns `(rows, settled)`. Each row is one hyperparameter's achievable,
    banked and remaining value; *settled* is True when the remaining total is
    rounding, which is a finding — this trial has taken essentially everything
    there was — and not a figure.
    """
    shares = result.hyperparameter_importance.get(metric, {})
    total = result.hyperparameter_tunability_total.get(metric, 0.0)
    if not shares or not total or not ablation:
        return [], False

    rows = []
    for name, share in shares.items():
        achievable = share * total
        # Signed on purpose. A negative one says this trial's value for that
        # hyperparameter is *worse* than the default — drift picked up while
        # chasing whichever one mattered — and flooring it here would hide the
        # one finding that is actionable on its own.
        banked = ablation.get(name, 0.0)
        rows.append({
            "name": name,
            "achievable": achievable,
            "banked": banked,
            # Floored, because a negative amount left is not an amount left.
            "remaining": max(0.0, achievable - banked),
            "banked_share": (banked / achievable) if achievable else 0.0,
            # Past the estimated ceiling: the search found something 10,000
            # random draws did not. Marked rather than quietly floored, because
            # it says something — this axis is not merely finished, it is
            # finished past where the estimate could see.
            "beyond": banked > achievable > 0,
        })
    rows.sort(key=lambda row: row["remaining"], reverse=True)
    left = sum(row["remaining"] for row in rows)
    return rows, left <= SETTLED * (total or 1.0)
