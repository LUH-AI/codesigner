"""Questions about an experiment's results answered from its stored trials, not
read off its pages: one trial, the trials ranked or filtered, a few side by
side, and a fixed set of analyses — the scores, how the search progressed, the
initial design, how much of the space was explored, how each hyperparameter
went with the score, importance, the metrics against each other, durations
and failures.

Everything here works on `Trials`, built from the experiment's rebuilt result
(experiments/services/snapshot.py); nothing touches the database. The tools
that ask these questions are in assistant/tools/experiment.py. Numbers are
rounded to four significant figures: enough to compare, short enough to read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from core.optimizers.base import HP_GAME_FIELDS, REQUESTED_ORIGIN
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS, STATUS_TIMEOUT
from core.priors import initial_design_notice


class ResultsError(ValueError):
    """A question the trials cannot answer: a metric, trial or field there is
    not. Its message is said to the model."""


#: How a trial's configuration arose, by phase.
DESIGN, GUIDED, RANDOM, REQUESTED, SEARCH, UNKNOWN = (
    "initial_design", "model_guided", "random_interleaved", "requested", "search", "unknown")

STATUS = {STATUS_SUCCESS: "ok", STATUS_CRASHED: "crashed", STATUS_TIMEOUT: "timeout"}

#: The orders `rows` lists trials in.
ORDERS = ("best", "worst", "first", "last", "slowest", "failed")

#: The comparisons a `rows` condition may make.
OPERATORS = ("=", "!=", "<", "<=", ">", ">=", "in")

#: The analyses `analyze` runs, each with what it answers.
ANALYSES = {
    "scores": "the metric's spread over all trials, by phase and by run",
    "progress": "when the best score improved, and how much the last trials added",
    "initial_design": "the initial design: its size and settings, how well it covered the "
                      "space, and what the model-guided trials added after it",
    "coverage": "how much of each hyperparameter's range was tried, and whether the best "
                "trials sit at an edge of it",
    "effects": "how each hyperparameter's values went with the score across the trials",
    "importance": "the stored HyperSHAP importance and interactions",
    "metrics": "how the metrics, and the trials' durations, go with one another",
    "durations": "how long trials took, by phase and run, and what made them slow",
    "failures": "which trials failed, why, and where in the space",
}


def _r(value):
    """*value* to four significant figures; None for missing or not finite."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(value):
        return None
    return float(f"{value:.4g}")


def _plain(config: dict) -> dict:
    return {k: _r(v) if isinstance(v, (int, float, np.number)) else v for k, v in config.items()}


def _spearman(xs, ys):
    """Spearman's ρ between *xs* and *ys*, or None with fewer than five pairs or
    nothing varying."""
    from scipy.stats import spearmanr

    if len(xs) < 5 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rho = spearmanr(xs, ys).statistic
    return _r(rho)


@dataclass
class Trials:
    """An experiment's trials and what reading them needs: its result, its
    search space (None when the model is gone), the optimizer's current name
    (never a former alias) and settings, and its runs as `{index, first, last, status, stopped_by,
    stopping}` with 1-based inclusive trial numbers."""

    result: object
    space: object = None
    optimizer: str = ""
    settings: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)

    @property
    def trials(self):
        return self.result.trials

    @cached_property
    def design(self) -> dict | None:
        """The initial design's planned size, or None without one."""
        return initial_design_notice(self.result)

    def metric(self, name: str | None) -> str:
        """*name* if the trials were scored on it, else the primary metric."""
        name = name or self.result.primary_metric
        known = {m for t in self.trials for m in t.scores}
        if self.trials and name not in known:
            raise ResultsError(f"the trials have no metric {name!r}; they have {sorted(known)}")
        return name

    def better(self, metric: str):
        return self.result.metric(metric).better

    def higher_is_better(self, metric: str) -> bool:
        return bool(self.result.metric(metric).higher_is_better)

    def phase(self, index: int) -> str:
        """How trial *index* (0-based) arose."""
        origin = self.trials[index].origin or ""
        if origin == REQUESTED_ORIGIN:
            return REQUESTED
        if origin.startswith("Initial Design"):
            return DESIGN
        if origin.startswith("Acquisition Function Maximizer"):
            return GUIDED
        if self.optimizer != "SMAC":
            return SEARCH
        if origin.startswith("Random Search"):
            return RANDOM
        if not origin and self.design:
            return DESIGN if index < self.design["size"] else GUIDED
        return UNKNOWN

    def run_of(self, number: int) -> int | None:
        """The run (1-based) trial *number* belongs to."""
        for run in self.runs:
            if run.get("first") and run["first"] <= number <= run["last"]:
                return run["index"]
        return None

    def index_of(self, number: int) -> int:
        for index, trial in enumerate(self.trials):
            if trial.trial == number:
                return index
        raise ResultsError(f"there is no trial {number}; trials run from "
                           f"{self.trials[0].trial} to {self.trials[-1].trial}"
                           if self.trials else "the experiment has no trials yet")

    def score(self, index: int, metric: str):
        """Trial *index*'s score on *metric*, or None when it failed or has none."""
        trial = self.trials[index]
        value = trial.scores.get(metric)
        if trial.failed or value is None or not math.isfinite(value):
            return None
        return float(value)

    def scored(self, metric: str, indices=None) -> list[tuple[int, float]]:
        """`(index, score)` of the trials with a score on *metric*."""
        indices = range(len(self.trials)) if indices is None else indices
        return [(i, s) for i in indices if (s := self.score(i, metric)) is not None]

    def ranked(self, metric: str) -> list[int]:
        """The scored trials' indices, best first."""
        pairs = self.scored(metric)
        pairs.sort(key=lambda p: p[1], reverse=self.higher_is_better(metric))
        return [i for i, _ in pairs]

    def hyperparameters(self) -> list:
        """The search space's hyperparameters, or those the trials set."""
        if self.space is not None:
            return list(self.space.values())
        return []

    def hp_names(self) -> list[str]:
        if self.space is not None:
            return list(self.space.keys())
        return sorted({k for t in self.trials for k in t.config})

    def row(self, index: int, metrics=()) -> dict:
        """Trial *index* in brief."""
        trial = self.trials[index]
        out = {"trial": trial.trial, "phase": self.phase(index),
               "status": STATUS.get(trial.run_info.get("status", STATUS_SUCCESS), "failed"),
               "scores": {m: _r(self.score(index, m)) for m in metrics},
               "config": _plain(trial.config), "seconds": _r(trial.duration)}
        run = self.run_of(trial.trial)
        if run is not None:
            out["run"] = run
        return out


def _stats(values: list[float], higher_is_better: bool) -> dict:
    if not values:
        return {"n": 0}
    xs = np.asarray(values, dtype=float)
    return {"n": len(xs), "best": _r(xs.max() if higher_is_better else xs.min()),
            "worst": _r(xs.min() if higher_is_better else xs.max()),
            "mean": _r(xs.mean()), "median": _r(np.median(xs)), "std": _r(xs.std()),
            "q25": _r(np.quantile(xs, 0.25)), "q75": _r(np.quantile(xs, 0.75))}


def _gain(t: Trials, metric: str, before, after):
    """How much better *after* is than *before*, positive when it improved."""
    if before is None or after is None:
        return None
    return _r(after - before if t.higher_is_better(metric) else before - after)


# ── single trials and lists of them ─────────────────────────────────────────

def trial(t: Trials, number: int, metric: str | None = None) -> dict:
    """Everything stored about trial *number*."""
    metric = t.metric(metric)
    index = t.index_of(number)
    found = t.trials[index]
    out = t.row(index)
    out["scores"] = {m: _r(t.score(index, m)) for m in found.scores}
    out["origin"] = found.origin or None
    ranked = t.ranked(metric)
    if index in ranked:
        out["rank"] = {"metric": metric, "place": ranked.index(index) + 1, "of": len(ranked)}
        best = t.score(ranked[0], metric)
        out["behind_best"] = _gain(t, metric, t.score(index, metric), best)
        earlier = [s for _, s in t.scored(metric, range(index))]
        out["new_best_when_run"] = bool(not earlier or all(
            t.better(metric)(t.score(index, metric), s) for s in earlier))
    if found.failed:
        out["failure"] = found.failure or None
        if found.traceback:
            out["traceback_tail"] = found.traceback[-1500:]
    return out


def _matches(t: Trials, index: int, condition: dict) -> bool:
    name, op, value = condition["field"], condition["op"], condition.get("value")
    row = t.trials[index]
    if name in row.config:
        actual = row.config[name]
    elif name in row.scores:
        actual = t.score(index, name)
    elif name == "phase":
        actual = t.phase(index)
    elif name == "status":
        actual = STATUS.get(row.run_info.get("status", STATUS_SUCCESS), "failed")
    elif name == "run":
        actual = t.run_of(row.trial)
    elif name == "seconds":
        actual = row.duration
    elif name == "trial":
        actual = row.trial
    else:
        return False
    if op == "in":
        return actual in (value if isinstance(value, list) else [value])
    if op in ("=", "!="):
        same = actual == value or str(actual) == str(value)
        return same if op == "=" else not same
    if actual is None:
        return False
    try:
        a, b = float(actual), float(value)
    except (TypeError, ValueError):
        a, b = str(actual), str(value)
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _check_conditions(t: Trials, where):
    known = set(t.hp_names()) | {m for x in t.trials for m in x.scores} | {
        "phase", "status", "run", "seconds", "trial"}
    for condition in where:
        if condition.get("field") not in known:
            raise ResultsError(f"{condition.get('field')!r} is not a field of the trials; they "
                               f"have {sorted(known)}")
        if condition.get("op") not in OPERATORS:
            raise ResultsError(f"op must be one of {list(OPERATORS)}")


def rows(t: Trials, order: str = "best", metric: str | None = None, limit: int = 10,
         where: list | None = None) -> dict:
    """The trials in *order*, those matching every condition in *where*."""
    metric = t.metric(metric)
    where = where or []
    _check_conditions(t, where)
    chosen = [i for i in range(len(t.trials)) if all(_matches(t, i, c) for c in where)]
    if order in ("best", "worst"):
        ranked = [i for i in t.ranked(metric) if i in set(chosen)]
        chosen = ranked if order == "best" else ranked[::-1]
    elif order == "last":
        chosen = chosen[::-1]
    elif order == "slowest":
        chosen = sorted(chosen, key=lambda i: t.trials[i].duration, reverse=True)
    elif order == "failed":
        chosen = [i for i in chosen if t.trials[i].failed]
    limit = max(1, min(int(limit), 25))
    metrics = [metric] + [m for m in ([] if metric == t.result.primary_metric
                                      else [t.result.primary_metric])]
    out = {"metric": metric, "higher_is_better": t.higher_is_better(metric),
           "matching": len(chosen), "of": len(t.trials),
           "trials": [t.row(i, metrics) for i in chosen[:limit]]}
    for listed, i in zip(out["trials"], chosen[:limit]):
        if t.trials[i].failed:
            listed["failure"] = t.trials[i].failure or None
    return out


def compare(t: Trials, numbers: list[int]) -> dict:
    """Trials *numbers* side by side: what they set differently and how they
    scored."""
    if not 2 <= len(numbers) <= 5:
        raise ResultsError("compare two to five trials")
    indices = [t.index_of(int(n)) for n in numbers]
    metrics = sorted({m for i in indices for m in t.trials[i].scores})
    names = t.hp_names()
    differs = [n for n in names if len({str(t.trials[i].config.get(n)) for i in indices}) > 1]
    return {
        "trials": [t.row(i, metrics) for i in indices],
        "hyperparameters_that_differ": differs,
        "hyperparameters_the_same": [n for n in names if n not in differs],
        "higher_is_better": {m: t.higher_is_better(m) for m in metrics},
    }


# ── the analyses ─────────────────────────────────────────────────────────────

def analyze(t: Trials, analysis: str, metric: str | None = None,
            hyperparameter: str | None = None) -> dict:
    """Run *analysis*, one of `ANALYSES`."""
    if analysis not in ANALYSES:
        raise ResultsError(f"analysis must be one of {list(ANALYSES)}")
    if not t.trials:
        return {"analysis": analysis, "trials": 0, "note": "the experiment has no trials yet"}
    metric = t.metric(metric)
    if hyperparameter and hyperparameter not in t.hp_names():
        raise ResultsError(f"there is no hyperparameter {hyperparameter!r}; there are "
                           f"{t.hp_names()}")
    found = _ANALYSES[analysis](t, metric, hyperparameter)
    return {"analysis": analysis, "metric": metric,
            "higher_is_better": t.higher_is_better(metric), **found}


def _scores(t: Trials, metric, _hp) -> dict:
    hib = t.higher_is_better(metric)
    best = t.ranked(metric)
    by_phase = {}
    for phase in (DESIGN, GUIDED, RANDOM, REQUESTED, SEARCH, UNKNOWN):
        indices = [i for i in range(len(t.trials)) if t.phase(i) == phase]
        if indices:
            by_phase[phase] = _stats([s for _, s in t.scored(metric, indices)], hib)
    by_run = {}
    for run in t.runs:
        indices = [i for i, x in enumerate(t.trials) if t.run_of(x.trial) == run["index"]]
        if indices:
            by_run[run["index"]] = _stats([s for _, s in t.scored(metric, indices)], hib)
    return {"overall": _stats([s for _, s in t.scored(metric)], hib),
            "failed": sum(x.failed for x in t.trials),
            "best_trial": t.trials[best[0]].trial if best else None,
            "by_phase": by_phase, "by_run": by_run}


def _improvements(t: Trials, metric) -> list[dict]:
    out, best = [], None
    for i, score in t.scored(metric):
        if best is None or t.better(metric)(score, best):
            out.append({"trial": t.trials[i].trial, "score": _r(score), "phase": t.phase(i),
                        "gain": _gain(t, metric, best, score)})
            best = score
    return out


def _best_by(t: Trials, metric, count: int):
    """The best score among the first *count* trials."""
    pairs = t.scored(metric, range(min(count, len(t.trials))))
    if not pairs:
        return None
    values = [s for _, s in pairs]
    return max(values) if t.higher_is_better(metric) else min(values)


def _progress(t: Trials, metric, _hp) -> dict:
    n = len(t.trials)
    steps = _improvements(t, metric)
    last = steps[-1]["trial"] if steps else None
    marks = sorted({max(1, round(n * q)) for q in (0.25, 0.5, 0.75, 1.0)})
    at = {f"after_{m}_trials": _r(_best_by(t, metric, m)) for m in marks}
    quarter = max(1, n // 4)
    final = _best_by(t, metric, n)
    out = {
        "trials": n, "improvements": steps, "best": _r(final),
        "best_so_far": at,
        "trials_since_last_improvement": (t.trials[-1].trial - last) if last else None,
        "gain_in_last_quarter": _gain(t, metric, _best_by(t, metric, n - quarter), final),
        "gain_over_whole_search": _gain(t, metric, _best_by(t, metric, 1), final),
        "runs": [{k: run.get(k) for k in ("index", "first", "last", "status", "stopped_by")}
                 for run in t.runs],
    }
    halves = (t.scored(metric, range(n // 2)), t.scored(metric, range(n // 2, n)))
    if all(halves):
        out["median_first_half"] = _r(np.median([s for _, s in halves[0]]))
        out["median_second_half"] = _r(np.median([s for _, s in halves[1]]))
    return out


def _unit(hp, values) -> list[float] | None:
    """*values* of *hp* on the unit interval as the optimizer sees it (a log
    scale for a log hyperparameter); None for a category."""
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter

    if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
        return None
    out = []
    for value in values:
        try:
            out.append(float(hp.to_vector(value)))
        except (TypeError, ValueError):
            continue
    return out


def _spread(units: list[float]) -> dict:
    """How evenly points cover [0, 1]: the share of ten equal bins with a
    point, and the widest stretch without one."""
    if not units:
        return {"bins_hit_of_10": 0, "widest_gap": 1.0}
    edges = sorted([0.0] + list(units) + [1.0])
    return {"bins_hit_of_10": len({min(int(u * 10), 9) for u in units}),
            "widest_gap": _r(max(b - a for a, b in zip(edges, edges[1:])))}


def _initial_design(t: Trials, metric, _hp) -> dict:
    if t.optimizer != "SMAC":
        return {"applies": False,
                "why": f"{t.optimizer or 'This optimizer'} chooses every trial the same way; "
                       "there is no initial design and no model-guided phase."}
    design = [i for i in range(len(t.trials)) if t.phase(i) == DESIGN]
    after = [i for i in range(len(t.trials)) if t.phase(i) in (GUIDED, RANDOM)]
    recorded = (t.result.metadata or {}).get("initial_design") or {}
    keys = ("initial_design", "use_trial_cap", "trial_cap", "use_share_cap", "share_cap",
            "initial_points_use_max", "use_default_config")
    out = {"applies": True,
           "planned_size": t.design["size"] if t.design else None,
           "kind": recorded.get("name") or t.settings.get("initial_design"),
           "per_hyperparameter": recorded.get("n_configs_per_hyperparameter"),
           "settings": {k: t.settings.get(k) for k in keys if k in t.settings},
           "design_trials": len(design), "trials_after_design": len(after),
           "hyperparameters": len(t.hp_names())}
    best_design = [s for _, s in t.scored(metric, design)]
    best_after = [s for _, s in t.scored(metric, after)]
    pick = max if t.higher_is_better(metric) else min
    if best_design:
        out["best_in_design"] = _r(pick(best_design))
    if best_after:
        out["best_after_design"] = _r(pick(best_after))
        out["gain_after_design"] = _gain(t, metric, pick(best_design), pick(best_after)) \
            if best_design else None
        out["trials_after_design_beating_its_best"] = sum(
            t.better(metric)(s, pick(best_design)) for s in best_after) if best_design else None
    ranked = t.ranked(metric)
    if ranked:
        out["best_trial"] = {"trial": t.trials[ranked[0]].trial, "phase": t.phase(ranked[0])}
    if not after:
        out["note"] = ("No trial ran after the design: the design used the whole budget, so "
                       "the model never chose a configuration.")
    coverage = {}
    for hp in t.hyperparameters():
        units = _unit(hp, [t.trials[i].config.get(hp.name) for i in design
                           if hp.name in t.trials[i].config])
        if units is None:
            values = [t.trials[i].config.get(hp.name) for i in design]
            coverage[hp.name] = {"values_tried": sorted({str(v) for v in values if v is not None}),
                                 "choices": [str(c) for c in hp.choices]}
        else:
            coverage[hp.name] = _spread(units)
    if coverage:
        out["design_coverage"] = coverage
    return out


def _bounds(hp) -> dict:
    lower, upper = getattr(hp, "lower", None), getattr(hp, "upper", None)
    return {"lower": _r(lower), "upper": _r(upper), "log": bool(getattr(hp, "log", False))}


def _coverage(t: Trials, metric, only) -> dict:
    ranked = t.ranked(metric)
    top = ranked[:max(1, len(ranked) // 10)]
    out = {"top_trials": len(top), "hyperparameters": {}}
    for hp in t.hyperparameters():
        if only and hp.name != only:
            continue
        values = [x.config.get(hp.name) for x in t.trials if x.config.get(hp.name) is not None]
        units = _unit(hp, values)
        if units is None:
            tried = {str(v) for v in values}
            out["hyperparameters"][hp.name] = {
                "choices": [str(c) for c in hp.choices],
                "never_tried": [str(c) for c in hp.choices if str(c) not in tried],
                "best_trials_use": sorted({str(t.trials[i].config.get(hp.name)) for i in top})}
            continue
        top_values = [t.trials[i].config.get(hp.name) for i in top]
        top_units = _unit(hp, top_values)
        found = {**_bounds(hp), "tried_min": _r(min(values)), "tried_max": _r(max(values)),
                 **_spread(units),
                 "best_trials_range": [_r(min(top_values)), _r(max(top_values))]}
        if ranked:
            best = t.trials[ranked[0]].config.get(hp.name)
            found["best_value"] = _r(best)
            unit = _unit(hp, [best])
            if unit and (unit[0] <= 0.05 or unit[0] >= 0.95):
                found["best_at_edge"] = "lower" if unit[0] <= 0.05 else "upper"
        out["hyperparameters"][hp.name] = found
    edges = [n for n, f in out["hyperparameters"].items() if f.get("best_at_edge")]
    if edges:
        out["note"] = ("The best value of " + ", ".join(edges) + " lies within 5% of a bound; "
                       "better values may lie beyond the range searched.")
    return out


def _effects(t: Trials, metric, only) -> dict:
    hib = t.higher_is_better(metric)
    pairs = t.scored(metric)
    out = {"trials": len(pairs), "hyperparameters": {},
           "note": "Association across the trials the optimizer chose, not cause: after the "
                   "initial design it samples where it expects good scores."}
    for name in t.hp_names():
        if only and name != only:
            continue
        hp = t.space[name] if t.space is not None else None
        values = [(t.trials[i].config.get(name), s) for i, s in pairs
                  if t.trials[i].config.get(name) is not None]
        if not values:
            continue
        numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
                      for v, _ in values)
        if hp is not None and _unit(hp, [values[0][0]]) is None:
            numeric = False
        if numeric:
            rho = _spearman([float(v) for v, _ in values], [s for _, s in values])
            found = {"spearman_with_score": rho}
            if rho is not None:
                found["reads"] = ("higher values went with better scores"
                                  if (rho > 0) == hib else "lower values went with better scores")
                if abs(rho) < 0.2:
                    found["reads"] = "little monotone relation"
            ordered = sorted(values, key=lambda p: float(p[0]))
            thirds = [ordered[k * len(ordered) // 3:(k + 1) * len(ordered) // 3] for k in range(3)]
            found["by_third_of_values"] = [
                {"from": _r(part[0][0]), "to": _r(part[-1][0]), "n": len(part),
                 "median_score": _r(np.median([s for _, s in part]))}
                for part in thirds if part]
        else:
            groups = {}
            for v, s in values:
                groups.setdefault(str(v), []).append(s)
            found = {"by_value": {v: _stats(scores, hib) | {"n": len(scores)}
                                  for v, scores in sorted(groups.items())}}
        out["hyperparameters"][name] = found
    return out


def _importance(t: Trials, metric, _hp) -> dict:
    result = t.result
    games = {}
    for game, names in HP_GAME_FIELDS.items():
        shares = (getattr(result, names["importance"], None) or {}).get(metric) or {}
        if not shares:
            continue
        found = {"importance": {k: _r(v) for k, v in
                                sorted(shares.items(), key=lambda kv: -abs(kv[1] or 0))}}
        warning = (getattr(result, names["warning"], None) or {}).get(metric)
        if warning:
            found["warning"] = warning
        grid = (getattr(result, names["interactions"], None) or {}).get(metric) or {}
        pairs = {}
        for a, row in grid.items():
            for b, value in row.items():
                if a < b and value is not None:
                    pairs[f"{a} × {b}"] = value
        if pairs:
            found["top_interactions"] = {k: _r(v) for k, v in sorted(
                pairs.items(), key=lambda kv: -abs(kv[1]))[:5]}
        games[game] = found
    if not games:
        return {"computed": False,
                "note": "Importance has not been computed for this metric. The dashboard's "
                        "importance figure computes it; until then, the effects analysis "
                        "gives a rougher view."}
    total = (result.hyperparameter_tunability_total or {}).get(metric)
    out = {"computed": True, "games": games,
           "reading": "tunability: how much tuning each hyperparameter from its default can "
                      "gain; mistunability: how much a bad value can lose; sensitivity: how "
                      "much the score varies with it. Values are shares of the whole."}
    if total:
        out["tunability_total_in_metric_units"] = _r(total)
    return out


def _metrics(t: Trials, metric, _hp) -> dict:
    names = sorted({m for x in t.trials for m in x.scores})
    ok = [i for i in range(len(t.trials)) if not t.trials[i].failed]
    pairs = {}
    for a_pos, a in enumerate(names):
        for b in names[a_pos + 1:]:
            both = [(t.score(i, a), t.score(i, b)) for i in ok]
            both = [(x, y) for x, y in both if x is not None and y is not None]
            rho = _spearman([x for x, _ in both], [y for _, y in both])
            if rho is not None:
                pairs[f"{a} ~ {b}"] = rho
    seconds = {}
    for name in names:
        both = [(t.score(i, name), t.trials[i].duration) for i in ok]
        both = [(x, y) for x, y in both if x is not None]
        rho = _spearman([x for x, _ in both], [y for _, y in both])
        if rho is not None:
            seconds[name] = rho
    best = {}
    for name in names:
        ranked = t.ranked(name)
        if ranked:
            best[name] = {"trial": t.trials[ranked[0]].trial,
                          "score": _r(t.score(ranked[0], name)),
                          "higher_is_better": t.higher_is_better(name)}
    return {"spearman_between_metrics": dict(sorted(pairs.items(), key=lambda kv: -abs(kv[1]))),
            "spearman_with_seconds": seconds, "best_trial_per_metric": best,
            "same_best_trial": len({b["trial"] for b in best.values()}) == 1}


def _durations(t: Trials, metric, _hp) -> dict:
    seconds = [x.duration for x in t.trials]
    out = {"total_seconds": _r(sum(seconds)),
           "per_trial": _stats(seconds, higher_is_better=False) | {
               "slowest": _r(max(seconds)), "fastest": _r(min(seconds))},
           "slowest_trials": [t.row(i, [metric]) for i in
                              sorted(range(len(t.trials)), key=lambda i: -seconds[i])[:3]]}
    for key in ("best", "worst"):
        out["per_trial"].pop(key, None)
    by_phase = {}
    for i, s in enumerate(seconds):
        by_phase.setdefault(t.phase(i), []).append(s)
    out["mean_seconds_by_phase"] = {k: _r(np.mean(v)) for k, v in by_phase.items()}
    with_hp = {}
    for name in t.hp_names():
        values = [(x.config.get(name), x.duration) for x in t.trials]
        values = [(v, s) for v, s in values if isinstance(v, (int, float)) and not isinstance(
            v, bool)]
        rho = _spearman([float(v) for v, _ in values], [s for _, s in values])
        if rho is not None:
            with_hp[name] = rho
    out["spearman_seconds_with_hyperparameter"] = dict(
        sorted(with_hp.items(), key=lambda kv: -abs(kv[1])))
    timeouts = sum(x.run_info.get("status") == STATUS_TIMEOUT for x in t.trials)
    if timeouts:
        out["timed_out"] = timeouts
    return out


def _failures(t: Trials, metric, _hp) -> dict:
    failed = [i for i, x in enumerate(t.trials) if x.failed]
    out = {"failed": len(failed), "of": len(t.trials)}
    if not failed:
        return out
    statuses, reasons = {}, {}
    for i in failed:
        x = t.trials[i]
        status = STATUS.get(x.run_info.get("status"), "failed")
        statuses[status] = statuses.get(status, 0) + 1
        reason = (x.failure or status).strip().splitlines()[0][:200] if (x.failure or status) \
            else status
        reasons.setdefault(reason, []).append(x.trial)
    out["by_status"] = statuses
    out["by_reason"] = [{"reason": r, "count": len(v), "trials": v[:10]}
                        for r, v in sorted(reasons.items(), key=lambda kv: -len(kv[1]))]
    ok = [i for i, x in enumerate(t.trials) if not x.failed]
    where = {}
    for name in t.hp_names():
        bad = [t.trials[i].config.get(name) for i in failed]
        good = [t.trials[i].config.get(name) for i in ok]
        bad = [v for v in bad if v is not None]
        good = [v for v in good if v is not None]
        if not bad:
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bad + good):
            where[name] = {"failed_range": [_r(min(bad)), _r(max(bad))],
                           "succeeded_range": [_r(min(good)), _r(max(good))] if good else None}
        else:
            values = sorted({str(v) for v in bad + good})
            where[name] = {v: {"failed": sum(str(b) == v for b in bad),
                               "succeeded": sum(str(g) == v for g in good)} for v in values}
    out["where_in_the_space"] = where
    out["note"] = "A failed trial is scored as a model that knows nothing; it is left out of " \
                  "every score above."
    return out


_ANALYSES = {"scores": _scores, "progress": _progress, "initial_design": _initial_design,
             "coverage": _coverage, "effects": _effects, "importance": _importance,
             "metrics": _metrics, "durations": _durations, "failures": _failures}
