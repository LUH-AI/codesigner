import logging
import tempfile
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration
from smac import Scenario
from smac.main.exceptions import ConfigurationSpaceExhaustedException
from smac.runhistory.dataclasses import TrialInfo

from ...analytics import compute_hp_games, pair_trials_with_scores
from ...metrics import metric_for, to_cost
from ..base import (
    REQUESTED_ORIGIN, STOPPED_BY_EXHAUSTED, OptimizationResult, game_fields, merge_stopping,
)
from ..trial import evaluate_trial
from . import history
from .facade import FacadeBuilder, scenario_budget
from .params import STRATEGIES
from .priors import (
    PRIOR_JUDGING_SAMPLES_PER_HYPERPARAMETER, apply_priors, stamped, waits_for_the_design,
)
from .surrogate import CONFIDENCE_MIN_TRIALS, RunSurrogate, confidence_nothing_better
from .walks import record_walks, walks_to

logger = logging.getLogger(__name__)

#: `SLICE_CANDIDATES` is also the extra budget of a reconstructed scenario, so
#: the figure quotes the decay factor the walk uses.
SLICE_CANDIDATES = 12


class SMACOptimizer(FacadeBuilder):
    name = "SMAC"

    #: `aliases` keeps the name from when SMAC always used the Gaussian-process
    #: facade.
    aliases = ("SMAC (BlackBox)",)

    #: The surrogate can report its confidence (see `incumbent_confidence`).
    supports_confidence_stopping = True

    #: A changed metric invalidates the surrogate carried between runs.
    fits_surrogate = True

    def acquisition_function(self):
        return {"kind": self._acquisition, "xi": float(self._acquisition_xi or 0.0)}

    @classmethod
    def judges_priors(cls, params):
        """`judges_priors` only accepts the GP, because the forest's log-scaled
        lower confidence bound maps to near-zero cost everywhere and every prior
        would pass."""
        strategy = cls.known_params(params).get("search_strategy")
        return (strategy if strategy in STRATEGIES else "gp") == "gp"

    def refit_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None):
        """A GP run is read through a GP, since a generic forest would make
        every surrogate look alike. The model trains on cost and answers in
        metric units, or on *values* as given, and a failed refit returns
        `(None, warning)`.
        """
        data = pair_trials_with_scores(config_space, trials, metric_name, values=values)
        if len(data) < 2:
            return None, "Not enough trials to fit a surrogate."

        metric = None if values is not None else metric_for(metric_name)
        X = np.array([config.get_array() for config, _ in data])
        y = np.array([float(target) if metric is None else to_cost(metric, target)
                      for _, target in data])

        try:
            with tempfile.TemporaryDirectory() as directory:
                scenario = Scenario(config_space, n_trials=max(len(y), 1), seed=seed,
                                    deterministic=True, output_directory=Path(directory))
                model = self._surrogate(STRATEGIES[self._search_strategy], scenario,
                                        log_y=False)
                model.train(X, y)
        except Exception as error:  # noqa: BLE001 — any refit failure falls back
            return None, f"Could not rebuild this run's surrogate: {error}"

        return RunSurrogate(model, metric), None

    def slice_challengers(self, config_space, trials, metric_name: str, seed: int = 0,
                          priors=None, count=SLICE_CANDIDATES, previous_result=None,
                          budget=None, with_walks=False):
        """`slice_challengers` builds the facade with `_facade`, replays the
        trials, applies priors and calls `ask()`, returning `(configs, origins)`
        in the search's own rank order, or None. The initial design keeps the
        size the run recorded, since a rebuild sized for an unbounded budget
        would propose space-filling samples. Successive calls to `ask()` without
        a `tell()` walk the maximizer's cached challenger list. *with_walks*
        adds each candidate's local-search path (see `record_walks`), or None
        for one reached without climbing.
        """
        if not trials:
            return None
        try:
            with self._rebuilt(config_space, trials, metric_name, seed, previous_result,
                               len(trials) + max(count, 1), "ranking candidates") as (smac, scenario):
                apply_priors(smac, scenario, config_space, priors, budget=budget)
                walks = record_walks(smac) if with_walks else None
                asked = [smac.ask() for _ in range(count)]
            configs = [dict(info.config) for info in asked]
            origins = [info.config.origin or "" for info in asked]
            if walks is None:
                return configs, origins
            return configs, origins, walks_to(asked, walks)
        except Exception as error:  # noqa: BLE001 — a diagnostic must not cost the figure
            logger.warning("Could not ask this run's optimizer what is next: %s", error)
            return None

    def evaluate_prior(self, config_space, trials, metric_name: str, seed: int = 0,
                       priors=None, tolerance: float = 0.15, previous_result=None,
                       budget=None, top_k: int = 10):
        """SMAC's `ClimbingComparisonPolicy` runs DynaBO's safeguard on demand,
        climbing from the best *top_k* draws of the belief and of the
        incumbent's neighbourhood and rejecting the belief when its climbs end
        worse by more than *tolerance*. The facade is rebuilt as in
        `slice_challengers`, and the verdict comes from `add_prior`. *tolerance*
        is positive, in raw objective units, and passed negated as the policy
        expects. An "unjudged" verdict means the policy had nothing to compare
        and accepted by rule, which `_RecordingPolicy` detects.
        """
        if not priors:
            return {"verdict": "unjudged", "reason": "no prior is stated"}
        if not trials:
            return {"verdict": "unjudged",
                    "reason": "nothing has been run yet, so there is no model to "
                              "judge this against"}
        try:
            from smac.acquisition.weight import ClimbingComparisonPolicy
        except ImportError:
            return {"verdict": "unjudged",
                    "reason": "this SMAC has no climbing comparison policy"}

        class _RecordingPolicy(ClimbingComparisonPolicy):
            compared = None

            def compare(self, *args, **kwargs):
                self.compared = super().compare(*args, **kwargs)
                return self.compared

        try:
            # DynaBO draws the neighbourhood as wide as the belief.
            policy = _RecordingPolicy(
                threshold=-abs(tolerance), top_k=int(top_k),
                n_samples_per_hyperparameter=PRIOR_JUDGING_SAMPLES_PER_HYPERPARAMETER,
                neighbourhood_std="prior")
            with self._rebuilt(config_space, trials, metric_name, seed, previous_result,
                               len(trials) + 1, "judging a prior") as (smac, scenario):
                applied = apply_priors(
                    smac, scenario, config_space, priors, budget=budget, acceptance=policy)
        except Exception as error:  # noqa: BLE001 — a diagnostic must not cost the figure
            logger.warning("Could not judge the stated prior: %s", error)
            return {"verdict": "unjudged", "reason": str(error)}

        if applied is None:
            return {"verdict": "unjudged", "reason": "no prior is stated"}
        if applied.get("applied") and policy.compared is None:
            return {"verdict": "unjudged",
                    "reason": "there is no model or incumbent to judge this against"}
        if applied.get("applied"):
            return {"verdict": "accepted",
                    "reason": "the surrogate does not think this region is worse "
                              "than where the search is already looking"}
        if applied.get("rejected"):
            return {"verdict": "rejected", "reason": applied.get("reason", "")}
        # A prior that failed to apply is no judgement on the belief.
        return {"verdict": "unjudged", "reason": applied.get("reason", "")}

    def serialize_result(self, result: OptimizationResult) -> dict:
        return history.serialized(super().serialize_result(result), result)

    def deserialize_result(self, d: dict) -> OptimizationResult:
        return history.restored(super().deserialize_result(d), d.get("optimizer_state") or {})

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials=None, previous_result=None, seed: int = 0, cancel_event=None,
                 stopping: dict | None = None, splits=None, priors=None,
                 configurations=None):
        # A surrogate fitted to another objective must be rebuilt, so the
        # history is replayed below.
        splits, previous_result, collector = self._begin(
            previous_result, primary_metric, n_trials, stopping, splits,
            X_train, y_train, X_val, y_val)
        criteria = merge_stopping(n_trials, stopping)

        # Each run uses a fresh directory and replays history, which costs
        # little and avoids SMAC's `overwrite=False` console prompt that hangs a
        # worker.
        trial_offset = len(previous_result.trials) if previous_result else 0
        output_dir = tempfile.mkdtemp()

        config_space = model.get_config_space(seed=seed)
        scenario = Scenario(
            config_space,
            name="ihpo",
            # SMAC sizes its initial design as a fraction of the real budget.
            n_trials=scenario_budget(trial_offset, criteria.get("max_trials")),
            deterministic=True,
            seed=seed,
            output_directory=Path(output_dir),
            **self._scenario_extras(),
        )

        def _unreachable(config, seed: int = 0) -> float:
            raise RuntimeError("SMAC called target_function unexpectedly in ask/tell mode")

        smac = self._facade(scenario, _unreachable, previous_result)

        wants_confidence = "incumbent_confidence" in criteria

        # A fresh facade needs the evaluated trials to avoid starting from zero.
        if previous_result is not None:
            history.replay(smac, config_space, previous_result.trials, primary_metric,
                           seed, replay_metric=collector.metric)

        # `add_prior` anchors the decay at the finished trials it sees and
        # keeps the anchor fixed afterwards, so the prior goes in after the
        # replay. A belief
        # that waits goes in once the initial design is exhausted, since the
        # design trials are beyond its reach and fading through them would
        # weaken it.
        waiting = waits_for_the_design(priors)
        # The design end is an absolute count from the start of the search,
        # since a resume continues inside the design.
        design_end = (self._initial_points(scenario, previous_result)
                      if waiting else 0)
        prior_report = None
        if not waiting:
            prior_report = stamped(apply_priors(smac, scenario, config_space, priors), smac)

        # Requested configurations are told to SMAC like its own, since the
        # model learns the same from them.
        def tell_requested(config, all_scores, run_info):
            chosen = Configuration(config_space, values=config)
            chosen.origin = REQUESTED_ORIGIN
            smac.tell(TrialInfo(config=chosen, seed=seed), history.trial_value(
                to_cost(collector.metric, all_scores[primary_metric]), run_info))

        self.evaluate_requested(
            configurations, model=model, splits=splits, metrics=metrics,
            primary_metric=primary_metric, seed=seed, collector=collector,
            cancel_event=cancel_event, told=tell_requested)

        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            # The prior goes in before the ask, so the first model-chosen trial
            # is weighted, and the count comes from SMAC's runhistory, where
            # `add_prior` takes its anchor.
            if prior_report is None and len(smac.runhistory) >= design_end:
                prior_report = stamped(apply_priors(smac, scenario, config_space, priors),
                                       smac)
            try:
                info = smac.ask()
            except ConfigurationSpaceExhaustedException:
                # A small discrete space can be exhausted, which completes the
                # search.
                collector.stopped_by = STOPPED_BY_EXHAUSTED
                break
            config = dict(info.config)
            all_scores, run_info = evaluate_trial(model, config, splits, metrics, seed=seed,
                                                   keep=self.keep_model is not None)
            smac.tell(info, history.trial_value(
                to_cost(collector.metric, all_scores[primary_metric]), run_info))
            collector.record(config, all_scores[primary_metric], all_scores,
                             run_info=run_info, origin=info.config.origin or "")

            # SMAC trains inside `ask`, so the model is one trial stale, which
            # suffices for a stopping heuristic.
            if wants_confidence and len(collector.results) >= CONFIDENCE_MIN_TRIALS:
                collector.note_confidence(confidence_nothing_better(
                    smac, config_space, collector.incumbent_config))

        incumbent = smac.intensifier.get_incumbent()
        all_trials = (previous_result.trials if previous_result else []) + collector.results

        games = compute_hp_games(config_space, all_trials, metrics, seed=seed,
                                 cancel_event=cancel_event, budget=self.analytics_max_coalitions,
                                 wanted=self.analytics_wanted)

        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=dict(incumbent) if incumbent else (all_trials[-1].config if all_trials else {}),
            # `scores` contains the rebased metric, with `score` as the fallback
            # for an old one-metric file.
            best_score=collector.metric.best(
                (t.scores.get(primary_metric, t.score) for t in all_trials), default=0.0),
            **game_fields(games),
            # `metadata` stays in memory, and `experiments/services/run.py`
            # writes the prior event, since `core/` knows nothing of a Run.
            metadata={"smac_output_dir": str(output_dir),
                      "stopped_by": collector.stopped_by,
                      "prior": prior_report},
        )
