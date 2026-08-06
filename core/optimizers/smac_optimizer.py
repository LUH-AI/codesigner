import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration
from smac import Scenario, BlackBoxFacade
from smac.utils.configspace import convert_configurations_to_array
from smac.runhistory import StatusType
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from ..paths import safe_join
from .base import BaseOptimizer, OptimizationResult, TrialCollector, merge_stopping, rebase_history
from ..splits import holdout
from .timing import STATUS_SUCCESS
from .trial import evaluate_trial

logging.getLogger("smac").setLevel(logging.WARNING)

# Fixed budget keeps the Scenario hash stable across runs; the while-loop controls actual stopping.
_SMAC_MAX_TRIALS = 100_000

# How many configurations to ask the surrogate about when answering "is anything
# left better than the incumbent?". The space is sampled rather than covered, so
# this trades a little cost for a tighter bound; a thousand predictions from a
# fitted GP is milliseconds beside a trial.
_CONFIDENCE_SAMPLES = 1000

# Trials before the surrogate is allowed to end a run. A GP fitted on a handful
# of points is confident in the way a straight line through two points is: the
# posterior is narrow because there is nothing to contradict it.
_CONFIDENCE_MIN_TRIALS = 10


def _normal_cdf(z):
    """Standard normal CDF, without pulling scipy in for one function."""
    from math import erf, sqrt

    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(z)


class SMACOptimizer(BaseOptimizer):
    """Bayesian optimization via SMAC's BlackBox facade.

    Uses a Gaussian Process surrogate to guide trial selection, treating the
    objective as a black box (no gradient or structural assumptions).
    """

    name = "SMAC (BlackBox)"

    #: The GP is the whole point of this optimizer, so it can be asked.
    supports_confidence_stopping = True

    def serialize_result(self, result: OptimizationResult) -> dict:
        output_dir = result.metadata.get("smac_output_dir", "")
        root = Path(output_dir) if output_dir else None

        if not root or not root.exists():
            return super().serialize_result(result)

        rh_files = list(root.rglob("runhistory.json"))
        if not rh_files:
            return super().serialize_result(result)

        rh_path = rh_files[0]
        rh = json.loads(rh_path.read_text(encoding="utf-8"))

        trial_by_id = {t.trial: t for t in result.trials}
        config_key_to_id = {
            tuple(sorted(t.config.items())): str(t.trial) for t in result.trials
        }

        for entry in rh["data"]:
            t = trial_by_id.get(entry["config_id"])
            if t is not None:
                entry["scores"] = t.scores
                entry["incumbent_score"] = t.incumbent_score
                incumbent_key = tuple(sorted(t.incumbent_config.items()))
                entry["incumbent_config_id"] = int(
                    config_key_to_id.get(incumbent_key, str(entry["config_id"]))
                )

        best_key = tuple(sorted(result.best_config.items())) if result.best_config else ()
        best_config_id = config_key_to_id.get(best_key) or (
            str(result.trials[-1].trial) if result.trials else "0"
        )

        optimizer_state = {
            f.relative_to(root).as_posix(): json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(root.rglob("*"))
            if f.is_file() and f != rh_path
        }

        return {
            "stats": rh["stats"],
            "data": rh["data"],
            "configs": rh["configs"],
            "config_origins": rh.get("config_origins", {}),
            "optimizer_state": optimizer_state,
            "primary_metric": result.primary_metric,
            "best_score": result.best_score,
            "best_config_id": best_config_id,
            "hyperparameter_importance": result.hyperparameter_importance,
            "hyperparameter_importance_warning": result.hyperparameter_importance_warning,
            "trials_limit": result.trials_limit,
        }

    def deserialize_result(self, d: dict) -> OptimizationResult:
        optimizer_state = d.get("optimizer_state", {})

        if not optimizer_state:
            return super().deserialize_result(d)

        output_dir = Path(tempfile.mkdtemp())
        first_key = next(iter(optimizer_state))
        subdir = "/".join(first_key.split("/")[:-1])
        smac_dir = safe_join(output_dir, subdir) if subdir else output_dir
        smac_dir.mkdir(parents=True, exist_ok=True)

        # Write runhistory.json; data entries include .ihpo extensions which SMAC ignores.
        (smac_dir / "runhistory.json").write_text(
            json.dumps({
                "stats": d.get("stats", {}),
                "data": d.get("data", []),
                "configs": d.get("configs", {}),
                "config_origins": d.get("config_origins", {}),
            }),
            encoding="utf-8",
        )

        # Write remaining SMAC files; update scenario.json's output_directory to new path.
        # Each key is confined to output_dir: they come from the .ihpo, and an
        # absolute or ../ key would otherwise be written wherever it pointed.
        scenario_key = f"{subdir}/scenario.json"
        for rel, content in optimizer_state.items():
            dest = safe_join(output_dir, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel == scenario_key:
                content = {**content, "output_directory": str(smac_dir)}
            dest.write_text(json.dumps(content), encoding="utf-8")

        result = super().deserialize_result(d)
        result.metadata["smac_output_dir"] = str(output_dir)
        return result

    def _confidence_nothing_better(self, smac, config_space, incumbent_cost: float):
        """How sure the surrogate is that no remaining configuration wins.

        This is what a Bayesian optimizer already knows and normally spends on
        choosing the next trial. The Gaussian process gives a posterior mean and
        standard deviation for any point in the space, so for a candidate *x*
        the chance it beats the incumbent is `Φ((c_inc − μ(x)) / σ(x))` — costs
        are minimized, so beating means lower. Take the best chance any sampled
        candidate has, and one minus it is the confidence that none of them does.

        Returns None when there is no answer to give: the model has not been
        fitted yet (SMAC trains it lazily, inside `ask`, and not before its
        initial design is done), or predicting failed. The criterion then simply
        does not fire, rather than a missing model reading as certainty.

        Two honest limits. It is confidence **under the surrogate's own model** —
        a GP with its own assumptions, and one that is poorly calibrated early
        on, which is why a handful of trials cannot end a run this way. And the
        space is *sampled*, not covered, so a narrow optimum that no sample lands
        near is not accounted for.
        """
        selector = getattr(smac.intensifier, "_config_selector", None)
        model = getattr(selector, "_model", None)
        if model is None:
            return None

        try:
            candidates = config_space.sample_configuration(_CONFIDENCE_SAMPLES)
            if not isinstance(candidates, list):
                candidates = [candidates]
            mean, std = model.predict(
                convert_configurations_to_array(candidates), covariance_type="std")
        except Exception:  # noqa: BLE001 — an unfitted or unhappy model means "no answer"
            return None
        if std is None:
            return None

        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.asarray(std, dtype=float).reshape(-1)
        # A candidate the model is certain about beats the incumbent or does not;
        # the floor keeps that from dividing by zero on the way to saying so.
        z = (incumbent_cost - mean) / np.maximum(std, 1e-12)
        best_chance = float(_normal_cdf(z).max())
        return 1.0 - best_chance

    def _replay(self, smac, config_space, trials, primary_metric: str, seed: int) -> None:
        """Tell SMAC an earlier run's trials, so a rebuilt surrogate knows them.

        Used when the stored SMAC state cannot be resumed — the optimized metric
        changed, or the run directory is gone. Without this the search would
        start blind while the page still showed the accumulated history.

        `save=False`: SMAC writes its runhistory on every `tell`, and rewriting
        a growing file once per replayed trial is pointless when the first real
        trial saves the lot. If the run is cancelled before any real trial, no
        runhistory is written and `serialize_result` falls back to rebuilding
        from the trials themselves, which is correct — just without SMAC state.
        """
        for t in trials:
            try:
                config = Configuration(config_space, values=t.config)
            except Exception:
                # A configuration the space no longer admits (the model's search
                # space changed under the experiment). Not tellable, and not
                # worth failing the run over.
                continue
            info = t.run_info
            smac.tell(
                TrialInfo(config=config, seed=seed),
                TrialValue(
                    cost=1.0 - t.scores.get(primary_metric, t.score),
                    time=info.get("time", 0.0), cpu_time=info.get("cpu_time", 0.0),
                    starttime=info.get("starttime", 0.0), endtime=info.get("endtime", 0.0),
                    status=StatusType(info.get("status", STATUS_SUCCESS)),
                ),
                save=False,
            )

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials=None, previous_result=None, seed: int = 0, cancel_event=None,
                 stopping: dict | None = None, splits=None):
        # One fold unless the caller divided the data itself; see core.splits.
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)

        # A changed metric makes the stored surrogate worse than useless: it was
        # fitted on costs from a different objective, and resuming would mix the
        # two in one model. Rebuild instead, replaying the history below.
        previous_result, metric_changed = rebase_history(previous_result, primary_metric)

        if previous_result is not None:
            stored_dir = previous_result.metadata.get("smac_output_dir", "")
            trial_offset = len(previous_result.trials)
            if stored_dir and Path(stored_dir).exists() and not metric_changed:
                output_dir = stored_dir
                overwrite = False
            else:
                output_dir = tempfile.mkdtemp()
                overwrite = True
        else:
            output_dir = tempfile.mkdtemp()
            trial_offset = 0
            overwrite = True

        collector = TrialCollector(
            trial_offset=trial_offset,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=merge_stopping(n_trials, stopping),
        )

        config_space = model.get_config_space(seed=seed)
        scenario = Scenario(
            config_space,
            name="ihpo",
            n_trials=_SMAC_MAX_TRIALS,
            deterministic=True,
            seed=seed,
            output_directory=output_dir,
        )

        def _unreachable(config, seed: int = 0) -> float:
            raise RuntimeError("SMAC called target_function unexpectedly in ask/tell mode")

        smac = BlackBoxFacade(scenario, _unreachable, overwrite=overwrite)

        wants_confidence = "incumbent_confidence" in merge_stopping(n_trials, stopping)

        # A fresh facade knows nothing, so anything already evaluated has to be
        # handed to it — otherwise the search would begin from zero while the
        # page still shows the accumulated trials.
        if overwrite and previous_result is not None:
            self._replay(smac, config_space, previous_result.trials, primary_metric, seed)

        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            info = smac.ask()
            config = dict(info.config)
            all_scores, run_info = evaluate_trial(model, config, splits, metrics, seed=seed)
            cost = 1.0 - all_scores[primary_metric]
            # Feed the measured timing into SMAC so its runhistory (which the
            # serialize override copies verbatim) carries the real values. The
            # status is passed through rather than assumed: a configuration the
            # model died on is something the surrogate should learn, not a
            # success that happened to score zero.
            smac.tell(info, TrialValue(
                cost=cost, time=run_info["time"], cpu_time=run_info["cpu_time"],
                starttime=run_info["starttime"], endtime=run_info["endtime"],
                status=StatusType(run_info["status"]),
            ))
            collector.record(config, all_scores[primary_metric], all_scores, run_info=run_info)

            # Only worth asking if someone is listening, and only once there is
            # enough history for the answer to mean anything. The model is one
            # trial stale — SMAC trains it inside `ask` — which for a stopping
            # heuristic is close enough and avoids re-fitting it here.
            if wants_confidence and len(collector.results) >= _CONFIDENCE_MIN_TRIALS:
                collector.note_confidence(self._confidence_nothing_better(
                    smac, config_space, 1.0 - collector.incumbent_score))

        incumbent = smac.intensifier.get_incumbent()
        all_trials = (previous_result.trials if previous_result else []) + collector.results

        hp_importance = {}
        hp_warning = {}
        for metric_name in metrics:
            imp, warn = self.compute_hp_importance(config_space, all_trials, metric_name, seed=seed)
            hp_importance[metric_name] = imp
            hp_warning[metric_name] = warn

        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=dict(incumbent) if incumbent else (all_trials[-1].config if all_trials else {}),
            # From `scores`, not `score`: after a metric change the history has
            # been re-read, and a trial that could not be (an old file with one
            # metric) must not contribute a score for a metric it never had.
            best_score=max((t.scores.get(primary_metric, t.score) for t in all_trials), default=0.0),
            hyperparameter_importance=hp_importance,
            hyperparameter_importance_warning=hp_warning,
            metadata={"smac_output_dir": str(output_dir),
                      "stopped_by": collector.stopped_by},
        )
