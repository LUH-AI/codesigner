import json
import logging
import tempfile
from pathlib import Path

from smac import Scenario, BlackBoxFacade
from smac.runhistory import StatusType
from smac.runhistory.dataclasses import TrialValue

from ..paths import safe_join
from .base import BaseOptimizer, OptimizationResult, TrialCollector, TrialResult
from .trial import evaluate_trial

logging.getLogger("smac").setLevel(logging.WARNING)

# Fixed budget keeps the Scenario hash stable across runs; the while-loop controls actual stopping.
_SMAC_MAX_TRIALS = 100_000


class SMACOptimizer(BaseOptimizer):
    """Bayesian optimization via SMAC's BlackBox facade.

    Uses a Gaussian Process surrogate to guide trial selection, treating the
    objective as a black box (no gradient or structural assumptions).
    """

    name = "SMAC (BlackBox)"

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

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials, previous_result=None, seed: int = 0, cancel_event=None):
        if previous_result is not None:
            stored_dir = previous_result.metadata.get("smac_output_dir", "")
            trial_offset = len(previous_result.trials)
            if stored_dir and Path(stored_dir).exists():
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
            target_new_trials=n_trials,
            trial_offset=trial_offset,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
        )

        scenario = Scenario(
            model.get_config_space(seed=seed),
            name="ihpo",
            n_trials=_SMAC_MAX_TRIALS,
            deterministic=True,
            seed=seed,
            output_directory=output_dir,
        )

        def _unreachable(config, seed: int = 0) -> float:
            raise RuntimeError("SMAC called target_function unexpectedly in ask/tell mode")

        smac = BlackBoxFacade(scenario, _unreachable, overwrite=overwrite)

        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            info = smac.ask()
            config = dict(info.config)
            all_scores, run_info = evaluate_trial(
                model, config, X_train, y_train, X_val, y_val, metrics, seed=seed)
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

        incumbent = smac.intensifier.get_incumbent()
        all_trials = (previous_result.trials if previous_result else []) + collector.results

        config_space = model.get_config_space(seed=seed)
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
            best_score=max((r.score for r in all_trials), default=0.0),
            hyperparameter_importance=hp_importance,
            hyperparameter_importance_warning=hp_warning,
            metadata={"smac_output_dir": str(output_dir)},
        )
