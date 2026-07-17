from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from hypershap import ExplanationTask, HyperSHAP

@dataclass
class OptimizerParam:
    """Describes one user-configurable parameter of an optimizer, for form rendering."""
    name: str                               # kwarg name passed to __init__
    label: str                              # human-readable label shown in the form
    type: str                               # "int", "float", or "select"
    default: Any
    min: Any = None                         # lower bound for int / float
    max: Any = None                         # upper bound for int / float
    choices: List[Any] = field(default_factory=list)  # options for select


@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    scores: Dict[str, float]    # score for every metric
    score: float                # primary metric score (used internally)
    incumbent_score: float      # running incumbent score
    incumbent_config: Dict[str, Any]


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


class TrialCollector:
    """Reusable bookkeeper for optimizer trial results.

    Tracks the per-trial results list and the running incumbent as trials
    complete.  Optimizer-specific callbacks should inherit from (or delegate
    to) this class and call ``record()`` once per evaluated trial.

    Parameters
    ----------
    target_new_trials:
        Stop collecting (``done`` becomes True) after this many new trials.
    trial_offset:
        Number of trials already recorded in a previous run; used to produce
        globally-sequential trial numbers when resuming.
    initial_best_score:
        Best primary-metric score seen before this run (``-inf`` for a fresh
        run).  Ensures the incumbent is correct relative to full history.
    initial_best_config:
        Config that produced ``initial_best_score``.
    """

    def __init__(
        self,
        target_new_trials: int,
        trial_offset: int = 0,
        initial_best_score: float = float("-inf"),
        initial_best_config: Optional[Dict[str, Any]] = None,
    ):
        self.results: List[TrialResult] = []
        self._target_new_trials = target_new_trials
        self._trial_offset = trial_offset
        self._incumbent_score = initial_best_score
        self._incumbent_config = initial_best_config

    @property
    def done(self) -> bool:
        """True once the target number of new trials has been collected."""
        return len(self.results) >= self._target_new_trials

    def record(
        self,
        config: Dict[str, Any],
        score: float,
        all_scores: Dict[str, float],
    ) -> TrialResult:
        """Record one completed trial, update the incumbent, return the TrialResult."""
        if score > self._incumbent_score:
            self._incumbent_score = score
            self._incumbent_config = config

        trial = TrialResult(
            trial=self._trial_offset + len(self.results) + 1,
            config=config,
            scores=all_scores,
            score=score,
            incumbent_score=self._incumbent_score,
            incumbent_config=self._incumbent_config or config,
        )
        self.results.append(trial)
        return trial


class BaseOptimizer(ABC):
    """Base class for all hyperparameter optimizers."""

    params_schema: List[OptimizerParam] = []

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def optimize(
        self,
        model,
        X_train, y_train,
        X_val, y_val,
        metrics: dict,
        primary_metric: str,
        n_trials: int,
        previous_result: Optional["OptimizationResult"] = None,
        seed: int = 0,
        cancel_event=None,
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
            "config_origins": {str(t.trial): self.name for t in result.trials},
            "optimizer_state": {},
            "primary_metric": result.primary_metric,
            "best_score": result.best_score,
            "best_config_id": best_config_id,
            "hyperparameter_importance": result.hyperparameter_importance,
            "hyperparameter_importance_warning": result.hyperparameter_importance_warning,
            "trials_limit": result.trials_limit,
        }

    def deserialize_result(self, d: dict) -> "OptimizationResult":
        """Reconstruct an OptimizationResult from a runhistory-mirrored result dict.

        Inverse of serialize_result.  Optimizers that override serialize_result
        to embed extra state should override this to restore it.
        """
        configs = d.get("configs", {})
        data = d.get("data", [])
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
            ))

        best_config_id = str(d.get("best_config_id") or (str(trials[-1].trial) if trials else "0"))
        return OptimizationResult(
            trials=trials,
            primary_metric=primary_metric,
            best_config=configs.get(best_config_id, {}),
            best_score=d.get("best_score", 0.0),
            hyperparameter_importance=d.get("hyperparameter_importance", {}),
            hyperparameter_importance_warning=d.get("hyperparameter_importance_warning", {}),
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

    def compute_hp_importance(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """Estimate hyperparameter importance from a completed list of trials.

        Returns (importance_dict, warning_message).  warning_message is None
        when HyperSHAP succeeds.
        """
        import numpy as np
        from ConfigSpace import Configuration

        params = list(config_space.keys())

        data: list[tuple] = []
        for t in trials:
            try:
                cfg = Configuration(config_space, values=t.config)
                data.append((cfg, t.scores[metric_name]))
            except Exception:
                continue

        if len(data) < 2:
            uniform = 1.0 / len(params) if params else 0.0
            return (
                {p: uniform for p in params},
                "Not enough trials for importance estimation; showing uniform weights.",
            )

        try:
            task = ExplanationTask.from_data(config_space, data)
            hs = HyperSHAP(task)
            iv = hs.tunability()
            order1 = iv.get_n_order(order=1).dict_values
            raw = {params[idx]: abs(val) for (idx,), val in order1.items()}
            total = sum(raw.values()) or 1.0
            return {k: v / total for k, v in raw.items()}, None
        except Exception as e:
            warning = f"HyperSHAP failed ({e}) — falling back to surrogate feature importances."

        try:
            from sklearn.ensemble import RandomForestRegressor
            X = np.array([cfg.get_array() for cfg, _ in data])
            y = np.array([score for _, score in data])
            rf = RandomForestRegressor(n_estimators=100, random_state=seed)
            rf.fit(X, y)
            importances = rf.feature_importances_
            total = importances.sum() or 1.0
            return {p: float(importances[i] / total) for i, p in enumerate(params)}, warning
        except Exception as e2:
            warning += f" Surrogate fallback also failed ({e2}); showing uniform weights."
            uniform = 1.0 / len(params) if params else 0.0
            return {p: uniform for p in params}, warning
