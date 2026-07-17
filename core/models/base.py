from abc import ABC, abstractmethod

from ConfigSpace import ConfigurationSpace


class BaseModel(ABC):
    """Interface for classifier models used in HPO experiments."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name shown in the UI and stored in .ihpo files."""
        ...

    @abstractmethod
    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        """Return the hyperparameter search space for this model."""
        ...

    @abstractmethod
    def train_evaluate(
        self,
        config: dict,
        X_train, y_train,
        X_val, y_val,
        metrics: dict,   # {metric_name: callable}
        seed: int = 0,
    ) -> dict:           # {metric_name: score}
        """Train on the given split with config and return scores for all metrics."""
        ...
