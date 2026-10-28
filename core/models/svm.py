"""`SVM` wraps scikit-learn's SVMs."""

from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from codesigner_model import OutputBaseModel

from .base import Tunable


class SVM(OutputBaseModel):
    """`build` wraps `SVC` in five-fold sigmoid calibration only when
    ``data["probabilities"]`` is set, since calibrating costs several fits.
    """

    name = "SVM"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn",)
    accepts_missing = False
    scale_sensitive = True
    costly_probabilities = True

    def build(self, hyperparameters, data, seed=0):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC, SVR

        shared: dict[str, Any] = dict(
            C=float(hyperparameters["C"]),
            kernel=hyperparameters["kernel"],
            gamma=hyperparameters["gamma"],
            degree=int(hyperparameters["degree"]),
            tol=float(hyperparameters["tol"]),
            max_iter=int(hyperparameters["max_iter"]),
        )
        if self.task == "regression":
            svm = SVR(epsilon=float(hyperparameters.get("epsilon", 0.1)), **shared)
        else:
            svm = SVC(random_state=seed, **shared)
            if data.get("probabilities"):
                svm = CalibratedClassifierCV(svm, ensemble=False)  # pyright: ignore[reportArgumentType]
        return svm


class SVMModel(Tunable, SVM):

    #: The name in `.ihpo` files from before SVM regression.
    aliases = ("SVM Classifier",)

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Float(      "C",        (0.01, 100.0), default=1.0,  log=True),
            Categorical("kernel",   ["rbf", "linear", "poly", "sigmoid"], default="rbf"),
            Categorical("gamma",    ["scale", "auto"],            default="scale"),
            Integer(    "degree",   (2, 5),         default=3),
            Float(      "tol",      (1e-5, 1e-1),   default=1e-3, log=True),
            Integer(    "max_iter", (200, 5000),    default=1000),
        ])
        if self.task == "regression":
            cs.add(Float("epsilon", (1e-3, 1.0), default=0.1, log=True))
        return cs
