# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn", "ConfigSpace"]
# ///
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.svm import SVC, SVR

from .base import BaseModel


class SVMModel(BaseModel):
    """scikit-learn's support vector machine: `SVC` for classification, `SVR`
    for regression, which also tunes the width of its insensitive tube."""

    name = "SVM"
    #: What it was called while it could only classify, which every `.ihpo`
    #: written then still says.
    aliases = ("SVM Classifier",)
    tasks = ("classification", "regression")

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

    def fit(self, config, X_train, y_train, seed: int = 0, *, probability: bool = False):
        """The SVM *config* describes, trained on all of ``(X_train, y_train)``."""
        shared = dict(
            C=float(config["C"]),
            kernel=config["kernel"],
            gamma=config["gamma"],
            degree=int(config["degree"]),
            tol=float(config["tol"]),
            max_iter=int(config["max_iter"]),
        )
        if self.task == "regression":
            clf = SVR(epsilon=float(config.get("epsilon", 0.1)), **shared)
        else:
            clf = SVC(random_state=seed, probability=probability, **shared)
        clf.fit(X_train, y_train)
        return clf

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        return self.fit(config, X_train, y_train, seed,
                        probability=False).predict(X_val)

    def fit_predict_proba(self, config, X_train, y_train, X_val, seed: int = 0):
        """An SVM has no probabilities of its own — `probability=True` fits an
        internal five-fold Platt calibration on top, which costs several times
        what the plain fit costs.

        That price is exactly why this is a separate method rather than a flag
        on `fit_predict`: a run that never asks for a probability metric never
        pays it, and the two methods are free to fit differently because they
        are answering different questions.

        A consequence worth knowing: `predict` on a calibrated SVC can disagree
        with `argmax` of its own `predict_proba`, so the labels here are the
        calibrated model's, not the plain one's. They are the labels that go
        with these probabilities, which is what a trial scoring both needs.
        """
        if self.task == "regression":
            raise NotImplementedError("a support vector regression has no class probabilities")
        clf = self.fit(config, X_train, y_train, seed, probability=True)
        return clf.predict(X_val), clf.predict_proba(X_val), clf.classes_
