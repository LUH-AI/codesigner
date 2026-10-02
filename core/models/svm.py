"""Support vector machine: scikit-learn's, classifying or regressing."""

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from .base import OutputBaseModel, Tunable


class SVM(OutputBaseModel):
    """scikit-learn's support vector machine: `SVC` for classification, `SVR`
    for regression. It refuses gaps and measures distances, so its columns are
    filled in and scaled first.

    An SVM has no probabilities of its own: asked for them, it fits an internal
    five-fold Platt calibration on top, which costs several times what the
    plain fit costs, so `build` adds it only when ``data["probabilities"]``
    says they will be asked for. A calibrated SVC's labels can disagree with
    the argmax of its own probabilities; they are the labels that go with
    those probabilities, which is what a trial scoring both needs.
    """

    name = "SVM"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn",)
    accepts_missing = False
    scale_sensitive = True

    def build(self, hyperparameters, data, seed=0):
        from sklearn.svm import SVC, SVR

        shared = dict(
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
            svm = SVC(random_state=seed, probability=bool(data.get("probabilities")), **shared)
        return svm


class SVMModel(Tunable, SVM):
    """The SVM, tuned over its kernel and penalties; for a regression also the
    width of its insensitive tube."""

    #: What it was called while it could only classify, which every `.ihpo`
    #: written then still says.
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
