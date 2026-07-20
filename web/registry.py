"""The models, optimizers, and metrics the application offers.

These are the choices presented in forms and used to reconstruct experiments
from a snapshot. Keys are the human-readable names stored in .ihpo files.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from core.models import RandomForestModel, SVMModel
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

MODELS = {
    "Random Forest": RandomForestModel(),
    "SVM Classifier": SVMModel(),
}

OPTIMIZERS = {
    "SMAC": SMACOptimizer(),
    "Random Search": RandomOptimizer(),
    "Grid Search": GridOptimizer(),
}

METRICS = {
    "accuracy":      lambda y, yp: accuracy_score(y, yp),
    "f1":            lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
    "precision":     lambda y, yp: precision_score(y, yp, average="weighted", zero_division=0),
    "recall(macro)": lambda y, yp: recall_score(y, yp, average="macro", zero_division=0),
}
