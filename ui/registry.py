"""The models, optimizers, and metrics the application offers.

These are the choices presented in forms and used to reconstruct experiments
from a snapshot. Keys are the human-readable names stored in .ihpo files.
"""

from core.metrics import METRICS
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

# Re-exported: the metrics belong to the domain layer now that the optimizers
# do the scoring, but the application's vocabulary keeps them here alongside
# the models and optimizers it offers.
__all__ = ["MODELS", "OPTIMIZERS", "METRICS"]
