"""The models, optimizers and metrics the application offers.

Keys are the human-readable names stored in `.ihpo` files, which is what makes
this the thing `io.build_experiment` resolves a snapshot against.

Lives in `core` rather than `ui` because everything in it is a core object and
none of it needs the web layer — and because a run executed somewhere else
entirely has to resolve the same names with no Django installation to do it in
(see `cluster/run_experiment.py`). Duplicating the three dicts there instead
would work until somebody added a model, at which point the cluster would refuse
that model alone, and say only that it "is not available".

`ui.registry` re-exports this, so the application's own vocabulary is unchanged.
"""

from .metrics import METRICS
from .models import RandomForestModel, SVMModel
from .optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

MODELS = {
    "Random Forest": RandomForestModel(),
    "SVM Classifier": SVMModel(),
}

OPTIMIZERS = {
    "SMAC": SMACOptimizer(),
    "Random Search": RandomOptimizer(),
    "Grid Search": GridOptimizer(),
}

__all__ = ["MODELS", "OPTIMIZERS", "METRICS"]
