from typing import Any

from smac import BlackBoxFacade, HyperparameterOptimizationFacade
from smac.initial_design import (
    DefaultInitialDesign, LatinHypercubeInitialDesign, RandomInitialDesign, SobolInitialDesign,
)

from ..base import OptimizerParam

#: Each facade bundles a surrogate, acquisition function, maximizer and encoder
#: chosen to work together.
STRATEGIES = {
    "gp": BlackBoxFacade,
    "rf": HyperparameterOptimizationFacade,
}

#: The `mcmc` fit draws kernel hyperparameters as an ensemble, which is better
#: calibrated and about ten times slower than the single likelihood fit of
#: `vanilla`.
GP_MODELS = ("vanilla", "mcmc")

#: `INITIAL_POINTS` names the form group for the two initial-design caps and the
#: choice of the binding cap, since all three set how many starting points SMAC
#: samples.
INITIAL_POINTS = "initial_points"


#: A Sobol or seeded random draw is a sequence, and a resumed run asking for
#: more points gets the old ones plus the rest. A Latin hypercube shifts every
#: point when `n` changes, so a resume under it keeps its size.
EXTENDABLE_DESIGNS = frozenset({"sobol", "random", "default_only"})

INITIAL_DESIGNS = {
    "sobol": SobolInitialDesign,
    "latin_hypercube": LatinHypercubeInitialDesign,
    "random": RandomInitialDesign,
    "default_only": DefaultInitialDesign,
}


#: `FACADE_DEFAULTS` reads each blank setting's default from the installed SMAC
#: facade signature, which keeps the export current. The `rf_` entries resolve
#: only under `rf`, and `signature_default` returns None otherwise.
FACADE_DEFAULTS = {
    "random_probability": ("get_random_design", "probability"),
    "challengers": ("get_acquisition_maximizer", "challengers"),
    "local_search_iterations": ("get_acquisition_maximizer", "local_search_iterations"),
    "retrain_after": ("get_config_selector", "retrain_after"),
    "rf_trees": ("get_model", "n_trees"),
    "rf_max_depth": ("get_model", "max_depth"),
    "rf_min_samples_split": ("get_model", "min_samples_split"),
    "rf_min_samples_leaf": ("get_model", "min_samples_leaf"),
    "rf_feature_ratio": ("get_model", "ratio_features"),
}


def signature_default(function, argument) -> Any:
    """`signature_default` looks up the default of *argument*, or None."""
    import inspect

    parameter = inspect.signature(function).parameters.get(argument)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


PARAMS_SCHEMA = [
    OptimizerParam("search_strategy", "Search strategy", "select", "gp",
                   choices=["gp", "rf"]),
    # SMAC takes the smaller of a budget share and a count, and both caps
    # can be switched off or swapped to the larger (see `_initial_points`).
    OptimizerParam("use_share_cap", "Use share cap", "bool", True,
                   group=INITIAL_POINTS),
    OptimizerParam("share_cap", "Share cap", "float", None,
                   min=0.01, max=1.0, group=INITIAL_POINTS,
                   enabled_by=("use_share_cap",)),
    OptimizerParam("use_trial_cap", "Use trial cap", "bool", True,
                   group=INITIAL_POINTS),
    OptimizerParam("trial_cap", "Trial cap", "int", None,
                   min=1, max=10_000, group=INITIAL_POINTS,
                   enabled_by=("use_trial_cap",)),
    OptimizerParam("initial_points_use_max", "Use the larger of the two",
                   "bool", False, group=INITIAL_POINTS,
                   enabled_by=("use_share_cap", "use_trial_cap")),
    OptimizerParam("random_probability", "Random trial rate",
                   "float", None, min=0.0, max=1.0),
    OptimizerParam("use_default_config", "Include the model's defaults",
                   "bool", False),

    OptimizerParam("initial_design", "Sampling method", "select",
                   "sobol", advanced=True,
                   choices=["sobol", "latin_hypercube", "random", "default_only"]),
    OptimizerParam("acquisition", "Acquisition function",
                   "select", "ei", advanced=True, choices=["ei", "pi"]),
    OptimizerParam("acquisition_xi", "Improvement margin", "float", 0.0,
                   min=0.0, max=1.0, advanced=True),
    OptimizerParam("challengers", "Candidates per trial", "int",
                   None, min=1, max=100_000, advanced=True),
    OptimizerParam("local_search_iterations", "Local search iterations", "int",
                   None, min=1, max=1_000, advanced=True),
    OptimizerParam("retrain_after", "Refit interval", "int",
                   None, min=1, max=100, advanced=True),

    # Every surrogate setting carries the `rf_` prefix and a "surrogate"
    # label, because the Random Forest model's own `max_depth` and
    # `min_samples_split` can be on screen at once.
    OptimizerParam("rf_trees", "Surrogate trees", "int", None,
                   min=2, max=1_000, advanced=True,
                   depends_on=("search_strategy", "rf")),
    OptimizerParam("rf_max_depth", "Surrogate tree depth", "int", None,
                   min=1, max=1_000, advanced=True,
                   depends_on=("search_strategy", "rf")),
    OptimizerParam("rf_min_samples_split", "Surrogate split threshold", "int",
                   None, min=2, max=100, advanced=True,
                   depends_on=("search_strategy", "rf")),
    OptimizerParam("rf_min_samples_leaf", "Surrogate leaf size", "int", None,
                   min=1, max=100, advanced=True,
                   depends_on=("search_strategy", "rf")),
    # Above 1.0 SMAC computes `max_features = 0`.
    OptimizerParam("rf_feature_ratio", "Surrogate feature ratio", "float",
                   None, min=0.05, max=1.0, advanced=True,
                   depends_on=("search_strategy", "rf")),
    OptimizerParam("rf_bootstrapping", "Bootstrap the surrogate's trees",
                   "bool", True, advanced=True,
                   depends_on=("search_strategy", "rf")),

    OptimizerParam("gp_model_type", "Surrogate fitting", "select", "vanilla",
                   choices=["vanilla", "mcmc"], advanced=True,
                   depends_on=("search_strategy", "gp")),
    OptimizerParam("gp_restarts", "Surrogate fit restarts", "int", None,
                   min=1, max=100, advanced=True,
                   depends_on=("search_strategy", "gp")),
    OptimizerParam("gp_normalize_y", "Normalise surrogate targets", "bool",
                   True, advanced=True,
                   depends_on=("search_strategy", "gp")),
]
