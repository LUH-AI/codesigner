"""`core.analytics` reads a search's trials through surrogates, the HyperSHAP
games, and slices and samples of the space."""

from .surrogate import pair_trials_with_scores, fit_surrogate, predict_mean_std
from .slices import (
    PREDICTION_SAMPLES, IncumbentSlice, sample_prediction_space, slice_axis,
    compute_incumbent_slice,
)
from .effects import (
    compute_hp_ablation, compute_local_effects, compute_partial_dependence,
    compute_surrogate_uncertainty,
)
from .games import (
    HP_GAMES, EAGER_MAX_COALITIONS, eager_analytics_budget_exceeded, compute_hp_games,
    compute_hp_game,
)
from .progress import SETTLED, tuning_progress
