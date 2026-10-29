"""`catalog.py` declares the figures, and `plots` builds their Plotly plots.
"""

from .base import (
    ACQUISITION, CARD, DASHBOARD, DATA, HYPERSHAP, LONG, MISC, OVERVIEW, SHAPES, SQUAT, STACK,
    TABLE, TABS, Figure, Panel, autocompute_key, deferred_computations,
)
from .catalog import (
    FIGURES, FIGURES_BY_KEY, HP_GAME_FIELDS, HP_GAME_HELP, HP_GAME_LABELS,
)
from .plots import (
    ACCENT_COLOR,
    PRIOR_COLOR,
    MARKER_COLOR,
    NEGATIVE_COLOR,
    PROPOSAL_COLOR,
    SELECTION_COLOR,
    UNCERTAINTY_SCALE,
    configuration_cube_plot,
    configuration_projection_plot,
    acquisition_slice_plots,
    hyperparameter_ablation_plot,
    hyperparameter_importance_plot,
    hyperparameter_interactions_bar_plot,
    hyperparameter_interactions_heatmap_plot,
    hyperparameter_graph_plot,
    hyperparameter_orders_plot,
    hyperparameter_progress_plot,
    hyperparameter_upset_plot,
    incumbent_scores,
    local_effects_plot,
    parallel_coordinates_plot,
    forecast_plot,
    partial_dependence_plot,
    slice_span,
    performance_over_time_plot,
    trial_duration_plot,
)
