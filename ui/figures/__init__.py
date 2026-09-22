"""Figures: the analytics panels on an experiment page.

`catalog.py` declares which figures exist; `plots.py` builds the Plotly plot
for each figure that draws one. Their templates live in
`ui/templates/ui/figures/`, one per figure, named for its key.
"""

from .base import (
    DOUBLE, FULL, HALF, SINGLE, Figure, autocompute_key, deferred_computations,
)
from .catalog import (
    FIGURES, FIGURES_BY_KEY, HP_GAME_FIELDS, HP_GAME_HELP, HP_GAME_LABELS,
)
from .plots import (
    BELIEF_COLOR,
    MARKER_COLOR,
    NEGATIVE_COLOR,
    SELECTION_COLOR,
    UNCERTAINTY_SCALE,
    configuration_cube_plot,
    configuration_projection_plot,
    acquisition_slice_plot,
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
    partial_dependence_plot,
    performance_over_time_plot,
    trial_duration_plot,
)

__all__ = [
    "FIGURES",
    "BELIEF_COLOR",
    "MARKER_COLOR",
    "NEGATIVE_COLOR",
    "SELECTION_COLOR",
    "UNCERTAINTY_SCALE",
    "FIGURES_BY_KEY",
    "DOUBLE",
    "FULL",
    "HALF",
    "SINGLE",
    "HP_GAME_FIELDS",
    "HP_GAME_HELP",
    "HP_GAME_LABELS",
    "Figure",
    "autocompute_key",
    "deferred_computations",
    "configuration_cube_plot",
    "configuration_projection_plot",
    "hyperparameter_ablation_plot",
    "hyperparameter_importance_plot",
    "hyperparameter_interactions_bar_plot",
    "hyperparameter_interactions_heatmap_plot",
    "hyperparameter_graph_plot",
    "hyperparameter_orders_plot",
    "hyperparameter_progress_plot",
    "hyperparameter_upset_plot",
    "incumbent_scores",
    "local_effects_plot",
    "parallel_coordinates_plot",
    "acquisition_slice_plot",
    "partial_dependence_plot",
    "performance_over_time_plot",
    "trial_duration_plot",
]
