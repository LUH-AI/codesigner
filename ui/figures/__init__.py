"""Figures: the analytics panels on an experiment page.

`catalog.py` declares which figures exist; `plots.py` builds the Plotly plot
for each figure that draws one. Their templates live in
`ui/templates/ui/figures/`, one per figure, named for its key.
"""

from .base import FULL, HALF, Figure
from .catalog import FIGURES, FIGURES_BY_KEY, HP_GAME_FIELDS
from .plots import (
    configuration_cube_plot,
    hyperparameter_ablation_plot,
    hyperparameter_importance_plot,
    hyperparameter_interactions_bar_plot,
    hyperparameter_interactions_heatmap_plot,
    incumbent_scores,
    parallel_coordinates_plot,
    performance_over_time_plot,
    trial_duration_plot,
)

__all__ = [
    "FIGURES",
    "FIGURES_BY_KEY",
    "FULL",
    "HALF",
    "HP_GAME_FIELDS",
    "Figure",
    "configuration_cube_plot",
    "hyperparameter_ablation_plot",
    "hyperparameter_importance_plot",
    "hyperparameter_interactions_bar_plot",
    "hyperparameter_interactions_heatmap_plot",
    "incumbent_scores",
    "parallel_coordinates_plot",
    "performance_over_time_plot",
    "trial_duration_plot",
]
