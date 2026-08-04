"""Figures: the analytics panels on an experiment page.

`catalog.py` declares which figures exist; `plots.py` builds the Plotly plot
for each figure that draws one. Their templates live in
`ui/templates/ui/figures/`, one per figure, named for its key.
"""

from .base import FULL, HALF, Figure
from .catalog import FIGURES, FIGURES_BY_KEY
from .plots import (
    error_over_time_plot,
    hyperparameter_importance_plot,
    incumbent_performance_plot,
    incumbent_scores,
    trial_duration_plot,
)

__all__ = [
    "FIGURES",
    "FIGURES_BY_KEY",
    "FULL",
    "HALF",
    "Figure",
    "error_over_time_plot",
    "hyperparameter_importance_plot",
    "incumbent_performance_plot",
    "incumbent_scores",
    "trial_duration_plot",
]
