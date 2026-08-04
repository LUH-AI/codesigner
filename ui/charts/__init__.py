"""Charts: the analytics panels on an experiment page.

`catalog.py` declares which charts exist; `figures.py` builds the Plotly
figures for the ones that draw a figure. Their templates live in
`ui/templates/ui/charts/`, one per chart, named for its key.
"""

from .catalog import CHARTS, CHARTS_BY_KEY, Chart
from .figures import (
    error_over_time_figure,
    hyperparameter_importance_figure,
    incumbent_performance_figure,
    incumbent_scores,
    trial_duration_figure,
)

__all__ = [
    "CHARTS",
    "CHARTS_BY_KEY",
    "Chart",
    "error_over_time_figure",
    "hyperparameter_importance_figure",
    "incumbent_performance_figure",
    "incumbent_scores",
    "trial_duration_figure",
]
