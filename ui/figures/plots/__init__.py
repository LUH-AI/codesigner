"""`plots` defines one Plotly builder per drawn figure, named for its `catalog.py`
figure, such as `performance_over_time_plot`. The builders are pure functions of
an OptimizationResult and a metric that return a Figure or None, with no request
state. A per-metric builder returns None when any trial lacks a score for the
metric (see `OptimizationResult.has_every_score`).
"""

from .style import (
    MARKER_COLOR, SELECTION_COLOR, ACCENT_COLOR, NEGATIVE_COLOR, PROPOSAL_COLOR,
    UNCERTAINTY_SCALE, FAILURE_SYMBOL, FAILURE_LINE_WIDTH, FAILURE_LINE_WIDTH_RINGED,
    FAILURE_SIZE, FAILURE_FILL, RECOLOR, OUTLINE, LINE, OVERLAY,
)
from .performance import (
    incumbent_scores, performance_over_time_plot, trial_duration_plot, forecast_plot,
)
from .importance import (
    hyperparameter_importance_plot, hyperparameter_progress_plot,
    hyperparameter_ablation_plot,
)
from .interactions import (
    hyperparameter_interactions_heatmap_plot, hyperparameter_interactions_bar_plot,
    hyperparameter_graph_plot, hyperparameter_upset_plot, hyperparameter_orders_plot,
)
from .space import (
    configuration_cube_plot, configuration_projection_plot, parallel_coordinates_plot,
)
from .surrogate import (
    partial_dependence_plot, PRIOR_COLOR, SIGMA_STEPS, FICTION_SIGMA, PRIOR_AXIS_TOP,
    slice_span, acquisition_slice_plots, local_effects_plot,
)
