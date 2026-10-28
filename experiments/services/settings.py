"""`resolve_settings` takes an experiment's own `settings` unless it inherits,
then the global defaults, then `SETTING_DEFAULTS`.
"""

from experiments.models import GlobalSettings

#: `GAME_DISPLAY_FIGURES` display the eager HyperSHAP games, and a run computes
#: the games only when one of them is on.
GAME_DISPLAY_FIGURES = (
    "hyperparameter_importance",
    "interactions_heatmap", "interactions_top_pairs", "interactions_graph",
    "interactions_coalitions", "interactions_orders",
)

# Built-in fallbacks; other display keys live in ui/experiment_settings.py.
SETTING_DEFAULTS = {
    "ice_max_curves": 100,
    "local_effects_max_trials": 100,
    # `prior_acceptance_tolerance` is how far, as a share of
    # `priors.metric_scale`, a stated belief's region may score below the
    # searched region before "Evaluate prior" refuses it. The default is
    # DynaBO's own threshold on a [0, 1] metric.
    "prior_acceptance_tolerance": 0.15,
    # Climbs averaged per side; SMAC's `ClimbingComparisonPolicy.top_k` default.
    "prior_acceptance_top_k": 10,
    # Best-N trial models kept for export; see trial_models.py.
    "keep_best_trial_models": 5,
    "keep_last_run_trial_models": False,
    # Exported time format; "" defers to the exporter's preference.
    "export_timestamps": "",
    **{f"show_{key}": True for key in GAME_DISPLAY_FIGURES},
}

#: Out-of-range input is clamped instead of failing the form.
SETTING_BOUNDS = {
    "ice_max_curves": (0, 10_000),
    "local_effects_max_trials": (0, 10_000),
    # Zero accepts nothing below the incumbent's neighbourhood, and one accepts
    # all but the worthless.
    "prior_acceptance_tolerance": (0.0, 1.0),
    "prior_acceptance_top_k": (1, 100),
    "keep_best_trial_models": (0, 100),
}

#: `SETTING_CHOICES` lists word-valued settings; any other posted value becomes
#: the default.
SETTING_CHOICES = {
    "export_timestamps": ("", "absolute", "relative", "none"),
}


def eager_analytics_wanted(settings: dict) -> bool:
    """`eager_analytics_wanted` feeds the optimizer's `analytics_wanted` flag
    before a run (`run.py`).
    """
    return any(settings.get(f"show_{key}") for key in GAME_DISPLAY_FIGURES)


def global_defaults(schema=SETTING_DEFAULTS) -> dict:
    stored = GlobalSettings.get_solo().default_experiment_settings
    return {key: stored.get(key, default) for key, default in schema.items()}


def resolve_settings(exp, schema=SETTING_DEFAULTS) -> dict:
    defaults = global_defaults(schema)
    if exp.use_default_settings:
        return defaults
    return {key: exp.settings.get(key, defaults[key]) for key in schema}
