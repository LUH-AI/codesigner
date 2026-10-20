"""Effective per-experiment settings, resolving overrides over global defaults.

Precedence per key: an experiment's own `settings` (when it isn't inheriting)
→ the global default experiment settings → the built-in `SETTING_DEFAULTS`.
"""

from experiments.models import GlobalSettings

#: The figures that *display* the eager HyperSHAP games. With every one of them
#: switched off there is nothing to compute them for — parallel coordinates and
#: partial dependence also read them, but only to order axes and pick a default
#: hyperparameter, and both already fall back cleanly when they are empty.
GAME_DISPLAY_FIGURES = (
    "hyperparameter_importance",
    "interactions_heatmap", "interactions_top_pairs", "interactions_graph",
    "interactions_coalitions", "interactions_orders",
)

# The experiment-settings schema and its built-in fallbacks:
#   ice_max_curves        — how many trials the PDP figure predicts and draws
#   local_effects_max_trials — how many trials the beeswarm explains
#   show_<figure>         — only the game figures here, which a run reads; the
#                           rest, and autocompute_<name>, are ui/experiment_settings.py's
SETTING_DEFAULTS = {
    "ice_max_curves": 100,
    "local_effects_max_trials": 100,
    # How much worse than where the search is already looking a stated belief's
    # region may score before "Evaluate prior" refuses it. DynaBO's safeguard
    # states this as a negative threshold; here it is the positive slack, which
    # is the direction it reads as a preference — larger is more forgiving, and
    # 0 refuses anything the model does not think is an improvement.
    #
    # A share of the metric's scale (`experiments.services.priors.metric_scale`), so it means the
    # same for every objective. For a metric on [0, 1] that is DynaBO's own
    # threshold, which is where the default comes from.
    "prior_acceptance_tolerance": 0.15,
    # How many of the best configurations each side of "Evaluate prior" climbs
    # from and is averaged over — SMAC's `ClimbingComparisonPolicy.top_k`, and
    # its default. Fewer lets one lucky climb decide; more dilutes the best
    # with the merely good.
    "prior_acceptance_top_k": 10,
    # How many trials' fitted models a run keeps for "Export trial
    # parameters": the best this many by the run's metric — and, with the
    # other, every trial of the last run, until the next. See
    # experiments/services/trial_models.py.
    "keep_best_trial_models": 5,
    "keep_last_run_trial_models": False,
    # How an exported file carries this experiment's times: "absolute",
    # "relative" or "none" — or "" for whatever the person exporting prefers.
    # See experiments/services/timestamps.py.
    "export_timestamps": "",
    **{f"show_{key}": True for key in GAME_DISPLAY_FIGURES},
}

#: Bounds for the settings that are numbers rather than checkboxes, as
#: (minimum, maximum). A value outside them is clamped rather than refused: this
#: is a display preference typed into a box, not an instruction worth failing a
#: form over. 0 is the floor everywhere and reads as "no limit".
SETTING_BOUNDS = {
    "ice_max_curves": (0, 10_000),
    "local_effects_max_trials": (0, 10_000),
    # A share of the metric's scale. Zero means "accept nothing the model scores
    # below the incumbent's neighbourhood", one accepts all but the worthless.
    "prior_acceptance_tolerance": (0.0, 1.0),
    # At least one climb, or there is nothing to compare.
    "prior_acceptance_top_k": (1, 100),
    "keep_best_trial_models": (0, 100),
}

#: Settings that are one of a few words rather than a number or a checkbox,
#: and the words they may be. Anything else posted is the default.
SETTING_CHOICES = {
    "export_timestamps": ("", "absolute", "relative", "none"),
}


def eager_analytics_wanted(settings: dict) -> bool:
    """Whether anything on the page will show the eager HyperSHAP games.

    Read by `run.py` before a run, and handed to the optimizer as
    `analytics_wanted`.
    """
    return any(settings.get(f"show_{key}") for key in GAME_DISPLAY_FIGURES)


def global_defaults(schema=SETTING_DEFAULTS) -> dict:
    """The global default experiment settings, backfilled with *schema*."""
    stored = GlobalSettings.get_solo().default_experiment_settings
    return {key: stored.get(key, default) for key, default in schema.items()}


def resolve_settings(exp, schema=SETTING_DEFAULTS) -> dict:
    """The effective settings for *exp* — its overrides, or the global defaults."""
    defaults = global_defaults(schema)
    if exp.use_default_settings:
        return defaults
    return {key: exp.settings.get(key, defaults[key]) for key in schema}
