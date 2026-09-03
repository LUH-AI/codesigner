"""Effective per-experiment settings, resolving overrides over global defaults.

Precedence per key: an experiment's own `settings` (when it isn't inheriting)
→ the global default experiment settings → the built-in `SETTING_DEFAULTS`.
"""

from ..figures import FIGURES, autocompute_key, deferred_computations
from ..models import GlobalSettings

# The experiment-settings schema and its built-in fallbacks:
#   ice_max_curves        — how many trials the PDP figure predicts and draws
#   local_effects_max_trials — how many trials the beeswarm explains
#   show_<figure>         — draw that figure on an experiment page
#   autocompute_<name>    — compute that deferred analytic as soon as the figure
#                           needs it, rather than waiting to be asked
# The last two are derived from the catalog rather than listed here, so
# declaring a figure — or declaring that a figure defers something — is all it
# takes to give it a setting.
#
# Together the flags cover three states without a tri-state widget: figure off =
# never computed; on + autocompute = computed the moment the figure needs it;
# on + no autocompute = a "Compute" button, and a page reload computes nothing.
# The last of those is the default, so opening an experiment costs nothing until
# it is asked to.
#
# "Never computed" now covers the eager games too, not just the deferred ones:
# with both figures that display them switched off, `ui/services/run.py` tells
# the optimizer not to bother — see BaseOptimizer.analytics_wanted.
SETTING_DEFAULTS = {
    "ice_max_curves": 100,
    "local_effects_max_trials": 100,
    **{figure.setting_key: True for figure in FIGURES},
    **{autocompute_key(name): False for name, _label in deferred_computations()},
}

#: Bounds for the settings that are numbers rather than checkboxes, as
#: (minimum, maximum). A value outside them is clamped rather than refused: this
#: is a display preference typed into a box, not an instruction worth failing a
#: form over. 0 is the floor everywhere and reads as "no limit".
SETTING_BOUNDS = {
    "ice_max_curves": (0, 10_000),
    "local_effects_max_trials": (0, 10_000),
}

#: The figures that *display* the eager HyperSHAP games. With every one of them
#: switched off there is nothing to compute them for — parallel coordinates and
#: partial dependence also read them, but only to order axes and pick a default
#: hyperparameter, and both already fall back cleanly when they are empty.
GAME_DISPLAY_FIGURES = (
    "hyperparameter_importance",
    "interactions_heatmap", "interactions_top_pairs", "interactions_graph",
    "interactions_coalitions", "interactions_orders",
)


def eager_analytics_wanted(settings: dict) -> bool:
    """Whether anything on the page will show the eager HyperSHAP games.

    Read by `ui/services/run.py` before a run, and handed to the optimizer as
    `analytics_wanted`.
    """
    return any(settings.get(f"show_{key}") for key in GAME_DISPLAY_FIGURES)


def global_defaults() -> dict:
    """The global default experiment settings, backfilled with SETTING_DEFAULTS."""
    stored = GlobalSettings.get_solo().default_experiment_settings
    return {key: stored.get(key, default) for key, default in SETTING_DEFAULTS.items()}


def resolve_settings(exp) -> dict:
    """The effective settings for *exp* — its overrides, or the global defaults."""
    defaults = global_defaults()
    if exp.use_default_settings:
        return defaults
    return {key: exp.settings.get(key, defaults[key]) for key in SETTING_DEFAULTS}
