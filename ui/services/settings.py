"""Effective per-experiment settings, resolving overrides over global defaults.

Precedence per key: an experiment's own `settings` (when it isn't inheriting)
→ the global default experiment settings → the built-in `SETTING_DEFAULTS`.
"""

from ..figures import FIGURES
from ..models import GlobalSettings

# The experiment-settings schema and its built-in fallbacks:
#   export_absolute_times — keep absolute timestamps when exporting an .ihpo
#   show_<figure>          — draw that figure on an experiment page
# The figure flags come from the catalog rather than being listed here, so
# declaring a figure is all it takes to give it a setting.
SETTING_DEFAULTS = {
    "export_absolute_times": True,
    **{figure.setting_key: True for figure in FIGURES},
}


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
