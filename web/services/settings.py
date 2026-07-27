"""Effective per-experiment settings, resolving overrides over global defaults.

Precedence per key: an experiment's own `settings` (when it isn't inheriting)
→ the global default experiment settings → the built-in `SETTING_DEFAULTS`.
"""

from ..models import GlobalSettings

# The experiment-settings schema and its built-in fallbacks. The first (and
# for now only) setting controls whether absolute timestamps are kept when an
# .ihpo is exported.
SETTING_DEFAULTS = {
    "export_absolute_times": True,
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
