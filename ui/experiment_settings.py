"""`SETTING_DEFAULTS` adds a `show_<figure>` and an `autocompute_<name>` flag
per catalog entry to the run's settings. A figure that is on without autocompute
shows a Compute button, which is the default.
"""

from experiments.services import settings as run_settings

from .figures import FIGURES, autocompute_key, deferred_computations

SETTING_DEFAULTS = {
    **run_settings.SETTING_DEFAULTS,
    **{figure.setting_key: True for figure in FIGURES},
    **{autocompute_key(name): False for name, _label in deferred_computations()},
    # The incumbent's HyperSHAP games and the cube's acquisition sample are
    # drawn on page open anyway.
    autocompute_key("hypershap_baseline"): True,
    autocompute_key("acquisition_slice"): True,
}


def global_defaults() -> dict:
    return run_settings.global_defaults(SETTING_DEFAULTS)


def resolve_settings(exp) -> dict:
    return run_settings.resolve_settings(exp, SETTING_DEFAULTS)
