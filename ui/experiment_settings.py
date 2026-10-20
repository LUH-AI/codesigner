"""The experiment-settings schema as the pages see it: the run's own settings
(`experiments/services/settings.py`) and a switch per figure.

- `show_<figure>`: draw that figure on an experiment page
- `autocompute_<name>`: compute that deferred analytic as soon as the figure
  needs it, rather than waiting to be asked

Both are derived from the catalog rather than listed here, so declaring a
figure — or declaring that a figure defers something — is all it takes to give
it a setting.

Together the flags cover three states without a tri-state widget: figure off =
never computed; on + autocompute = computed the moment the figure needs it;
on + no autocompute = a "Compute" button, and a page reload computes nothing.
The last of those is the default, so opening an experiment costs nothing until
it is asked to.
"""

from experiments.services import settings as run_settings

from .figures import FIGURES, autocompute_key, deferred_computations

SETTING_DEFAULTS = {
    **run_settings.SETTING_DEFAULTS,
    **{figure.setting_key: True for figure in FIGURES},
    **{autocompute_key(name): False for name, _label in deferred_computations()},
    # The exceptions. Measuring the HyperSHAP games from the incumbent is what
    # the page opens on, so waiting to be asked would open every experiment on
    # six blank figures. And the acquisition function's sample is the one the
    # configuration cube opens its field on, so it is drawn anyway.
    autocompute_key("hypershap_baseline"): True,
    autocompute_key("acquisition_slice"): True,
}


def global_defaults() -> dict:
    return run_settings.global_defaults(SETTING_DEFAULTS)


def resolve_settings(exp) -> dict:
    return run_settings.resolve_settings(exp, SETTING_DEFAULTS)
