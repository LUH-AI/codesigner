"""Effective settings resolve from per-experiment overrides over global defaults.

An experiment either inherits the global default experiment settings
(`use_default_settings=True`) or overrides them (its `settings` dict wins, per
key, falling back to the global default). The global default itself falls back
to the built-in `SETTING_DEFAULTS`.
"""

from ui.models import Experiment, GlobalSettings
from ui.services.settings import SETTING_DEFAULTS, resolve_settings


def _exp(**kw):
    base = dict(name="s", model_name="Random Forest", optimizer_name="Random Search",
                metric_names=["accuracy"], seed=0)
    base.update(kw)
    return Experiment.objects.create(**base)


def test_fresh_experiment_uses_builtin_defaults():
    exp = _exp()
    assert exp.use_default_settings is True
    assert resolve_settings(exp) == SETTING_DEFAULTS
    assert resolve_settings(exp)["export_absolute_times"] is True


def test_inherits_changed_global_default():
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"export_absolute_times": False}
    gs.save()
    exp = _exp()  # inherits
    assert resolve_settings(exp)["export_absolute_times"] is False


def test_per_experiment_override_wins_over_global():
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"export_absolute_times": True}
    gs.save()
    exp = _exp(use_default_settings=False, settings={"export_absolute_times": False})
    assert resolve_settings(exp)["export_absolute_times"] is False


def test_override_missing_key_falls_back_to_global():
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"export_absolute_times": False}
    gs.save()
    exp = _exp(use_default_settings=False, settings={})  # overriding, but empty
    assert resolve_settings(exp)["export_absolute_times"] is False


def test_global_settings_is_a_singleton():
    a = GlobalSettings.get_solo()
    b = GlobalSettings.get_solo()
    assert a.pk == b.pk
    assert GlobalSettings.objects.count() == 1
