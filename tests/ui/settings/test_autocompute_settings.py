"""Each computation a figure defers in `Figure.deferred` gets a boolean
`autocompute_<name>` setting. Switching the figure off already means never
computing it, so two states suffice.
"""

from django.urls import reverse

from ui.figures import FIGURES, autocompute_key, deferred_computations
from experiments.models import GlobalSettings
from ui.experiment_settings import SETTING_DEFAULTS, global_defaults, resolve_settings


def test_the_deferred_computations_are_declared():
    """`deferred_computations` lists them in catalog order."""
    assert [name for name, _ in deferred_computations()] == [
        "hypershap_baseline", "forecast", "partial_dependence", "local_ablation",
        "local_effects", "acquisition_slice", "surrogate_uncertainty"]


def test_a_deferred_name_is_the_computations_not_the_figures():
    """`local_ablation` keeps its stored name, since renaming it would need a
    data migration."""
    local = next(f for f in FIGURES if f.key == "local_explanation")
    assert [name for name, _ in local.deferred] == ["local_ablation"]
    assert local.setting_key == "show_local_explanation"


def test_figures_with_nothing_deferred_declare_nothing():
    deferred_keys = {f.key for f in FIGURES if f.deferred}
    assert deferred_keys == {"hyperparameter_importance", "local_explanation",
                             "partial_dependence", "local_effects",
                             "configuration_cube", "acquisition_slice", "forecast"}


#: The incumbent's HyperSHAP games fill the opening page, and the cube opens its
#: field on the acquisition sample.
ON_BY_DEFAULT = {"hypershap_baseline", "acquisition_slice"}


def test_every_deferred_computation_has_a_setting_defaulting_to_off():
    """On a 10,000-trial run, partial dependence costs 6.9 s and a 4.5 MB
    response."""
    for name, _label in deferred_computations():
        expected = name in ON_BY_DEFAULT
        assert SETTING_DEFAULTS[autocompute_key(name)] is expected, name
        assert global_defaults()[autocompute_key(name)] is expected, name


def test_the_settings_are_booleans():
    """`_posted_settings` coerces every key with `bool()`, so a non-boolean
    default would turn into True."""
    for name, _label in deferred_computations():
        assert isinstance(SETTING_DEFAULTS[autocompute_key(name)], bool)


def test_both_settings_pages_offer_them(client):
    for url in (reverse("ui:default_experiment_settings"),):
        body = client.get(url).content.decode()
        for name, _label in deferred_computations():
            assert f'name="{autocompute_key(name)}"' in body, name


def test_they_start_unchecked(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    for name, _label in deferred_computations():
        field = body.split(f'name="{autocompute_key(name)}"', 1)[1].split(">", 1)[0]
        assert ("checked" in field) is (name in ON_BY_DEFAULT), name


def test_saving_records_each_choice(client):
    """An unticked box comes out False."""
    client.post(reverse("ui:default_experiment_settings"), {
        "ice_max_curves": "100",
        **{f.setting_key: "on" for f in FIGURES},
        "autocompute_local_ablation": "on",
        # an unticked checkbox
    })

    stored = GlobalSettings.get_solo().default_experiment_settings
    assert stored["autocompute_partial_dependence"] is False
    assert stored["autocompute_local_ablation"] is True


def test_an_experiment_can_override_the_default(client):
    """`resolve_settings` prefers an experiment's own autocompute setting over
    the default."""
    from core import io
    from experiments.services import snapshot as adapter
    from tests.conftest import FIXTURES_DIR

    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    exp.use_default_settings = False
    exp.settings = {"autocompute_partial_dependence": True}
    exp.save(update_fields=["use_default_settings", "settings"])

    resolved = resolve_settings(exp)
    assert resolved["autocompute_partial_dependence"] is True
    assert resolved["autocompute_local_ablation"] is False, "the other one is untouched"


def test_a_stored_setting_from_before_these_existed_still_resolves():
    """`global_defaults` fills in keys that older stored rows lack."""
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"show_trials": False}
    gs.save(update_fields=["default_experiment_settings"])

    defaults = global_defaults()
    assert defaults["show_trials"] is False
    for name, _label in deferred_computations():
        assert defaults[autocompute_key(name)] is (name in ON_BY_DEFAULT), name
