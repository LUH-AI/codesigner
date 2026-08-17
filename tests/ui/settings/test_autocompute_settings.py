"""Per-computation control over the analytics that are worked out on request.

Two figures don't ship their numbers with the page, because those numbers depend
on a choice rather than only on the metric: the importance figure's "Local" game
(which trial) and partial dependence (which hyperparameter). Each declares itself
in `Figure.deferred` and gets an `autocompute_<name>` setting.

Booleans, not a three-way choice. "Never compute this" is already expressible by
switching the figure off, so two states is the honest shape — which also means
`_posted_settings`' blanket `bool()` needs no special case.

The keys name the *computation*, not the figure, because local ablation is a view
*of* the importance figure: `autocompute_hyperparameter_importance` would claim
the importance numbers are deferred, when they are computed once at run completion
and stored.
"""

from django.urls import reverse

from ui.figures import FIGURES, autocompute_key, deferred_computations
from ui.models import GlobalSettings
from ui.services.settings import SETTING_DEFAULTS, global_defaults, resolve_settings


def test_the_two_deferred_computations_are_declared():
    assert [name for name, _ in deferred_computations()] == [
        "local_ablation", "partial_dependence"]


def test_a_deferred_name_is_the_computations_not_always_the_figures():
    """local_ablation belongs to the importance figure, whose other nine views are
    precomputed — so the setting cannot be named after the figure."""
    importance = next(f for f in FIGURES if f.key == "hyperparameter_importance")
    assert [name for name, _ in importance.deferred] == ["local_ablation"]
    assert importance.setting_key == "show_hyperparameter_importance"


def test_figures_with_nothing_deferred_declare_nothing():
    deferred_keys = {f.key for f in FIGURES if f.deferred}
    assert deferred_keys == {"hyperparameter_importance", "partial_dependence"}


def test_every_deferred_computation_has_a_setting_defaulting_to_on():
    """On by default: this is a way to opt out of today's behaviour, not a change
    to it."""
    for name, _label in deferred_computations():
        assert SETTING_DEFAULTS[autocompute_key(name)] is True
        assert global_defaults()[autocompute_key(name)] is True


def test_the_settings_are_booleans():
    """The whole reason no tri-state was built — `_posted_settings` coerces every
    settings key with `bool()`, so a non-boolean here would silently become True."""
    for name, _label in deferred_computations():
        assert isinstance(SETTING_DEFAULTS[autocompute_key(name)], bool)


def test_both_settings_pages_offer_them(client):
    """test_settings_page_parity already requires every SETTING_DEFAULTS key on
    both pages; this names them so a failure points here rather than at parity."""
    for url in (reverse("ui:default_experiment_settings"),):
        body = client.get(url).content.decode()
        for name, _label in deferred_computations():
            assert f'name="{autocompute_key(name)}"' in body, name


def test_they_start_checked(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    for name, _label in deferred_computations():
        field = body.split(f'name="{autocompute_key(name)}"', 1)[1].split(">", 1)[0]
        assert "checked" in field, name


def test_saving_records_each_choice(client):
    """Posting with partial dependence's box unticked stores that one off."""
    client.post(reverse("ui:default_experiment_settings"), {
        "export_absolute_times": "on",
        **{f.setting_key: "on" for f in FIGURES},
        "autocompute_local_ablation": "on",
        # autocompute_partial_dependence omitted — an unticked checkbox
    })

    stored = GlobalSettings.get_solo().default_experiment_settings
    assert stored["autocompute_partial_dependence"] is False
    assert stored["autocompute_local_ablation"] is True


def test_an_experiment_can_override_the_default(client):
    """Per-experiment, like every other figure setting — one wide model shouldn't
    force the choice on every other experiment."""
    from core import io
    from ui.services import snapshot as adapter
    from tests.conftest import FIXTURES_DIR

    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    exp.use_default_settings = False
    exp.settings = {"autocompute_partial_dependence": False}
    exp.save(update_fields=["use_default_settings", "settings"])

    resolved = resolve_settings(exp)
    assert resolved["autocompute_partial_dependence"] is False
    assert resolved["autocompute_local_ablation"] is True, "the other one is untouched"


def test_a_stored_setting_from_before_these_existed_still_resolves():
    """`resolve_settings` backfills, so no migration was needed — every settings
    row written before this phase simply reads as on."""
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"export_absolute_times": False}
    gs.save(update_fields=["default_experiment_settings"])

    defaults = global_defaults()
    assert defaults["export_absolute_times"] is False
    for name, _label in deferred_computations():
        assert defaults[autocompute_key(name)] is True
