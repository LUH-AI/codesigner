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


def test_the_deferred_computations_are_declared():
    """In catalog order — partial dependence, the local explanation, then the
    beeswarm, which is the most expensive of the three."""
    assert [name for name, _ in deferred_computations()] == [
        "partial_dependence", "local_ablation", "local_effects"]


def test_a_deferred_name_is_the_computations_not_the_figures():
    """`local_ablation` is the computation the local-explanation figure fetches,
    and it keeps that name: it is stored in settings, and it was named for the
    computation back when the figure was one view of the importance figure.
    Renaming it to match the figure would be a data migration for nothing."""
    local = next(f for f in FIGURES if f.key == "local_explanation")
    assert [name for name, _ in local.deferred] == ["local_ablation"]
    assert local.setting_key == "show_local_explanation"


def test_figures_with_nothing_deferred_declare_nothing():
    deferred_keys = {f.key for f in FIGURES if f.deferred}
    assert deferred_keys == {"local_explanation", "partial_dependence",
                             "local_effects"}


def test_every_deferred_computation_has_a_setting_defaulting_to_off():
    """Off by default, so opening or reloading an experiment page computes
    nothing at all until it is asked to.

    These were on when the setting shipped, on the argument that it was an
    opt-out rather than a change. The measurements said otherwise: partial
    dependence fires on page open and costs 6.9 s and a 4.5 MB response on a
    10,000-trial run, and local ablation is 1.4 s at the same size. Neither is
    worth spending on a page the reader may not be looking at."""
    for name, _label in deferred_computations():
        assert SETTING_DEFAULTS[autocompute_key(name)] is False
        assert global_defaults()[autocompute_key(name)] is False


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


def test_they_start_unchecked(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    for name, _label in deferred_computations():
        field = body.split(f'name="{autocompute_key(name)}"', 1)[1].split(">", 1)[0]
        assert "checked" not in field, name


def test_saving_records_each_choice(client):
    """Posting with partial dependence's box unticked stores that one off."""
    client.post(reverse("ui:default_experiment_settings"), {
        "ice_max_curves": "100",
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
    exp.settings = {"autocompute_partial_dependence": True}
    exp.save(update_fields=["use_default_settings", "settings"])

    resolved = resolve_settings(exp)
    assert resolved["autocompute_partial_dependence"] is True
    assert resolved["autocompute_local_ablation"] is False, "the other one is untouched"


def test_a_stored_setting_from_before_these_existed_still_resolves():
    """`resolve_settings` backfills, so no migration was needed — every settings
    row written before this phase simply reads as on."""
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"show_trials": False}
    gs.save(update_fields=["default_experiment_settings"])

    defaults = global_defaults()
    assert defaults["show_trials"] is False
    for name, _label in deferred_computations():
        assert defaults[autocompute_key(name)] is False
