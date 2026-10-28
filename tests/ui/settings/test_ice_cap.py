"""`ice_max_curves` caps the trials that partial dependence predicts. The cap
limits the work, so the drawn mean covers only the sample.
"""

from django.urls import reverse

from core.analytics import _sample_evenly
from experiments.models import GlobalSettings
from experiments.services.settings import SETTING_BOUNDS
from ui.experiment_settings import SETTING_DEFAULTS, resolve_settings


# Sampler

def test_zero_means_every_trial():
    assert _sample_evenly(list(range(50)), 0) == list(range(50))


def test_a_limit_above_the_trial_count_changes_nothing():
    assert _sample_evenly([1, 2, 3], 10) == [1, 2, 3]


def test_the_sample_spans_the_run_rather_than_its_first_k():
    """A trial history explores first and exploits later, so a prefix would
    misstate the average."""
    assert _sample_evenly(list(range(100)), 5) == [0, 25, 50, 74, 99]


def test_the_first_and_last_trials_are_always_drawn():
    for limit in (2, 3, 7, 33):
        sample = _sample_evenly(list(range(100)), limit)
        assert sample[0] == 0 and sample[-1] == 99, limit
        assert len(sample) == limit


def test_a_limit_of_one_keeps_the_last_trial():
    """The most recent curve informs most."""
    assert _sample_evenly(list(range(10)), 1) == [9]


def test_the_sample_is_deterministic():
    """A random sample would flicker between reloads."""
    items = list(range(1000))
    assert _sample_evenly(items, 40) == _sample_evenly(items, 40)


# Setting

def test_it_defaults_to_deepcaves_own_cap():
    assert SETTING_DEFAULTS["ice_max_curves"] == 100


def test_it_is_a_number_not_a_checkbox():
    assert not isinstance(SETTING_DEFAULTS["ice_max_curves"], bool)
    assert isinstance(SETTING_DEFAULTS["ice_max_curves"], int)


def test_both_settings_pages_offer_it(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    assert 'name="ice_max_curves"' in body
    assert 'type="number"' in body


def test_saving_a_number_stores_it(client):
    from ui.figures import FIGURES

    client.post(reverse("ui:default_experiment_settings"), { "ice_max_curves": "250",
        "local_effects_max_trials": "100",
        **{f.setting_key: "on" for f in FIGURES},
    })

    assert GlobalSettings.get_solo().default_experiment_settings["ice_max_curves"] == 250


def test_an_out_of_range_number_is_clamped_not_refused(client):
    """A display preference never fails a form."""
    from ui.figures import FIGURES

    client.post(reverse("ui:default_experiment_settings"), { "ice_max_curves": "99999",
        **{f.setting_key: "on" for f in FIGURES},
    })

    high = SETTING_BOUNDS["ice_max_curves"][1]
    assert GlobalSettings.get_solo().default_experiment_settings["ice_max_curves"] == high


def test_unreadable_input_falls_back_to_the_default_not_to_zero(client):
    """Zero means no limit, so nonsense falls back to the default."""
    from ui.figures import FIGURES

    client.post(reverse("ui:default_experiment_settings"), { "ice_max_curves": "not a number",
        **{f.setting_key: "on" for f in FIGURES},
    })

    stored = GlobalSettings.get_solo().default_experiment_settings["ice_max_curves"]
    assert stored == SETTING_DEFAULTS["ice_max_curves"]


def test_the_beeswarm_has_its_own_cap():
    """`ice_max_curves` caps surrogate predictions, and
    `local_effects_max_trials` caps whole ablation games."""
    assert SETTING_DEFAULTS["local_effects_max_trials"] == 100
    assert "local_effects_max_trials" in SETTING_BOUNDS


def test_both_caps_are_offered_on_the_settings_page(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    for name in ("ice_max_curves", "local_effects_max_trials"):
        assert f'name="{name}"' in body, name


def test_a_settings_row_from_before_this_existed_backfills():
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"show_trials": False}
    gs.save(update_fields=["default_experiment_settings"])

    from ui.experiment_settings import global_defaults
    assert global_defaults()["ice_max_curves"] == 100
