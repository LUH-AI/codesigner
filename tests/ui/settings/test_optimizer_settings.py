"""The form renders whatever `params_schema` declares, and a new setting needs
no template change. Settings are fixed at creation, like `cv_folds`, and
describe every trial in the experiment.
"""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, get_optimizer_step, post_new_experiment
from experiments.models import Experiment, Run

pytestmark = pytest.mark.django_db


def _experiment(optimizer_name="SMAC", params=None, **overrides):
    from experiments.services import snapshot as adapter

    snapshot = {
        "version": "0.1.0", "name": "exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": optimizer_name,
        "optimizer_params": params or {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    snapshot.update(overrides)
    return adapter.experiment_from_snapshot(snapshot, adopt_paths=True)


def _settings(client, exp):
    return client.get(reverse("ui:experiment_settings", args=[exp.pk])).content.decode()


# Create form

def test_the_create_form_offers_the_search_settings(client):
    html = get_optimizer_step(client).content.decode()

    assert 'name="opt_search_strategy"' in html
    assert 'name="opt_share_cap"' in html
    assert 'name="opt_use_default_config"' in html


def test_the_advanced_ones_are_folded_away(client):
    html = get_optimizer_step(client).content.decode()

    assert "<details" in html and "Advanced search settings" in html
    advanced = _without_scripts(html).split("Advanced search settings", 1)[1]
    assert 'name="opt_retrain_after"' in advanced
    assert 'name="opt_search_strategy"' not in advanced


def test_creating_stores_what_was_chosen(client):
    post_new_experiment(client, {
        "name": "configured", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "opt_search_strategy": "rf", "opt_share_cap": "0.4",
        "opt_use_default_config": "on",
    })

    stored = Experiment.objects.get(data__name="configured").data.optimizer_params
    assert stored["search_strategy"] == "rf"
    assert stored["share_cap"] == 0.4
    assert stored["use_default_config"] is True


def test_creating_without_touching_them_stores_the_defaults(client):
    post_new_experiment(client, {
        "name": "plain", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
    })

    stored = Experiment.objects.get(data__name="plain").data.optimizer_params
    assert stored["search_strategy"] == "gp"
    # The two strategies disagree on unset values, so the one in use decides.
    assert stored["challengers"] is None


# Fixed after creation

def _create(client, **overrides):
    data = {"name": "created", "model_name": "Random Forest",
            "optimizer_name": "SMAC", "seed": "0",
            "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification"}
    data.update(overrides)
    post_new_experiment(client, data)
    return Experiment.objects.get(data__name=data["name"]).data.optimizer_params


def test_the_settings_page_does_not_offer_them(client):
    """Changing the settings mid-experiment would leave the recorded trials
    describing a search that no longer runs.
    """
    exp = _experiment(params={"search_strategy": "rf"})

    assert "Configure optimizer" not in _settings(client, exp)


def test_posting_them_at_the_settings_page_changes_nothing(client):
    """Accepting unrendered fields would let anyone who knew the names edit
    them."""
    exp = _experiment(params={"search_strategy": "gp"})

    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"save_search": "1", "opt_search_strategy": "rf"})
    exp.refresh_from_db()

    assert exp.data.optimizer_params["search_strategy"] == "gp"


def test_a_value_out_of_range_is_clamped_rather_than_refused(client):
    assert _create(client, opt_share_cap="5")["share_cap"] == 1.0


def test_an_unreadable_value_falls_back_to_the_default(client):
    assert _create(client, opt_acquisition_xi="lots")["acquisition_xi"] == 0.0


def test_an_unreadable_value_with_no_default_falls_back_to_blank(client):
    """Blank means whatever SMAC does, the same as the default."""
    assert _create(client, opt_share_cap="lots")["share_cap"] is None


# Renames

def test_an_experiment_stored_under_the_old_name_still_works(client):
    """The alias must resolve, since an `.ihpo` on another disk still names
    "SMAC (BlackBox)"."""
    exp = _experiment(optimizer_name="SMAC (BlackBox)")

    assert client.get(
        reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200
    assert client.get(
        reverse("ui:experiment_settings", args=[exp.pk])).status_code == 200


def test_a_stored_setting_the_optimizer_no_longer_has_is_ignored(client):
    """A withdrawn setting would reach `__init__` as an unexpected keyword, and
    the page catches only ValueError."""
    exp = _experiment(params={"search_strategy": "gp", "withdrawn_long_ago": 7})

    assert client.get(
        reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200


# Panels per optimizer

def _without_scripts(html):
    """`_without_scripts` drops every `<script>` block, because the panel's
    script also mentions the scraped `name="opt_…"` attributes.
    """
    import re

    return re.sub(r"<script\b.*?</script>", "", html, flags=re.S)


def _panels(html):
    """`_panels` maps each `data-optimizer` key to (hidden, fields)."""
    import re

    html = _without_scripts(html)
    starts = list(re.finditer(r'<div data-optimizer="([^"]+)"( hidden)?>', html))
    edges = [m.end() for m in starts] + [len(html)]
    return {
        m.group(1): (bool(m.group(2)),
                     re.findall(r'name="(opt_[a-z_]+)"', html[edges[i]:edges[i + 1]]))
        for i, m in enumerate(starts)
    }


def test_every_optimizer_with_settings_gets_a_panel(client):
    """The dropdown swaps panels without a request, and Random Search declares
    no settings."""
    panels = _panels(get_optimizer_step(client).content.decode())

    assert set(panels) == {"SMAC", "Grid Search"}


def test_only_the_selected_optimizers_panel_is_shown(client):
    panels = _panels(get_optimizer_step(client).content.decode())

    assert panels["SMAC"][0] is False
    assert panels["Grid Search"][0] is True


def test_the_panel_is_the_same_bordered_box_as_the_figure_settings(client):
    html = get_optimizer_step(client).content.decode()

    assert '<fieldset class="check-group">' in html
    assert "<legend>" in html


def test_grid_search_settings_are_reachable_at_last(client):
    panels = _panels(get_optimizer_step(client).content.decode())

    assert panels["Grid Search"][1] == ["opt_numeric_steps"]


def test_creating_with_grid_search_stores_what_its_panel_said(client):
    post_new_experiment(client, {
        "name": "gridded", "model_name": "Random Forest",
        "optimizer_name": "Grid Search", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "opt_numeric_steps": "9",
    })

    exp = Experiment.objects.get(data__name="gridded")
    assert exp.data.optimizer_params == {"numeric_steps": 9}


def test_the_hidden_panels_settings_do_not_leak_into_the_chosen_one(client):
    """Hand-posted fields from another optimizer fall outside its schema."""
    post_new_experiment(client, {
        "name": "clean", "model_name": "Random Forest",
        "optimizer_name": "Grid Search", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "opt_numeric_steps": "4", "opt_search_strategy": "rf",
    })

    assert Experiment.objects.get(data__name="clean").data.optimizer_params == {"numeric_steps": 4}


def test_an_optimizer_whose_panel_was_hidden_gets_its_own_defaults(client):
    """The parser reads a missing value as its default, so an unopened panel
    stores what it showed."""
    post_new_experiment(client, {
        "name": "untouched", "model_name": "Random Forest",
        "optimizer_name": "Grid Search", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
    })

    assert Experiment.objects.get(data__name="untouched").data.optimizer_params == {"numeric_steps": 5}


def test_the_optimizer_step_reopens_on_what_was_saved(client):
    page = get_optimizer_step(client)
    url = page.request["PATH_INFO"]
    client.post(url, {"optimizer_name": "Grid Search", "opt_numeric_steps": "8"})
    html = client.get(url).content.decode()

    panels = _panels(html)
    assert panels["Grid Search"][0] is False
    assert panels["SMAC"][0] is True
    assert 'name="opt_numeric_steps"' in html and 'value="8"' in html


def test_an_unchosen_panel_is_inert_as_rendered_not_only_hidden(client):
    """`disabled` on the fieldset keeps a panel out of the POST without
    JavaScript, since two optimizers could one day share a setting name."""
    import re

    html = get_optimizer_step(client).content.decode()
    fieldsets = re.findall(
        r'<div data-optimizer="([^"]+)"( hidden)?>\s*\n?\s*'
        r'<fieldset class="check-group"( disabled)?>', html)

    assert fieldsets, "no panels found"
    for key, hidden, disabled in fieldsets:
        assert bool(hidden) == bool(disabled), f"{key}: hidden and disabled disagree"


def test_the_switch_toggles_the_fieldset_and_not_just_its_inputs(client):
    """A disabled fieldset overrides its children."""
    html = get_optimizer_step(client).content.decode()

    assert 'querySelectorAll("input, select, textarea, fieldset")' in html
    assert 'select.addEventListener("change", apply)' in html


# Labels and help

def _smac_panel(client):
    html = get_optimizer_step(client).content.decode()
    start = html.index('data-optimizer="SMAC"')
    return html[start:html.index("data-optimizer", start + 10)]


def test_a_setting_is_named_and_not_described_in_place(client):
    """The label names the setting, and the explanation stays hidden until asked
    for."""
    assert '<p class="caption">' not in _smac_panel(client)


def test_the_explanation_is_behind_a_marked_control(client):
    """The help sits behind a circled mark, since a `title` gives no sign until
    hovered."""
    panel = _smac_panel(client)

    assert '<button type="button" class="info-mark"' in panel
    assert ('<span class="info-bubble" role="tooltip" id="opt_share_cap_help">'
            "Fraction of the run") in panel


def test_the_explanation_is_readable_without_a_pointer_too(client):
    """The bubble describes the control for screen readers, and the mark is a
    button reachable by keyboard."""
    panel = _smac_panel(client)
    mark = panel[panel.index('class="info-mark"'):]

    assert 'aria-describedby="opt_share_cap_help"' in panel
    assert 'tabindex="0"' in mark[:mark.index(">")]


def _control(panel, name):
    """`_control` returns one control's opening tag."""
    at = panel.index(f'id="opt_{name}"')
    return panel[panel.rindex("<", 0, at):panel.index(">", at) + 1]


def test_a_decimal_field_is_not_a_stepper_at_all(client):
    """A step of one covers a fraction's whole range."""
    panel = _smac_panel(client)
    control = _control(panel, "share_cap")

    assert control.startswith("<input")
    assert 'type="text"' in control and 'inputmode="decimal"' in control
    assert "step=" not in control


def test_a_whole_number_field_still_steps_by_one(client):
    panel = _smac_panel(client)
    control = _control(panel, "trial_cap")

    assert 'type="number"' in control and 'step="1"' in control


def test_an_empty_field_shows_the_value_it_would_actually_use(client):
    """The placeholder shows the default itself, taken from
    `SMACOptimizer.strategy_defaults`.
    """
    html = get_optimizer_step(client).content.decode()
    panel = _smac_panel(client)

    def placeholder(fragment, name):
        import re
        at = fragment.index(f'id="opt_{name}"')
        tag = fragment[fragment.rindex("<", 0, at):fragment.index(">", at)]
        found = re.search(r'placeholder="([^"]*)"', tag)
        return found.group(1) if found else ""

    assert placeholder(panel, "retrain_after") == "1", "the Gaussian process's own"
    assert placeholder(panel, "share_cap") == "0.25"
    assert placeholder(panel, "trial_cap") == "10 per hyperparameter", (
        "blank here is SMAC's own per-hyperparameter bound, not a strategy default")
    assert placeholder(html[html.index('data-optimizer="Grid Search"'):], "numeric_steps") == ""


def test_a_strategy_dependent_default_carries_both_strategies_values(client):
    """The selector changes without a request, so the field carries both
    strategies' defaults."""
    import json
    import re

    panel = _smac_panel(client)
    at = panel.index('id="opt_random_probability"')
    field = panel[panel.rindex('<div class="field"', 0, at):at]
    found = re.search(r"data-strategy-defaults=\"([^\"]*)\"", field)

    assert found, "the field must carry every strategy's default"
    assert json.loads(found.group(1).replace("&quot;", '"')) == {"gp": "0.08447", "rf": "0.2"}


def test_a_bounded_setting_shows_its_range(client):
    """`min` and `max` speak only once a value is wrong."""
    panel = _smac_panel(client)
    at = panel.index('id="opt_random_probability"')
    field = panel[panel.rindex('<div class="field"', 0, at):panel.index("</div>", at)]

    assert "0–1" in field


# Initial points

def test_both_caps_are_offered_side_by_side(client):
    """The caps sit together, giving "the smaller of these" a meaning."""
    panel = _smac_panel(client)
    block = panel[panel.index('<fieldset class="param-group">'):]

    assert "Initial points" in block
    for name in ("use_share_cap", "share_cap", "use_trial_cap", "trial_cap",
                 "initial_points_use_max"):
        assert f'name="opt_{name}"' in block, name
    assert block.count('class="param-column"') == 2


def test_both_caps_are_on_and_both_values_blank_to_begin_with(client):
    panel = _smac_panel(client)

    for name in ("use_share_cap", "use_trial_cap"):
        assert " checked" in _control(panel, name), name
    for name in ("share_cap", "trial_cap"):
        assert 'value=""' in _control(panel, name), name


def test_the_smaller_of_the_two_is_the_default(client):
    """SMAC takes the smaller; the larger is arithmetic of ours."""
    assert " checked" not in _control(_smac_panel(client), "initial_points_use_max")


def test_use_default_is_a_control_and_not_a_setting(client):
    """The checkbox is never submitted, since an empty field already means
    SMAC's own."""
    panel = _smac_panel(client)
    row = panel[panel.index('data-default-for="opt_share_cap"'):]
    row = row[:row.index("</div>")]

    assert "Use default" in row
    assert " checked" in row
    assert "name=" not in row, "not a setting; nothing to submit"


def test_creating_stores_the_caps(client):
    post_new_experiment(client, {
        "name": "capped", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "opt_use_share_cap": "on", "opt_share_cap": "0.4",
        "opt_use_trial_cap": "on", "opt_trial_cap": "12",
        "opt_initial_points_use_max": "on",
    })

    params = Experiment.objects.get(data__name="capped").data.optimizer_params
    assert params["share_cap"] == 0.4
    assert params["trial_cap"] == 12
    assert params["initial_points_use_max"] is True


def test_a_cap_switched_off_is_stored_as_off(client):
    post_new_experiment(client, {
        "name": "uncapped", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "opt_use_trial_cap": "on",
    })

    params = Experiment.objects.get(data__name="uncapped").data.optimizer_params
    assert params["use_share_cap"] is False
    assert params["use_trial_cap"] is True


# Surrogate settings

def test_the_surrogate_settings_are_all_there(client):
    panel = _smac_panel(client)

    for name in ("rf_trees", "rf_max_depth", "rf_min_samples_split",
                 "rf_min_samples_leaf", "rf_feature_ratio", "rf_bootstrapping",
                 "gp_model_type", "gp_restarts", "gp_normalize_y"):
        assert f'name="opt_{name}"' in panel, name


def test_they_are_all_folded_away(client):
    advanced = _smac_panel(client).split("Advanced search settings", 1)[1]

    assert 'name="opt_rf_trees"' in advanced
    assert 'name="opt_gp_model_type"' in advanced


def test_every_surrogate_setting_says_surrogate(client):
    """The demo Random Forest model has two setting names that the forest
    surrogate also has."""
    from ui.optimizer_labels import LABELS
    from core.optimizers import SMACOptimizer

    scoped = [p.name for p in SMACOptimizer.params_schema if p.depends_on]

    assert scoped, "no strategy-scoped settings found"
    for name in scoped:
        label = str(LABELS[name][0])
        assert "urrogate" in label, f"{name}: {label!r} does not say which model"


def test_a_setting_declares_when_it_applies(client):
    """`data-when` lets the form hide fields generically."""
    panel = _smac_panel(client)

    at = panel.index('id="opt_rf_trees"')
    field = panel[panel.rindex('<div class="field"', 0, at):at]

    assert 'data-when="search_strategy=rf"' in field


def test_the_panel_wires_its_own_switch(client):
    """The panel's script limits itself to its own fieldset, since the create
    page renders one panel per optimizer."""
    html = get_optimizer_step(client).content.decode()

    assert 'querySelectorAll("[data-when]")' in html
    assert 'document.currentScript.closest("fieldset")' in html


def test_a_setting_for_the_other_strategy_is_hidden_and_not_disabled(client):
    """Disabling would drop the value from the POST, and switching strategy and
    back would reset it."""
    html = get_optimizer_step(client).content.decode()

    hiding = html[html.index("[data-when]"):html.index("[data-enabled-by]")]

    assert "field.hidden = !control || control.value !== wanted" in hiding
    assert "disabled" not in hiding, "hiding must not drop the value from the POST"


def test_creating_stores_the_surrogate_settings(client):
    post_new_experiment(client, {
        "name": "forested", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "opt_search_strategy": "rf", "opt_rf_trees": "64",
        "opt_rf_feature_ratio": "0.5",
    })

    params = Experiment.objects.get(data__name="forested").data.optimizer_params
    assert params["rf_trees"] == 64
    assert params["rf_feature_ratio"] == 0.5


def test_a_feature_ratio_above_one_is_clamped_on_the_way_in(client):
    """SMAC computes `max_features = 0` above 1.0, and only the optimizer's cap
    covers a hand-edited `.ihpo`."""
    post_new_experiment(client, {
        "name": "greedy", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "opt_search_strategy": "rf", "opt_rf_feature_ratio": "5",
    })

    assert Experiment.objects.get(data__name="greedy").data.optimizer_params["rf_feature_ratio"] == 1.0


def test_the_slow_fitting_method_says_so_where_it_is_chosen(client):
    panel = _smac_panel(client)

    assert '<option value="mcmc">MCMC (slow)</option>' in panel
