"""Editing how an experiment's search runs.

`OptimizerParam` has described these settings since Step 2 and nothing rendered
them — the create form's docstring has said "editing them is a later step" for
eight steps. The form is generic: it loops what the optimizer declares, so a new
setting is a line in `params_schema` and no template changes.

They are editable *between* runs, not fixed at creation like `cv_folds`. The
distinction is whether past trials stay comparable, and they do: the search
strategy chooses which configurations to try, not how they are measured.
"""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR
from ui.models import Experiment, Run

pytestmark = pytest.mark.django_db


def _experiment(optimizer_name="SMAC", params=None, **overrides):
    from ui.services import snapshot as adapter

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


# ── the create form ──────────────────────────────────────────────────────────

def test_the_create_form_offers_the_search_settings(client):
    html = client.get(reverse("ui:new_experiment")).content.decode()

    assert 'name="opt_search_strategy"' in html
    assert 'name="opt_exploration_ratio"' in html
    assert 'name="opt_use_default_config"' in html


def test_the_advanced_ones_are_folded_away(client):
    """Four settings anyone can reason about, and six that need you to already
    know what they do. Both present, only one group unfolded."""
    html = client.get(reverse("ui:new_experiment")).content.decode()

    assert "<details" in html and "Advanced search settings" in html
    advanced = html.split("Advanced search settings", 1)[1]
    assert 'name="opt_retrain_after"' in advanced
    assert 'name="opt_search_strategy"' not in advanced


def test_creating_stores_what_was_chosen(client):
    client.post(reverse("ui:new_experiment"), {
        "name": "configured", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "opt_search_strategy": "rf", "opt_exploration_ratio": "0.4",
        "opt_use_default_config": "on",
    })

    stored = Experiment.objects.get(name="configured").optimizer_params
    assert stored["search_strategy"] == "rf"
    assert stored["exploration_ratio"] == 0.4
    assert stored["use_default_config"] is True


def test_creating_without_touching_them_stores_the_defaults(client):
    client.post(reverse("ui:new_experiment"), {
        "name": "plain", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
    })

    stored = Experiment.objects.get(name="plain").optimizer_params
    assert stored["search_strategy"] == "gp"
    # Unset stays unset rather than becoming a number of ours: the two
    # strategies disagree about these, and the one in use should decide.
    assert stored["challengers"] is None


# ── editing afterwards ───────────────────────────────────────────────────────

def test_the_settings_page_shows_what_is_stored(client):
    exp = _experiment(params={"search_strategy": "rf", "exploration_ratio": 0.4})

    html = _settings(client, exp)

    assert 'value="rf" selected' in html
    assert 'value="0.4"' in html


def test_saving_changes_them(client):
    exp = _experiment()

    client.post(reverse("ui:experiment_settings", args=[exp.pk]), {
        "save_search": "1", "opt_search_strategy": "rf",
        "opt_exploration_ratio": "0.5", "opt_acquisition": "pi",
    })
    exp.refresh_from_db()

    assert exp.optimizer_params["search_strategy"] == "rf"
    assert exp.optimizer_params["exploration_ratio"] == 0.5
    assert exp.optimizer_params["acquisition"] == "pi"


def test_saving_the_search_leaves_the_display_settings_alone(client):
    """Two forms on one page, two fields on the model. The display settings
    have inherit/override/promote-to-default semantics and these have none, so
    saving one must not reach into the other."""
    exp = _experiment()
    exp.settings = {"export_absolute_times": False}
    exp.use_default_settings = False
    exp.save()

    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"save_search": "1", "opt_search_strategy": "rf"})
    exp.refresh_from_db()

    assert exp.settings == {"export_absolute_times": False}
    assert exp.use_default_settings is False


def test_they_cannot_be_changed_under_a_running_search(client):
    """Rebuilding the optimizer mid-run would change what the run is using
    halfway through it."""
    exp = _experiment(params={"search_strategy": "gp"})
    Run.objects.create(experiment=exp, stopping={"max_trials": 5},
                       primary_metric="accuracy", status="running")

    html = _settings(client, exp)
    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"save_search": "1", "opt_search_strategy": "rf"})
    exp.refresh_from_db()

    assert "A run is in flight" in html
    assert exp.optimizer_params["search_strategy"] == "gp"


def test_a_value_out_of_range_is_clamped_rather_than_refused(client):
    exp = _experiment()

    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"save_search": "1", "opt_exploration_ratio": "5"})
    exp.refresh_from_db()

    assert exp.optimizer_params["exploration_ratio"] == 1.0


def test_an_unreadable_value_falls_back_to_the_default(client):
    exp = _experiment()

    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"save_search": "1", "opt_exploration_ratio": "lots"})
    exp.refresh_from_db()

    assert exp.optimizer_params["exploration_ratio"] == 0.25


def test_an_optimizer_with_nothing_to_configure_offers_nothing(client):
    """Random search declares an empty schema, so the section is absent rather
    than an empty box with a save button."""
    exp = _experiment(optimizer_name="Random Search")

    assert "Configure optimizer" not in _settings(client, exp)


# ── surviving a rename ───────────────────────────────────────────────────────

def test_an_experiment_stored_under_the_old_name_still_works(client):
    """SMAC was called "SMAC (BlackBox)" while it was hardwired to that facade.
    Rows were migrated; an .ihpo on someone else's disk was not, so the alias
    has to resolve."""
    exp = _experiment(optimizer_name="SMAC (BlackBox)")

    assert "Configure optimizer" in _settings(client, exp)
    assert client.get(
        reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200


def test_a_stored_setting_the_optimizer_no_longer_has_is_ignored(client):
    """Renaming or withdrawing a setting would otherwise reach `__init__` as an
    unexpected keyword — raised from the page that rebuilds the result to draw
    it, which catches ValueError and nothing else."""
    exp = _experiment(params={"search_strategy": "gp", "withdrawn_long_ago": 7})

    assert client.get(
        reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200


# ── the create page's panel per optimizer ────────────────────────────────────

def _panels(html):
    """Each `data-optimizer` panel on the page: key → (hidden, field names)."""
    import re

    starts = list(re.finditer(r'<div data-optimizer="([^"]+)"( hidden)?>', html))
    edges = [m.end() for m in starts] + [len(html)]
    return {
        m.group(1): (bool(m.group(2)),
                     re.findall(r'name="(opt_[a-z_]+)"', html[edges[i]:edges[i + 1]]))
        for i, m in enumerate(starts)
    }


def test_every_optimizer_with_settings_gets_a_panel(client):
    """All of them are rendered so the dropdown can swap between them without a
    request. Random Search declares nothing, so it contributes nothing rather
    than an empty box."""
    panels = _panels(client.get(reverse("ui:new_experiment")).content.decode())

    assert set(panels) == {"SMAC", "Grid Search"}


def test_only_the_selected_optimizers_panel_is_shown(client):
    panels = _panels(client.get(reverse("ui:new_experiment")).content.decode())

    assert panels["SMAC"][0] is False
    assert panels["Grid Search"][0] is True


def test_the_panel_is_the_same_bordered_box_as_the_figure_settings(client):
    """`check-group` — the fieldset the figures checkboxes live in. It already
    carries number inputs elsewhere, so this needs no CSS of its own."""
    html = client.get(reverse("ui:new_experiment")).content.decode()

    assert '<fieldset class="check-group">' in html
    assert "<legend>" in html


def test_grid_search_settings_are_reachable_at_last(client):
    """`numeric_steps` has round-tripped correctly since Step 2 and no page has
    ever offered it — the create form only ever rendered SMAC's schema."""
    panels = _panels(client.get(reverse("ui:new_experiment")).content.decode())

    assert panels["Grid Search"][1] == ["opt_numeric_steps"]


def test_creating_with_grid_search_stores_what_its_panel_said(client):
    client.post(reverse("ui:new_experiment"), {
        "name": "gridded", "model_name": "Random Forest",
        "optimizer_name": "Grid Search", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "opt_numeric_steps": "9",
    })

    exp = Experiment.objects.get(name="gridded")
    assert exp.optimizer_params == {"numeric_steps": 9}


def test_the_hidden_panels_settings_do_not_leak_into_the_chosen_one(client):
    """The browser disables them so they are never submitted. Even posted by
    hand they belong to another optimizer and are not in its schema."""
    client.post(reverse("ui:new_experiment"), {
        "name": "clean", "model_name": "Random Forest",
        "optimizer_name": "Grid Search", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "opt_numeric_steps": "4", "opt_search_strategy": "rf",
    })

    assert Experiment.objects.get(name="clean").optimizer_params == {"numeric_steps": 4}


def test_an_optimizer_whose_panel_was_hidden_gets_its_own_defaults(client):
    """A disabled input is not submitted, and the parser reads a missing value
    as that setting's default — so picking an optimizer without opening its
    panel stores exactly what it would have shown."""
    client.post(reverse("ui:new_experiment"), {
        "name": "untouched", "model_name": "Random Forest",
        "optimizer_name": "Grid Search", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
    })

    assert Experiment.objects.get(name="untouched").optimizer_params == {"numeric_steps": 5}


def test_a_validation_error_keeps_the_optimizer_and_its_values(client):
    """The page comes back with the same optimizer selected and the numbers
    still in their boxes, rather than resetting to the first one."""
    html = client.post(reverse("ui:new_experiment"), {
        "name": "", "model_name": "Random Forest",
        "optimizer_name": "Grid Search", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "opt_numeric_steps": "8",
    }).content.decode()

    panels = _panels(html)
    assert panels["Grid Search"][0] is False
    assert panels["SMAC"][0] is True
    assert 'name="opt_numeric_steps"' in html and 'value="8"' in html


def test_an_unchosen_panel_is_inert_as_rendered_not_only_hidden(client):
    """`hidden` is a browser hint; `disabled` is what keeps a panel's values out
    of the POST. Rendering it on the fieldset means the page is correct with no
    JavaScript at all, and the switch is an enhancement rather than the
    mechanism — which matters, because two optimizers could one day declare a
    setting under the same name."""
    import re

    html = client.get(reverse("ui:new_experiment")).content.decode()
    fieldsets = re.findall(
        r'<div data-optimizer="([^"]+)"( hidden)?>\s*\n?\s*'
        r'<fieldset class="check-group"( disabled)?>', html)

    assert fieldsets, "no panels found"
    for key, hidden, disabled in fieldsets:
        assert bool(hidden) == bool(disabled), f"{key}: hidden and disabled disagree"


def test_the_switch_toggles_the_fieldset_and_not_just_its_inputs(client):
    """A disabled fieldset overrides its children, so re-enabling the inputs one
    by one would leave the panel dead."""
    html = client.get(reverse("ui:new_experiment")).content.decode()

    assert 'querySelectorAll("input, select, textarea, fieldset")' in html
    assert "wireChoice" in html


# ── what the fields say, and where they say it ───────────────────────────────

def _smac_panel(client):
    html = client.get(reverse("ui:new_experiment")).content.decode()
    start = html.index('data-optimizer="SMAC"')
    return html[start:html.index("data-optimizer", start + 10)]


def test_a_setting_is_named_and_not_described_in_place(client):
    """A dozen settings with a line of prose under each is a wall to read past
    while filling in a form. The label is the setting's name; the explanation is
    hidden until someone wants it."""
    assert '<p class="caption">' not in _smac_panel(client)


def test_the_explanation_is_behind_a_marked_control(client):
    """A circled i beside the name. A `title` on the label would have been
    invisible until hovered, so nothing on the page would say there was
    anything to read."""
    panel = _smac_panel(client)

    assert '<button type="button" class="info-mark"' in panel
    assert ('<span class="info-bubble" role="tooltip" id="opt_exploration_ratio_help">'
            'Fraction of the budget sampled') in panel


def test_the_explanation_is_readable_without_a_pointer_too(client):
    """The bubble is the control's description, so the words are announced on
    reaching the field whether or not anyone opens it — and the mark is a
    button, so opening it takes a keypress rather than a hover."""
    panel = _smac_panel(client)
    mark = panel[panel.index('class="info-mark"'):]

    assert 'aria-describedby="opt_exploration_ratio_help"' in panel
    assert 'tabindex="0"' in mark[:mark.index(">")]


def _control(panel, name):
    """The whole opening tag of one setting's control."""
    at = panel.index(f'id="opt_{name}"')
    return panel[panel.rindex("<", 0, at):panel.index(">", at) + 1]


def test_a_decimal_field_is_not_a_stepper_at_all(client):
    """A number input's arrows step by one, which on a fraction between 0 and 1
    is the entire range. Worse than useless: they make the field look like it
    wants a whole number."""
    panel = _smac_panel(client)
    control = _control(panel, "exploration_ratio")

    assert control.startswith("<input")
    assert 'type="text"' in control and 'inputmode="decimal"' in control
    assert "step=" not in control


def test_a_whole_number_field_still_steps_by_one(client):
    """Where one is a real increment the arrows are worth having, and the
    difference between the two kinds of field is then visible."""
    panel = _smac_panel(client)
    control = _control(panel, "exploration_trials")

    assert 'type="number"' in control and 'step="1"' in control


def test_an_empty_field_says_what_filling_it_in_would_displace(client):
    """The placeholder names the search strategy as the source of the default,
    which is only true of the settings that declare none of their own — and not
    of every one of those. Grid Search has no strategy at all."""
    html = client.get(reverse("ui:new_experiment")).content.decode()
    panel = _smac_panel(client)

    def placeholder(fragment, name):
        import re
        at = fragment.index(f'id="opt_{name}"')
        tag = fragment[fragment.rindex("<", 0, at):fragment.index(">", at)]
        found = re.search(r'placeholder="([^"]*)"', tag)
        return found.group(1) if found else ""

    assert placeholder(panel, "retrain_after") == "search strategy default"
    assert placeholder(panel, "exploration_ratio") == "", "it has a default of its own"
    assert placeholder(panel, "exploration_trials") == "uses the share", (
        "blank here falls back to the share, not to anything the strategy picks")
    assert placeholder(html[html.index('data-optimizer="Grid Search"'):], "numeric_steps") == ""


# ── the exploration budget, said either way ──────────────────────────────────

def test_both_ways_of_saying_how_long_to_explore_are_offered(client):
    panel = _smac_panel(client)

    assert 'name="opt_exploration_ratio"' in panel
    assert 'name="opt_exploration_trials"' in panel


def test_creating_with_a_count_stores_the_count(client):
    client.post(reverse("ui:new_experiment"), {
        "name": "counted", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "opt_exploration_trials": "12",
    })

    assert Experiment.objects.get(name="counted").optimizer_params["exploration_trials"] == 12


def test_leaving_the_count_blank_stores_nothing_for_it(client):
    """Blank means "use the share", and the share is the default. A zero here
    would read as "explore not at all"."""
    client.post(reverse("ui:new_experiment"), {
        "name": "shared", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "0",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "opt_exploration_trials": "",
    })

    params = Experiment.objects.get(name="shared").optimizer_params
    assert params["exploration_trials"] is None
    assert params["exploration_ratio"] == 0.25
