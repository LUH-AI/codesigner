"""Data Handling's pages: reachable from the experiment, under it in the trail,
with their own sidebar — and, for now, only saying what they will do."""

import re

import pytest
from django.urls import reverse

from datahandling.sections import SECTIONS
from ui.models import Experiment
from ui.permissions import OpenPolicy
from tests.conftest import post_new_experiment


class HidesEverything(OpenPolicy):
    """A policy that sees no experiment at all."""

    def experiments(self, request):
        return Experiment.objects.none()


def _exp(name="Iris tuning"):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def _url(exp, section):
    if section is SECTIONS[0]:
        return reverse("datahandling:data_overview", args=[exp.pk])
    return reverse("datahandling:data_section", args=[exp.pk, section.slug])


def _trail(html):
    nav = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]
    return [re.sub(r"\s+", " ", m).strip()
            for m in re.findall(r"<(?:a|span)[^>]*>(.*?)</(?:a|span)>", nav, re.S)]


def _sidebar(html):
    return html.split('<nav class="sidebar">', 1)[1].split("</nav>", 1)[0]


@pytest.mark.parametrize("section", SECTIONS, ids=[s.slug for s in SECTIONS])
def test_every_section_renders_what_it_plans(client, section):
    """What: each section's page shows its title and every capability it lists.
    How: fetches the page and looks for each capability's title in it."""
    exp = _exp()
    response = client.get(_url(exp, section))

    assert response.status_code == 200
    html = response.content.decode()
    assert str(section.title) in html
    for capability in section.planned:
        assert str(capability.title) in html


def test_the_first_sections_slug_is_not_a_second_address(client):
    """What: the overview is only reachable at the data root.
    How: requests the overview's slug as a section and expects a 404."""
    exp = _exp()
    url = reverse("datahandling:data_section", args=[exp.pk, SECTIONS[0].slug])
    assert client.get(url).status_code == 404


def test_an_unknown_section_is_a_404(client):
    """What: a slug no section has is not a page.
    How: requests a made-up slug."""
    exp = _exp()
    url = reverse("datahandling:data_section", args=[exp.pk, "no-such-section"])
    assert client.get(url).status_code == 404


def test_an_invisible_experiment_has_no_data_pages(client, settings):
    """What: the data pages exist only for experiments the reader can see.
    How: with a policy that hides everything, both kinds of page 404."""
    exp = _exp()
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    assert client.get(_url(exp, SECTIONS[0])).status_code == 404
    assert client.get(_url(exp, SECTIONS[1])).status_code == 404


def test_every_section_sits_under_data_handling(client):
    """What: the trail reads Experiments / <name> / Data Handling whichever
    section is showing — the sections are tabs on one page, as the dashboard's
    figures are. How: reads the breadcrumb labels off two sections' pages."""
    exp = _exp()
    for section in SECTIONS[:2]:
        html = client.get(_url(exp, section)).content.decode()
        assert _trail(html) == ["Experiments", "Iris tuning", "Data Handling"]


def test_the_sections_are_tabs_on_the_page(client):
    """What: the sections are the page's tabs, drawn as the dashboard's are,
    each with its own address, the one asked for selected and its panel the
    one showing. How: reads the tab bar and the panels of one section's page."""
    exp = _exp(name="tabbed")
    section = SECTIONS[2]
    html = client.get(_url(exp, section)).content.decode()
    bar = html.split('class="fig-tabbar"', 1)[1].split("</div>\n        </div>", 1)[0]

    for s in SECTIONS:
        assert f'data-url="{_url(exp, s)}"' in bar
    assert re.search(r'data-tab="%s"[^>]*aria-selected="true"' % section.slug, bar, re.S)
    shown = re.findall(r'class="fig-panel"\s+id="fig-panel-([\w-]+)"', html)
    assert shown == [section.slug]
    assert "data-sections" not in _sidebar(html)


def test_the_experiments_heading_is_shared(client):
    """What: Data Handling and the timeline open under the same heading and
    buttons as the dashboard. How: compares the heading block of each page,
    less what is particular to one request — the rename form's token and the
    page it returns to."""
    exp = _exp()

    def heading(url):
        html = client.get(url).content.decode()
        block = re.sub(r"\s+", " ", html.split("<main", 1)[1].split('class="caption ownership"', 1)[0]
                       .split("</div>\n    </div>", 1)[0].split("</nav>", 1)[-1])
        return re.sub(r'name="(csrfmiddlewaretoken|next)" value="[^"]*"', "", block)

    dashboard = heading(reverse("ui:experiment_detail", args=[exp.pk]))
    assert "Experiment: Iris tuning" in dashboard and "Export" in dashboard
    assert heading(_url(exp, SECTIONS[0])) == dashboard
    assert heading(reverse("ui:experiment_timeline", args=[exp.pk])) == dashboard


def test_the_experiment_page_links_to_its_data(client):
    """What: the open experiment's Data Handling view is one click away from
    its dashboard. How: looks for the overview's URL in the dashboard's
    sidebar."""
    exp = _exp()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert f'href="{_url(exp, SECTIONS[0])}"' in _sidebar(html)


def test_the_section_slugs_are_unique():
    """What: no two sections share an address.
    How: compares the number of slugs with the number of distinct ones."""
    slugs = [s.slug for s in SECTIONS]
    assert len(slugs) == len(set(slugs))


def _section(slug):
    return next(s for s in SECTIONS if s.slug == slug)


def test_processing_choices_start_on_the_models_own_defaults(client):
    """What: each processing control lists the real choices, the model's own
    default among them named "(Random Forest default)" and selected, with no
    separate "as the model needs" entry; it is posted as "auto", so it keeps
    following the model. How: reads the page for a Random Forest experiment."""
    exp = _exp()
    html = client.get(_url(exp, SECTIONS[0])).content.decode()

    def options(name):
        select = html.split(f'name="{name}"', 1)[1].split("</select>", 1)[0]
        return re.findall(r'<option value="(\w+)"( selected)?[^>]*>([^<]+)</option>', select)

    missing = options("missing")
    assert ("auto", " selected", "Leave for the model (Random Forest default)") in missing
    assert [label for _, _, label in missing] == ["Fill in", "Leave for the model (Random Forest default)"]
    assert ("auto", " selected", "Leave as they are (Random Forest default)") in options("scale")
    assert ("auto", " selected", "One column per label (Random Forest default)") in options("labels")
    assert "As the model needs" not in html


def test_a_model_that_cannot_take_gaps_cannot_be_given_them(client):
    """What: for a model that cannot take a missing value, filling in is its
    default and leaving gaps is offered disabled.
    How: reads the page for an SVM experiment."""
    exp = _exp()
    exp.data.model_name = "SVM"
    exp.data.save()
    html = client.get(_url(exp, SECTIONS[0])).content.decode()
    select = html.split('name="missing"', 1)[1].split("</select>", 1)[0]

    assert re.search(r'value="auto" selected[^>]*>Fill in \(SVM default\)', select)
    assert re.search(r'value="keep"[^>]*disabled[^>]*>Leave for the model', select)
    assert re.search(r'value="auto" selected[^>]*>Standardize \(SVM default\)',
                     html.split('name="scale"', 1)[1].split("</select>", 1)[0])


def test_a_processing_choice_is_saved_and_travels_in_the_file(client):
    """What: choosing "standardize" on Features is stored, and the experiment's
    .ihpo records it in its dataset section. How: posts the form, then builds
    the snapshot."""
    from ui.services.snapshot import snapshot_from_experiment

    exp = _exp()
    client.post(_url(exp, _section("features")), {"scale": "standardize", "labels": "auto"})
    exp.refresh_from_db()

    assert exp.data.processing["scale"] == "standardize"
    assert snapshot_from_experiment(exp)["dataset"]["processing"] == {
        "missing": "auto", "scale": "standardize", "labels": "auto"}


def test_processing_is_fixed_once_there_are_trials(client):
    """What: an experiment with trials keeps the processing they were scored
    under — the controls are disabled and a post changes nothing.
    How: gives the experiment a result with one trial and posts a change."""
    exp = _exp()
    exp.data.result = {
        "stats": {"submitted": 1, "finished": 1, "running": 0},
        "data": [{"config_id": 1, "cost": 0.5, "time": 1.0, "scores": {"accuracy": 0.5},
                  "incumbent_config_id": 1}],
        "configs": {"1": {"max_depth": 5}}, "config_origins": {}, "optimizer_state": {},
        "primary_metric": "accuracy", "best_score": 0.5, "best_config_id": "1",
    }
    exp.data.save()
    url = _url(exp, _section("features"))

    assert "disabled" in client.get(url).content.decode()
    client.post(url, {"scale": "standardize"})
    exp.refresh_from_db()
    assert not exp.data.processing


@pytest.mark.django_db
def test_a_run_is_given_the_experiments_processing(client):
    """What: the model a run builds is told the experiment's processing, so the
    choice on the page is the one its trials are scored under.
    How: creates iris with an SVM, sets scaling off, and builds the run's model."""
    from core import io
    from ui.registry import METRICS, MODELS, OPTIMIZERS
    from ui.services.snapshot import snapshot_from_experiment

    from tests.conftest import DATASETS_DIR

    post_new_experiment(client, {
        "name": "svm", "task": "classification", "model_name": "SVM",
        "optimizer_name": "Random Search", "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "seed": 0})
    exp = Experiment.objects.get(data__name="svm")
    exp.data.processing = {"scale": "none"}
    exp.data.save()

    _, built = io.build_experiment(snapshot_from_experiment(exp), METRICS, MODELS, OPTIMIZERS)
    assert built["model"].processing == {"missing": "auto", "scale": "none", "labels": "auto"}


# ── panels: the same kind of thing as a figure ──────────────────────────────

def test_data_panels_and_figures_are_one_type():
    """What: a Data Handling panel and a dashboard figure are both `Panel`s,
    with the same shape, width and template machinery.
    How: checks the class hierarchy and the derived attributes."""
    from datahandling.panels import DATA_PANELS
    from ui.figures import FIGURES, Panel

    assert all(issubclass(p, Panel) for p in DATA_PANELS + FIGURES)
    for panel in DATA_PANELS:
        assert panel.page == "data" and panel.template and panel.columns()


def test_every_data_panel_is_placed_once_on_its_tab():
    """What: the layout file places each Data Handling panel on its section's
    tab, and a file missing one is refused.
    How: reads the file's data table, then a copy with one panel removed."""
    from datahandling.panels import DATA_PANELS
    from ui import layout

    arranged = layout.read_data_default()
    for panel in DATA_PANELS:
        assert [s["key"] for s in arranged[panel.home_tab]].count(panel.key) == 1, panel.key


def test_a_layout_file_without_a_data_panel_is_refused(tmp_path):
    """What: like a figure, a Data Handling panel the file forgets is an error.
    How: writes the file without the scaling panel and reads it."""
    from ui import layout

    text = layout.DEFAULT_PATH.read_text()
    broken = tmp_path / "layout.toml"
    broken.write_text("\n".join(l for l in text.splitlines() if '"scaling"' not in l))
    with pytest.raises(layout.LayoutFileError, match="scaling is not placed"):
        layout.read_data_default(broken)


def _panel(html, slug):
    return html.split(f'id="fig-panel-{slug}"', 1)[1].split('class="fig-panel', 1)[0]


def test_a_created_experiments_overview_describes_its_data(client):
    """What: once created, Overview is about the dataset and holds no
    controls; missing values sits on Cleaning, scaling and label columns on
    Features, each tab saving its own. How: reads the three tabs' panels."""
    exp = _exp()
    html = client.get(_url(exp, SECTIONS[0])).content.decode()
    overview, cleaning, features = (_panel(html, s) for s in ("overview", "cleaning", "features"))

    assert "<select" not in overview and "<form" not in overview
    assert 'data-key="profile"' in overview
    assert 'name="missing"' in cleaning and cleaning.count('type="submit"') == 1
    assert 'name="scale"' in features and 'name="labels"' in features
    assert 'class="fig-grip"' not in html


def test_a_drafts_overview_is_the_choices_to_make_first(client):
    """What: step 2 of a draft's setup opens on an Overview of its own — the
    three choices to make before a run, in one form, and none of the dataset
    cards. Each control is drawn once, in its own tab, and moved into the
    Overview's slot while it shows. How: saves step 1 of a draft and reads
    step 2's Overview and Features panels."""
    from tests.ui.experiments.test_drafts import _save, _setup
    from tests.conftest import start_draft

    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    html = client.get(reverse("ui:setup_data", args=[exp.pk])).content.decode()
    overview = _panel(html, "overview")

    for key in ("missing_values", "scaling", "label_columns"):
        assert f'class="fig-slot shape-card" data-key="{key}"' in overview
    assert 'data-key="profile"' not in overview
    assert "data-purpose" not in html
    assert overview.count("<form") == 1 and overview.count('type="submit"') == 1
    assert html.count('name="scale"') == 1 and 'name="scale"' in _panel(html, "features")


def test_a_setup_overview_naming_an_unknown_panel_is_refused(tmp_path):
    """What: the draft Overview's arrangement is checked like the others; a
    panel nobody defines is an error. How: renames one in a copy of the file."""
    from ui import layout

    text = layout.DEFAULT_PATH.read_text()
    head, setup = text.split("[setup]", 1)
    broken = tmp_path / "layout.toml"
    broken.write_text(head + "[setup]" + setup.replace('"scaling"', '"scalingg"'))
    with pytest.raises(layout.LayoutFileError, match="scalingg"):
        layout.read_setup_default(broken)
