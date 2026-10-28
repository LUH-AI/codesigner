"""Data Handling's pages sit under the experiment in the trail and show its
processing choices."""

import re

import pytest
from django.urls import reverse

from ui.figures.data_handling import SECTIONS
from experiments.models import Experiment
from access.policy import OpenPolicy
from tests.conftest import post_new_experiment


class HidesEverything(OpenPolicy):
    """`HidesEverything` extends `OpenPolicy` to hide everything."""

    def experiments(self, request):
        return Experiment.objects.none()


def _exp(name="Iris tuning"):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def _url(exp, section):
    if section is SECTIONS[0]:
        return reverse("ui:data_overview", args=[exp.pk])
    return reverse("ui:data_section", args=[exp.pk, section.slug])


def _trail(html):
    nav = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]
    return [re.sub(r"\s+", " ", m).strip()
            for m in re.findall(r"<(?:a|span)[^>]*>(.*?)</(?:a|span)>", nav, re.S)]


def _sidebar(html):
    return html.split('<nav class="sidebar">', 1)[1].split("</nav>", 1)[0]


@pytest.mark.parametrize("section", SECTIONS, ids=[s.slug for s in SECTIONS])
def test_every_section_renders_what_it_plans(client, section):
    """Each section shows its title and every planned capability."""
    exp = _exp()
    response = client.get(_url(exp, section))

    assert response.status_code == 200
    html = response.content.decode()
    assert str(section.title) in html
    for capability in section.planned:
        assert str(capability.title) in html


def test_the_first_sections_slug_is_not_a_second_address(client):
    """The overview lives only at the data root."""
    exp = _exp()
    url = reverse("ui:data_section", args=[exp.pk, SECTIONS[0].slug])
    assert client.get(url).status_code == 404


def test_an_unknown_section_is_a_404(client):
    exp = _exp()
    url = reverse("ui:data_section", args=[exp.pk, "no-such-section"])
    assert client.get(url).status_code == 404


def test_an_invisible_experiment_has_no_data_pages(client, settings):
    """The data pages exist only for experiments the reader can see."""
    exp = _exp()
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    assert client.get(_url(exp, SECTIONS[0])).status_code == 404
    assert client.get(_url(exp, SECTIONS[1])).status_code == 404


def test_every_section_sits_under_data_handling(client):
    """The sections are tabs on one page, as the dashboard's figures are."""
    exp = _exp()
    for section in SECTIONS[:2]:
        html = client.get(_url(exp, section)).content.decode()
        assert _trail(html) == ["Experiments", "Iris tuning", "Data Handling"]


def test_the_sections_are_tabs_on_the_page(client):
    """Each tab has its own address, and the requested one is selected and
    showing."""
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
    """The comparison drops the rename form's token and return page."""
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
    exp = _exp()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert f'href="{_url(exp, SECTIONS[0])}"' in _sidebar(html)


def test_the_section_slugs_are_unique():
    slugs = [s.slug for s in SECTIONS]
    assert len(slugs) == len(set(slugs))


def _section(slug):
    return next(s for s in SECTIONS if s.slug == slug)


def test_processing_choices_start_on_the_models_own_defaults(client):
    """The model's default is posted as "auto", so the choice keeps following
    the model."""
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
    """Leaving gaps is offered disabled for such a model."""
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
    """The `.ihpo` records the choice in its dataset section."""
    from experiments.services.snapshot import snapshot_from_experiment

    exp = _exp()
    client.post(_url(exp, _section("features")), {"scale": "standardize", "labels": "auto"})
    exp.refresh_from_db()

    assert exp.data.processing["scale"] == "standardize"
    assert snapshot_from_experiment(exp)["dataset"]["processing"] == {
        "missing": "auto", "scale": "standardize", "labels": "auto"}


def test_processing_is_fixed_once_there_are_trials(client):
    """Trials keep the processing they were scored under."""
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
    """The page's choice is the one a run's trials are scored under."""
    from core import io
    from core.registry import METRICS, MODELS, OPTIMIZERS
    from experiments.services.snapshot import snapshot_from_experiment

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


# Panels

def test_data_panels_and_figures_are_one_type():
    """Panels and figures share shape, width and template machinery."""
    from ui.figures.data_handling import DATA_PANELS
    from ui.figures import FIGURES, Panel

    assert all(issubclass(p, Panel) for p in DATA_PANELS + FIGURES)
    for panel in DATA_PANELS:
        assert panel.page == "data" and panel.template and panel.columns()


def test_every_data_panel_is_placed_once_on_its_tab():
    from ui.figures.data_handling import DATA_PANELS
    from ui import layout

    arranged = layout.read_data_default()
    for panel in DATA_PANELS:
        assert [s["key"] for s in arranged[panel.home_tab]].count(panel.key) == 1, panel.key


def test_a_layout_file_without_a_data_panel_is_refused(tmp_path):
    """A forgotten panel is an error, like a forgotten figure."""
    from ui import layout

    text = layout.DEFAULT_PATH.read_text()
    broken = tmp_path / "layout.toml"
    broken.write_text("\n".join(l for l in text.splitlines() if '"scaling"' not in l))
    with pytest.raises(layout.LayoutFileError, match="scaling is not placed"):
        layout.read_data_default(broken)


def _panel(html, slug):
    return html.split(f'id="fig-panel-{slug}"', 1)[1].split('class="fig-panel', 1)[0]


def test_a_created_experiments_overview_describes_its_data(client):
    """Overview holds no controls; each other tab saves its own."""
    exp = _exp()
    html = client.get(_url(exp, SECTIONS[0])).content.decode()
    overview, cleaning, features = (_panel(html, s) for s in ("overview", "cleaning", "features"))

    assert "<select" not in overview and "<form" not in overview
    assert 'data-key="profile"' in overview
    assert 'name="missing"' in cleaning and cleaning.count('type="submit"') == 1
    assert 'name="scale"' in features and 'name="labels"' in features
    assert 'class="fig-grip"' not in html


def test_a_drafts_overview_is_the_choices_to_make_first(client):
    """Each control is drawn once in its own tab and moved into the Overview's
    slot while shown."""
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
    """The draft Overview's arrangement is checked like the others."""
    from ui import layout

    text = layout.DEFAULT_PATH.read_text()
    head, setup = text.split("[setup]", 1)
    broken = tmp_path / "layout.toml"
    broken.write_text(head + "[setup]" + setup.replace('"scaling"', '"scalingg"'))
    with pytest.raises(layout.LayoutFileError, match="scalingg"):
        layout.read_setup_default(broken)
