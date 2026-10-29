"""Each stored game contains gains over the config space default. The incumbent and
selected-trial baselines depend on the metric and the click, so
`hypershap_figures` computes them on request.
"""

import pytest
from django.urls import reverse

from core import io
from ui.figures import HP_GAME_FIELDS
from experiments.services import snapshot as adapter
from ui.experiment_settings import SETTING_DEFAULTS

from tests.conftest import FIXTURES_DIR

GAME_FIGURES = {"hyperparameter_importance", "interactions_heatmap", "interactions_top_pairs",
                "interactions_graph", "interactions_coalitions", "interactions_orders"}


def _experiment(fixture="test2.ihpo"):
    """`_experiment` imports an experiment from a stored .ihpo fixture."""
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / fixture).read_bytes()))


def _figures(client, exp, metric="accuracy", idx=8):
    url = reverse("ui:hypershap_figures", args=[exp.pk])
    return client.get(url + f"?metric={metric}&idx={idx}")


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def test_the_games_come_back_for_every_game_figure(client):
    """One request covers every game figure for every game, so switching game
    costs no second request.
    """
    exp = _experiment()
    data = _figures(client, exp).json()
    assert exp.data.result is not None
    hps = set(next(iter(exp.data.result["configs"].values())))

    assert set(data["plots"]) == GAME_FIGURES
    for game in HP_GAME_FIELDS:
        assert data["plots"]["interactions_heatmap"][game] is not None
        assert data["plots"]["hyperparameter_importance"][f"{game}-bar"] is not None
        assert game in data["warnings"]
        assert set(data["importance"][game]) == hps
        assert sum(data["importance"][game].values()) == pytest.approx(1.0)


def test_the_games_differ_from_the_stored_default_ones(client):
    """The comparison uses mistunability, since tunability from the incumbent
    can be near zero everywhere.
    """
    exp = _experiment("analytics.ihpo")
    stored = client.get(reverse("ui:metric_figures", args=[exp.pk]) + "?metric=accuracy").json()
    fetched = _figures(client, exp, idx=0).json()["plots"]

    def z(plots):
        return plots["interactions_heatmap"]["mistunability"]["data"][0]["z"]

    assert z(fetched) != z(stored)


def test_a_switched_off_figure_is_not_computed(client):
    from experiments.models import GlobalSettings

    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"show_interactions_graph": False}
    gs.save(update_fields=["default_experiment_settings"])

    data = _figures(client, _experiment()).json()

    assert "interactions_graph" not in data["plots"]
    assert "hyperparameter_importance" in data["plots"]


def test_rejects_an_unknown_metric(client):
    assert _figures(client, _experiment(), metric="not-a-metric").status_code == 400


@pytest.mark.parametrize("idx", ["-1", "9999", "not-a-number", ""])
def test_rejects_a_trial_that_is_not_there(client, idx):
    assert _figures(client, _experiment(), idx=idx).status_code == 400


def test_a_trial_explained_against_itself_says_so_rather_than_drawing_zeros(client):
    """A trial compared with itself gets no figure and a warning, since a
    waterfall of zeros would read as a finding.
    """
    exp = _experiment()
    url = reverse("ui:trial_ablation", args=[exp.pk]) + "?metric=accuracy&idx=8&baseline=8"
    data = client.get(url).json()

    assert data["figure"] is None
    assert data["warning"]


def test_a_trial_can_be_explained_against_another_trial(client):
    exp = _experiment()
    base = reverse("ui:trial_ablation", args=[exp.pk]) + "?metric=accuracy&idx=0"
    from_default = client.get(base).json()
    from_best = client.get(base + "&baseline=8").json()

    assert from_best["warning"] is None
    assert from_best["figure"]["data"][0]["type"] == "waterfall"
    assert from_best["figure"]["data"][0]["y"] != from_default["figure"]["data"][0]["y"]


@pytest.mark.parametrize("baseline", ["-1", "9999", "not-a-number"])
def test_the_local_explanation_rejects_a_baseline_that_is_not_there(client, baseline):
    exp = _experiment()
    url = (reverse("ui:trial_ablation", args=[exp.pk])
           + f"?metric=accuracy&idx=0&baseline={baseline}")
    assert client.get(url).status_code == 400


def test_local_effects_can_be_measured_from_a_trial(client):
    """The beeswarm moves with the baseline."""
    exp = _experiment()
    base = reverse("ui:local_effects", args=[exp.pk]) + "?metric=accuracy"
    from_default = client.get(base).json()
    from_best = client.get(base + "&baseline=8").json()

    assert from_best["figure"] is not None
    assert from_best["figure"]["data"] != from_default["figure"]["data"]


def test_the_sidebar_section_is_hypershap_and_opens_on_the_incumbent(client):
    """The sidebar's "Compare against" selector offers default, incumbent and
    selected, with incumbent preselected.
    """
    html = _page(client, _experiment("analytics.ihpo"))
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]
    select = sidebar.split('id="hypershap-baseline"', 1)[1].split("</select>", 1)[0]

    assert "<h2>HyperSHAP</h2>" in sidebar
    for value in ("default", "incumbent", "selected"):
        assert f'value="{value}"' in select
    assert 'value="incumbent" selected' in select
    assert 'value="default" selected' not in select


def test_the_baseline_computation_fetches_without_being_asked_by_default():
    """`autocompute_hypershap_baseline` defaults on because six figures open on
    its result.
    """
    assert SETTING_DEFAULTS["autocompute_hypershap_baseline"] is True


def test_a_click_moves_the_figures_when_the_selected_trial_is_the_baseline(client):
    """Under "Selected", `selectionMoved` refreshes every HyperSHAP figure, and
    both `selectTrial` and `selectProposal` call it.
    """
    html = _page(client, _experiment("analytics.ihpo"))
    def body(name):
        return html.split(f"function {name}(", 1)[1].split("\n            }\n", 1)[0]

    moved = body("selectionMoved")
    assert 'currentBaseline() === "selected"' in moved
    assert "refreshHyperSHAP(metric)" in moved
    assert "refreshLocalEffects(metric)" in moved
    assert "refreshLocalAblation(metric)" in moved
    assert "selectionMoved(metric)" in body("selectTrial")
    assert "selectionMoved(metric)" in body("selectProposal")


def test_the_table_has_a_column_for_every_baseline(client):
    """Each table has default, incumbent and selected columns, and the page
    fills the empty two as games arrive.
    """
    html = _page(client, _experiment("analytics.ihpo"))
    table = html.split('class="importance-table"', 1)[1].split("</table>", 1)[0]
    header = table.split("</thead>", 1)[0]
    row = table.split("<tr data-hp=", 1)[1].split("</tr>", 1)[0]

    assert header.index("Default") < header.index("Incumbent") < header.index("Selected")
    assert '<td data-baseline="incumbent"></td>' in row
    assert '<td data-baseline="selected"></td>' in row
    assert row.split('<td data-baseline="default">', 1)[1].split("</td>", 1)[0].strip()


def test_the_bars_offer_every_baseline_to_compare_with(client):
    """The bar comparison offers none plus all three baselines and starts at
    none; the page hides the active baseline.
    """
    html = _page(client, _experiment("analytics.ihpo"))
    group = html.split("data-bar-comparison", 1)[1].split("</span>", 1)[0]

    for value in ("none", "selected", "incumbent", "default"):
        assert f'name="importance-comparison" value="{value}"' in group
    assert 'value="none" checked' in group
