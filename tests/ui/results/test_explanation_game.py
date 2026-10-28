"""One sidebar selector picks the explanation game for every HyperSHAP figure.
The result stores all three games' interactions and Möbius terms, which
`compute_hp_games` computes anyway.
"""

import json

from django.urls import reverse

from core import io
from ui.figures import FIGURES, HP_GAME_FIELDS, HP_GAME_LABELS
from experiments.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR, export_ihpo

#: Selector order, with tunability and its mirror mistunability paired.
GAMES = ["tunability", "mistunability", "sensitivity"]


def _experiment():
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def test_every_game_names_all_four_of_its_fields():
    """`HP_GAME_FIELDS` must name all four fields per game, since every figure
    reads the game the sidebar names."""
    assert list(HP_GAME_FIELDS) == GAMES
    for game, fields in HP_GAME_FIELDS.items():
        assert set(fields) == {"importance", "warning", "interactions", "moebius"}
        assert game in HP_GAME_LABELS


def test_the_selector_is_in_the_sidebar_and_not_in_any_figure(client):
    html = _page(client, _experiment())
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]
    content = html.split("<main", 1)[-1]

    assert "<h2>HyperSHAP</h2>" in sidebar
    assert 'id="explanation-game"' in sidebar
    for game in GAMES:
        assert f'value="{game}"' in sidebar
    assert 'id="explanation-game"' not in content
    assert "view-select-hyperparameter_importance-game" not in html


def test_the_interaction_figures_ship_a_payload_per_game(client):
    """Switching game costs no round trip, because the page carries every game's
    payload."""
    exp = _experiment()
    html = _page(client, exp)
    plots = json.loads(html.split('id="metric-plots-data"', 1)[1]
                       .split(">", 1)[1].split("</script>", 1)[0])
    metric = next(iter(plots))

    for figure in FIGURES:
        if not figure.key.startswith("interactions_"):
            continue
        assert set(plots[metric][figure.key]) == set(GAMES), figure.key


def test_the_other_two_games_carry_their_own_interactions(client):
    exp = _experiment()
    html = _page(client, exp)
    plots = json.loads(html.split('id="metric-plots-data"', 1)[1]
                       .split(">", 1)[1].split("</script>", 1)[0])
    metric = next(iter(plots))
    grids = {game: plots[metric]["interactions_heatmap"][game]["data"][0]["z"]
             for game in GAMES}

    for game in GAMES:
        assert grids[game], f"{game} has no interaction grid"
    assert grids["tunability"] != grids["sensitivity"]
    assert grids["tunability"] != grids["mistunability"]


def test_an_exported_file_carries_every_game(client):
    exp = _experiment()
    body = json.loads(
        export_ihpo(client, exp.pk).content)

    for game, fields in HP_GAME_FIELDS.items():
        for part in ("interactions", "moebius"):
            assert body["result"][fields[part]], f"{game} {part}"


def test_a_file_written_before_this_still_opens(client):
    """Older .ihpo files lack the sensitivity and mistunability fields, and the
    page reads that as no interactions."""
    snapshot = io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes())
    for field in ("hyperparameter_sensitivity_interactions",
                  "hyperparameter_sensitivity_moebius",
                  "hyperparameter_mistunability_interactions",
                  "hyperparameter_mistunability_moebius"):
        snapshot["result"].pop(field, None)
    exp = adapter.experiment_from_snapshot(snapshot)

    html = _page(client, exp)
    plots = json.loads(html.split('id="metric-plots-data"', 1)[1]
                       .split(">", 1)[1].split("</script>", 1)[0])
    metric = next(iter(plots))

    assert plots[metric]["interactions_heatmap"]["tunability"]
    assert plots[metric]["interactions_heatmap"]["sensitivity"] is None


def test_the_local_explanation_is_its_own_figure(client):
    """`local_explanation` covers one trial and has no interactions, so it stays
    a separate figure outside the game selector."""
    local = next(f for f in FIGURES if f.key == "local_explanation")
    html = _page(client, _experiment())

    assert local.plot(None) is None, "fetched, never precomputed"
    assert local.deferred, "and it waits to be asked"
    assert 'data-figure="local_explanation"' in html
    assert "Local (selected trial)" not in html, "no longer an option on a selector"


def test_the_projection_opens_on_pls(client):
    """`configuration_cube` declares PLS in the catalog and in the markup, so
    the selector names the view the figure draws.
    """
    from ui.figures import FIGURES_BY_KEY

    figure = FIGURES_BY_KEY["configuration_cube"]
    html = _page(client, _experiment())
    picker = html.split('id="view-select-configuration_cube"', 1)[1].split("</select>", 1)[0]

    assert figure.default_view == "pls"
    assert figure.opening_view() == "pls"
    assert figure.default_view != figure.views[0], "and the selector still reads simplest-first"
    assert f'<option value="{figure.default_view}" selected>' in picker
    assert picker.count("selected") == 1
