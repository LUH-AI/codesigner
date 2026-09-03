"""One explanation game, chosen once, answered by every HyperSHAP figure.

The game selector used to live inside the importance figure and move only that
figure. Every interaction figure read tunability whatever it said — so reading
the importance figure under sensitivity beside the heatmap under tunability was
comparing two different questions and looking like one answer.

It is now a sidebar section, and all three games' interactions and Möbius terms
are stored. That costs storage and no time at all: `compute_hp_games` always
computed the three, and for a long time two were discarded for want of a field.
"""

import json

from django.urls import reverse

from core import io
from ui.figures import FIGURES, HP_GAME_FIELDS, HP_GAME_LABELS
from ui.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR, export_ihpo

#: In the selector's own order — mistunability is tunability's mirror, what
#: there is to lose against what there is to gain, and the two read as a pair.
GAMES = ["tunability", "mistunability", "sensitivity"]


def _experiment():
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def test_every_game_names_all_four_of_its_fields():
    """A game is not only an importance. The same HyperSHAP call produces the
    order-2 grid and the Möbius decomposition, and a figure reading any of them
    reads the game the sidebar names — so the mapping has to be complete or a
    figure silently falls back to tunability."""
    assert list(HP_GAME_FIELDS) == GAMES
    for game, fields in HP_GAME_FIELDS.items():
        assert set(fields) == {"importance", "warning", "interactions", "moebius"}
        assert game in HP_GAME_LABELS


def test_the_selector_is_in_the_sidebar_and_not_in_any_figure(client):
    """One choice about the experiment, not one per figure."""
    html = _page(client, _experiment())
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]
    content = html.split("<main", 1)[-1]

    assert "Explanation Game" in sidebar
    assert 'id="explanation-game"' in sidebar
    for game in GAMES:
        assert f'value="{game}"' in sidebar
    assert 'id="explanation-game"' not in content
    assert "view-select-hyperparameter_importance-game" not in html


def test_the_interaction_figures_ship_a_payload_per_game(client):
    """Switching game must not cost a round trip, so every game's numbers ride
    along — the same "one payload per option" the view machinery already does
    for the importance figure's renderings."""
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
    """The point of the whole change: sensitivity's heatmap is sensitivity's,
    not tunability's shown under another name."""
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
    """They are stored, so they survive a round trip — an experiment reopened
    from its file answers the same three questions it did before."""
    exp = _experiment()
    body = json.loads(
        export_ihpo(client, exp.pk).content)

    for game, fields in HP_GAME_FIELDS.items():
        for part in ("interactions", "moebius"):
            assert body["result"][fields[part]], f"{game} {part}"


def test_a_file_written_before_this_still_opens(client):
    """Two of the six fields are absent from every .ihpo written earlier, which
    has to read as "that game has no interactions" rather than as an error."""
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
    """It explains one trial rather than the search, and has no interactions to
    give the five figures that read them — so it is not a fourth game."""
    local = next(f for f in FIGURES if f.key == "local_explanation")
    html = _page(client, _experiment())

    assert local.plot(None) is None, "fetched, never precomputed"
    assert local.deferred, "and it waits to be asked"
    assert 'data-figure="local_explanation"' in html
    assert "Local (selected trial)" not in html, "no longer an option on a selector"


def test_the_projection_opens_on_pls(client):
    """It answers the question the page is actually about — which directions
    through this space moved the metric — where axes needs three decisions
    before it says anything at all.

    Declared twice, in the catalog and in the markup, because the page has to
    open on the view its selector claims to be on: a selector saying one thing
    beside a figure drawing another is the desync `readViewSelectors` exists to
    prevent, and this is where it would come from.
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
