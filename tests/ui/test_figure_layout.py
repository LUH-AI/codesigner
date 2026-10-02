"""How a reader arranges an experiment page, and how that is kept.

*What:* an arrangement is each tab's figures in order, each with a width in
grid columns (`ui/layout.py`). It is normalised against the catalog whenever it
is read or written, so a stored one stays valid as figures are added or change
shape. Each reader has their own per experiment; one who has arranged nothing
sees the site default, and without one of those the catalog's own order.

*How:* `normalize` directly, for what it repairs; the save endpoint through the
client, as different readers, for whose row a request writes and who may set
the site default.
"""

import json

import pytest
from django.urls import reverse

from access.models import Group, Membership
from ui import layout as layouts
from ui.figures import FIGURES, FIGURES_BY_KEY, OVERVIEW, SHAPES, TABS
from ui.models import DefaultFigureLayout, Experiment, FigureLayout

pytestmark = pytest.mark.django_db


def _keys(layout, tab):
    return [s["key"] for s in layout["tabs"][tab]]


# ── the default ──────────────────────────────────────────────────────────────

def test_every_figure_has_a_home_and_a_shape():
    """The rot guard: a figure added to the catalog has to say where it goes
    and what shape it is, or it has no place on the page."""
    for figure in FIGURES:
        if figure.in_sidebar:
            continue
        assert figure.home_tab in TABS and figure.home_tab != OVERVIEW, figure.key
        assert figure.shape in SHAPES, figure.key


def test_the_default_is_the_file():
    """config/figure_layout.toml, figure for figure, columns counted from 1
    there and from 0 here. It ships with the code, so every install opens on
    it; there is no other default."""
    import tomllib

    raw = tomllib.loads(layouts.DEFAULT_PATH.read_text(encoding="utf-8"))
    layout = layouts.default_layout()

    for tab in TABS:
        assert [(s["key"], s["w"], s["c"] + 1) for s in layout["tabs"][tab]] == [
            (e["figure"], e["width"], e["column"]) for e in raw[tab]], tab


def test_the_file_places_every_figure():
    """The rot guard the app itself applies at start-up: a figure added to the
    catalog and not placed in the file stops the app, rather than turning up
    wherever some other order would put it."""
    assert layouts.check() == []


def test_the_app_refuses_a_file_that_is_missing_or_wrong(tmp_path):
    good = layouts.DEFAULT_PATH.read_text(encoding="utf-8")
    cases = {
        "missing": None,
        "unreadable": "overview = [",
        "a figure left out": good.replace('{ figure = "prior",', '# { figure = "prior",'),
        "on the wrong tab": good.replace('figure = "partial_dependence"', 'figure = "prior"'),
        "too wide": good.replace('{ figure = "trials",                width = 2, column = 3 }',
                                 '{ figure = "trials", width = 4, column = 1 }'),
        "across the middle": good.replace(
            '{ figure = "local_effects",             width = 2, column = 3 }',
            '{ figure = "local_effects", width = 2, column = 2 }'),
    }
    for name, text in cases.items():
        path = tmp_path / f"{name}.toml"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        with pytest.raises(layouts.LayoutFileError):
            layouts.read_default(path)


# ── what normalising repairs ─────────────────────────────────────────────────

@pytest.mark.parametrize("stored", [None, [], "x", {"tabs": "x"}, {"version": 1}])
def test_anything_that_is_not_an_arrangement_is_the_default(stored):
    assert layouts.normalize(stored) == layouts.default_layout()


def test_unknown_figures_and_duplicates_are_dropped():
    layout = layouts.default_layout()
    layout["tabs"]["misc"].insert(0, {"key": "gone_figure", "w": 2})
    layout["tabs"]["misc"].append(dict(layout["tabs"]["misc"][1]))

    fixed = layouts.normalize(layout)

    assert "gone_figure" not in _keys(fixed, "misc")
    assert len(_keys(fixed, "misc")) == len(set(_keys(fixed, "misc")))


def test_a_figure_cannot_be_moved_off_its_home_tab():
    """Only Overview holds figures from other tabs."""
    layout = layouts.default_layout()
    layout["tabs"]["hypershap"].append({"key": "trials", "w": 2})

    fixed = layouts.normalize(layout)

    assert "trials" not in _keys(fixed, "hypershap")
    assert "trials" in _keys(fixed, "misc")


@pytest.mark.parametrize("key, asked, kept", [
    ("hyperparameter_importance", 9, 4),   # squat: 1, 2 or 4
    ("hyperparameter_importance", 3, 2),   # never 3: the widest that fits
    ("hyperparameter_importance", 0, 1),
    ("parallel_coordinates", 1, 2),        # long: 2 or 4
    ("parallel_coordinates", 3, 2),
    ("trials", 4, 2),                      # the table is always 2
    ("trial_duration", "wide", 1),         # nonsense: its opening width
])
def test_widths_are_held_to_the_shape(key, asked, kept):
    layout = layouts.default_layout()
    tab = FIGURES_BY_KEY[key].home_tab
    for slot in layout["tabs"][tab]:
        if slot["key"] == key:
            slot["w"] = asked

    fixed = layouts.normalize(layout)

    assert {s["key"]: s["w"] for s in fixed["tabs"][tab]}[key] == kept


@pytest.mark.parametrize("w, c, kept", [
    (1, 3, 3),         # a one-column figure may start in any column
    (2, 2, 2),         # a two-column one in 1 or 3 (0-based 0 or 2)…
    (2, 1, None),      # …never across the middle
    (4, 0, 0),
    (4, 2, None),
    (1, 4, None),      # off the grid
    (1, "2", None),    # not a number
    (1, True, None),
])
def test_a_column_is_kept_only_where_the_width_can_start(w, c, kept):
    """The column a figure stays in on the four-column grid, so that something
    above it growing moves it down rather than across. One its width cannot
    start in is dropped, and the page finds it one."""
    layout = layouts.default_layout()
    slot = next(s for s in layout["tabs"]["hypershap"] if s["key"] == "hyperparameter_importance")
    slot["w"], slot["c"] = w, c

    fixed = next(s for s in layouts.normalize(layout)["tabs"]["hypershap"]
                 if s["key"] == "hyperparameter_importance")

    assert fixed.get("c") == kept


def test_the_page_carries_each_column(client):
    """On the slot, for layout.js to place it by; absent where none is kept."""
    from core import io
    from tests.conftest import FIXTURES_DIR
    from ui.services import snapshot

    exp = snapshot.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    layout = layouts.default_layout()
    for slot in layout["tabs"]["misc"]:
        if slot["key"] == "trial_duration":
            slot["c"] = 3
        if slot["key"] == "best_configuration":
            slot.pop("c")
    layouts.save(exp, None, layout)

    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    misc = html.split('id="fig-panel-misc"', 1)[1]

    duration = misc[misc.index('data-key="trial_duration"'):][:300]
    assert 'data-c="3"' in duration
    best = misc[misc.index('data-key="best_configuration"'):][:300]
    assert "data-c=" not in best.split(">", 1)[0]


def test_a_figure_new_to_the_catalog_is_added_where_it_belongs():
    """Absent from its home tab means the arrangement predates it."""
    layout = layouts.default_layout()
    layout["tabs"]["misc"] = [s for s in layout["tabs"]["misc"] if s["key"] != "trials"]
    layout["tabs"][OVERVIEW] = [s for s in layout["tabs"][OVERVIEW] if s["key"] != "trials"]

    fixed = layouts.normalize(layout)

    assert _keys(fixed, "misc")[-1] == "trials"
    assert "trials" in _keys(fixed, OVERVIEW), "the default file pins it"


def test_but_one_missing_only_from_overview_was_unpinned():
    layout = layouts.default_layout()
    layout["tabs"][OVERVIEW] = [s for s in layout["tabs"][OVERVIEW] if s["key"] != "trials"]

    assert "trials" not in _keys(layouts.normalize(layout), OVERVIEW)


def test_overview_may_hold_any_figure():
    layout = layouts.default_layout()
    layout["tabs"][OVERVIEW].append({"key": "interactions_graph", "w": 4})

    assert {"key": "interactions_graph", "w": 4} in layouts.normalize(layout)["tabs"][OVERVIEW]


def test_a_switched_off_figure_keeps_its_place():
    """Skipped on the page, kept in the arrangement, so turning it back on puts
    it back where it was."""
    shown = [f for f in FIGURES if f.key != "trial_duration"]
    layout = layouts.default_layout()

    placed = {t["tab"]: [s["figure"].key for s in t["slots"]]
              for t in layouts.placed(layout, shown)}

    assert "trial_duration" not in placed["misc"]
    assert "trial_duration" in _keys(layouts.normalize(layout), "misc")


# ── whose arrangement ────────────────────────────────────────────────────────

@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


@pytest.fixture
def lab(django_user_model):
    group = Group.objects.create(name="lab", user_limit=9)
    people = {}
    for name in ("ana", "ben", "cleo"):
        user = django_user_model.objects.create_user(username=name, password="pw")
        Membership.objects.create(user=user, group=group,
                                  role=Membership.LEAD if name == "ben" else Membership.MEMBER)
        people[name] = django_user_model.objects.get(pk=user.pk)
    exp = Experiment.objects.create(
        name="wine", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=people["ana"])
    return {"exp": exp, **people}


def _save(client, exp, body):
    return client.post(reverse("ui:save_layout", args=[exp.pk]),
                       data=json.dumps(body), content_type="application/json")


def _narrowed():
    layout = layouts.default_layout()
    layout["tabs"]["misc"].reverse()
    return layout


def test_a_reader_keeps_their_own_arrangement(client, hosted, lab):
    client.force_login(lab["ana"])

    resp = _save(client, lab["exp"], {"layout": _narrowed()})

    assert resp.status_code == 200
    assert layouts.layout_for(lab["exp"], lab["ana"]) == layouts.normalize(_narrowed())
    assert layouts.layout_for(lab["exp"], lab["ben"]) == layouts.default_layout(), \
        "another reader's page is untouched"


def test_a_reader_who_may_only_look_can_still_arrange(client, hosted, lab):
    """Ben leads the group and reaches Ana's experiment without any grant to
    change it. The arrangement is his, not the experiment's."""
    client.force_login(lab["ben"])

    assert _save(client, lab["exp"], {"layout": _narrowed()}).status_code == 200
    assert FigureLayout.objects.get().user == lab["ben"]


def test_somebody_who_cannot_see_it_cannot_arrange_it(client, hosted, lab):
    client.force_login(lab["cleo"])

    assert _save(client, lab["exp"], {"layout": _narrowed()}).status_code == 404
    assert not FigureLayout.objects.exists()


def test_the_page_opens_as_it_was_left(client, hosted, lab):
    """On an experiment with results, which is when the page has tabs."""
    from core import io
    from tests.conftest import FIXTURES_DIR
    from ui.services import snapshot

    exp = snapshot.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()), owner=lab["ana"])
    client.force_login(lab["ana"])
    _save(client, exp, {"layout": _narrowed()})

    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).context

    from ui.figures import FIGURES_BY_KEY

    misc = next(t for t in page["figure_tabs"] if t["tab"] == "misc")
    # The first of the reader's order that this experiment shows: a figure for
    # forecasts only (Figure.forecast_only) has no slot on its page.
    first = next(s["key"] for s in _narrowed()["tabs"]["misc"]
                 if not FIGURES_BY_KEY[s["key"]].forecast_only)
    assert misc["slots"][0]["figure"].key == first


def test_reset_goes_back_to_the_site_default(client, hosted, lab):
    client.force_login(lab["ana"])
    _save(client, lab["exp"], {"layout": _narrowed()})

    resp = _save(client, lab["exp"], {"reset": True})

    assert json.loads(resp.content)["layout"] == layouts.default_layout()
    assert not FigureLayout.objects.exists()


def test_the_site_default_is_what_a_new_reader_sees(client, hosted, lab):
    assert layouts.layout_for(lab["exp"], lab["ben"]) == layouts.default_layout()


def test_the_page_cannot_set_the_site_default(client, hosted, lab):
    """There is no button for it, and the endpoint does not take it from
    anybody: what is sent is kept as the sender's own."""
    client.force_login(lab["ana"])

    _save(client, lab["exp"], {"layout": _narrowed(), "site_default": True})

    assert layouts.site_default() == layouts.default_layout()
    assert layouts.layout_for(lab["exp"], lab["ana"]) == layouts.normalize(_narrowed())


def test_the_settings_page_resets_it(client, hosted, lab):
    """An ordinary form there, so it goes back to the experiment afterwards."""
    client.force_login(lab["ana"])
    _save(client, lab["exp"], {"layout": _narrowed()})

    page = client.get(reverse("ui:experiment_settings", args=[lab["exp"].pk])).content.decode()
    resp = client.post(reverse("ui:save_layout", args=[lab["exp"].pk]), {"reset": "1"})

    assert reverse("ui:save_layout", args=[lab["exp"].pk]) in page
    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:experiment_detail", args=[lab["exp"].pk])
    assert not FigureLayout.objects.exists()


# ── a reader's own default ───────────────────────────────────────────────────

def _settings_post(client, exp, button):
    return client.post(reverse("ui:save_layout", args=[exp.pk]), {button: "1"})


@pytest.fixture
def second(lab):
    return Experiment.objects.create(
        name="iris", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=lab["ana"])


def test_a_reader_can_make_an_arrangement_their_default(client, hosted, lab, second):
    """Every experiment they have not arranged opens on it; nobody else's
    does, and nor does one they have."""
    client.force_login(lab["ana"])
    _save(client, lab["exp"], {"layout": _narrowed()})

    resp = _settings_post(client, lab["exp"], "make_default")

    assert resp.status_code == 302
    assert layouts.layout_for(second, lab["ana"]) == layouts.normalize(_narrowed())
    assert layouts.layout_for(second, lab["ben"]) == layouts.default_layout()


def test_resetting_an_experiment_goes_to_the_readers_default(client, hosted, lab, second):
    client.force_login(lab["ana"])
    _save(client, second, {"layout": _narrowed()})
    _settings_post(client, second, "make_default")
    _save(client, lab["exp"], {"layout": layouts.default_layout()})

    _settings_post(client, lab["exp"], "reset")

    assert layouts.layout_for(lab["exp"], lab["ana"]) == layouts.normalize(_narrowed())


def test_resetting_the_default_goes_back_to_the_sites(client, hosted, lab, second):
    client.force_login(lab["ana"])
    mine = layouts.default_layout()
    mine["tabs"]["hypershap"].reverse()
    layouts.save_user_default(lab["ana"], mine)

    _settings_post(client, lab["exp"], "reset_default")

    assert layouts.layout_for(second, lab["ana"]) == layouts.default_layout()


def test_without_accounts_there_is_one_readers_default(client):
    """One reader, one row — and the site's default, the file, is untouched by
    it either way."""
    exp = Experiment.objects.create(
        name="solo", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    _save(client, exp, {"layout": _narrowed()})

    _settings_post(client, exp, "make_default")
    _settings_post(client, exp, "make_default")

    assert DefaultFigureLayout.objects.get().user is None
    assert layouts.user_default(None) == layouts.normalize(_narrowed())
    assert layouts.site_default() == layouts.default_layout()
    _settings_post(client, exp, "reset_default")
    assert not DefaultFigureLayout.objects.exists()


def test_the_settings_page_offers_all_three(client, hosted, lab):
    client.force_login(lab["ana"])

    page = client.get(reverse("ui:experiment_settings", args=[lab["exp"].pk])).content.decode()

    for name in ("reset", "make_default", "reset_default"):
        assert f'name="{name}"' in page, name


@pytest.mark.parametrize("body", [b"not json", b"[]", b"{}", b'{"layout": 3}'])
def test_a_malformed_request_is_refused(client, hosted, lab, body):
    client.force_login(lab["ana"])

    resp = client.post(reverse("ui:save_layout", args=[lab["exp"].pk]),
                       data=body, content_type="application/json")

    assert resp.status_code == 400


def test_without_accounts_there_is_one_arrangement(client):
    exp = Experiment.objects.create(
        name="solo", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)

    _save(client, exp, {"layout": _narrowed()})
    _save(client, exp, {"layout": layouts.default_layout()})

    row = FigureLayout.objects.get()
    assert row.user is None and row.layout == layouts.default_layout()
