"""What the assistant knows of an experiment's page beyond its form: which tab
of figures shows, what every tab holds, the trials table and the selected
trial (assistant/static/assistant/act.js reports them, `BrowserPage.view`
keeps them); pointing at a tab, a figure or a trial; and asking the page what
it shows when there is no record of it (`acts.current`).
"""

import json

import pytest
from django.urls import reverse

from assistant import acts
from assistant.gateway import Context
from assistant.models import BrowserPage, ClientAct, Conversation
from assistant.prompts import system_prompt
from assistant.turns import tools_for
from core import io
from experiments.services import snapshot
from tests.conftest import FIXTURES_DIR

from .evals.page import SimulatedPage

pytestmark = pytest.mark.django_db

VIEW = {
    "tab": "hypershap",
    "tabs": [
        {"name": "overview", "label": "Overview", "figures": [
            {"key": "trials", "label": "Trials (40)", "home": False}]},
        {"name": "hypershap", "label": "HyperSHAP", "figures": [
            {"key": "hyperparameter_importance", "label": "Importance", "home": True}]},
        {"name": "misc", "label": "Misc", "figures": [
            {"key": "trials", "label": "Trials (40)", "home": True}]},
    ],
    "outside_tabs": [{"key": "selected_configuration", "label": "Selected"}],
    "trials": {"count": 40, "first": 1, "last": 40},
    "selected_trial": 7,
}


@pytest.fixture
def on(settings):
    settings.ASSISTANT_ENABLED = True
    settings.ASSISTANT_ACT_SECONDS = 0


@pytest.fixture
def guided():
    """The guided SMAC fixture: 40 trials over two runs."""
    return snapshot.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "smac_iris_guided.ihpo").read_bytes()))


def _open(client, exp, view=VIEW):
    """A conversation about *exp*, its dashboard reported open with *view*."""
    client.get("/")
    client.session.save()
    conversation = Conversation.objects.create(browser=client.session.session_key,
                                               experiment=exp, phase=Conversation.DASHBOARD)
    client.post(reverse("assistant:page"),
                json.dumps({"path": reverse("ui:experiment_detail", args=[exp.pk]),
                            "title": "Dashboard", "fields": [], "targets": [], "view": view}),
                content_type="application/json")
    return conversation


def _run(conversation, tool, **args):
    return tools_for(conversation).execute(tool, args, Context(conversation))


def test_a_page_reports_its_tabs_figures_and_trials(client, on, guided):
    """What: the view a page reports is kept as sent, and when it was reported.
    How: a dashboard reported on its HyperSHAP tab with trial 7 selected."""
    _open(client, guided)
    page = BrowserPage.objects.get()

    assert page.view == VIEW
    assert page.reported_at is not None


def test_a_reported_view_is_kept_to_its_fields(client, on, guided):
    """What: anything a view holds beyond its fields is dropped, values are cut
    to size, and a trials table of anything but whole numbers is left out.
    How: a view with a stray field, a long label and a trials count as text."""
    _open(client, guided, {"tab": "misc", "secret": "x", "tabs": [
        {"name": "misc", "label": "L" * 900, "figures": [{"key": "trials", "script": "y"}]}],
        "trials": {"count": "40", "first": 1, "last": 40}})
    view = BrowserPage.objects.get().view

    assert set(view) == {"tab", "tabs", "outside_tabs"}
    assert len(view["tabs"][0]["label"]) == 500
    assert view["tabs"][0]["figures"] == [{"key": "trials", "label": None}]


def test_look_at_page_names_the_tab_and_where_each_figures_numbers_are_read(client, on, guided):
    """What: look_at_page gives the tab showing, every tab's figures with the
    target that points at each and the tool that reads the numbers behind it,
    the trials table and the selected trial.
    How: the reported dashboard, then the tool."""
    seen = json.loads(_run(_open(client, guided), "look_at_page").output)

    assert seen["tab_showing"] == "hypershap" and seen["tabs"][1]["target"] == "tab:hypershap"
    importance = seen["tabs"][1]["figures"][0]
    assert importance["target"] == "figure:hyperparameter_importance"
    assert importance["numbers_from"] == "analyze_experiment(analysis=importance)"
    assert seen["trials_table"]["target"] == "figure:trials"
    assert {"count": 40, "first": 1, "last": 40}.items() <= seen["trials_table"].items()
    assert "performance_over_time" in seen["trials_table"]["figures_showing_a_trial"]
    assert seen["selected_trial"] == 7


def test_the_system_prompt_says_which_tab_shows_and_what_is_on_it(client, on, guided):
    """What: the model is told the tab showing, its figures by target, the
    other tabs and the selected trial, without calling a tool.
    How: the system prompt of a conversation whose dashboard is on HyperSHAP."""
    told = system_prompt(_open(client, guided))

    assert 'showing the "HyperSHAP" tab (tab:hypershap)' in told
    assert "Importance (figure:hyperparameter_importance)" in told
    assert "Its other tabs: overview, misc." in told
    assert "The figures show trial 7." in told


@pytest.mark.parametrize("target", ["tab:misc", "figure:trials", "figure:selected_configuration"])
def test_the_assistant_points_at_tabs_and_figures(client, on, guided, target):
    """What: highlight sends a tab or a figure, in a tab or outside them, that
    the page reported.
    How: each kind of target on the reported dashboard."""
    _run(_open(client, guided), "highlight", target=target)

    assert ClientAct.objects.get().args == {"target": target}


@pytest.mark.parametrize("target", ["figure:trials", "figure:selected_configuration"])
def test_a_trial_is_shown_in_a_figure_that_shows_the_selection(client, on, guided, target):
    """What: highlight with `trial` sends the figure and the trial's number, for
    any figure that shows the selected trial.
    How: the trials table, then the sidebar's selected configuration, with
    trial 40."""
    _run(_open(client, guided), "highlight", target=target, trial=40)

    assert ClientAct.objects.get().args == {"target": target, "trial": 40}


@pytest.mark.parametrize("target, trial, said", [
    ("tab:timeline", None, "tabs are"), ("figure:forecast", None, "no figure 'forecast'"),
    ("figure:trials", 41, "trials 1 to 40"),
    ("figure:hyperparameter_importance", 3, "shows the selected trial"),
    ("tab:misc", 3, "shows the selected trial"),
    ("trial:3", None, "pass the trial's number as `trial`")])
def test_what_the_page_does_not_have_is_not_pointed_at(client, on, guided, target, trial, said):
    """What: a tab, figure or trial the page did not report is refused, saying
    what there is, and so is a trial asked for in a figure or tab that does not
    show the selected trial, or named as a target.
    How: a missing tab, a figure off the page, a trial past the table, a trial
    in the importance figure and in a tab, and the old `trial:` target."""
    args = {"target": target} if trial is None else {"target": target, "trial": trial}
    refused = _run(_open(client, guided), "highlight", **args)

    assert said in refused.output and not ClientAct.objects.exists()


def test_a_trial_needs_trials_on_the_page(client, on, guided):
    """What: selecting a trial on a page that lists none is refused.
    How: a dashboard reported without a trials table."""
    refused = _run(_open(client, guided, {**VIEW, "trials": None}), "highlight",
                   target="figure:trials", trial=3)

    assert "lists no trials" in refused.output


def test_navigate_opens_a_dashboard_tab(client, on, guided):
    """What: navigate with a tab sends the dashboard's address with the tab, and
    refuses a tab on another page or one the dashboard lacks.
    How: from the timeline, the HyperSHAP tab, then a tab of the timeline and
    an unknown tab."""
    conversation = _open(client, guided)
    BrowserPage.objects.update(path=reverse("ui:experiment_timeline", args=[guided.pk]))
    _run(conversation, "navigate", page="dashboard", tab="hypershap")
    elsewhere = _run(conversation, "navigate", page="timeline", tab="hypershap")
    unknown = _run(conversation, "navigate", page="dashboard", tab="nowhere")

    url = reverse("ui:experiment_detail", args=[guided.pk]) + "#hypershap"
    assert ClientAct.objects.get().args == {"url": url}
    assert "only the dashboard" in elsewhere.output and "tabs are" in unknown.output


def test_without_a_record_the_page_is_asked_what_it_shows(client, on, guided, monkeypatch):
    """What: with no record of the person's page, `current` asks the page to
    report and uses what it reports; with a record, nothing is asked.
    How: a page that answers the report act while it waits, then a second
    call."""
    client.get("/")
    client.session.save()
    conversation = Conversation.objects.create(browser=client.session.session_key,
                                               experiment=guided, phase=Conversation.DASHBOARD)
    page = SimulatedPage(client, conversation.browser)

    def answered(act, seconds):
        page.open(reverse("ui:experiment_detail", args=[guided.pk]) + "#misc")
        act.status = ClientAct.DONE
        act.save(update_fields=["status"])
        return act

    monkeypatch.setattr(acts, "wait", answered)
    found = acts.current(conversation)
    again = acts.current(conversation)

    assert found is not None and found.view["tab"] == "misc"
    assert again == found
    assert ClientAct.objects.filter(kind=ClientAct.REPORT).count() == 1


def test_the_dashboard_has_what_the_page_script_reads(client, on, guided):
    """What: the dashboard's markup has what act.js reads for the view: tabs
    with their names, figure slots with their keys and home tabs, figure cards
    with their titles, and trial rows with their numbers.
    How: the rendered dashboard read as act.js would (tests/assistant/evals/
    page.py), opened on the HyperSHAP tab."""
    client.get("/")
    client.session.save()
    page = SimulatedPage(client, client.session.session_key)
    page.open(reverse("ui:experiment_detail", args=[guided.pk]) + "#hypershap")
    view = page.view
    tabs = {t["name"]: t for t in view["tabs"]}

    assert view["tab"] == "hypershap"
    assert {"overview", "hypershap", "acquisition", "misc"} <= set(tabs)
    homes = {f["key"] for t in tabs.values() for f in t["figures"] if f["home"]}
    assert {"trials", "hyperparameter_importance"} <= homes
    assert any(f["label"].startswith("Trials") for f in tabs["misc"]["figures"])
    assert view["trials"] == {"count": 40, "first": 1, "last": 40}


def test_a_highlight_says_exactly_what_it_outlined_and_what_shows_now(client, on, guided,
                                                                       monkeypatch):
    """What: a highlight carried out is answered with its one target, the tab
    the page shows afterwards, and for a trial that every figure shows it.
    How: a simulated dashboard on the Misc tab carrying out a highlight of a
    tab, then of trial 12."""
    client.get("/")
    client.session.save()
    conversation = Conversation.objects.create(browser=client.session.session_key,
                                               experiment=guided, phase=Conversation.DASHBOARD)
    page = SimulatedPage(client, conversation.browser)
    page.open(reverse("ui:experiment_detail", args=[guided.pk]) + "#misc")

    def carried_out(act, seconds):
        page.carry_out(act)
        return act

    monkeypatch.setattr(acts, "wait", carried_out)
    tab = _run(conversation, "highlight", target="tab:hypershap").output
    trial = _run(conversation, "highlight", target="figure:trials", trial=12).output

    assert tab.startswith("Outlined tab:hypershap") and "The HyperSHAP tab is showing." in tab
    # The trials table is pinned to Overview, so that copy is the one shown.
    assert "The Overview tab is showing." in trial and "Trial 12 is selected" in trial


def test_navigate_does_not_reopen_the_page_already_open(client, on, guided):
    """What: navigating to the page the person has open is refused, pointing at
    highlight for its tabs and figures.
    How: navigate to the dashboard, with and without a tab, from the dashboard."""
    conversation = _open(client, guided)
    plain = _run(conversation, "navigate", page="dashboard")
    tabbed = _run(conversation, "navigate", page="dashboard", tab="misc")

    assert "already has that page open" in plain.output and "highlight" in tabbed.output
    assert not ClientAct.objects.exists()
