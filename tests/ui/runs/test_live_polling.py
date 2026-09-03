"""What the run-status poll costs, and what it carries.

Three things that were wrong once the figures started updating live:

- it asked every two seconds however long a trial took, so a run of ten-minute
  trials paid three hundred requests per new row;
- it kept asking with the tab in the background, where nobody could see the
  answer;
- and the trials table, alone among the trial-based figures, did not move —
  the figures beside it filled in while the table stayed at whatever it held
  when the page loaded.
"""

import json

import pytest
from django.urls import reverse

from tests.ui.runs.test_live_figures import _live_payload, _poll, ran_experiment  # noqa: F401


# ── how often it asks ───────────────────────────────────────────────────────

def test_the_interval_never_drops_below_the_write_interval():
    """The run writes at most every `PARTIAL_RESULT_SECONDS`, so a faster poll
    is a request that can only ever answer "nothing yet"."""
    from ui import views
    from ui.services.run import PARTIAL_RESULT_SECONDS

    assert views.POLL_MIN_SECONDS >= PARTIAL_RESULT_SECONDS


@pytest.mark.parametrize("durations,expected", [
    ([], 5),                       # nothing measured yet
    ([0.05] * 5, 5),               # demo-dataset trials: the floor holds
    ([12.0] * 5, 12),              # a real dataset: one ask per trial
    ([600.0] * 5, 30),             # and a ceiling, so a long run still feels live
])
def test_the_interval_follows_the_trials(durations, expected):
    """Polling ten times per trial is ten times the traffic for one new row."""
    from ui import views

    class _Exp:
        result = {"data": [{"time": d} for d in durations]}

    assert views._poll_seconds(_Exp()) == expected


def test_one_pathological_trial_does_not_slow_the_whole_run():
    """The median of the recent few, not the mean of all of them."""
    from ui import views

    class _Exp:
        result = {"data": [{"time": 6.0}] * 9 + [{"time": 4000.0}]}

    assert views._poll_seconds(_Exp()) == 6


@pytest.mark.django_db
def test_the_fragment_carries_its_own_interval(client, ran_experiment):
    """poll.js re-reads `hx-trigger` off each replacement, so the interval
    adapts per swap with no state on either side."""
    body = _poll(client, ran_experiment, trials=1)

    assert "hx-trigger=\"every " in body
    assert "every 2s" not in body, "the old fixed interval"


def test_polling_stops_while_the_tab_is_hidden():
    """Browsers throttle background timers but do not stop the requests, so a
    forgotten tab kept asking for figures nobody could see. Held rather than
    skipped, so the poll fires the moment the tab comes back."""
    from pathlib import Path

    source = Path("ui/static/ui/poll.js").read_text(encoding="utf-8")

    assert "document.hidden" in source
    assert "visibilitychange" in source
    # Navigating away needs nothing — the timer lives on the page it started
    # from — and the comment says so, so nobody adds a second mechanism for it.
    assert "Navigating away" in source


# ── and what it carries for the table ───────────────────────────────────────

@pytest.mark.django_db
def test_the_poll_carries_only_the_rows_the_page_is_missing(client, ran_experiment):
    """Appended, not swapped: the table is the one thing you are *using* while
    reading a chart, and replacing it under someone mid-sort would lose the
    sort, the page, the scroll position and the row they had picked."""
    exp = ran_experiment
    stored = len(exp.result["data"])

    live = _live_payload(_poll(client, exp, trials=stored - 2))

    assert live["rows_html"].count("<tr ") == 2
    assert f'data-trial-idx="{stored - 1}"' in live["rows_html"]
    assert 'data-trial-idx="0"' not in live["rows_html"], "already on the page"


@pytest.mark.django_db
def test_the_rows_come_from_the_same_partial_the_table_was_built_from(
        client, ran_experiment):
    """One template, so the two cannot drift into formatting a duration or a
    failure differently."""
    exp = ran_experiment
    live = _live_payload(_poll(client, exp, trials=len(exp.result["data"]) - 1))
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    last = live["rows_html"].strip()
    assert last.startswith("<tr ")
    # The very same row is in the page's own table.
    idx = last[last.index('data-trial-idx="'):].split('"')[1]
    assert f'data-trial-idx="{idx}"' in page


@pytest.mark.django_db
def test_no_rows_are_sent_when_the_table_is_switched_off(client, ran_experiment):
    """The endpoint must not become a way to fetch what the page deliberately
    did not build."""
    exp = ran_experiment
    exp.use_default_settings = False
    exp.settings = {"show_trials": False}
    exp.save(update_fields=["use_default_settings", "settings"])

    live = _live_payload(_poll(client, exp, trials=1))

    assert "rows_html" not in live


@pytest.mark.django_db
def test_the_page_appends_them_rather_than_replacing_the_table(client, ran_experiment):
    body = client.get(
        reverse("ui:experiment_detail", args=[ran_experiment.pk])).content.decode()

    assert "appendTrialRows" in body
    assert "trials:appended" in body, "the sort is re-applied after an append"


# ── the shade that means "selected" ─────────────────────────────────────────

def test_the_selection_shade_is_the_one_that_was_actually_on_screen():
    """Trial performance faded its points to 0.7 so a failure's cross could sit
    among them at full strength, and the selected point faded with them — over
    Plotly's own plot background that composited down to a soft coral, while
    every other figure painted the same constant at full opacity and came out
    visibly redder. Resolved towards the one that had been on screen."""
    from ui.figures.plots import SELECTION_COLOR

    def over(fg, bg, alpha):
        f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(alpha * x + (1 - alpha) * y):02X}"
                             for x, y in zip(f, b))

    # What it used to render as: the old constant, faded, over Plotly's default
    # `plot_bgcolor`.
    assert over("#EF553B", "#E5ECF6", 0.7) == SELECTION_COLOR


def test_the_page_and_the_figures_agree_on_it():
    """`--selected` in app.css and SELECTION_COLOR in plots.py are one shade
    with two homes — the page draws the selection highlight itself."""
    from pathlib import Path

    from ui.figures.plots import SELECTION_COLOR

    css = Path("ui/static/ui/app.css").read_text(encoding="utf-8")

    assert f"--selected: {SELECTION_COLOR};" in css


def test_the_sidebar_no_longer_marks_the_current_experiment_in_the_brand_red():
    """Two different jobs that looked alike: --primary means "this is an
    action", and this means "this is the one you picked"."""
    from pathlib import Path

    css = Path("ui/static/ui/app.css").read_text(encoding="utf-8")
    active = css[css.index(".tab-list .tab.active {"):]
    active = active[:active.index("}")]

    assert "var(--selected)" in active
    assert "var(--primary)" not in active


# ── the fold ────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_a_fresh_experiment_opens_with_the_run_settings_showing(client):
    """There is nothing else to do with the page yet, and a closed fold makes
    starting a run look like it needs finding."""
    from tests.ui.runs.test_run_views import _experiment

    exp = _experiment()
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert '<details class="run-config" open>' in body


@pytest.mark.django_db
def test_an_experiment_with_results_keeps_it_folded_away(client, ran_experiment):
    """Once there are results the metric selector above is the control you keep
    coming back to, and the run settings are a decision taken once."""
    ran_experiment.runs.all().update(status="done")
    body = client.get(
        reverse("ui:experiment_detail", args=[ran_experiment.pk])).content.decode()

    assert '<details class="run-config">' in body
