"""The status poll sets its interval from trial durations and holds while the
tab is hidden, and it appends new table rows.
"""

import json

import pytest
from django.urls import reverse

from tests.ui.runs.test_live_figures import _live_payload, _poll, ran_experiment  # noqa: F401


def test_the_interval_never_drops_below_the_write_interval():
    """Any poll within `PARTIAL_RESULT_SECONDS` would find nothing new."""
    from ui.views import common
    from experiments.services.run import PARTIAL_RESULT_SECONDS

    assert common.POLL_MIN_SECONDS >= PARTIAL_RESULT_SECONDS


@pytest.mark.parametrize("durations,expected", [
    ([], 5),                       # nothing measured
    ([0.05] * 5, 5),               # floor holds
    ([12.0] * 5, 12),              # one ask per trial
    ([600.0] * 5, 30),             # ceiling keeps long runs live
])
def test_the_interval_follows_the_trials(durations, expected):
    from ui.views import common

    class _Exp:
        class data:
            result = {"data": [{"time": d} for d in durations]}

    assert common._poll_seconds(_Exp()) == expected


def test_one_pathological_trial_does_not_slow_the_whole_run():
    """The median of recent trials ignores outliers."""
    from ui.views import common

    class _Exp:
        class data:
            result = {"data": [{"time": 6.0}] * 9 + [{"time": 4000.0}]}

    assert common._poll_seconds(_Exp()) == 6


@pytest.mark.django_db
def test_the_fragment_carries_its_own_interval(client, ran_experiment):
    """`poll.js` re-reads `hx-trigger` per swap, keeping no state."""
    body = _poll(client, ran_experiment, trials=1)

    assert "hx-trigger=\"every " in body
    assert "every 2s" not in body, "the old fixed interval"


def test_polling_stops_while_the_tab_is_hidden():
    """The poll holds in a hidden tab, since background tabs still send
    requests."""
    from pathlib import Path

    source = Path("ui/static/ui/poll.js").read_text(encoding="utf-8")

    assert "document.hidden" in source
    assert "visibilitychange" in source
    # Navigating away needs nothing, and poll.js documents that.
    assert "Navigating away" in source


# rows for the table

@pytest.mark.django_db
def test_the_poll_carries_only_the_rows_the_page_is_missing(client, ran_experiment):
    """Rows are appended so a reader keeps their sort, page, scroll and
    selection."""
    exp = ran_experiment
    stored = len(exp.data.result["data"])

    live = _live_payload(_poll(client, exp, trials=stored - 2))

    assert live["rows_html"].count("<tr ") == 2
    assert f'data-trial-idx="{stored - 1}"' in live["rows_html"]
    assert 'data-trial-idx="0"' not in live["rows_html"], "already on the page"


@pytest.mark.django_db
def test_the_rows_come_from_the_same_partial_the_table_was_built_from(
        client, ran_experiment):
    """One template formats live rows and table rows alike."""
    exp = ran_experiment
    live = _live_payload(_poll(client, exp, trials=len(exp.data.result["data"]) - 1))
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    last = live["rows_html"].strip()
    assert last.startswith("<tr ")
    idx = last[last.index('data-trial-idx="'):].split('"')[1]
    assert f'data-trial-idx="{idx}"' in page


@pytest.mark.django_db
def test_no_rows_are_sent_when_the_table_is_switched_off(client, ran_experiment):
    """A hidden table gets no rows through the poll."""
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


# the selection shade

def test_the_selection_shade_is_the_one_that_was_actually_on_screen():
    """`SELECTION_COLOR` is the faded coral that trial performance showed, now
    used at full opacity everywhere."""
    from ui.figures.plots import SELECTION_COLOR

    def over(fg, bg, alpha):
        f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(alpha * x + (1 - alpha) * y):02X}"
                             for x, y in zip(f, b))

    # old constant at 0.7 over Plotly's `plot_bgcolor`
    assert over("#EF553B", "#E5ECF6", 0.7) == SELECTION_COLOR


def test_the_page_and_the_figures_agree_on_it():
    """The page draws the highlight itself, so `--selected` must equal
    `SELECTION_COLOR`."""
    from pathlib import Path

    from ui.figures.plots import SELECTION_COLOR

    css = Path("ui/static/ui/app.css").read_text(encoding="utf-8")

    assert f"--selected: {SELECTION_COLOR};" in css


def test_the_sidebar_no_longer_marks_the_current_experiment_in_the_brand_red():
    """`--primary` marks actions, and `--selected` marks the picked item."""
    from pathlib import Path

    css = Path("ui/static/ui/app.css").read_text(encoding="utf-8")
    active = css[css.index(".tab-list .tab.active {"):]
    active = active[:active.index("}")]

    assert "var(--selected)" in active
    assert "var(--primary)" not in active


def test_selected_is_one_colour_in_both_themes():
    """The dark theme leaves `--selected` alone, matching the figures."""
    import re
    from pathlib import Path

    css = Path("ui/static/ui/app.css").read_text(encoding="utf-8")
    dark_blocks = re.findall(r':root\[data-theme="(?:dark|system)"\]\s*\{([^}]*)\}', css)

    assert dark_blocks, "the dark theme's colours are where this expects them"
    for block in dark_blocks:
        assert "--selected" not in block


@pytest.mark.parametrize("rule", [
    '.fig-tab[aria-selected="true"]',
    "nav.rail .rail-icon.active",
    "table.trials tbody tr.selected",
])
def test_every_selected_state_uses_it(rule):
    from pathlib import Path

    css = Path("ui/static/ui/app.css").read_text(encoding="utf-8")
    body = css[css.index(rule + " {"):]
    body = body[:body.index("}")]

    assert "var(--selected" in body
    assert "var(--primary)" not in body


@pytest.mark.django_db
def test_a_fresh_experiment_opens_with_the_run_settings_showing(client):
    """A fresh experiment has nothing else to do but run."""
    from tests.ui.runs.test_run_views import _experiment

    exp = _experiment()
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert '<details class="run-config" open>' in body


@pytest.mark.django_db
def test_an_experiment_with_results_keeps_it_folded_away(client, ran_experiment):
    """With results, the metric selector is the recurring control."""
    ran_experiment.runs.all().update(status="done")
    body = client.get(
        reverse("ui:experiment_detail", args=[ran_experiment.pk])).content.decode()

    assert '<details class="run-config">' in body
