"""What the page does with a trial that produced no measurement.

The figures mark it (see test_failed_trials.py); this is the rest of the
surfacing. Three things a reader needs and none of which a mark on a chart can
carry: which row of the table it was, that the big number in the sidebar is a
placeholder rather than a result, and why it failed — the last of which is a
traceback, which is a file to open rather than a panel to read.

Built by editing a stored result rather than by running a model that crashes: a
failure is a `status` and an `additional_info` on the trial's data entry, so
everything here is reachable without an optimizer. `scratch/fabricate_failures.py`
does the same thing to a copy of a real experiment, for looking at.
"""

import json

import pytest
from django.urls import reverse

from core import io
from core.optimizers.timing import STATUS_CRASHED, STATUS_TIMEOUT
from ui.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR

TRACEBACK = 'Traceback (most recent call last):\n  File "m.py"\nValueError: no'


@pytest.fixture
def experiment():
    """An experiment whose trial at index 3 crashed and index 5 timed out.

    The crash carries a traceback and the timeout does not, which is what
    `evaluate_trial` records: a deadline expiring says nothing about where the
    model was when it ran out.
    """
    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    data = exp.result["data"]
    metrics = list(exp.metric_names)

    data[3]["status"] = STATUS_CRASHED
    data[3]["cost"] = 1.0
    data[3]["scores"] = {m: 0.0 for m in metrics}
    data[3]["additional_info"] = {"error": "ValueError: no", "traceback": TRACEBACK}

    data[5]["status"] = STATUS_TIMEOUT
    data[5]["cost"] = 1.0
    data[5]["scores"] = {m: 0.0 for m in metrics}
    data[5]["additional_info"] = {"error": "past its 600s deadline"}

    exp.save(update_fields=["result"])
    return exp


# ── the table ────────────────────────────────────────────────────────────────

def test_the_table_marks_the_failed_rows(client, experiment):
    """Tinted rather than crossed: a row is a fill. Its zeros are shown as
    stored — blanking them would leave the score columns unsortable and hide
    that the search was handed those numbers."""
    html = client.get(reverse("ui:experiment_detail", args=[experiment.pk])).content.decode()
    rows = [r for r in html.split("<tr ") if "data-trial-idx" in r]

    failed = [i for i, r in enumerate(rows) if 'class="failed"' in r]
    assert failed == [3, 5]
    assert "ValueError: no" in rows[3], "the reason is in reach without leaving the table"


# ── the sidebar ──────────────────────────────────────────────────────────────

def _panel(client, exp, idx):
    return client.get(
        reverse("ui:trial_panel", args=[exp.pk]) + f"?metric=accuracy&idx={idx}"
    ).content.decode()


def test_the_panel_says_the_score_is_not_a_result(client, experiment):
    """The number above it reads as a measurement of 0.0 otherwise, which is the
    one thing it is not."""
    body = _panel(client, experiment, 3)

    assert "placeholder" in body
    assert "ValueError: no" in body
    assert "metric-value failed" in body, "struck through, not hidden"


def test_the_panel_links_the_traceback_when_there_is_one(client, experiment):
    """A dozen lines of paths and frames is the wrong shape for a sidebar."""
    assert "trial-traceback" in _panel(client, experiment, 3)


def test_the_panel_offers_no_link_when_there_is_no_traceback(client, experiment):
    """The timeout. A link to a 404 is worse than no link, and the reason on its
    own is the whole story for a deadline that expired."""
    body = _panel(client, experiment, 5)

    assert "past its 600s deadline" in body
    assert "trial-traceback" not in body


def test_a_trial_that_succeeded_says_nothing_about_failure(client, experiment):
    """The notice is absent rather than empty, so the panel keeps its shape for
    the trials that are the normal case."""
    body = _panel(client, experiment, 0)

    assert "placeholder" not in body
    assert "metric-value failed" not in body


# ── the traceback itself ─────────────────────────────────────────────────────

def _traceback(client, exp, idx):
    return client.get(
        reverse("ui:trial_traceback", args=[exp.pk]) + f"?idx={idx}")


def test_the_traceback_is_served_as_plain_text(client, experiment):
    """Something to scroll, search and paste elsewhere — and inline, so a click
    opens it rather than downloading it."""
    response = _traceback(client, experiment, 3)

    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/plain")
    assert response["Content-Disposition"].startswith("inline")
    body = response.content.decode()
    assert TRACEBACK in body
    assert "ValueError: no" in body, "the reason heads the file"


def test_a_trial_with_no_traceback_has_no_page(client, experiment):
    """Which covers three cases at once: it succeeded, it timed out, or its
    traceback was left out of an export."""
    assert _traceback(client, experiment, 5).status_code == 404
    assert _traceback(client, experiment, 0).status_code == 404


def test_an_index_outside_the_run_is_refused(client, experiment):
    assert _traceback(client, experiment, 9999).status_code == 400
    assert _traceback(client, experiment, "nope").status_code == 400


# ── and what leaves in an export ─────────────────────────────────────────────

def _exported(client, exp, **answers):
    response = client.post(reverse("ui:experiment_export", args=[exp.pk]), answers)
    return json.loads(response.content.decode())["result"]["data"]


def test_the_export_asks_about_tracebacks(client, experiment):
    """A second question beside the timestamps one, because a traceback is the
    same kind of thing: a fact about this machine rather than about the search."""
    body = client.get(reverse("ui:experiment_export", args=[experiment.pk])).content.decode()

    assert 'name="tracebacks" value="keep"' in body
    assert "absolute paths" in body


def test_the_export_does_not_ask_when_there_is_nothing_to_send(client):
    """Asking about something the file does not contain is a question with no
    answer."""
    clean = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    body = client.get(reverse("ui:experiment_export", args=[clean.pk])).content.decode()

    assert 'name="tracebacks"' not in body
    assert 'name="timestamps"' in body, "the other question is always asked"


def test_the_traceback_is_left_out_unless_it_is_asked_for(client, experiment):
    """Unticked is the safer answer, so the default strips it."""
    data = _exported(client, experiment, timestamps="keep")

    assert "traceback" not in data[3]["additional_info"]
    assert data[3]["additional_info"]["error"] == "ValueError: no", (
        "the reason is the trial's own result and stays either way")


def test_the_traceback_travels_when_it_is_asked_for(client, experiment):
    """So an .ihpo sent to someone can carry the crash they need to see."""
    data = _exported(client, experiment, timestamps="keep", tracebacks="keep")

    assert data[3]["additional_info"]["traceback"] == TRACEBACK


def test_the_two_answers_are_independent(client, experiment):
    """Two questions, not one with two settings: keeping the traceback must not
    smuggle the timestamps out with it."""
    data = _exported(client, experiment, tracebacks="keep")

    assert data[3]["additional_info"]["traceback"] == TRACEBACK
    assert "starttime" not in data[3]
    assert "endtime" not in data[3]


def test_a_failure_survives_a_round_trip(client, experiment):
    """The status and the reason are the trial's result, so re-importing the file
    has to produce a run the page marks the same way."""
    from core.optimizers import RandomOptimizer

    body = client.post(reverse("ui:experiment_export", args=[experiment.pk]),
                       {"timestamps": "keep", "tracebacks": "keep"}).content
    reimported = adapter.experiment_from_snapshot(io.parse(body))
    result = RandomOptimizer().deserialize_result(reimported.result)

    assert [i for i, t in enumerate(result.trials) if t.failed] == [3, 5]
    assert result.trials[3].traceback == TRACEBACK
    assert result.trials[5].traceback == "", "it never had one"
