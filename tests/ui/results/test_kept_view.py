"""Where the reader was, kept across the navigations a run causes.

Starting a run, running a proposal, cancelling one and the reload when one ends
are navigations, and a navigation opens the page afresh. None of them is the
reader asking to start over, so the page writes down its tab, its controls, the
selected trial and the 3D camera before each, and puts them back on the page it
lands on. Asserted on the markup and the page script, which is where it lives.
Uses the test2.ihpo fixture as a SMAC experiment, so every control is offered.
"""

from pathlib import Path

from django.urls import reverse

from core import io
from ui.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR


def _page(client, *, running=False):
    from ui.models import Run

    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    exp.data.optimizer_name = "SMAC"
    exp.data.save(update_fields=["optimizer_name"])
    if running:
        Run.objects.create(experiment=exp, stopping={"max_trials": 3},
                           primary_metric="accuracy", status="running")
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def test_the_forms_that_start_or_stop_a_run_keep_the_view():
    """The run form, and while a run is going its cancel and give-up buttons,
    are marked for the page to note the view before they navigate."""
    side = Path("ui/templates/ui/_side_evaluation.html").read_text()
    status = Path("ui/templates/ui/_run_status.html").read_text()

    assert "{% url 'ui:experiment_run' summary.pk %}\" data-keeps-view>" in side
    for action in ("run_cancel", "run_force_stop"):
        form = status.split(f"'ui:{action}'", 1)[1].split(">", 1)[0]
        assert "data-keeps-view" in form, action


def test_the_proposal_form_keeps_the_view():
    """Run as Trial is a run starting too."""
    template = Path("ui/templates/ui/_selected_proposal_inner.html").read_text()
    assert "data-run-proposal data-keeps-view" in template


def test_the_reload_after_a_run_is_announced_first():
    """poll.js says it is about to reload before it does, so the page can note
    the view the reload would otherwise lose."""
    poll = Path("ui/static/ui/poll.js").read_text()
    at = poll.index("window.location.reload()")
    assert 'new CustomEvent("poll:reloading")' in poll[at - 300:at]


def test_the_view_is_put_back_before_anything_reads_a_control(client):
    """The kept view is read and its controls restored at the very top of the
    page script, and the opening defaults step aside for it."""
    html = _page(client)
    script = html.split("(function () {", 1)[1]

    assert script.index("const keptView") < script.index("const metricPlots")
    assert "if (!keptView) {" in html
    assert 'document.addEventListener("poll:reloading", keepView);' in html
    assert 'event.target.matches("form[data-keeps-view]")' in html
