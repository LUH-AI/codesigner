"""The page saves its tab, controls, selected trial and 3D camera before each
navigation a run causes, and restores them on landing.
"""

from pathlib import Path

from django.urls import reverse

from core import io
from experiments.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR


def _page(client, *, running=False):
    from experiments.models import Run

    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    exp.data.optimizer_name = "SMAC"
    exp.data.save(update_fields=["optimizer_name"])
    if running:
        Run.objects.create(experiment=exp, stopping={"max_trials": 3},
                           primary_metric="accuracy", status="running")
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def test_the_forms_that_start_or_stop_a_run_keep_the_view():
    """The run form and the cancel and give-up buttons carry `data-keeps-view`."""
    side = Path("ui/templates/ui/figures/evaluation.html").read_text()
    status = Path("ui/templates/ui/_run_status.html").read_text()

    assert "{% url 'ui:experiment_run' summary.pk %}\" data-keeps-view>" in side
    for action in ("run_cancel", "run_force_stop"):
        form = status.split(f"'ui:{action}'", 1)[1].split(">", 1)[0]
        assert "data-keeps-view" in form, action


def test_the_proposal_form_keeps_the_view():
    template = Path("ui/templates/ui/_selected_proposal_inner.html").read_text()
    assert "data-run-proposal data-keeps-view" in template


def test_the_reload_after_a_run_is_announced_first():
    """`poll.js` fires `poll:reloading` before reloading so the page can save
    the view."""
    poll = Path("ui/static/ui/poll.js").read_text()
    at = poll.index("window.location.reload()")
    assert 'new CustomEvent("poll:reloading")' in poll[at - 300:at]


def test_the_view_is_put_back_before_anything_reads_a_control(client):
    """The page script restores the kept view before any control is read."""
    html = _page(client)
    script = html.split("(function () {", 1)[1]

    assert script.index("const keptView") < script.index("const metricPlots")
    assert "if (!keptView) {" in html
    assert 'document.addEventListener("poll:reloading", keepView);' in html
    assert 'event.target.matches("form[data-keeps-view]")' in html


def test_slow_figures_mark_their_cards_computing():
    """What: every request that computes a figure goes through `cardBusy`, so
    its card shows the computing spinner, and the Compute explanations form
    marks the HyperSHAP cards. How: the page script, the acquisition and
    next-trials scripts and the sidebar template are read for each wrapped
    request."""
    detail = Path("ui/templates/ui/experiment_detail.html").read_text()
    for url in ("trialAblationUrl", "hypershapUrl", "partialDependenceUrl", "localEffectsUrl",
                "forecastUrl", "uncertaintyUrl", "predictionUrl"):
        assert f"cardBusy(" in detail.split(f"fetch({url}", 1)[0].rsplit("\n", 2)[-1] \
            or f"cardBusy(measuredFigures, fetch({url}" in detail, url
    assert detail.count("cardBusy(") >= 7
    acquisition = Path("ui/static/ui/acquisition.js").read_text()
    assert 'cardBusy("acquisition_slice"' in acquisition and 'cardBusy("prior"' in acquisition
    assert "window.cardBusy(cards, fetch(" in Path("ui/static/ui/next_trials.js").read_text()
    side = Path("ui/templates/ui/_side_explanation.html").read_text()
    assert "game-figures-data" in side and "cardBusy(keys" in side
    assert "ui/busy.js" in Path("ui/templates/ui/_experiment_page.html").read_text()
