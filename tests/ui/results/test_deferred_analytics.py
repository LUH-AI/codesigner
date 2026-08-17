"""What the experiment page does with the autocompute settings.

With one switched off, the figure renders a "Compute" button and the page script
is told not to fetch — so opening or reloading the experiment computes nothing
until asked. That is the whole point of the setting, and it is the direct answer
to "you shouldn't be recomputing or refitting anything on reloading the page".

The gating itself is client-side (the fetch is), so these check the two things the
server is responsible for: shipping the flags, and shipping the button markup.
The behaviour they drive is verified in a browser — see the phase walkthrough.
"""

import json

from core import io
from ui.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR


def _experiment(settings=None):
    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    if settings is not None:
        exp.use_default_settings = False
        exp.settings = settings
        exp.save(update_fields=["use_default_settings", "settings"])
    return exp


def _autocompute_payload(body):
    """The flags as the page script will parse them."""
    chunk = body.split('id="autocompute-data"', 1)[1]
    return json.loads(chunk.split(">", 1)[1].split("</script>", 1)[0])


def test_the_flags_reach_the_page_as_json(client):
    """Through json_script like every other payload, so a Python dict can't land
    in the page as `{'x': True}` — which parses as neither JSON nor JavaScript."""
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert _autocompute_payload(body) == {"local_ablation": True,
                                          "partial_dependence": True}


def test_switching_one_off_is_visible_to_the_page(client):
    exp = _experiment({"autocompute_partial_dependence": False})
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert _autocompute_payload(body) == {"local_ablation": True,
                                          "partial_dependence": False}


def test_both_figures_ship_a_compute_button(client):
    """Rendered always and revealed by the script, rather than conditionally
    rendered — the flag is read in one place (the fetch), so the markup doesn't
    need to duplicate the decision."""
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert 'id="pdp-compute-btn"' in body
    assert 'id="ablation-compute-btn"' in body
    # …and each prompt starts hidden, so an autocomputing page never flashes it.
    for prompt_id in ('id="pdp-compute"', 'id="ablation-compute"'):
        tag = body.split(prompt_id, 1)[1].split(">", 1)[0]
        assert "hidden" in tag, prompt_id


def test_a_pending_figure_is_cleared_not_declared_empty(client):
    """"Nothing has been asked for yet" is not "we looked and there is nothing".

    The page has a separate `clearPlot` for the former precisely so the figure's
    "No … data available" caption doesn't contradict the Compute prompt above it.
    """
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert "function clearPlot(" in body
    assert 'clearPlot("partial_dependence")' in body
    assert 'clearPlot("hyperparameter_importance")' in body


def test_the_endpoints_still_work_when_autocompute_is_off(client):
    """The setting governs whether the *page* asks, not whether the server will
    answer — the Compute button hits the same endpoint."""
    exp = _experiment({"autocompute_partial_dependence": False,
                       "autocompute_local_ablation": False})
    hp = next(iter(exp.result["configs"].values()))
    first_hp = list(hp.keys())[0]

    pdp = client.get(f"/experiments/{exp.pk}/partial-dependence/"
                     f"?metric=accuracy&hp={first_hp}")
    ablation = client.get(f"/experiments/{exp.pk}/trial-ablation/?metric=accuracy&idx=0")

    assert pdp.status_code == 200 and pdp.json()["figure"]
    assert ablation.status_code == 200 and ablation.json()["figure"]
