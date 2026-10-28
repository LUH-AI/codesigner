"""The server renders the autocompute flags and the Compute button markup, and
the browser skips the fetch until a click.
"""

import json

from core import io
from experiments.services import snapshot as adapter

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
    """`_autocompute_payload` parses the autocompute flags."""
    chunk = body.split('id="autocompute-data"', 1)[1]
    return json.loads(chunk.split(">", 1)[1].split("</script>", 1)[0])


def test_the_flags_reach_the_page_as_json(client):
    """`json_script` keeps a Python dict repr like `{'x': True}` off the page."""
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert _autocompute_payload(body) == {"local_ablation": False,
                                          "partial_dependence": False,
                                          "acquisition_slice": True,
                                          "local_effects": False,
                                          "surrogate_uncertainty": False,
                                          "forecast": False,
                                          "hypershap_baseline": True}


def test_switching_one_on_is_visible_to_the_page(client):
    exp = _experiment({"autocompute_partial_dependence": True})
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert _autocompute_payload(body) == {"local_ablation": False,
                                          "partial_dependence": True,
                                          "acquisition_slice": True,
                                          "local_effects": False,
                                          "surrogate_uncertainty": False,
                                          "forecast": False,
                                          "hypershap_baseline": True}


def test_both_figures_ship_a_compute_button(client):
    """The buttons always render and the script reveals them."""
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert 'id="pdp-compute-btn"' in body
    assert 'id="ablation-compute-btn"' in body
    # Prompts start hidden, so autocompute never flashes them.
    for prompt_id in ('id="pdp-compute"', 'id="ablation-compute"'):
        tag = body.split(prompt_id, 1)[1].split(">", 1)[0]
        assert "hidden" in tag, prompt_id


def test_a_pending_figure_is_cleared_not_declared_empty(client):
    """`clearPlot` avoids a "No … data available" caption under the Compute
    prompt.
    """
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert "function clearPlot(" in body
    assert 'clearPlot("partial_dependence")' in body
    assert 'clearPlot("local_explanation")' in body


def test_the_endpoints_still_work_when_autocompute_is_off(client):
    exp = _experiment({"autocompute_partial_dependence": False,
                       "autocompute_local_ablation": False})
    assert exp.data.result is not None
    hp = next(iter(exp.data.result["configs"].values()))
    first_hp = list(hp.keys())[0]

    pdp = client.get(f"/experiments/{exp.pk}/partial-dependence/"
                     f"?metric=accuracy&hp={first_hp}")
    ablation = client.get(f"/experiments/{exp.pk}/trial-ablation/?metric=accuracy&idx=0")

    assert pdp.status_code == 200 and pdp.json()["figure"]
    assert ablation.status_code == 200 and ablation.json()["figure"]
