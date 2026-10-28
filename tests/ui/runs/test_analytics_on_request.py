"""A cancelled or partial run skips the HyperSHAP games, and the page offers to
compute them later. The games are written back onto the stored result, because
they fill fields the result already carries.
"""

import pytest
from django.urls import reverse

from tests.ui.runs.test_run_views import no_thread  # noqa: F401


def _blank_the_games(exp):
    """`_blank_the_games` leaves the result as a cancelled run would."""
    stored = exp.data.result
    for field in ("hyperparameter_importance", "hyperparameter_sensitivity",
                  "hyperparameter_mistunability", "hyperparameter_interactions",
                  "hyperparameter_moebius", "hyperparameter_sensitivity_interactions",
                  "hyperparameter_sensitivity_moebius",
                  "hyperparameter_mistunability_interactions",
                  "hyperparameter_mistunability_moebius"):
        if field in stored:
            stored[field] = {m: {} for m in exp.data.metric_names}
    stored["hyperparameter_importance_warning"] = {
        m: "Importance analytics were skipped because the run was cancelled."
        for m in exp.data.metric_names}
    exp.data.result = stored
    exp.data.save(update_fields=["result"])
    return exp


@pytest.fixture
def cancelled_experiment(client):
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    return _blank_the_games(_ran_experiment(client))


# the offer, only where it applies

@pytest.mark.django_db
def test_the_page_offers_to_compute_what_is_missing(client, cancelled_experiment):
    body = client.get(reverse("ui:experiment_detail",
                              args=[cancelled_experiment.pk])).content.decode()

    assert "compute-analytics" in body
    assert reverse("ui:experiment_compute_analytics",
                   args=[cancelled_experiment.pk]) in body


@pytest.mark.django_db
def test_the_button_sits_with_the_game_selector_not_inside_a_figure(client,
                                                                   cancelled_experiment):
    """One computation fills six figures, so the button belongs to the game
    selector.
    """
    body = client.get(reverse("ui:experiment_detail",
                              args=[cancelled_experiment.pk])).content.decode()

    section = body[body.index("<h2>HyperSHAP</h2>"):]
    section = section[:section.index("</section>")]
    assert 'class="compute-analytics"' in section

    card = body[body.index('data-figure="hyperparameter_importance"'):]
    card = card[:card.index("</section>")]
    assert "compute-analytics" not in card


@pytest.mark.django_db
def test_the_waiting_figures_point_at_the_button_and_say_nothing_else(client,
                                                                     cancelled_experiment):
    """Each of the six waiting figures shows one short line pointing at the
    button.
    """
    body = client.get(reverse("ui:experiment_detail",
                              args=[cancelled_experiment.pk])).content.decode()

    assert body.count("Press Compute explanations to see results.") == 6
    assert "The explanations were not computed for this run" not in body


@pytest.mark.django_db
def test_a_figure_with_nothing_to_wait_for_says_so_in_its_own_words(client):
    """A run that computed its games keeps the usual empty caption.
    """
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    exp = _ran_experiment(client)
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "Press Compute explanations to see results." not in body
    assert "No interaction data available." in body


@pytest.mark.django_db
def test_a_run_that_computed_them_is_not_offered_it(client):
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    exp = _ran_experiment(client)
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "compute-analytics" not in body


@pytest.mark.django_db
def test_it_is_not_offered_while_a_run_is_in_flight(client, cancelled_experiment):
    """Asking during a run would race its partial writes."""
    cancelled_experiment.runs.all().update(status="running")

    body = client.get(reverse("ui:experiment_detail",
                              args=[cancelled_experiment.pk])).content.decode()

    assert "compute-analytics" not in body


# asking

@pytest.mark.django_db
def test_asking_fills_the_games_in_on_the_stored_result(client, cancelled_experiment):
    """The result matches what a completed run writes, so the page and export
    need no new path."""
    from experiments.models import Experiment

    before = cancelled_experiment.data.result["hyperparameter_importance"]
    assert not any(before.values()), "nothing to begin with"

    resp = client.post(reverse("ui:experiment_compute_analytics",
                               args=[cancelled_experiment.pk]))

    assert resp.status_code == 302
    after = Experiment.objects.get(pk=cancelled_experiment.pk).data.result
    assert after is not None
    assert any(after["hyperparameter_importance"].values())
    for game_field in ("hyperparameter_sensitivity", "hyperparameter_mistunability"):
        assert any(after[game_field].values()), game_field


@pytest.mark.django_db
def test_asking_keeps_every_trial(client, cancelled_experiment):
    from experiments.models import Experiment

    before = cancelled_experiment.data.result["data"]
    client.post(reverse("ui:experiment_compute_analytics", args=[cancelled_experiment.pk]))
    stored = Experiment.objects.get(pk=cancelled_experiment.pk).data.result
    assert stored is not None
    after = stored["data"]

    assert len(after) == len(before)
    assert [e["scores"] for e in after] == [e["scores"] for e in before]


@pytest.mark.django_db
def test_the_offer_goes_away_once_it_has_been_taken(client, cancelled_experiment):
    client.post(reverse("ui:experiment_compute_analytics", args=[cancelled_experiment.pk]))

    body = client.get(reverse("ui:experiment_detail",
                              args=[cancelled_experiment.pk])).content.decode()

    assert "compute-analytics" not in body


@pytest.mark.django_db
def test_asking_during_a_run_changes_nothing(client, cancelled_experiment):
    from experiments.models import Experiment

    cancelled_experiment.runs.all().update(status="running")
    before = cancelled_experiment.data.result

    client.post(reverse("ui:experiment_compute_analytics", args=[cancelled_experiment.pk]))

    assert Experiment.objects.get(pk=cancelled_experiment.pk).data.result == before


@pytest.mark.django_db
def test_it_is_post_only(client, cancelled_experiment):
    resp = client.get(reverse("ui:experiment_compute_analytics",
                              args=[cancelled_experiment.pk]))
    assert resp.status_code == 405
