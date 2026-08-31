"""Explanations for a run that never computed any.

`docs/plan_today.md` Step 3, third part.

`compute_hp_games` declines outright for a cancelled run — pressing Cancel used
to still buy the full analytics bill, so on a wide model you waited minutes for
a run you had just stopped. A partial write during a run leaves them out for the
same reason at finer grain. Both were permanent: the numbers were computed once,
at run completion, or never, and the only way to get them was to resume the run.

So they become askable afterwards. What that costs is the same bill, paid when
somebody wants it — which is the whole difference, and it is why the eager skip
is now a deferral rather than a refusal.

Written back onto the stored result rather than returned as a payload, unlike
the three fetched figures: those answer "which trial, which hyperparameter",
which has no small precomputable set of answers, while these are exactly the
fields the result already carries.
"""

import pytest
from django.urls import reverse

from tests.ui.runs.test_run_views import no_thread  # noqa: F401


def _blank_the_games(exp):
    """Leave the result exactly as a cancelled run leaves it: the trials, and
    every game field empty with a reason in the warning."""
    stored = exp.result
    for field in ("hyperparameter_importance", "hyperparameter_sensitivity",
                  "hyperparameter_mistunability", "hyperparameter_interactions",
                  "hyperparameter_moebius", "hyperparameter_sensitivity_interactions",
                  "hyperparameter_sensitivity_moebius",
                  "hyperparameter_mistunability_interactions",
                  "hyperparameter_mistunability_moebius"):
        if field in stored:
            stored[field] = {m: {} for m in exp.metric_names}
    stored["hyperparameter_importance_warning"] = {
        m: "Importance analytics were skipped because the run was cancelled."
        for m in exp.metric_names}
    exp.result = stored
    exp.save(update_fields=["result"])
    return exp


@pytest.fixture
def cancelled_experiment(client):
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    return _blank_the_games(_ran_experiment(client))


# ── the page offers it, and only when it applies ────────────────────────────

@pytest.mark.django_db
def test_the_page_offers_to_compute_what_is_missing(client, cancelled_experiment):
    body = client.get(reverse("ui:experiment_detail",
                              args=[cancelled_experiment.pk])).content.decode()

    assert "compute-analytics" in body
    assert reverse("ui:experiment_compute_analytics",
                   args=[cancelled_experiment.pk]) in body


@pytest.mark.django_db
def test_a_run_that_computed_them_is_not_offered_it(client):
    """Nothing is missing, so there is nothing to ask for."""
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    exp = _ran_experiment(client)
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "compute-analytics" not in body


@pytest.mark.django_db
def test_it_is_not_offered_while_a_run_is_in_flight(client, cancelled_experiment):
    """The run will compute them itself when it ends, and asking now would race
    its own partial writes."""
    cancelled_experiment.runs.all().update(status="running")

    body = client.get(reverse("ui:experiment_detail",
                              args=[cancelled_experiment.pk])).content.decode()

    assert "compute-analytics" not in body


# ── and asking works ────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_asking_fills_the_games_in_on_the_stored_result(client, cancelled_experiment):
    """The result becomes what a completed run would have written, so the page
    renders it with no live-update path of its own and the export carries it."""
    from ui.models import Experiment

    before = cancelled_experiment.result["hyperparameter_importance"]
    assert not any(before.values()), "nothing to begin with"

    resp = client.post(reverse("ui:experiment_compute_analytics",
                               args=[cancelled_experiment.pk]))

    assert resp.status_code == 302
    after = Experiment.objects.get(pk=cancelled_experiment.pk).result
    assert any(after["hyperparameter_importance"].values())
    for game_field in ("hyperparameter_sensitivity", "hyperparameter_mistunability"):
        assert any(after[game_field].values()), game_field


@pytest.mark.django_db
def test_asking_keeps_every_trial(client, cancelled_experiment):
    """It is computed *from* the trials, and must not disturb them — the run's
    own record is the thing being filled in, not replaced."""
    from ui.models import Experiment

    before = cancelled_experiment.result["data"]
    client.post(reverse("ui:experiment_compute_analytics", args=[cancelled_experiment.pk]))
    after = Experiment.objects.get(pk=cancelled_experiment.pk).result["data"]

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
    from ui.models import Experiment

    cancelled_experiment.runs.all().update(status="running")
    before = cancelled_experiment.result

    client.post(reverse("ui:experiment_compute_analytics", args=[cancelled_experiment.pk]))

    assert Experiment.objects.get(pk=cancelled_experiment.pk).result == before


@pytest.mark.django_db
def test_it_is_post_only(client, cancelled_experiment):
    resp = client.get(reverse("ui:experiment_compute_analytics",
                              args=[cancelled_experiment.pk]))
    assert resp.status_code == 405
