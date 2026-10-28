"""Changing requests need POST, and uploads stay unserved.
"""

from django.urls import reverse

from experiments.models import Experiment, Run


def _experiment_with_active_run():
    exp = Experiment.objects.create(
        name="r", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy",
                       status="running")
    return exp


def test_cancelling_a_run_requires_a_post(client):
    exp = _experiment_with_active_run()

    resp = client.get(reverse("ui:run_cancel", args=[exp.pk]))

    assert resp.status_code == 405
    assert not exp.runs.filter(cancel_requested=True).exists()


def test_cancelling_still_works_on_post(client):
    exp = _experiment_with_active_run()

    resp = client.post(reverse("ui:run_cancel", args=[exp.pk]))

    assert resp.status_code == 302
    assert exp.runs.filter(cancel_requested=True).exists()


def test_uploads_are_not_served_once_the_instance_has_accounts(settings):
    """Serving the flat MEDIA_ROOT would publish every dataset at a guessable
    URL."""
    from importlib import reload

    from django.urls import clear_url_caches
    import config.urls

    settings.DEBUG = True
    settings.REQUIRE_LOGIN = True
    reload(config.urls)
    clear_url_caches()
    try:
        assert not any("media" in str(p.pattern) for p in config.urls.urlpatterns)
    finally:
        settings.REQUIRE_LOGIN = False
        reload(config.urls)
        clear_url_caches()
