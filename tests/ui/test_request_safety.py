"""Requests that change something must say so, and uploads stay unserved.

Two properties that only bite once the app is reachable by more than its
operator, but cost nothing to hold now.
"""

from django.urls import reverse

from ui.models import Experiment, Run


def _experiment_with_active_run():
    exp = Experiment.objects.create(
        name="r", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy",
                       status="running")
    return exp


def test_cancelling_a_run_requires_a_post(client):
    """On GET this used to cancel, which means an <img> tag or a link
    prefetcher could stop someone's run and CSRF never entered into it."""
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
    """MEDIA_ROOT is one flat directory shared by everyone, so serving it from
    disk would publish every user's dataset at a guessable URL."""
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
