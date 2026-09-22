"""The one way a view is allowed to reach an experiment.

`ui/permissions.py` exists so that adding real rules later is a matter of
writing a policy, not of finding every view that looks an experiment up. That
only holds if there is no second way in, so the first test here walks the
URLconf and fails on any experiment route that skipped the decorator — the check
that keeps the seam honest as routes are added.

The rest pin the behaviour a real policy will depend on: the queryset is what
decides existence (so an invisible experiment 404s rather than 403s, and does
not appear in the sidebar or a breadcrumb), and a refused action on a visible
experiment is a 403.
"""

import pytest
from django.core.exceptions import PermissionDenied
from django.test import RequestFactory
from django.urls import reverse

from ui import permissions, urls
from ui.models import Experiment
from ui.permissions import OpenPolicy, experiment_view


def _experiment(name="perm") -> Experiment:
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


# ── the audit ────────────────────────────────────────────────────────────────

#: `<int:pk>` routes whose pk is **not** an experiment, and what it is instead.
#:
#: Named one at a time on purpose. The audit's whole value is that a new route
#: is guilty until somebody says otherwise, so the way past it is an entry here
#: — which is a line in a diff — rather than a pattern loose enough to let the
#: next one through unnoticed.
NOT_EXPERIMENTS = {
    "group_remove_person": "a membership, removed by that group's lead",
    "site_job_stop": "a run, stopped by a site admin from outside its group",
}


def _experiment_routes():
    """Every route that names an experiment in its path."""
    return [p for p in urls.urlpatterns
            if "<int:pk>" in str(p.pattern) and p.name not in NOT_EXPERIMENTS]


def test_the_exceptions_are_real_routes():
    """So the list cannot rot into a blanket excuse.

    A name that no longer exists would sit here forever quietly widening what
    the audit skips, and the next route to be called the same thing would
    inherit the exemption.
    """
    names = {p.name for p in urls.urlpatterns if "<int:pk>" in str(p.pattern)}

    missing = set(NOT_EXPERIMENTS) - names
    assert not missing, f"no such route(s): {sorted(missing)}"


def test_the_exceptions_do_not_reach_an_experiment_through_the_pk():
    """The reason they are allowed out: their pk addresses something else.

    `site_job_stop` takes a run because a site admin has no experiment
    visibility for `@experiment_view` to check — and giving them some so a
    decorator would fit would defeat the separation it exists for.
    """
    from ui import panels

    assert {p.name: p.callback for p in urls.urlpatterns
            if p.name in NOT_EXPERIMENTS}.keys() == NOT_EXPERIMENTS.keys()
    for name in NOT_EXPERIMENTS:
        route = next(p for p in urls.urlpatterns if p.name == name)
        assert route.callback.__module__ == panels.__name__, (
            f"{name} is not a panel route any more; re-check what its pk means")


def test_there_are_experiment_routes_to_audit():
    """Guards the audit below: a pattern-matching mistake that found nothing
    would otherwise leave it passing vacuously forever."""
    assert len(_experiment_routes()) >= 8


def test_every_experiment_route_goes_through_the_policy():
    """A `<int:pk>` route that looks the experiment up itself has silently
    opted out of every rule a policy might impose. This is what makes the seam
    a boundary rather than a convention."""
    undecorated = [p.name for p in _experiment_routes()
                   if getattr(p.callback, "experiment_action", None) is None]

    assert not undecorated, (
        f"experiment route(s) missing @experiment_view: {undecorated}")


def test_each_route_declares_a_known_action():
    for pattern in _experiment_routes():
        action = pattern.callback.experiment_action
        assert action in permissions.ACTIONS, f"{pattern.name}: {action!r}"


def test_the_declared_actions_match_what_the_routes_do():
    """Reading the URLconf should tell you what a route is for. Pinned so a
    later policy that refuses, say, RUN refuses the right set of routes."""
    declared = {p.name: p.callback.experiment_action for p in _experiment_routes()}

    assert declared == {
        "experiment_detail": permissions.VIEW,
        "trial_panel": permissions.VIEW,
        "trial_ablation": permissions.VIEW,
        "trial_traceback": permissions.VIEW,
        "partial_dependence": permissions.VIEW,
        "acquisition_slice": permissions.VIEW,
        "metric_figures": permissions.VIEW,
        "local_effects": permissions.VIEW,
        "surrogate_uncertainty": permissions.VIEW,
        "run_status": permissions.VIEW,
        "env_status": permissions.VIEW,
        "experiment_run": permissions.RUN,
        # Computing the explanations a cancelled run skipped: it spends this
        # machine's CPU and writes to the stored result, which is what RUN
        # covers. VIEW would have made reading an experiment a way to make it
        # compute.
        "experiment_compute_analytics": permissions.RUN,
        "prepare_env": permissions.RUN,
        "run_cancel": permissions.RUN,
        "experiment_settings": permissions.EDIT,
        "experiment_share": permissions.EDIT,
        "experiment_delete": permissions.DELETE,
        "experiment_export": permissions.EXPORT,
        "run_force_stop": permissions.RUN,
    }


def test_an_unknown_action_is_a_mistake_at_import_time():
    """A typo in an action would otherwise be a permission that is never
    checked, in a decorator that looks like it is checking one."""
    with pytest.raises(ValueError):
        experiment_view("delte")


# ── the default policy ───────────────────────────────────────────────────────

def test_the_default_policy_allows_everything(client):
    """Nothing changes for an install with no accounts, which is most of them."""
    exp = _experiment()
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200
    assert OpenPolicy().may(None, exp, permissions.VIEW)


def test_the_policy_is_read_from_settings(settings):
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"
    assert isinstance(permissions.policy(), HidesEverything)

    settings.EXPERIMENT_POLICY = "ui.permissions.OpenPolicy"
    assert isinstance(permissions.policy(), OpenPolicy)


# ── what a restrictive policy gets ───────────────────────────────────────────

class HidesEverything(OpenPolicy):
    """Sees nothing at all."""

    def experiments(self, request):
        return Experiment.objects.none()


class RefusesEverything(OpenPolicy):
    """Sees everything, permits nothing."""

    def may(self, request, experiment, action):
        return False


def test_an_invisible_experiment_is_a_404_not_a_403(client, settings):
    """Existence is the secret. A 403 on a URL that 404s for a stranger tells
    them the experiment is real, which is the leak the queryset split avoids."""
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 404


def test_a_refused_action_on_a_visible_experiment_is_a_403(client, settings):
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.RefusesEverything"

    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 403


def test_the_decorator_raises_permission_denied_rather_than_returning_it(settings):
    """So a project-wide 403 handler and the usual middleware apply, instead of
    each view inventing its own refusal page."""
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.RefusesEverything"

    @experiment_view(permissions.VIEW)
    def view(request, experiment):
        raise AssertionError("should not have been called")

    with pytest.raises(PermissionDenied):
        view(RequestFactory().get("/"), pk=exp.pk)


def test_an_invisible_experiment_is_not_in_the_sidebar(client, settings):
    _experiment(name="hidden-from-the-rail")
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    assert b"hidden-from-the-rail" not in client.get(reverse("ui:home")).content


def test_an_invisible_experiment_is_not_named_in_a_breadcrumb(client, settings):
    """The trail is built from the URL, so without this it would name an
    experiment straight out of the database on a page that 404s."""
    exp = _experiment(name="hidden-from-the-trail")
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    html = client.get(
        reverse("ui:appearance") + f"?next=/experiments/{exp.pk}/").content

    assert b"hidden-from-the-trail" not in html


def test_the_view_is_handed_the_experiment_not_the_pk():
    """The reason this is a decorator: a view is never given a pk it could look
    up without asking."""
    exp = _experiment()
    seen = {}

    @experiment_view(permissions.VIEW)
    def view(request, experiment):
        seen["got"] = experiment
        return "ok"

    assert view(RequestFactory().get("/"), pk=exp.pk) == "ok"
    assert seen["got"].pk == exp.pk


pytestmark = pytest.mark.django_db
