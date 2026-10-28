"""Every experiment route goes through `@experiment_view`, so rules live in one
policy. An invisible experiment 404s, and a refused action on a visible one
gives 403.
"""

import pytest
from django.core.exceptions import PermissionDenied
from django.test import RequestFactory
from django.urls import reverse

from access import policy as access_policy
from access.policy import OpenPolicy
from experiments.models import Experiment
from ui import urls
from ui.permissions import CREATED, ONLY, experiment_view


def _experiment(name="perm") -> Experiment:
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


# Audit

#: Routes listed here take a pk that names another object, and each new
#: exemption shows as a line in a diff.
NOT_EXPERIMENTS = {
    "group_remove_person": "a membership, removed by that group's lead",
    "group_set_role": "a membership, whose role the primary lead sets",
    "group_transfer_primary": "a membership, receiving the primary lead role",
    "site_job_stop": "a run, stopped by a site admin from outside its group",
}

#: Routes listed here reach an experiment through `GroupPolicy.trash` instead of
#: `@experiment_view`. A binned experiment sits outside
#: `policy().experiments()`, since it is in no group.
THROUGH_THE_BIN = {
    "site_trash_rehome": "put back into a group by a site admin",
    "site_trash_delete": "discarded for good by a site admin",
    # A person's bin goes through `policy().bin()` for the same reason.
    "bin_restore": "put back by one of its people",
    "bin_download": "taken away as an .ihpo by one of its people",
    "bin_dismiss": "taken out of one person's bin",
}


def _experiment_routes():
    """`_experiment_routes` lists every `<int:pk>` route outside both exemption
    lists."""
    exempt = set(NOT_EXPERIMENTS) | set(THROUGH_THE_BIN)
    return [p for p in urls.urlpatterns
            if "<int:pk>" in str(p.pattern) and p.name not in exempt]


def test_the_exceptions_are_real_routes():
    """A stale name would widen the audit's skips and pass to the next route of
    that name.
    """
    names = {p.name for p in urls.urlpatterns if "<int:pk>" in str(p.pattern)}

    missing = set(NOT_EXPERIMENTS) - names
    assert not missing, f"no such route(s): {sorted(missing)}"


def test_the_exceptions_do_not_reach_an_experiment_through_the_pk():
    """A site admin has no experiment visibility for `@experiment_view` to
    check.
    """
    from ui.views import groups, site

    assert {p.name: p.callback for p in urls.urlpatterns
            if p.name in NOT_EXPERIMENTS}.keys() == NOT_EXPERIMENTS.keys()
    for name in NOT_EXPERIMENTS:
        route = next(p for p in urls.urlpatterns if p.name == name)
        assert route.callback.__module__ in (groups.__name__, site.__name__), (
            f"{name} is not a panel route any more; re-check what its pk means")


def test_the_bin_routes_resolve_through_the_bin(client):
    """The check reads the source, since a hand-rolled lookup would look like a
    policy call from outside.
    """
    import inspect

    from ui.views import bin as bin_views
    from ui.views import site

    # A deleted group's bin and a person's bin.
    through = {site.__name__: "policy().trash(request)",
               bin_views.__name__: "_binned(request, pk)"}
    assert "policy().bin(request)" in inspect.getsource(bin_views._binned)

    for name in THROUGH_THE_BIN:
        route = next(p for p in urls.urlpatterns if p.name == name)
        module = route.callback.__module__
        assert module in through, f"{name}: re-check what its pk means"
        assert through[module] in inspect.getsource(route.callback), (
            f"{name} no longer resolves its experiment through a bin's "
            f"queryset, so nothing is deciding whether it exists")


def test_the_bin_routes_are_real_routes():
    names = {p.name for p in urls.urlpatterns if "<int:pk>" in str(p.pattern)}

    assert not set(THROUGH_THE_BIN) - names


def test_there_are_experiment_routes_to_audit():
    """A matching mistake would leave the audit passing vacuously."""
    assert len(_experiment_routes()) >= 8


def test_every_experiment_route_goes_through_the_policy():
    """A route that looks up its own experiment escapes every policy rule."""
    undecorated = [p.name for p in _experiment_routes()
                   if getattr(p.callback, "experiment_action", None) is None]

    assert not undecorated, (
        f"experiment route(s) missing @experiment_view: {undecorated}")


def test_each_route_declares_a_known_action():
    for pattern in _experiment_routes():
        action = getattr(pattern.callback, "experiment_action")
        assert action in access_policy.ACTIONS, f"{pattern.name}: {action!r}"


def test_the_declared_actions_match_what_the_routes_do():
    """A policy that refuses RUN must refuse the right routes."""
    declared = {p.name: getattr(p.callback, "experiment_action") for p in _experiment_routes()}

    assert declared == {
        "experiment_detail": access_policy.VIEW,
        "trial_panel": access_policy.VIEW,
        "proposal_panel": access_policy.VIEW,
        "trial_ablation": access_policy.VIEW,
        "trial_traceback": access_policy.VIEW,
        "partial_dependence": access_policy.VIEW,
        "forecast": access_policy.VIEW,
        "acquisition_slice": access_policy.VIEW,
        "metric_figures": access_policy.VIEW,
        "hypershap_figures": access_policy.VIEW,
        "surrogate_prediction": access_policy.VIEW,
        "next_trials": access_policy.VIEW,
        "local_effects": access_policy.VIEW,
        "surrogate_uncertainty": access_policy.VIEW,
        "run_status": access_policy.VIEW,
        "env_status": access_policy.VIEW,
        "experiment_run": access_policy.RUN,
        "run_proposal": access_policy.RUN,
        "experiment_optimizer": access_policy.EDIT,
        "experiment_rename": access_policy.RENAME,
        # Computing analytics spends CPU and writes the result, and starts no
        # run.
        "experiment_compute_analytics": access_policy.COMPUTE,
        "prepare_env": access_policy.RUN,
        # Judging a prior costs seconds of compute and writes nothing.
        "evaluate_prior": access_policy.COMPUTE,
        # Resetting a prior changes the next run.
        "reset_prior": access_policy.EDIT,
        "run_cancel": access_policy.RUN,
        # Stating a prior changes what the next run searches.
        "save_prior": access_policy.EDIT,
        # A reader's own row, so viewers may arrange it.
        "save_layout": access_policy.VIEW,
        "experiment_settings": access_policy.EDIT,
        "experiment_timeline": access_policy.VIEW,
        # Contributors get everything except sharing.
        "experiment_share": access_policy.SHARE,
        "experiment_transfer": access_policy.SHARE,
        "experiment_delete": access_policy.DELETE,
        "experiment_export": access_policy.EXPORT,
        # The tuned model leaves with the export.
        "experiment_export_model": access_policy.EXPORT,
        "experiment_export_trial_model": access_policy.EXPORT,
        "run_force_stop": access_policy.RUN,
        # Looking at the data is looking at the experiment.
        "data_overview": access_policy.VIEW,
        "data_section": access_policy.VIEW,
        # Draft setup changes the experiment.
        "setup": access_policy.EDIT,
        "setup_optimizer": access_policy.EDIT,
        "setup_data": access_policy.EDIT,
        "setup_priors": access_policy.EDIT,
        "setup_resume": access_policy.EDIT,
        "setup_create": access_policy.EDIT,
        # Discarding a draft deletes it for good, files and all.
        "setup_discard": access_policy.DELETE,
    }


def test_drafts_are_reached_only_where_they_are_set_up():
    """A draft is reachable only through its setup routes and the prior requests
    of the last step."""
    declared = {p.name: getattr(p.callback, "experiment_drafts") for p in _experiment_routes()}
    reach = {name for name, drafts in declared.items() if drafts != CREATED}

    assert reach == {"setup", "setup_optimizer", "setup_data", "setup_priors", "setup_resume", "setup_create",
                     "setup_discard", "save_prior", "reset_prior", "evaluate_prior",
                     "acquisition_slice"}
    assert {name for name, drafts in declared.items() if drafts == ONLY} == {
        "setup", "setup_optimizer", "setup_data", "setup_priors", "setup_resume", "setup_create",
        "setup_discard"}


def test_an_unknown_action_is_a_mistake_at_import_time():
    """A typo would otherwise be a permission nobody checks."""
    with pytest.raises(ValueError):
        experiment_view("delte")


# Default policy

def test_the_default_policy_allows_everything(client):
    """Installs without accounts see no change."""
    exp = _experiment()
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200
    assert OpenPolicy().may(None, exp, access_policy.VIEW)


def test_the_policy_is_read_from_settings(settings):
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"
    assert isinstance(access_policy.policy(), HidesEverything)

    settings.EXPERIMENT_POLICY = "access.policy.OpenPolicy"
    assert isinstance(access_policy.policy(), OpenPolicy)


# Restrictive policies

class HidesEverything(OpenPolicy):
    """`HidesEverything` shows nothing."""

    def experiments(self, request):
        return Experiment.objects.none()


class RefusesEverything(OpenPolicy):
    """`RefusesEverything` permits nothing."""

    def may(self, request, experiment, action):
        return False


def test_an_invisible_experiment_is_a_404_not_a_403(client, settings):
    """A 403 would confirm to a stranger that the experiment exists."""
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 404


def test_a_refused_action_on_a_visible_experiment_is_a_403(client, settings):
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.RefusesEverything"

    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 403


def test_the_decorator_raises_permission_denied_rather_than_returning_it(settings):
    """The project-wide 403 handler then applies."""
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.RefusesEverything"

    @experiment_view(access_policy.VIEW)
    def view(request, experiment):
        raise AssertionError("should not have been called")

    with pytest.raises(PermissionDenied):
        view(RequestFactory().get("/"), pk=exp.pk)


def test_an_invisible_experiment_is_not_in_the_sidebar(client, settings):
    _experiment(name="hidden-from-the-rail")
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    assert b"hidden-from-the-rail" not in client.get(reverse("ui:home")).content


def test_an_invisible_experiment_is_not_named_in_a_breadcrumb(client, settings):
    """The URL-built trail would otherwise name a hidden experiment."""
    exp = _experiment(name="hidden-from-the-trail")
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    html = client.get(
        reverse("ui:appearance") + f"?next=/experiments/{exp.pk}/").content

    assert b"hidden-from-the-trail" not in html


def test_the_view_is_handed_the_experiment_not_the_pk():
    """A view never receives a pk to look up unchecked."""
    exp = _experiment()
    seen = {}

    @experiment_view(access_policy.VIEW)
    def view(request, experiment):
        seen["got"] = experiment
        return "ok"

    assert view(RequestFactory().get("/"), pk=exp.pk) == "ok"
    assert seen["got"].pk == exp.pk


pytestmark = pytest.mark.django_db


# ── the same check, without a view ───────────────────────────────────────────

def test_resolving_without_a_view_answers_as_the_decorator_does(settings):
    """What: `resolve_experiment`, the decorator's check for code that is not a
    view, gives the experiment, a 404 for one the policy hides, and a 403 for
    an action it refuses.
    How: one experiment under the default policy and both restrictive ones."""
    from django.http import Http404

    from ui.permissions import resolve_experiment

    exp = _experiment()
    request = RequestFactory().get("/")
    assert resolve_experiment(request, exp.pk, access_policy.EDIT).pk == exp.pk

    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"
    with pytest.raises(Http404):
        resolve_experiment(request, exp.pk, access_policy.VIEW)

    settings.EXPERIMENT_POLICY = f"{__name__}.RefusesEverything"
    with pytest.raises(PermissionDenied):
        resolve_experiment(request, exp.pk, access_policy.VIEW)


def test_resolving_the_wrong_kind_is_not_found():
    """What: a created experiment asked for as a draft, or a draft as a created
    one, is not found — only a page turns that into a redirect.
    How: both mismatches through `resolve_experiment`, and the decorator's
    redirect for a GET."""
    from django.http import Http404

    from ui.permissions import ONLY, resolve_experiment

    created = _experiment()
    draft = _experiment()
    draft.draft = True
    draft.save(update_fields=["draft"])
    request = RequestFactory().get("/")

    with pytest.raises(Http404):
        resolve_experiment(request, created.pk, access_policy.EDIT, ONLY)
    with pytest.raises(Http404):
        resolve_experiment(request, draft.pk, access_policy.VIEW)

    @experiment_view(access_policy.VIEW)
    def view(request, experiment):
        raise AssertionError("should not have been called")

    assert view(request, pk=draft.pk).status_code == 302
