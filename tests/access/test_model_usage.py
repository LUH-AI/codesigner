"""Site-wide usage per model, where each model's best configurations land,
and deleting an account whose experiments are already in the bin."""

import pytest

from access.models import Group, Membership
from access.services import users as user_service
from experiments.models import Experiment, Run
from experiments.services import bin as bin_service
from experiments.services import site as site_service

pytestmark = pytest.mark.django_db

SPACE = {"hyperparameters": [
    {"type": "uniform_float", "name": "C", "lower": 0.01, "upper": 100.0, "log": True},
    {"type": "categorical", "name": "kernel", "choices": ["rbf", "linear"]}]}


def _experiment(owner, group, model, best=None):
    exp = Experiment.objects.create(
        name="x", model_name=model, optimizer_name="SMAC", metric_names=["accuracy"],
        seed=0, owner=owner, group=group)
    if best is not None:
        exp.data.config_space = SPACE
        exp.data.result = {"configs": {"7": best}, "best_config_id": 7,
                           "primary_metric": "accuracy",
                           "hyperparameter_importance": {"accuracy": {"C": 0.6, "kernel": 0.1}}}
        exp.data.save(update_fields=["config_space", "result"])
    return exp


@pytest.fixture
def lab(django_user_model):
    group = Group.objects.create(name="lab", user_limit=9)
    ana = django_user_model.objects.create_user("ana", password="pw")
    Membership.objects.create(user=ana, group=group)
    return group, ana


def test_usage_is_totalled_per_model(lab):
    """What: `usage_by_model` counts each model's experiments and sums its
    runs' trials and seconds, most seconds first. How: two SVM experiments
    and one Random Forest, with runs."""
    group, ana = lab
    for model, seconds in (("SVM", 30.0), ("SVM", 50.0), ("Random Forest", 10.0)):
        exp = _experiment(ana, group, model)
        Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                           trial_count=5, trial_seconds=seconds)

    rows = site_service.usage_by_model()

    assert [r["model"] for r in rows] == ["SVM", "Random Forest"]
    assert (rows[0]["experiments"], rows[0]["runs"], rows[0]["trials"], rows[0]["seconds"]) == (
        2, 2, 10, 80.0)
    assert rows[0]["people"] == 1 and rows[0]["groups"] == 1


def test_best_configurations_summarise_where_the_winners_land(lab):
    """What: `best_configurations` gives a numeric hyperparameter's median of
    best values and its place in the searched log range, a categorical one's
    winning shares, and orders them by mean importance. How: three SVM
    experiments with known best configurations."""
    group, ana = lab
    for c, kernel in ((0.1, "rbf"), (1.0, "rbf"), (10.0, "linear")):
        _experiment(ana, group, "SVM", best={"C": c, "kernel": kernel})

    svm = site_service.best_configurations()["SVM"]

    assert svm["experiments"] == 3
    c, kernel = svm["hyperparameters"]
    assert c["name"] == "C" and c["median"] == 1.0 and c["importance"] == pytest.approx(0.6)
    assert c["positions"]["median"] == pytest.approx(0.5)
    assert kernel["choices"][0] == {"choice": "rbf", "share": pytest.approx(2 / 3)}


def test_an_account_whose_experiments_are_binned_can_be_deleted(lab):
    """What: `delete_user` deletes an account whose only experiment is in the
    bin, which then waits in the site bin, and still raises for a live one.
    How: delete ana's experiment, then her account."""
    group, ana = lab
    exp = _experiment(ana, group, "SVM")
    live = _experiment(ana, group, "SVM")
    with pytest.raises(Exception, match="still owns experiments"):
        user_service.delete_user(ana)

    bin_service.delete(live)
    bin_service.delete(exp)
    user_service.delete_user(ana)

    assert set(site_service.trash().values_list("pk", flat=True)) == {exp.pk, live.pk}
