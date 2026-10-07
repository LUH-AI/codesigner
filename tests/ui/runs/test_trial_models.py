"""Trial models kept for export, and the Export trial parameters download.

A run keeps the fitted model of the best trials — the one its last fold
fitted, nothing trained again — and, if asked, every trial of the run until the
next. Each is in its library's own format beside a note saying it is partially
trained. Runs execute inline in tests (see `runs_execute_synchronously`).
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from django.urls import reverse

from ui.models import Experiment, TrialModel

from tests.conftest import DATASETS_DIR, FIXTURES_DIR, post_new_experiment


def _create(client, name="iris", model_name="Random Forest", dataset=DATASETS_DIR / "iris.csv",
            **settings_overrides):
    from django.core.files.uploadedfile import SimpleUploadedFile

    data = {"name": name, "task": "classification", "model_name": model_name,
            "optimizer_name": "Random Search", "seed": 0,
            "evaluation_scheme": "kfold", "evaluation_value": 3}
    if Path(dataset).parent == DATASETS_DIR:
        data["demo_dataset"] = str(dataset)
    else:
        data["dataset_file"] = SimpleUploadedFile(Path(dataset).name, Path(dataset).read_bytes(),
                                                  content_type="text/csv")
    post_new_experiment(client, data)
    exp = Experiment.objects.get(data__name=name)
    if settings_overrides:
        exp.use_default_settings = False
        exp.settings = settings_overrides
        exp.save()
    return exp


def _run(client, exp, trials):
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": trials, "optimize_metric": "accuracy"})
    exp.refresh_from_db()


def _download(client, exp, idx):
    return client.get(reverse("ui:experiment_export_trial_model", args=[exp.pk]),
                      {"metric": "accuracy", "idx": idx})


def test_a_run_keeps_the_best_trials_models_and_no_others(client):
    """What: with three to keep, exactly the three best trials by the run's
    metric have a model on disk, each with its note. How: runs six trials and
    compares what was kept with the scores."""
    from ui.views import _rebuild_result

    exp = _create(client, keep_best_trial_models=3)
    _run(client, exp, 6)

    trials = _rebuild_result(exp).trials
    best = sorted(trials, key=lambda t: (-t.score, t.trial))[:3]
    kept = TrialModel.objects.filter(experiment=exp)
    assert sorted(k.trial for k in kept) == sorted(t.trial for t in best)
    for k in kept:
        files = {p.name for p in Path(k.directory).iterdir()}
        assert {"model.skops", "trial.json"} <= files
        assert k.bytes == sum(p.stat().st_size for p in Path(k.directory).iterdir())


def test_keeping_the_last_run_keeps_it_until_the_next(client):
    """What: with "every trial of the last run" on, a run keeps all its
    trials; the next run then lets the first run's go, but for the best.
    How: two runs of three trials, keeping the best one."""
    exp = _create(client, keep_best_trial_models=1, keep_last_run_trial_models=True)
    _run(client, exp, 3)
    assert TrialModel.objects.filter(experiment=exp).count() == 3

    _run(client, exp, 3)
    kept = set(TrialModel.objects.filter(experiment=exp).values_list("trial", flat=True))
    assert {4, 5, 6} <= kept and len(kept & {1, 2, 3}) <= 1


def test_the_download_is_the_trials_model_marked_partially_trained(client, tmp_path):
    """What: the zip holds the trial's model in its library's format and a
    note that says it is partially trained, which trial and configuration it
    is, and how to load it — and the model loads and predicts.
    How: downloads the best trial's parameters and loads model.skops."""
    from skops.io import get_untrusted_types, load

    from ui.views import _rebuild_result

    exp = _create(client, keep_best_trial_models=2)
    _run(client, exp, 4)
    result = _rebuild_result(exp)
    idx = result.best_index("accuracy")

    response = _download(client, exp, idx)
    assert response.status_code == 200 and response["Content-Type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = {Path(n).name: n for n in archive.namelist()}
    note = json.loads(archive.read(names["trial.json"]))
    assert note["status"].startswith("partially trained")
    assert note["trial"] == result.trials[idx].trial
    assert note["hyperparameters"] == dict(result.trials[idx].config)
    assert "skops" in note["how_to_load"]

    path = tmp_path / "model.skops"
    path.write_bytes(archive.read(names["model.skops"]))
    model = load(path, trusted=get_untrusted_types(file=path))
    X = np.loadtxt(DATASETS_DIR / "iris.csv", delimiter=",", skiprows=1, usecols=range(4))
    assert set(model.predict(X)) <= set(note["classes"])


@pytest.mark.parametrize("model_name, filename", [
    ("LightGBM", "model.txt"), ("XGBoost", "model.json"), ("CatBoost", "model.cbm")])
def test_a_booster_is_kept_in_its_own_format_and_loads_there(client, model_name, filename):
    """What: a booster's parameters are its library's own file, which that
    library loads — LightGBM's can even be trained on from.
    How: keeps one trial of each booster on flats and loads the file."""
    exp = _create(client, name=model_name, model_name=model_name,
                  dataset=FIXTURES_DIR / "flats.csv", keep_best_trial_models=1)
    _run(client, exp, 1)
    kept = TrialModel.objects.get(experiment=exp)
    path = Path(kept.directory) / filename
    assert path.is_file()

    if model_name == "LightGBM":
        import lightgbm

        booster = lightgbm.Booster(model_file=str(path))
        rows = np.random.default_rng(0).random((5, booster.num_feature()))
        assert booster.predict(rows).shape[0] == 5
    elif model_name == "XGBoost":
        import xgboost

        booster = xgboost.Booster()
        booster.load_model(str(path))
        assert booster.num_features() > 0
    else:
        from catboost import CatBoost

        assert CatBoost().load_model(str(path)).tree_count_ > 0


def test_a_trial_whose_model_was_not_kept_says_why(client):
    """What: the page's button is disabled for a trial with no kept model and
    says why, and asking anyway is a 404 with the same reason.
    How: keeps one of three, then asks for a trial that is not it."""
    from ui.views import _rebuild_result

    exp = _create(client, keep_best_trial_models=1)
    _run(client, exp, 3)
    result = _rebuild_result(exp)
    kept = TrialModel.objects.get(experiment=exp).trial
    other = next(i for i, t in enumerate(result.trials) if t.trial != kept)

    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="export-trial-parameters"' in page and "was not kept" in page
    assert _download(client, exp, other).status_code == 404


def test_a_group_without_room_keeps_no_model_and_the_run_still_finishes(
        client, settings, django_user_model):
    """What: kept models never take the place of data: a group already at its
    limit keeps none, the run finishes all its trials, and its log says why.
    How: a group whose limit is one byte, a three-trial run."""
    from access.models import Group, Membership
    from ui.models import Run
    from ui.views import _rebuild_result

    settings.REQUIRE_LOGIN = True
    group = Group.objects.create(name="lab", user_limit=5, storage_limit_bytes=1)
    user = django_user_model.objects.create_user(username="ana", password="pw")
    Membership.objects.create(user=user, group=group, role="member")
    client.force_login(user)

    exp = _create(client, keep_best_trial_models=2)
    assert exp.group == group
    _run(client, exp, 3)

    assert len(_rebuild_result(exp).trials) == 3
    assert not TrialModel.objects.filter(experiment=exp).exists()
    events = Run.objects.filter(experiment=exp).latest("pk").events
    assert any(e["kind"] == "trial_models_not_kept" and e["count"] >= 1 for e in events)
