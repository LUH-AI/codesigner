"""ROC AUC and log loss, tracked with every other metric when they come free.

They are scored on class probabilities, so a classification experiment tracks
them when its model gives probabilities from its own fit — not for a model
with none, and not for the SVM, whose probabilities cost a calibration on every
trial. A run that would score them without any leaves them out, or refuses
when one is what it optimizes. Runs execute inline in tests (see
`runs_execute_synchronously`).
"""

import numpy as np
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core.metrics import METRICS
from core.models import BaseModel
from ui.services.run import scoreable

from tests.conftest import DATASETS_DIR, post_new_experiment
from tests.ui.custom_models.conftest import VALID_MODEL_SRC


def _create(client, **overrides):
    data = {
        "name": "iris", "model_name": "LightGBM", "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification", "evaluation_scheme": "kfold",
        "evaluation_value": 3, "seed": 0,
    }
    data.update(overrides)
    return post_new_experiment(client, data)


@pytest.mark.django_db
def test_a_model_with_probabilities_is_scored_on_them(client):
    """What: a classification experiment whose model gives probabilities
    tracks ROC AUC and log loss without being asked, and a run optimizing ROC
    AUC measures every trial. How: creates iris with LightGBM, runs three
    trials on roc_auc, and reads the scores back."""
    from ui.models import Experiment
    from ui.views import _rebuild_result

    assert _create(client).status_code == 302
    exp = Experiment.objects.get(data__name="iris")
    assert exp.data.metric_names[-2:] == ["roc_auc", "log_loss"]

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "roc_auc"})
    exp.refresh_from_db()
    result = _rebuild_result(exp)

    assert len(result.trials) == 3 and not any(t.failed for t in result.trials)
    # 0.5 is allowed: a configuration whose leaves need more rows than a fold
    # has cannot split, and predicts the class shares — a coin flip's AUC.
    assert all(0.5 <= t.scores["roc_auc"] <= 1 and t.scores["log_loss"] > 0
               for t in result.trials)
    assert max(t.scores["roc_auc"] for t in result.trials) > 0.9


@pytest.mark.django_db
def test_the_svm_is_spared_its_calibration(client):
    """What: the SVM gives probabilities only by calibrating on every trial,
    so its experiment tracks the label metrics alone.
    How: creates iris with the SVM and reads its metrics."""
    from ui.models import Experiment

    _create(client, model_name="SVM")
    names = Experiment.objects.get(data__name="iris").data.metric_names

    assert "balanced_accuracy" in names
    assert "roc_auc" not in names and "log_loss" not in names


@pytest.mark.django_db
def test_a_model_without_probabilities_is_created_without_them(client, settings):
    """What: an uploaded model with no `fit_predict_proba` is created as usual,
    tracking the label metrics. How: creates one from an upload."""
    from ui.models import Experiment

    settings.ALLOW_CUSTOM_MODELS = True
    upload = SimpleUploadedFile("m.py", VALID_MODEL_SRC.encode(), content_type="text/x-python")
    assert _create(client, name="labels only", model_name="", model_file=upload).status_code == 302

    names = Experiment.objects.get(data__name="labels only").data.metric_names
    assert "accuracy" in names and "roc_auc" not in names


@pytest.mark.django_db
def test_a_regression_experiment_has_no_class_probabilities(client):
    """What: a regression experiment tracks the regression metrics only —
    there are no classes to score probabilities of. How: creates diabetes."""
    from ui.models import Experiment

    _create(client, name="diabetes", demo_dataset=str(DATASETS_DIR / "diabetes.csv"),
            task="regression")

    assert Experiment.objects.get(data__name="diabetes").data.metric_names == ["rmse", "mae", "r2"]


class _LabelsOnly(BaseModel):
    name = "Labels only"

    def get_config_space(self, seed: int = 0):
        return None

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        return np.asarray(y_train)[: len(X_val)]


def test_a_run_leaves_out_what_its_model_cannot_score():
    """What: at run start, a model that gives no probabilities loses ROC AUC
    and log loss, and optimizing one of them is refused with the reason.
    How: asks `scoreable` with accuracy and with roc_auc as the run's metric."""
    metrics = {n: METRICS[n] for n in ("accuracy", "roc_auc", "log_loss")}

    assert list(scoreable(metrics, _LabelsOnly(), "accuracy")) == ["accuracy"]
    with pytest.raises(ValueError, match="class probabilities"):
        scoreable(metrics, _LabelsOnly(), "roc_auc")
