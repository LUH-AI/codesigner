"""`use_demo_dataset` and `use_demo_model` switch the form between demo and
upload inputs and default to upload. Only the page script reads them; validation
ignores them.
"""

from django.urls import reverse

from experiments.models import Experiment
from tests.conftest import post_new_experiment, get_new_experiment


def _demo_dataset_path():
    """`_demo_dataset_path` names iris since the models here only classify.
    """
    from core import datasets
    return datasets.demo_datasets()["iris"]


def test_dataset_toggle_present(client):
    body = get_new_experiment(client).content.decode()
    assert 'name="use_demo_dataset"' in body  # demo/upload checkbox


def test_dataset_checkbox_defaults_unchecked(client):
    """Upload is the initial mode."""
    body = get_new_experiment(client).content.decode()
    checkbox = body[body.index('name="use_demo_dataset"') - 60:body.index('name="use_demo_dataset"') + 60]
    assert "checked" not in checkbox


def test_the_dataset_label_says_which_it_is(client):
    """What: one label for the dataset, showing "Dataset (.csv)" while
    uploading, with the demo checkbox beside it and the demo wording in the
    label too, for the script to switch to.
    How: step 1's HTML around the label."""
    body = get_new_experiment(client).content.decode()
    label = body[body.index("data-dataset-label"):body.index('name="use_demo_dataset"')]

    assert 'data-showing="upload"' in label
    assert '<span data-when="upload">Dataset (.csv)</span>' in label
    assert '<span data-when="demo">Dataset (demo)</span>' in label
    assert "upload a CSV" not in body


def test_model_toggle_present_when_flag_on(client, settings):
    settings.ALLOW_CUSTOM_MODELS = True
    body = get_new_experiment(client).content.decode()
    assert 'name="use_demo_model"' in body


def test_model_toggle_absent_when_flag_off(client, settings):
    settings.ALLOW_CUSTOM_MODELS = False
    body = get_new_experiment(client).content.decode()
    assert 'name="use_demo_model"' not in body


def test_normal_submission_still_creates(client):
    resp = post_new_experiment(client, {
        "name": "toggle-ok",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(), "task": "classification",
        "seed": "0",
    })
    assert resp.status_code == 302
    assert Experiment.objects.filter(data__name="toggle-ok").exists()
