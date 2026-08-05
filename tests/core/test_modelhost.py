"""A model running in another process, driven over a pipe.

No uv and no network here: this interpreter stands in for a model environment,
since it already has numpy, ConfigSpace and the SDK. What is under test is the
conversation — that a model can be described without running a trial, that
predictions come back and get scored, and that every way the child can fail or
hang is contained rather than taken out on the run.
"""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from core.metrics import METRICS
from core.modelhost import ModelProcessError, TrialTimeout, describe, model_session
from core.modelhost.errors import ModelTrialError, TrialCancelled
from core.optimizers.timing import STATUS_CRASHED, STATUS_TIMEOUT, STATUS_SUCCESS
from core.optimizers.trial import evaluate_trial

X_TRAIN = np.array([[0.0], [1.0], [2.0], [3.0]])
Y_TRAIN = np.array(["a", "b", "a", "b"], dtype=object)
X_VAL = np.array([[4.0], [5.0]])
Y_VAL = np.array(["a", "b"], dtype=object)

HEADER = '''
# /// script
# dependencies = ["ConfigSpace"]
# ///
from ConfigSpace import ConfigurationSpace, Integer
from codesigner_model import BaseModel
'''


def _model(tmp_path: Path, body: str, name: str = "model.py") -> Path:
    """Write a model file whose fit_predict body is *body*."""
    method = textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8)
    source = HEADER + (
        "\n\nclass TestModel(BaseModel):\n"
        '    name = "Test Model"\n\n'
        "    def get_config_space(self, seed: int = 0):\n"
        "        cs = ConfigurationSpace(seed=seed)\n"
        '        cs.add([Integer("k", (1, 5), default=3)])\n'
        "        return cs\n\n"
        "    def fit_predict(self, config, X_train, y_train, X_val, seed=0):\n"
        + method + "\n"
    )
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _session(model_file, **kwargs):
    return model_session(sys.executable, model_file, X_TRAIN, Y_TRAIN, X_VAL, **kwargs)


class _Flag:
    """A cancel flag the test can flip once the child is up."""

    def __init__(self, value=False):
        self.value = value

    def is_set(self):
        return self.value


# ── describing a model without running it ────────────────────────────────────

def test_describe_reports_the_name_and_search_space(tmp_path):
    """How an experiment learns what it has, without a web request importing
    user code — and, since the class is instantiated to answer, whether it
    works at all."""
    hello = describe(sys.executable, _model(tmp_path, "return ['a'] * len(X_val)"))

    assert hello["name"] == "Test Model"
    assert hello["model_class"] == "TestModel"
    assert "k" in hello["config_space"]["hyperparameters"][0].values()


def test_describe_reports_a_file_that_will_not_import(tmp_path):
    """The failure the create form deliberately no longer looks for."""
    bad = tmp_path / "bad.py"
    bad.write_text("import a_package_that_does_not_exist\n", encoding="utf-8")

    with pytest.raises(ModelProcessError, match="ModuleNotFoundError"):
        describe(sys.executable, bad)


def test_describe_refuses_a_file_with_no_model(tmp_path):
    plain = tmp_path / "plain.py"
    plain.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ModelProcessError, match="No BaseModel subclass"):
        describe(sys.executable, plain)


def test_describe_refuses_a_file_with_two_models(tmp_path):
    """Guessing which one was meant is worse than saying so."""
    two = tmp_path / "two.py"
    two.write_text(HEADER + textwrap.dedent('''
        class First(BaseModel):
            name = "First"
            def get_config_space(self, seed=0): return ConfigurationSpace(seed=seed)
            def fit_predict(self, c, a, b, d, seed=0): return []

        class Second(BaseModel):
            name = "Second"
            def get_config_space(self, seed=0): return ConfigurationSpace(seed=seed)
            def fit_predict(self, c, a, b, d, seed=0): return []
    '''), encoding="utf-8")

    with pytest.raises(ModelProcessError, match="more than one model"):
        describe(sys.executable, two)


# ── a model file may import its neighbours ───────────────────────────────────

def test_a_model_can_import_a_module_beside_it(tmp_path):
    """A generated pipeline is more than one file more often than not, so the
    model file's own directory is on the path."""
    (tmp_path / "helper.py").write_text("LABEL = 'b'\n", encoding="utf-8")
    model = _model(tmp_path, "from helper import LABEL\nreturn [LABEL] * len(X_val)")

    with _session(model) as remote:
        assert list(remote.fit_predict({"k": 1}, None, None, None)) == ["b", "b"]


# ── the search space survives the crossing ───────────────────────────────────

def test_the_config_space_is_rebuilt_as_a_live_object(tmp_path):
    with _session(_model(tmp_path, "return ['a'] * len(X_val)")) as remote:
        space = remote.get_config_space(seed=0)
        assert list(space.keys()) == ["k"]
        assert (space["k"].lower, space["k"].upper) == (1, 5)


def test_the_rebuilt_space_samples_reproducibly(tmp_path):
    """Serializing loses the seed and RandomOptimizer samples from this, so it
    is re-applied on the way out rather than inherited."""
    with _session(_model(tmp_path, "return ['a'] * len(X_val)")) as remote:
        first = [dict(remote.get_config_space(seed=7).sample_configuration()) for _ in range(3)]
        second = [dict(remote.get_config_space(seed=7).sample_configuration()) for _ in range(3)]
    assert first == second


# ── trials ───────────────────────────────────────────────────────────────────

def test_predictions_come_back_and_score(tmp_path):
    with _session(_model(tmp_path, "return ['a', 'b']")) as remote:
        scores, run_info = evaluate_trial(
            remote, {"k": 1}, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, METRICS, seed=0)

    assert scores["accuracy"] == 1.0
    assert run_info["status"] == STATUS_SUCCESS


def test_the_child_reports_its_own_cpu_time(tmp_path):
    """This thread's CPU clock saw none of the work, so the child's measurement
    is the only true one."""
    with _session(_model(tmp_path, "return ['a', 'b']")) as remote:
        _, run_info = evaluate_trial(
            remote, {"k": 1}, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, METRICS, seed=0)

    assert remote.last_cpu_time is not None
    assert run_info["cpu_time"] == remote.last_cpu_time


def test_the_model_never_receives_the_validation_labels(tmp_path):
    """It gets X_train, y_train and X_val. Writing y_val down is impossible
    because it was never sent."""
    model = _model(tmp_path, """
        import json, pathlib
        pathlib.Path(%r).write_text(json.dumps({
            "n_train": len(X_train), "labels": sorted(set(y_train)), "n_val": len(X_val),
        }))
        return ['a'] * len(X_val)
    """ % str(tmp_path / "seen.json"))

    with _session(model) as remote:
        remote.fit_predict({"k": 1}, None, None, None)

    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen == {"n_train": 4, "labels": ["a", "b"], "n_val": 2}


def test_a_model_that_prints_does_not_corrupt_the_protocol(tmp_path):
    """stdout is the wire, so the child diverts the model's output to stderr."""
    with _session(_model(tmp_path, "print('hello from the model')\nreturn ['a', 'b']")) as remote:
        assert list(remote.fit_predict({"k": 1}, None, None, None)) == ["a", "b"]


def test_a_model_that_raises_fails_one_trial_and_the_run_continues(tmp_path):
    model = _model(tmp_path, """
        if config["k"] == 1:
            raise ValueError("k=1 is no good")
        return ['a', 'b']
    """)
    with _session(model) as remote:
        bad, bad_info = evaluate_trial(
            remote, {"k": 1}, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, METRICS, seed=0)
        good, good_info = evaluate_trial(
            remote, {"k": 2}, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, METRICS, seed=0)

    assert bad == {name: 0.0 for name in METRICS}
    assert bad_info["status"] == STATUS_CRASHED
    assert "k=1 is no good" in bad_info["additional_info"]["error"]
    # the child is still alive and answering
    assert good["accuracy"] == 1.0
    assert good_info["status"] == STATUS_SUCCESS


def test_unsendable_predictions_are_a_trial_failure(tmp_path):
    with _session(_model(tmp_path, "return [object(), object()]")) as remote:
        with pytest.raises(ModelTrialError, match="cannot be sent"):
            remote.fit_predict({"k": 1}, None, None, None)


# ── failure containment ──────────────────────────────────────────────────────

def test_a_wedged_model_is_killed_and_the_trial_times_out(tmp_path):
    model = _model(tmp_path, "import time\ntime.sleep(30)\nreturn ['a', 'b']")

    with _session(model, trial_timeout=1.0) as remote:
        with pytest.raises(TrialTimeout):
            remote.fit_predict({"k": 1}, None, None, None)
        assert not remote._process.alive, "the child should have been killed"


def test_a_timeout_is_recorded_as_a_timed_out_trial(tmp_path):
    model = _model(tmp_path, "import time\ntime.sleep(30)\nreturn ['a', 'b']")

    with _session(model, trial_timeout=1.0) as remote:
        scores, run_info = evaluate_trial(
            remote, {"k": 1}, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, METRICS, seed=0)

    assert scores == {name: 0.0 for name in METRICS}
    assert run_info["status"] == STATUS_TIMEOUT


def test_cancelling_mid_trial_stops_rather_than_failing_the_trial(tmp_path):
    """Cancellation could not reach a running trial before: the flag was only
    read between them, so a hanging model was unstoppable. It is distinct from a
    timeout because the loop should hand back the trials it has, not record this
    one as a failure."""
    model = _model(tmp_path, "import time\ntime.sleep(30)\nreturn ['a', 'b']")
    flag = _Flag()

    with _session(model, cancel=flag, trial_timeout=30.0) as remote:
        flag.value = True
        with pytest.raises(TrialCancelled):
            remote.fit_predict({"k": 1}, None, None, None)
        assert not remote._process.alive


def test_a_run_cancelled_before_it_starts_never_gets_a_child(tmp_path):
    """The flag is checked on the way in too, so an already-cancelled run does
    not pay for an interpreter."""
    with pytest.raises(TrialCancelled):
        with _session(_model(tmp_path, "return ['a', 'b']"), cancel=_Flag(True)):
            pass


def test_a_model_that_exits_mid_trial_is_reported(tmp_path):
    with _session(_model(tmp_path, "import os\nos._exit(1)")) as remote:
        with pytest.raises(ModelProcessError, match="stopped without answering"):
            remote.fit_predict({"k": 1}, None, None, None)


def test_the_child_is_gone_after_the_session(tmp_path):
    with _session(_model(tmp_path, "return ['a', 'b']")) as remote:
        process = remote._process
        assert process.alive
    assert not process.alive


def test_the_split_directory_is_removed_after_the_session(tmp_path):
    """The split is written to a private temp directory; it holds the dataset,
    so it goes away with the session."""
    session = _session(_model(tmp_path, "return ['a', 'b']"))
    with session as remote:
        arrays_dir = session._arrays_dir
        assert (arrays_dir / "X_train.npy").is_file()
        assert arrays_dir.stat().st_mode & 0o777 == 0o700
    assert not arrays_dir.exists()


# ── refusing what cannot be sent ─────────────────────────────────────────────

def test_a_non_numeric_feature_column_is_refused_with_a_reason(tmp_path):
    """Pickling is off, so an object feature array cannot cross. That dataset
    already fails at fit; this says so before a subprocess is spent on it."""
    X_text = np.array([["red"], ["blue"]], dtype=object)

    with pytest.raises(ValueError, match="not numeric"):
        with model_session(sys.executable, _model(tmp_path, "return ['a']"),
                           X_text, Y_TRAIN, X_text):
            pass
