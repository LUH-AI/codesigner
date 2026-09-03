import textwrap

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

# A minimal valid custom model: one integer hyperparameter, constant scores so
# a run is instant and deterministic. Mirrors the fixture in
# tests/core/test_model_loading.py. The PEP 723 header is what a real model
# needs to declare the environment it runs in.
VALID_MODEL_SRC = textwrap.dedent("""
    # /// script
    # dependencies = ["ConfigSpace"]
    # ///
    from ConfigSpace import ConfigurationSpace, Integer
    from core.models import BaseModel

    class MyModel(BaseModel):
        name = "My Custom Model"
        def get_config_space(self, seed: int = 0):
            cs = ConfigurationSpace(seed=seed)
            cs.add([Integer("k", (1, 5), default=3)])
            return cs
        def fit_predict(self, config, X_train, y_train, X_val, seed=0):
            return [next(iter(y_train))] * len(X_val)
""")

# A .py with no BaseModel subclass — must be rejected at upload time.
INVALID_MODEL_SRC = "x = 1\n"


@pytest.fixture
def model_upload():
    """A valid uploaded model .py (Django SimpleUploadedFile)."""
    return SimpleUploadedFile(
        "mymodel.py", VALID_MODEL_SRC.encode("utf-8"), content_type="text/x-python"
    )


@pytest.fixture
def bad_model_upload():
    """An uploaded .py that contains no BaseModel subclass."""
    return SimpleUploadedFile(
        "bad.py", INVALID_MODEL_SRC.encode("utf-8"), content_type="text/x-python"
    )
