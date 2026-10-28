import textwrap

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

# Constant scores keep runs instant; real models carry the PEP 723 header.
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

# No BaseModel subclass: upload must reject it.
INVALID_MODEL_SRC = "x = 1\n"


@pytest.fixture
def model_upload():
    """`model_upload` is a valid upload."""
    return SimpleUploadedFile(
        "mymodel.py", VALID_MODEL_SRC.encode("utf-8"), content_type="text/x-python"
    )


@pytest.fixture
def bad_model_upload():
    return SimpleUploadedFile(
        "bad.py", INVALID_MODEL_SRC.encode("utf-8"), content_type="text/x-python"
    )
