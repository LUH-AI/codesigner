"""`io.load_model_from_path` loads a user's model file.
"""

import textwrap

from core import datasets, io

from tests.conftest import DATASETS_DIR

_VALID_MODEL = textwrap.dedent("""
    from ConfigSpace import ConfigurationSpace, Integer
    from core.models import BaseModel

    class MyModel(BaseModel):
        name = "My Custom Model"
        def get_config_space(self, seed: int = 0):
            cs = ConfigurationSpace(seed=seed)
            cs.add([Integer("k", (1, 5), default=3)])
            return cs
        def fit_predict(self, config, X_train, y_train, X_val, seed=0):
            return ["a"] * len(X_val)
""")


def test_load_model_from_path_valid(tmp_path):
    f = tmp_path / "mymodel.py"
    f.write_text(_VALID_MODEL, encoding="utf-8")
    model, err = io.load_model_from_path(str(f))
    assert err is None
    assert model.name == "My Custom Model"


def test_load_model_from_path_missing_file():
    model, err = io.load_model_from_path("/nope/model.py")
    assert model is None and "File not found" in err


def test_load_model_from_path_no_subclass(tmp_path):
    f = tmp_path / "nomodel.py"
    f.write_text("x = 1\n", encoding="utf-8")
    model, err = io.load_model_from_path(str(f))
    assert model is None and "No BaseModel subclass" in err


def test_load_model_from_path_missing_abstract_methods(tmp_path):
    """`load_model_from_path` reports each missing method."""
    f = tmp_path / "incomplete.py"
    f.write_text(textwrap.dedent("""
        from core.models import BaseModel
        class Half(BaseModel):
            name = "Half"
            def get_config_space(self, seed: int = 0):
                from ConfigSpace import ConfigurationSpace
                return ConfigurationSpace(seed=seed)
    """), encoding="utf-8")
    model, err = io.load_model_from_path(str(f))
    assert model is None and "fit_predict" in err


def test_demo_datasets_includes_bundled_csvs():
    """Bundled CSVs appear as {stem: path}."""
    from pathlib import Path
    demos = datasets.demo_datasets()
    assert {"iris", "wine"} <= set(demos)
    assert all(Path(p).is_file() for p in demos.values())


def test_mounted_models_returns_stem_path_dict():
    """Mounted `.py` files appear as {stem: path}."""
    mounts = datasets.mounted_models()
    assert isinstance(mounts, dict)
    assert all(p.endswith(".py") for p in mounts.values())
