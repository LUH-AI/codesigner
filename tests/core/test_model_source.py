"""`inspect_model_source` reads an uploaded model from its syntax tree without
importing it.
"""

import pytest

from core.model_source import (
    MAX_SOURCE_BYTES,
    STARTER_HEADER,
    inspect_model_source,
)

HEADER = '# /// script\n# dependencies = ["scikit-learn"]\n# ///\n'

MODEL = '''
from codesigner_model import BaseModel

class MyModel(BaseModel):
    name = "My Model"

    def get_config_space(self, seed=0):
        return None

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        return []
'''


def _inspect(source: str):
    return inspect_model_source(source.encode())


def test_reads_name_class_and_dependencies():
    info, err = _inspect(HEADER + MODEL)

    assert err is None
    assert info is not None
    assert info.name == "My Model"
    assert info.class_name == "MyModel"
    assert info.dependencies == ["scikit-learn"]
    assert info.has_header is True


def test_reads_requires_python():
    info, _ = _inspect('# /// script\n# requires-python = ">=3.11"\n# ///\n' + MODEL)
    assert info is not None
    assert info.requires_python == ">=3.11"


def test_a_stdlib_only_model_needs_no_header():
    """A file needing nothing may declare nothing."""
    info, err = _inspect('''
import json

from codesigner_model import BaseModel

class Plain(BaseModel):
    name = "Plain"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert err is None
    assert info is not None
    assert info.dependencies == []
    assert info.has_header is False


@pytest.mark.parametrize("base", [
    "BaseModel",                 # usual import
    "codesigner_model.BaseModel",
    "sdk.BaseModel",             # aliased
])
def test_the_base_class_is_matched_by_name(base):
    """Without importing, only the trailing name identifies the base class."""
    info, err = _inspect(f'''
class M({base}):
    name = "M"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert err is None, err
    assert info is not None
    assert info.class_name == "M"


def test_an_oversized_file_is_refused_before_parsing():
    """The check runs inside a web request."""
    _, err = inspect_model_source(b"x = 1\n" * MAX_SOURCE_BYTES)
    assert err is not None
    assert "too large" in err


def test_non_utf8_is_refused():
    _, err = inspect_model_source(b"\xff\xfe\x00bad")
    assert err is not None
    assert "UTF-8" in err


def test_a_syntax_error_names_the_line():
    _, err = _inspect("class Broken(BaseModel)\n    pass\n")
    assert err is not None
    assert "not valid Python" in err
    assert "line 1" in err


def test_a_file_with_no_model_class_is_refused():
    """The wording matches the loader's message."""
    _, err = _inspect("x = 1\n")
    assert err == "No BaseModel subclass found in the file."


def test_a_name_that_is_not_a_literal_is_refused():
    """The experiment needs a literal name while its environment builds."""
    _, err = _inspect('''
class M(BaseModel):
    @property
    def name(self): return "computed"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert err is not None
    assert "literal name" in err
    assert "My Model" in err          # example literal name


def test_an_annotation_without_a_value_does_not_count_as_setting_it():
    """A bare annotation sets nothing."""
    info, err = _inspect('''
class M(BaseModel):
    tasks: tuple
    name: str
    name = "Annotated"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert err is None
    assert info is not None
    assert info.name == "Annotated"
    assert info.tasks == ("classification",)


def test_a_malformed_header_is_refused():
    _, err = _inspect('# /// script\n# dependencies = [oops\n# ///\n' + MODEL)
    assert err is not None
    assert "not valid TOML" in err


def test_dependencies_must_be_strings():
    _, err = _inspect('# /// script\n# dependencies = [1, 2]\n# ///\n' + MODEL)
    assert err is not None
    assert "list of strings" in err


def test_listing_the_sdk_is_refused_with_a_reason():
    """The SDK comes preinstalled in every model environment."""
    _, err = _inspect('# /// script\n# dependencies = ["codesigner-model"]\n# ///\n' + MODEL)
    assert err is not None
    assert "provided automatically" in err


def test_two_headers_are_refused():
    _, err = _inspect(HEADER + HEADER + MODEL)
    assert err is not None
    assert "more than one" in err


def test_an_undeclared_import_with_no_header_is_caught_early():
    """The import would fail minutes later, after the environment builds."""
    _, err = _inspect('''
import sklearn

class M(BaseModel):
    name = "M"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert err is not None
    assert "imports `sklearn`" in err
    assert STARTER_HEADER in err       # header to paste


def test_a_declared_dependency_is_never_second_guessed():
    """Any header ends the check because import and distribution names differ,
    as with sklearn and scikit-learn."""
    info, err = _inspect('# /// script\n# dependencies = ["scikit-learn"]\n# ///\n'
                         'import sklearn\nimport cv2\n' + MODEL)
    assert err is None
    assert info is not None
    assert info.dependencies == ["scikit-learn"]


def test_the_sdk_import_is_not_treated_as_undeclared():
    info, err = _inspect(MODEL)
    assert err is None
    assert info is not None
    assert info.has_header is False
