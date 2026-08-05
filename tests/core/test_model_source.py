"""Reading a model file without running it.

Everything the create form needs from an uploaded model — is it Python, is
there a model class, what is it called, what does it say it needs — is read from
the syntax tree instead of by importing the file. These pin what that can and
cannot answer, and that the messages a user sees are the useful ones.
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


# ── the happy path ───────────────────────────────────────────────────────────

def test_reads_name_class_and_dependencies():
    info, err = _inspect(HEADER + MODEL)

    assert err is None
    assert info.name == "My Model"
    assert info.class_name == "MyModel"
    assert info.dependencies == ["scikit-learn"]
    assert info.has_header is True


def test_reads_requires_python():
    info, _ = _inspect('# /// script\n# requires-python = ">=3.11"\n# ///\n' + MODEL)
    assert info.requires_python == ">=3.11"


def test_a_stdlib_only_model_needs_no_header():
    """Declaring nothing is fine as long as the file needs nothing."""
    info, err = _inspect('''
import json

from codesigner_model import BaseModel

class Plain(BaseModel):
    name = "Plain"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert err is None
    assert info.dependencies == []
    assert info.has_header is False


@pytest.mark.parametrize("base", [
    "BaseModel",                 # the usual import
    "codesigner_model.BaseModel",  # module-qualified
    "sdk.BaseModel",             # aliased import
])
def test_the_base_class_is_matched_by_name(base):
    """The file is never imported, so there is nothing to resolve the reference
    against — the trailing name is all there is to go on."""
    info, err = _inspect(f'''
class M({base}):
    name = "M"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert err is None, err
    assert info.class_name == "M"


# ── what it refuses, and what it says ────────────────────────────────────────

def test_an_oversized_file_is_refused_before_parsing():
    """Parsing is real work and this runs in a web request."""
    _, err = inspect_model_source(b"x = 1\n" * MAX_SOURCE_BYTES)
    assert "too large" in err


def test_non_utf8_is_refused():
    _, err = inspect_model_source(b"\xff\xfe\x00bad")
    assert "UTF-8" in err


def test_a_syntax_error_names_the_line():
    _, err = _inspect("class Broken(BaseModel)\n    pass\n")
    assert "not valid Python" in err
    assert "line 1" in err


def test_a_file_with_no_model_class_is_refused():
    """Worded exactly as the loader that used to do this, so the message a user
    sees does not depend on which check caught it."""
    _, err = _inspect("x = 1\n")
    assert err == "No BaseModel subclass found in the file."


def test_a_name_that_is_not_a_literal_is_refused():
    """A @property name cannot be read without running the file, and the
    experiment needs something to be called while its environment builds."""
    _, err = _inspect('''
class M(BaseModel):
    @property
    def name(self): return "computed"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert "literal name" in err
    assert "My Model" in err          # the message shows the fix


def test_a_malformed_header_is_refused():
    _, err = _inspect('# /// script\n# dependencies = [oops\n# ///\n' + MODEL)
    assert "not valid TOML" in err


def test_dependencies_must_be_strings():
    _, err = _inspect('# /// script\n# dependencies = [1, 2]\n# ///\n' + MODEL)
    assert "list of strings" in err


def test_listing_the_sdk_is_refused_with_a_reason():
    """It is installed into every model environment already, so listing it gets
    a confusing 'no such package' from the index instead."""
    _, err = _inspect('# /// script\n# dependencies = ["codesigner-model"]\n# ///\n' + MODEL)
    assert "provided automatically" in err


def test_two_headers_are_refused():
    _, err = _inspect(HEADER + HEADER + MODEL)
    assert "more than one" in err


# ── the one heuristic ────────────────────────────────────────────────────────

def test_an_undeclared_import_with_no_header_is_caught_early():
    """A model env holds only what the file declares, so this import is certain
    to fail — several minutes later, once the environment has been built."""
    _, err = _inspect('''
import sklearn

class M(BaseModel):
    name = "M"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, c, a, b, d, seed=0): return []
''')
    assert "imports `sklearn`" in err
    assert STARTER_HEADER in err       # and shows exactly what to paste


def test_a_declared_dependency_is_never_second_guessed():
    """Import names and distribution names disagree often enough — sklearn from
    scikit-learn, cv2 from opencv-python — that matching them would reject
    working files. A header present at all ends the questioning."""
    info, err = _inspect('# /// script\n# dependencies = ["scikit-learn"]\n# ///\n'
                         'import sklearn\nimport cv2\n' + MODEL)
    assert err is None
    assert info.dependencies == ["scikit-learn"]


def test_the_sdk_import_is_not_treated_as_undeclared():
    info, err = _inspect(MODEL)
    assert err is None
    assert info.has_header is False
