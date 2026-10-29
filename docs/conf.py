"""Sphinx configuration for codesigner's reference documentation.

AutoAPI reads the source rather than importing it, so the build needs none of
the application's dependencies, its Django settings or the SMAC branch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "Codesigner"

extensions = ["autoapi.extension", "myst_parser", "source_view", "doc_comments"]

autoapi_type = "python"
autoapi_dirs = ["../access", "../config", "../core", "../experiments", "../remote", "../ui",
                "../model_sdk/codesigner_model"]
autoapi_ignore = ["*migrations*"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autoapi_member_order = "bysource"
autoapi_add_toctree_entry = False
autoapi_keep_files = False

# Docstrings write names in single backticks; read them as references to
# Python objects, linked where they resolve.
default_role = "py:obj"

# Other projects' documentation, which the names in the source link into.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "django": ("https://docs.djangoproject.com/en/stable/", "https://docs.djangoproject.com/en/stable/_objects/"),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "smac": ("https://automl.github.io/SMAC3/main", None),
    "ConfigSpace": ("https://automl.github.io/ConfigSpace/latest", None),
}

html_theme = "furo"
html_title = "Codesigner"
exclude_patterns = ["_build"]
html_static_path = ["_static"]
html_css_files = ["source.css"]
