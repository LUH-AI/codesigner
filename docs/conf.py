"""Sphinx configuration for codesigner's reference documentation.

AutoAPI reads the source rather than importing it, so the build needs none of
the application's dependencies, its Django settings or the SMAC branch.
"""

project = "Codesigner"

# viewcode first: AutoAPI only serves it source if it is already loaded.
extensions = ["sphinx.ext.viewcode", "autoapi.extension", "myst_parser"]

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

html_theme = "furo"
html_title = "Codesigner"
exclude_patterns = ["_build"]
