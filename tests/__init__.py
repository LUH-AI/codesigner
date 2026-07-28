"""The suite mirrors the code's layering, then its features.

`core/` covers the domain layer and imports no Django — running it alone is
what proves that layer stands on its own. `ui/` covers the Django layer, one
directory per feature (experiments, runs, results, storage, custom_models,
settings, i18n, ops), with the database and an isolated MEDIA_ROOT supplied to
all of it by `ui/conftest.py`.
"""
