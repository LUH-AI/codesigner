import pytest


@pytest.fixture(autouse=True)
def _db_and_isolated_media(db, settings, tmp_path):
    """Web-layer tests need the database, and running an experiment now stores
    a dataset copy — give each test a throwaway MEDIA_ROOT so the real media/
    directory stays clean."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
