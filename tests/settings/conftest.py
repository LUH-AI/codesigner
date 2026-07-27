import pytest


@pytest.fixture(autouse=True)
def _db_and_media(db, settings, tmp_path):
    """Settings tests hit the DB (Experiment/GlobalSettings) and export files."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
