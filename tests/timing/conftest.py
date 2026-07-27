import pytest


@pytest.fixture(autouse=True)
def _db_and_media(db, settings, tmp_path):
    """The duration-column view test needs DB access + a throwaway MEDIA_ROOT;
    the core-level timing tests ignore it (harmless)."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
