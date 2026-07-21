import pytest


@pytest.fixture(autouse=True)
def _db_and_isolated_media(db, settings, tmp_path):
    """Every page renders the sidebar, which lists saved experiments from the
    database — so even the inspect-page tests need DB access. Isolated
    MEDIA_ROOT keeps the real media/ directory clean."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
