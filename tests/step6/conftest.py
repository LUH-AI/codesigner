import pytest


@pytest.fixture(autouse=True)
def _db_and_isolated_media(db, settings, tmp_path):
    """Run-execution and view tests touch the database and store datasets —
    give each test DB access and a throwaway MEDIA_ROOT."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
