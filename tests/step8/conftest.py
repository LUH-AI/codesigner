import pytest
from django.utils import translation


@pytest.fixture(autouse=True)
def _db_and_isolated_media(db, settings, tmp_path):
    """i18n view tests touch the database and store datasets — give each test
    DB access and a throwaway MEDIA_ROOT."""
    settings.MEDIA_ROOT = str(tmp_path / "media")


@pytest.fixture(autouse=True)
def _reset_language():
    """Each test starts from Django's default language and restores it after,
    so activate()/set_language side effects never leak between tests."""
    translation.activate("en")
    yield
    translation.deactivate_all()
