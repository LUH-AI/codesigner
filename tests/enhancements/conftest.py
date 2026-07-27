import pytest


@pytest.fixture(autouse=True)
def _db_media_flag(db, settings, tmp_path):
    """Enhancement tests touch the DB, store uploads, and one depends on the
    custom-model flag — give each a throwaway MEDIA_ROOT and the flag on."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
    settings.ALLOW_CUSTOM_MODELS = True
