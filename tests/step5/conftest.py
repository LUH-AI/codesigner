import pytest


@pytest.fixture(autouse=True)
def _isolated_media(settings, tmp_path):
    """Persisting an experiment stores a dataset copy — give each test a
    throwaway MEDIA_ROOT so the real media/ directory stays clean. (DB access
    is declared per-test with @pytest.mark.django_db.)"""
    settings.MEDIA_ROOT = str(tmp_path / "media")
