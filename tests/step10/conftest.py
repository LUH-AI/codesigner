import pytest


@pytest.fixture(autouse=True)
def _db_media_immediate(db, settings, tmp_path):
    """Step 10 run-queue tests need DB access, a throwaway MEDIA_ROOT, and huey
    in immediate mode so an enqueued task runs synchronously in-process (no
    consumer, and no real queue file touched). db_task skips connection
    handling in immediate mode, so it plays nicely with the test transaction."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
    from huey.contrib.djhuey import HUEY
    previous = HUEY.immediate
    HUEY.immediate = True
    yield
    HUEY.immediate = previous
