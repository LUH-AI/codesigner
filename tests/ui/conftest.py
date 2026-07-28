"""Shared setup for the Django-layer suite.

Everything under tests/ui/ exercises the web layer, so it all wants the same
three things: the database, a throwaway MEDIA_ROOT (creating or importing an
experiment stores a dataset copy, and the real media/ directory must stay
clean), and deterministic run execution. Each was previously repeated in a
per-directory conftest; it lives here once.

tests/core/ deliberately gets none of it — the domain layer runs without
Django, and its suite proves that by never touching these fixtures.
"""

import pytest


@pytest.fixture(autouse=True)
def db_and_isolated_media(db, settings, tmp_path):
    """Database access plus a per-test MEDIA_ROOT, for every web-layer test."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
    # Default-on in the app; pinned here so a test never depends on the ambient
    # value, and the tests that need it off can turn it off explicitly.
    settings.ALLOW_CUSTOM_MODELS = True


@pytest.fixture(autouse=True)
def huey_runs_inline():
    """Execute enqueued tasks in-process instead of handing them to a consumer.

    There is no consumer under test, so an enqueued run would never execute.
    Immediate mode runs it synchronously in the caller — combined with the
    root conftest's RUN_IMMEDIATE_IN_THREAD=False, a launched run has finished
    by the time the request returns and assertions are race-free. `db_task`
    skips its connection handling in immediate mode, so it leaves the test
    transaction alone.
    """
    from huey.contrib.djhuey import HUEY

    previous = HUEY.immediate
    HUEY.immediate = True
    yield
    HUEY.immediate = previous
