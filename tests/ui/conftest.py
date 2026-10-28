"""The autouse fixtures here give web-layer tests a database, a throwaway
MEDIA_ROOT and inline runs; `tests/experiments/conftest.py` imports them.
"""

import pytest


@pytest.fixture(autouse=True)
def db_and_isolated_media(db, settings, tmp_path):
    """`db_and_isolated_media` opens the database and isolates MEDIA_ROOT."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
    # The app default is off, so pin it on.
    settings.ALLOW_CUSTOM_MODELS = True


@pytest.fixture(autouse=True)
def huey_runs_inline():
    """`HUEY.immediate` runs enqueued tasks in the caller, since no consumer
    runs under test. The root conftest's `RUN_IMMEDIATE_IN_THREAD=False` then
    makes a launched run finish before the request returns. `db_task` leaves the
    test transaction alone in immediate mode.
    """
    from huey.contrib.djhuey import HUEY

    previous = HUEY.immediate
    HUEY.immediate = True
    yield
    HUEY.immediate = previous
