"""The experiments app is Django-layer code like ui's, so its tests get the same
database, media and queue setup, and the same stand-ins for uv."""

from tests.ui.conftest import db_and_isolated_media, huey_runs_inline  # noqa: F401
from tests.ui.model_envs.conftest import failing_uv, fake_uv, no_uv  # noqa: F401
