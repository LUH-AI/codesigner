"""In immediate mode, huey executes a task inside the request and would keep a
run past gunicorn's timeout. Therefore `enqueue` starts a daemon thread, which
tests disable (tests/conftest.py).
"""

import threading

from django.conf import settings
from huey.contrib.djhuey import HUEY


def enqueue(task, *args) -> None:
    """`enqueue` never waits on a consumer."""
    if settings.RUN_IMMEDIATE_IN_THREAD and HUEY.immediate:
        threading.Thread(target=task, args=args, daemon=True).start()
        return
    task(*args)
