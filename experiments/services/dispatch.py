"""Handing work to the background, whether or not there is a consumer.

With a consumer running, enqueuing returns at once and there is nothing to think
about. In immediate mode there is no consumer and huey executes the task **in the
caller** — which, for work started from a page, is the request. A run takes
minutes and preparing an environment can take longer, so either would hold the
response open past gunicorn's timeout.

So immediate-mode work goes to a daemon thread and the response returns. Tests
turn that off (see tests/conftest.py) to keep launches synchronous and
assertions race-free.
"""

import threading

from django.conf import settings
from huey.contrib.djhuey import HUEY


def enqueue(task, *args) -> None:
    """Run *task* in the background, returning immediately either way."""
    if settings.RUN_IMMEDIATE_IN_THREAD and HUEY.immediate:
        threading.Thread(target=task, args=args, daemon=True).start()
        return
    task(*args)
