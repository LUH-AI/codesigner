# A run nobody is executing

## What happened

Two runs sat at "running" for four days. Cancel did nothing to either.

Cancelling is **cooperative**: `run_cancel` sets `cancel_requested` on the row,
and `DbCancelFlag.is_set()` — polled by the optimizer between trials — reads it
back. It needs something to be doing the polling.

Nothing was. With no `run_huey` consumer, `HUEY.immediate` is on (it defaults to
`DEBUG`), and `services/dispatch.py` puts the task on a **daemon thread in the
web process** so the request can return. A daemon thread dies with its process,
and in development that process is `runserver` — which restarts every time a
file is saved. The runs started at 13:33; a save shortly after took them with
it. After that the flag had no reader, and the row had no way out of "running".

`sweep_stale_runs` already existed for exactly this, but it is a management
command run by the container entrypoint. A bare `runserver` never calls it, so in
development the hole was open the whole way.

## Two fixes, because there are two failures

**The dead run finishes itself.** `sweep_orphaned_runs`, called from the
run-status poll the page is already making. In immediate mode the executor *is*
this process, so a run whose `started_at` predates this process's own start has
nobody executing it and never will. That is an exact test rather than a timeout —
no guessing about how long a run ought to take, and no false positives. It is
wrong in consumer mode, where the executor outlives any web restart, so it checks
`HUEY.immediate` first and does nothing there; consumer restarts remain the
startup command's job. It runs once per process, because the answer cannot change
while the process lives.

**The live-but-unresponsive run can be given up on.** A worker that is still
running but wedged is not orphaned, and no amount of inference will free it. So
the status box escalates: Cancel first, and once that has been asked for and has
not worked, **Give up on it**.

It does not kill anything and does not claim to — the web process has no handle
on a worker in another thread or another process. What it does is stop the
experiment waiting. If the worker turns out to be alive after all, its last act
is a filtered update of that same row, which puts back whatever actually
happened.

Second rather than first on purpose: a cooperative cancel keeps the trials that
already ran, and giving up cannot.

## Verify

```
python -m pytest -q -m "not slow"     # 1023 passing
python manage.py check
```

The sweep is covered four ways — it finishes a run older than this process,
leaves a newer one alone, does nothing under a consumer, and runs once. The
escalation is covered by the order the two buttons appear in, that giving up
finishes the row and says why, that a run never asked to stop is not given up on
by reaching the URL directly, and that it is POST-only.

Worth knowing while developing: **saving a file kills any run in progress.**
That is `runserver`'s autoreloader and not something to fix — but it is why runs
disappear mid-flight during a working session, and now the page says so instead
of hanging.
