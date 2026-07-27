# Step 10 (part 1) — Background runs on a real task queue

**Status:** implemented, awaiting your sign-off.
**You can now:** run experiments through a durable **huey** task queue processed
by a separate `manage.py run_huey` consumer, instead of a thread inside the web
server. Nothing changes for you visually — the payoff is operational: the web
process stays light, queued runs survive a web restart, and the work can be
isolated (and later scaled) in its own process.

> Step 10 is being delivered in parts. This part is the task-queue swap; the
> Docker/compose + whitenoise/gunicorn + healthcheck + demo volume mounts +
> moving custom-model execution into the worker + the final parity walk follow.

---

## 1. The concept: a task queue

Until now, clicking **Run** spawned a background *thread inside the web process*
(Step 6). It worked, but the heavy optimization ran in the same process that
serves pages, and a web restart abandoned in-flight runs.

A **task queue** splits "decide to run work" from "do the work":

- The web process, on Run, drops a message — "execute run #42" — into a queue.
  Here the queue is a SQLite file (**SqliteHuey**), so there's no Redis and
  nothing extra to operate — it fits the single-box design.
- A separate long-lived process, the **consumer** (`manage.py run_huey`),
  watches the queue and actually runs the optimization.

Because we deliberately built run state on the database back in Step 6
(status, `cancel_requested`, timestamps), the swap touches almost nothing: the
web process only needs to change how it *launches* the work. Polling and cancel
already read/write the DB and are unchanged.

**Immediate mode.** huey can run a task inline instead of via the consumer
(`immediate=True`). We default it to `DEBUG`, so a lone `runserver` still works
in development with no second process; it's off in production, where the
consumer runs. Tests flip it on so an enqueued task runs synchronously and can
be asserted.

## 2. What changed

- **`config/settings.py`** — `huey.contrib.djhuey` added to `INSTALLED_APPS`;
  a `HUEY` config (`SqliteHuey`, `filename=BASE_DIR/huey.sqlite3`,
  `immediate=env.bool("HUEY_IMMEDIATE", default=DEBUG)`, `results=False`).
- **`web/tasks.py`** (new) — `run_experiment_task = @db_task()` wrapping the
  existing `execute_run`. `run_huey` autoloads each app's `tasks.py`, so the
  consumer finds it with no extra wiring; the web process imports it lazily when
  enqueuing.
- **`web/services/run.py`** — `start_background_run` now enqueues that task
  instead of `threading.Thread(...)` (and `import threading` is gone). This is
  the *only* launch-path change; `execute_run` (rebuild-from-DB, run with the
  `DbCancelFlag`, write result/status back) is byte-for-byte the same.
- **`sweep_stale_runs`** — now sweeps only `running` runs, not `pending` (see
  the deviation below).
- **`.gitignore`** — `huey.sqlite3*`.

## 3. Feature checklist

Source: Step 6's thread-based engine (the behavior to preserve). ✅ done ·
🔄 changed · ⏭ later in Step 10.

- ✅ Clicking Run executes the optimization out of band; the page stays
  responsive and polls to completion (unchanged UX)
- ✅ Run lifecycle preserved: create → execute → done, **resume** (trials
  continue), **cancel** mid-run, **error** capture — all still exercised by the
  existing `tests/step6` suite calling `execute_run` directly
- ✅ Runs execute in a **separate consumer process** (`manage.py run_huey`)
- ✅ Durable queue: a queued (`pending`) run **survives a web restart** and is
  still processed — a genuine improvement over the thread version
- ✅ Dev ergonomics: `immediate=DEBUG` means a solo `runserver` still runs tasks
  inline; set `HUEY_IMMEDIATE=false` to use the real consumer locally
- 🔄 **Stale-run sweep semantics** changed (below)
- ⏭ Custom-model execution moving *into* the worker (the trust-boundary
  isolation) — deferred to the Docker part of Step 10
- ⏭ Running the consumer as a second container process (compose) — deferred

## 4. Deviation: the stale-run sweep

Step 6's `sweep_stale_runs` marked **both** `pending` and `running` runs errored
on startup, because in-process threads didn't survive a restart. With a durable
queue that's no longer true: a `pending` run is still sitting in the queue and
the consumer *will* pick it up, so sweeping it would wrongly kill live work.
Only a `running` run is stale — its consumer died mid-execution — so the sweep
now targets `status="running"` alone. (It's meant to run once at
consumer/system startup; the exact wiring lands with the Docker entrypoint.)
The orphan-sweep test was updated to this contract.

## 5. The tests

- `tests/step10/test_task_queue.py`: enqueuing a run (immediate mode) drives it
  to `done` with results written (the launch → task → `execute_run` path); the
  task is a registered huey `db_task` (`schedule`/`call_local`); and
  `call_local` runs the experiment (the consumer's execution path).
- `tests/step6/test_orphan_sweep.py` (updated): `running` → error, `pending`
  left queued, `done` untouched.
- Full suite: **198 passed**, 5 slow deselected.

## 6. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python -m pytest tests/step10 tests/step6/test_orphan_sweep.py

# Real two-process path (not immediate):
HUEY_IMMEDIATE=false DEBUG=false .venv/bin/python manage.py run_huey     # terminal 1
HUEY_IMMEDIATE=false .venv/bin/python manage.py runserver                # terminal 2
# create an experiment with a dataset, click Run — the consumer executes it.
```

I verified the two-process path directly: `run_huey` booted and autoloaded
`web.tasks.run_experiment_task`; a run enqueued from a separate process came
back `pending`, and the consumer executed it to `done` (3 trials).
