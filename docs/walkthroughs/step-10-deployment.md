# Step 10 (part 2) — Package for deployment + final parity

**Status:** implemented, awaiting your sign-off.
**You can now (as an operator):** run Codesigner as two containers off one image
— a gunicorn web server and a huey worker — with static files served in-process
and demo datasets / models supplied by volume mounts. This closes the last
parity item.

Part 1 was the task-queue swap ([step-10-task-queue.md](step-10-task-queue.md));
this part is static serving, the healthcheck, the mounted-model source, and the
container packaging.

---

## 1. The concepts

- **WhiteNoise** serves static files from the app process, so there's no need
  for a separate nginx/CDN in a single-box deployment. `collectstatic` gathers
  and compresses files; the middleware serves them with cache headers. We use
  the *non-manifest* compressed storage so `{% static %}` also works in tests
  and `runserver` without a prior `collectstatic` (the manifest/hashed variant
  errors there, since it tries to hash files that haven't been collected).
- **gunicorn** is the production WSGI server (dev uses `runserver`).
- **Two processes, one image.** `docker-entrypoint.sh` dispatches on a role
  argument: `web` (migrate → gunicorn) and `worker` (`sweep_stale_runs` →
  `run_huey`). `docker-compose.yml` runs both, sharing a `data` volume (SQLite
  DB, huey queue, media) so they see the same run state.
- **Healthcheck.** `/healthz/` returns `ok` touching nothing, so compose can
  probe liveness.
- **Volume mounts.** `./datasets` and `./mounted_models` are bind-mounted;
  files dropped there appear as demo-dataset / mounted-model options — the
  operator supplies data without rebuilding the image.

## 2. What changed

- `config/settings.py` — WhiteNoise middleware (after SecurityMiddleware),
  `STATIC_ROOT`, `STORAGES` (WhiteNoise `CompressedStaticFilesStorage`), and
  env-configurable `MEDIA_ROOT` + huey `filename` so both processes share one
  data volume.
- `ui/views.py` + `ui/urls.py` — the `/healthz/` endpoint.
- `ui/forms.py` + `ui/views.py` + `new_experiment.html` — the mounted-model
  source (part of the parity item; see its own tests).
- New: `Dockerfile`, `docker-entrypoint.sh`, `docker-compose.yml`,
  `.dockerignore`, `.github/workflows/ci.yml`, `mounted_models/`. `requirements.txt`
  gains gunicorn/whitenoise and pins SMAC from git. `README.md` documents dev,
  the queue, Docker, and the trust model.

## 3. Feature checklist (Step 10's build items)

✅ done · ⚠️ built but not run here · 🔁 already covered.

- ✅ Runs on a real task queue with a separate consumer (part 1)
- ✅ Custom-model execution runs in the **worker** process (a consequence of
  part 1 — `execute_run` runs in the consumer; the create-time *validation*
  load still happens in the web process, noted in Step 9)
- ✅ WhiteNoise static serving + `collectstatic` (compressed variants produced)
- ✅ `/healthz/` healthcheck endpoint (tested)
- ✅ Mounted-model source + demo-dataset source, bind-mounted in compose
- ✅ Dockerfile with the swig/pyrfr build deps + SMAC git pin; two-role
  entrypoint; compose with shared data volume + healthcheck
- ✅ README (setup, dev, Docker, trust model); CI workflow (tests + translation
  compile + docker build)
- ⚠️ `docker build` / `docker run` end-to-end — **not executed in the build
  environment (no Docker daemon).** The image, compose, and entrypoint are
  written; everything runnable without Docker is verified (below).

## 4. Verification

Automated: **`pytest` — 209 passed** (full suite, including the slow SMAC
end-to-end tests). New this part: `tests/ui/ops/test_healthz.py` and
`tests/ui/custom_models/test_mounted_models.py`.

Ran here (no Docker needed):
- `manage.py collectstatic --noinput` → 133 files copied, compressed `.gz`
  variants produced.
- `manage.py check` → no issues (WhiteNoise middleware + STORAGES load).
- `config.wsgi:application` imports (gunicorn's target); `docker-entrypoint.sh`
  passes `sh -n`.
- `/healthz/` returns `ok`; the mounted-model dropdown renders live from a
  `mounted_models/*.py`.

**Not run here:** `docker build` and `docker compose up` — no Docker daemon in
this environment. To finish the sign-off on a Docker host:

```bash
cp .env.example .env && $EDITOR .env      # set SECRET_KEY
docker compose up --build
# open http://localhost:8000 — create → run → cancel → export → import
```

## 5. Final parity walk

Every functional item in [PARITY.md](../../PARITY.md) is implemented and
covered by the test suite and per-step live smokes: create (registry / uploaded
/ mounted model · demo / uploaded dataset), run (SMAC / Random / Grid), resume,
cancel, metric-change rules, the four analytics panels + click-to-select +
metric switching, `.ihpo` export/import (incl. read-only), delete, the sidebar
spinner, and en/de/es.

Two honest caveats on "done":
1. The **containerized** end-to-end (`docker build && docker run … create → run
   → cancel → export → import`) that Step 10 specifies as its final check has
   **not been executed here** — it needs a Docker host. The artifacts are in
   place to do it.
2. The plan's "**side-by-side against the running Streamlit app one final
   time**" is a manual pass best done by you with both apps open; the ticks
   reflect per-step verification, several of them side-by-side during the
   relevant step, but I have not re-walked all 22 items against a live Streamlit
   instance in one sitting.

So: **feature-complete and green in tests; the container run and the final
manual side-by-side are the two things left to close the book.**
