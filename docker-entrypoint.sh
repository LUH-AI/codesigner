#!/usr/bin/env sh
set -e

# One image, two roles (see docker-compose.yml):
#   web    — apply migrations, then serve the ASGI app via gunicorn and uvicorn's worker
#            (WhiteNoise serves static; the assistant's event stream holds no worker).
#   worker — clear runs orphaned by a previous consumer, queue a build of the
#            assistant's library when it is on and the library is stale, then
#            process the queue.
# The stale-run sweep lives on the worker: a "running" row is only stale when
# the consumer that owned it died, which is exactly a worker restart.
case "$1" in
  web)
    python manage.py migrate --noinput
    exec gunicorn config.asgi:application -k uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000 --workers 3 --timeout 120
    ;;
  worker)
    python manage.py sweep_stale_runs
    python manage.py sweep_stale_model_envs
    python manage.py assistant_knowledge ensure
    exec python manage.py run_huey
    ;;
  *)
    exec "$@"
    ;;
esac
