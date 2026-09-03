#!/usr/bin/env sh
set -e

# One image, two roles (see docker-compose.yml):
#   web    — apply migrations, then serve via gunicorn (WhiteNoise serves static).
#   worker — clear runs orphaned by a previous consumer, then process the queue.
# The stale-run sweep lives on the worker: a "running" row is only stale when
# the consumer that owned it died, which is exactly a worker restart.
case "$1" in
  web)
    python manage.py migrate --noinput
    exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 120
    ;;
  worker)
    python manage.py sweep_stale_runs
    python manage.py sweep_stale_model_envs
    exec python manage.py run_huey
    ;;
  *)
    exec "$@"
    ;;
esac
