#!/usr/bin/env bash
# Run codesigner locally. One script, one argument per configuration — the same
# shape as docker-entrypoint.sh, which dispatches the container's two roles.
#
#   ./run.sh                    transparent: no accounts, runs execute in-process
#   ./run.sh auth               the login wall, ownership and groups
#   ./run.sh auth --demo        ... plus a worked instance to sign in to
#   ./run.sh queue              the real queue instead of in-process runs
#   ./run.sh docker             web + worker in containers
#   ./run.sh docker --hosted    ... over Redis and Postgres (configuration C)
#
# Extra arguments reach `runserver`, so `./run.sh 8001` moves the port.
#
# Why a script rather than a README paragraph. `migrate` after a pull and
# `sweep_stale_runs` after a hard kill are both idempotent, both fast, and both
# only ever noticed by their absence — a missing column, or a row stuck at
# "running". And REQUIRE_LOGIN has to arrive as a process variable rather than
# sitting in .env, because config/settings.py reads .env and the test suite
# would inherit it: every account-agnostic test would silently become a
# login-wall test. Encoding that here is the point.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# The project's own interpreter, so this works without an activated venv.
PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python"

MODE="serve"
DEMO=false
HOSTED=false
CUSTOM_MODELS=false
EXTRA=()

case "${1:-}" in
    auth | queue | docker) MODE="$1"; shift ;;
    "" | -* | [0-9]*) ;;
    *) echo "run.sh: unknown mode '$1' (auth, queue, docker)" >&2; exit 2 ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --demo)          DEMO=true ;;
        --hosted)        HOSTED=true ;;
        --custom-models) CUSTOM_MODELS=true ;;
        -h | --help)     sed -n '2,13p' "$0" | cut -c3-; exit 0 ;;
        *)               EXTRA+=("$1") ;;
    esac
    shift
done

# ── containers ───────────────────────────────────────────────────────────────
# Nothing below applies: the image runs its own migrations and sweeps through
# docker-entrypoint.sh, and compose owns the environment.
if [ "$MODE" = docker ]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "run.sh: docker is not installed. The bare-metal modes above need" >&2
        echo "        no container; this one is for isolating model execution." >&2
        exit 1
    fi
    files=(-f docker-compose.yml)
    if $HOSTED; then files+=(-f docker-compose.hosted.yml); fi
    exec docker compose "${files[@]}" up --build "${EXTRA[@]}"
fi

# ── bare metal ───────────────────────────────────────────────────────────────
# Uploaded model .py files are arbitrary code execution, and outside a container
# there is nothing between them and this account. Off here regardless of .env,
# and `--custom-models` is the deliberate way past it.
export ALLOW_CUSTOM_MODELS=False
if $CUSTOM_MODELS; then
    export ALLOW_CUSTOM_MODELS=True
    echo "!! ALLOW_CUSTOM_MODELS=True outside a container." >&2
    echo "!! An uploaded model .py is executed as $(id -un), with this" >&2
    echo "!! filesystem and ~/.ssh in reach. './run.sh docker' isolates it." >&2
fi

if [ "$MODE" = auth ]; then
    export REQUIRE_LOGIN=True
fi

# The real queue, for exercising the consumer path a container or host would
# use. Two terminals by necessity: run_huey is a second long-running process,
# and sharing one terminal would interleave its log with the server's.
if [ "$MODE" = queue ]; then
    export HUEY_IMMEDIATE=false
fi

"$PY" manage.py migrate --noinput
"$PY" manage.py sweep_stale_runs

if $DEMO; then
    if [ "$MODE" != auth ]; then
        echo "note: --demo seeds accounts, but without 'auth' there is no login" >&2
        echo "      wall for them to matter at. Did you mean './run.sh auth --demo'?" >&2
    fi
    "$PY" manage.py seed_demo
fi

if [ "$MODE" = queue ]; then
    echo
    echo "Queue mode. In a second terminal:"
    echo "    HUEY_IMMEDIATE=false $PY manage.py run_huey"
    echo
fi

exec "$PY" manage.py runserver "${EXTRA[@]}"
