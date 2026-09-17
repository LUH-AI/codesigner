#!/usr/bin/env bash
# Put codesigner's run half on the cluster, and build the environment it needs.
#
# Idempotent: run it again after changing anything under core/ and it syncs the
# difference. --allow-existing rather than --clear, deliberately: clearing would
# delete the interpreter out from under any job currently running against it,
# and this is meant to be safe to run while work is in flight.
#
#   ./cluster/deploy.sh                 # uses CLUSTER_HOST / CLUSTER_ROOT
#   CLUSTER_HOST=kisski ./cluster/deploy.sh
#
# What crosses: core/, model_sdk/ and cluster/ — not ui/, not config/, not the
# database or media. `core/` imports no Django, which is what makes that enough.
set -euo pipefail

HOST="${CLUSTER_HOST:-kisski}"
ROOT="${CLUSTER_ROOT:-codesigner}"
PYTHON_VERSION="${CLUSTER_PYTHON:-3.12}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

echo "==> syncing to ${HOST}:${ROOT}"
ssh "$HOST" "mkdir -p '${ROOT}'"
# --relative keeps the three directories at their own names under ROOT.
# --delete so a file removed here is removed there; without it the cluster
# accumulates modules that no longer exist and imports keep working locally
# while meaning something else there.
rsync -az --relative --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='build' --exclude='*.egg-info' \
    core model_sdk cluster \
    "${HOST}:${ROOT}/"

echo "==> building the environment"
# uv is already on the cluster and the login node has PyPI egress, so the venv
# is built here rather than shipped. It lands on the shared filesystem, which is
# what lets every compute node use it without building anything itself.
# `core.models` imports `codesigner_model` — the model contract, which is its
# own dependency-free distribution and has to be installed, not merely present.
# --no-deps because it declares none by design, and letting the resolver look
# would make an offline cluster fetch a build backend it does not need.
ssh "$HOST" "cd '${ROOT}' && \
    uv venv --allow-existing --python '${PYTHON_VERSION}' >/dev/null && \
    uv pip install --quiet -r cluster/requirements-cluster.txt && \
    uv pip install --quiet --no-deps ./model_sdk"

echo "==> checking"
ssh "$HOST" "cd '${ROOT}' && .venv/bin/python -c \
    'from importlib.metadata import version; import core.io, core.registry; \
     print(\"smac\", version(\"smac\"), \"| models:\", sorted(core.registry.MODELS), \
           \"| optimizers:\", sorted(core.registry.OPTIMIZERS))'"
