#!/bin/bash
# The assistant's models on KISSKI, from this machine.
#
#   cluster/assistant/kisski.sh start    submit the job (serve.sbatch)
#   cluster/assistant/kisski.sh status   the job, and where it serves once ready
#   cluster/assistant/kisski.sh tunnel   forward the two servers to localhost and
#                                        write the settings Codesigner reads
#   cluster/assistant/kisski.sh stop     cancel the job
#
# Needs `ssh kisski` to work (an entry in ~/.ssh/config, and the university
# network or VPN). The settings go to $ENV_FILE, readable only by you; load them
# before starting Codesigner:  set -a; . ~/.config/codesigner/assistant.env; set +a
set -euo pipefail

HOST="${KISSKI_HOST:-kisski}"
REMOTE_DIR="${ASSISTANT_DIR:-codesigner-assistant}"
CHAT_LOCAL="${CHAT_LOCAL_PORT:-8000}"
EMBED_LOCAL="${EMBED_LOCAL_PORT:-8001}"
ENV_FILE="${ENV_FILE:-$HOME/.config/codesigner/assistant.env}"
HERE="$(cd "$(dirname "$0")" && pwd)"

remote() { ssh -o BatchMode=yes "$HOST" "$@" < /dev/null; }

case "${1:-}" in
start)
    remote "mkdir -p $REMOTE_DIR/logs"
    scp -q "$HERE/serve.sbatch" "$HOST:$REMOTE_DIR/serve.sbatch"
    remote "cd $REMOTE_DIR && sbatch serve.sbatch"
    ;;
status)
    remote "squeue -u \$USER -n codesigner-assistant -o '%i %T %M %N %R'; \
            if [ -s $REMOTE_DIR/endpoint ]; then echo; cat $REMOTE_DIR/endpoint; \
            else echo; echo 'not serving yet'; fi; \
            log=\$(ls -t $REMOTE_DIR/logs/serve-*.out 2>/dev/null | head -1); \
            if [ -n \"\$log\" ]; then echo; echo \"== \$log\"; tail -5 \"\$log\"; fi"
    ;;
stop)
    remote "scancel -u \$USER -n codesigner-assistant"
    ;;
tunnel)
    endpoint="$(remote "cat $REMOTE_DIR/endpoint 2>/dev/null" || true)"
    if [ -z "$endpoint" ]; then
        echo "Not serving yet: see '$0 status'." >&2
        exit 1
    fi
    value() { printf '%s\n' "$endpoint" | sed -n "s/^$1=//p"; }
    # The login node cannot reach ports on compute nodes, so the tunnel goes
    # through the node itself — allowed while our job runs on it — to the
    # servers' ports there. The node by the name Slurm knows it by, as the
    # same user and key as $HOST.
    node="$(remote "squeue -h -j $(value job) -o %N" 2>/dev/null || true)"
    node="${node:-$(value node)}"
    user="$(ssh -G "$HOST" | awk '/^user /{print $2; exit}')"
    key="$(ssh -G "$HOST" | awk '/^identityfile /{print $2; exit}')"
    mkdir -p "$(dirname "$ENV_FILE")"
    (umask 077 && {
        echo "ASSISTANT_ENABLED=True"
        echo "ASSISTANT_LLM_BASE_URL=http://127.0.0.1:$CHAT_LOCAL/v1"
        echo "ASSISTANT_LLM_MODEL=$(value chat_model)"
        echo "ASSISTANT_LLM_TOKEN=$(remote "cat $REMOTE_DIR/key")"
        echo "ASSISTANT_EMBED_BASE_URL=http://127.0.0.1:$EMBED_LOCAL/v1"
        echo "ASSISTANT_EMBED_MODEL=$(value embed_model)"
        echo "ASSISTANT_EMBED_TOKEN=$(remote "cat $REMOTE_DIR/key")"
    } > "$ENV_FILE")
    echo "Settings in $ENV_FILE. Forwarding localhost:$CHAT_LOCAL and :$EMBED_LOCAL through $node; Ctrl-C ends it."
    exec ssh -N -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new \
        -J "$HOST" -i "${key/#\~/$HOME}" \
        -L "$CHAT_LOCAL:127.0.0.1:$(value chat_port)" \
        -L "$EMBED_LOCAL:127.0.0.1:$(value embed_port)" "$user@$node"
    ;;
*)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
