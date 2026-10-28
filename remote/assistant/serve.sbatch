#!/bin/bash
# The assistant's models on one KISSKI H100, for development until an API is
# available: the chat model and the embedding model, each a vLLM server with the
# OpenAI API, behind one key.
#
# Submitted by scripts/assistant_kisski.sh, which copies this file to
# $ASSISTANT_DIR and tunnels to the servers once they answer. Ports are picked
# free on the node, since other jobs share it, and written with the node's name
# to $ASSISTANT_DIR/endpoint for the tunnel to read.
#SBATCH --job-name=codesigner-assistant
#SBATCH --partition=kisski-inference
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/serve-%j.out

set -euo pipefail

ASSISTANT_DIR="${ASSISTANT_DIR:-$HOME/codesigner-assistant}"
IMAGE="${IMAGE:-$ASSISTANT_DIR/vllm-v0.30.0.sif}"
CHAT_MODEL="${CHAT_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
CHAT_NAME="${CHAT_NAME:-qwen3-30b-a3b}"
CHAT_CONTEXT="${CHAT_CONTEXT:-32768}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_NAME="${EMBED_NAME:-qwen3-embedding-0.6b}"
ENDPOINT="$ASSISTANT_DIR/endpoint"

cd "$ASSISTANT_DIR"
export HF_HOME="$ASSISTANT_DIR/hf"

# One key for both servers, made once and readable only by its owner. In the
# environment rather than on the command line, where anyone on the node could
# read it.
if [ ! -s key ]; then
    (umask 077 && openssl rand -hex 32 > key)
fi
VLLM_API_KEY="$(cat key)"
export VLLM_API_KEY

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
CHAT_PORT="$(free_port)"
EMBED_PORT="$(free_port)"

serve() {
    apptainer exec --nv --bind "$ASSISTANT_DIR" "$IMAGE" vllm serve "$@" --host 0.0.0.0
}

answers() {
    curl -sf -o /dev/null -H "Authorization: Bearer $VLLM_API_KEY" "http://127.0.0.1:$1/v1/models"
}

wait_for() {
    local port="$1" pid="$2"
    until answers "$port"; do
        kill -0 "$pid" 2>/dev/null || { echo "the server on port $port stopped"; exit 1; }
        sleep 10
    done
}

cleanup() {
    rm -f "$ENDPOINT"
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT
rm -f "$ENDPOINT"

# The chat model first, taking most of the GPU; the embedding model fits in
# what is left, so it starts once the first has claimed its share.
serve "$CHAT_MODEL" --served-model-name "$CHAT_NAME" --port "$CHAT_PORT" \
    --max-model-len "$CHAT_CONTEXT" --gpu-memory-utilization 0.80 \
    --enable-auto-tool-choice --tool-call-parser hermes &
CHAT_PID=$!
wait_for "$CHAT_PORT" "$CHAT_PID"

serve "$EMBED_MODEL" --served-model-name "$EMBED_NAME" --port "$EMBED_PORT" \
    --runner pooling --convert embed --max-model-len 8192 --gpu-memory-utilization 0.10 &
EMBED_PID=$!
wait_for "$EMBED_PORT" "$EMBED_PID"

cat > "$ENDPOINT" <<EOF
node=${SLURMD_NODENAME:-$(hostname)}
job=${SLURM_JOB_ID:-}
chat_port=$CHAT_PORT
chat_model=$CHAT_NAME
embed_port=$EMBED_PORT
embed_model=$EMBED_NAME
EOF
echo "serving $CHAT_NAME on $(hostname -s):$CHAT_PORT and $EMBED_NAME on :$EMBED_PORT"

# Either server ending ends the job, so the endpoint file never names a dead one.
wait -n "$CHAT_PID" "$EMBED_PID"
