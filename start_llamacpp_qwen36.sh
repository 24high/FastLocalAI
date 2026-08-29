#!/usr/bin/env bash
#
# Startet Qwen3.6-35B-A3B UD-Q4_K_M mit llama.cpp (llama-server, CUDA-Docker).
#
# WARUM als Alternative zu Lucebox:
#   1. Warm-TTFT: llama-server haelt den KV-Cache token-genau im Slot und
#      verarbeitet bei Folgerequests NUR die neuen Tokens ("--cache-reuse").
#      Kein Sticky-Snapshot-Problem wie bei Lucebox -- die TTFT eines
#      Folgerequests haengt nur vom letzten Turn ab, nicht vom Gesamtkontext.
#   2. Cold-Prefill: Experten werden beim Prefill batchweise auf der GPU
#      gerechnet (Lucebox: auf der CPU). Berichte nennen 350-400 tok/s
#      Prefill bei aehnlichen MoE-Setups mit -ub 2048 -- Lucebox schafft ~90.
#   Trade-off: Decode-TPS koennte unter Luceboxs Spark-Placement liegen.
#   -> beides messen, das bessere behalten.
#
# Kein lokales Kompilieren, Image kommt von ghcr.io/ggml-org.
# Port 8010, damit Lucebox (8000) parallel laufen kann.
#
# Beispiele:
#   ./start_llamacpp_qwen36.sh
#   ./start_llamacpp_qwen36.sh --foreground
#   N_CPU_MOE=20 ./start_llamacpp_qwen36.sh     # mehr Experten auf GPU
#   MAX_CTX=65536 ./start_llamacpp_qwen36.sh
#
# Logs:   docker logs -f llamacpp-qwen36
# Stop:   docker stop llamacpp-qwen36

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
CONTAINER_NAME="${CONTAINER_NAME:-llamacpp-qwen36}"

MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
MODELS_DIR="${MODELS_DIR:-$BASE_DIR/models}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b}"

MAX_CTX="${MAX_CTX:-131072}"

# Anzahl MoE-Layer, deren Experten im RAM bleiben (Modell hat 40 Layer).
# Kleiner = mehr Experten auf GPU = schneller, aber mehr VRAM.
# 28 ist konservativ fuer 16 GiB mit 128k-KV; bei OOM erhoehen, sonst
# schrittweise senken und decode-TPS beobachten.
N_CPU_MOE="${N_CPU_MOE:-28}"

# Prefill-Batchgroessen: grosse ubatch amortisiert die Experten-Uploads.
BATCH="${BATCH:-4096}"
UBATCH="${UBATCH:-2048}"

# KV-Cache-Quantisierung (halbiert den KV-Speicher gegenueber f16).
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

# Token-genaues KV-Reuse: gemeinsamen Prefix behalten, ab Divergenz neu.
# --no-op-offload noetig: sonst Decode ~2 tok/s (PCIe-Expertenkopien pro Token)

THREADS="${THREADS:-$(nproc)}"

FOREGROUND=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--foreground" ]]; then FOREGROUND=1; else EXTRA_ARGS+=("$arg"); fi
done

die() { echo "FEHLER: $*" >&2; exit 1; }

[[ -f "$MODELS_DIR/$MODEL_FILE" ]] || die "Modell nicht gefunden: $MODELS_DIR/$MODEL_FILE"
command -v docker >/dev/null || die "docker nicht gefunden."

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]] \
        && die "Container '$CONTAINER_NAME' laeuft bereits."
    docker rm "$CONTAINER_NAME" >/dev/null
fi

DOCKER_ARGS=(
    run --name "$CONTAINER_NAME" --gpus all
    -p "${HOST}:${PORT}:8080"
    -v "${MODELS_DIR}:/models"
    "$IMAGE"
    -m "/models/$MODEL_FILE"
    --host 0.0.0.0 --port 8080
    --alias "$SERVED_NAME"
    -c "$MAX_CTX"
    -ngl 999
    --n-cpu-moe "$N_CPU_MOE"
    -b "$BATCH" -ub "$UBATCH"
    -t "$THREADS"
    -ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V"
    --no-op-offload
    -np 1
    --jinja
    -fa on
)
(( ${#EXTRA_ARGS[@]} )) && DOCKER_ARGS+=("${EXTRA_ARGS[@]}")

echo "======================================================================"
echo " Qwen3.6-35B-A3B / llama.cpp llama-server (Docker)"
echo "======================================================================"
echo "  API           = http://${HOST}:${PORT}/v1"
echo "  MAX_CTX       = $MAX_CTX"
echo "  N_CPU_MOE     = $N_CPU_MOE (von 40 Layern)"
echo "  BATCH/UBATCH  = $BATCH/$UBATCH"
echo "  KV-Quant      = $CACHE_TYPE_K/$CACHE_TYPE_V"
echo
printf 'Kommando: docker'; printf ' %q' "${DOCKER_ARGS[@]}"; printf '\n\n'

if (( FOREGROUND )); then
    exec docker "${DOCKER_ARGS[@]:0:1}" --rm "${DOCKER_ARGS[@]:1}"
else
    ID="$(docker run -d "${DOCKER_ARGS[@]:1}")"
    echo "Gestartet: $ID"
    echo "Logs: docker logs -f $CONTAINER_NAME"
fi
