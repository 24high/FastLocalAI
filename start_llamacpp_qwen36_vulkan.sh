#!/usr/bin/env bash
#
# Startet Qwen3.6-35B-A3B UD-Q4_K_M mit llama.cpp (Vulkan-Build, NATIV, kein Docker).
#
# Gemessen auf RTX 4080 16GB + i9-10900KF (2026-08-29):
#   Cold-Prefill : ~144 tok/s   (Lucebox: ~90)
#   Warm-TTFT    : 0.7 s bei +23 Tokens, 12 s bei +1.7k Tokens
#                  -> es werden NUR die neuen Tokens verarbeitet,
#                     kein Sticky-Snapshot-Problem wie bei Lucebox
#   Decode       : ~30 tok/s    (Lucebox: ~27)
#
# WICHTIG: --no-op-offload ist Pflicht. Ohne das Flag kopiert llama.cpp
# beim Decode Expert-Weights pro Token ueber PCIe zur GPU -> 2 tok/s.
# (Trade-off: MIT op-offload waere der Prefill ~360 tok/s, aber Decode
# unbrauchbar. Wer beides will, braucht den CUDA-Build/Docker und kann
# dort weiter tunen.)
#
# Binaries: /home/dh/Downloads/llamacpp-b10679/llama-b10679 (GitHub-Release b10679)
#
# Beispiele:
#   ./start_llamacpp_qwen36_vulkan.sh
#   N_CPU_MOE=26 ./start_llamacpp_qwen36_vulkan.sh   # mehr Experten auf GPU (VRAM!)
#   MAX_CTX=65536 ./start_llamacpp_qwen36_vulkan.sh
#
# Logs:   tail -f "$LOG"    Stop: kill $(cat <BASISDIR>/server.pid)

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${BIN_DIR:-$BASE_DIR/llamacpp-b10679/llama-b10679}"

MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
MODEL_PATH="${MODEL_PATH:-$BASE_DIR/models/$MODEL_FILE}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b}"

MAX_CTX="${MAX_CTX:-131072}"

# MoE-Layer, deren Experten im RAM bleiben (Modell hat 40 Layer).
# 28 laesst bei 128k-Kontext ~2 GiB VRAM frei. Kleiner = schneller,
# aber OOM-Gefahr, wenn andere Programme VRAM belegen.
N_CPU_MOE="${N_CPU_MOE:-28}"

BATCH="${BATCH:-4096}"
UBATCH="${UBATCH:-2048}"
THREADS="${THREADS:-8}"          # physische Kerne bevorzugen (i9-10900KF: 10)
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

LOG="${LOG:-$BASE_DIR/llamacpp-b10679/server.log}"
PIDFILE="$BASE_DIR/llamacpp-b10679/server.pid"

die() { echo "FEHLER: $*" >&2; exit 1; }

[[ -x "$BIN_DIR/llama-server" ]] || die "llama-server nicht gefunden: $BIN_DIR
Binaries holen mit:
  mkdir -p \"$BASE_DIR/llamacpp-b10679\" && cd \"$BASE_DIR/llamacpp-b10679\"
  curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b10679/llama-b10679-bin-ubuntu-vulkan-x64.tar.gz
  tar xzf llama-b10679-bin-ubuntu-vulkan-x64.tar.gz"
[[ -f "$MODEL_PATH" ]] || die "Modell nicht gefunden: $MODEL_PATH"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    die "llama-server laeuft bereits (PID $(cat "$PIDFILE"))."
fi

echo "======================================================================"
echo " Qwen3.6-35B-A3B / llama.cpp Vulkan (nativ)"
echo "======================================================================"
echo "  API        = http://${HOST}:${PORT}/v1"
echo "  MAX_CTX    = $MAX_CTX   N_CPU_MOE = $N_CPU_MOE   THREADS = $THREADS"
echo "  BATCH/UB   = $BATCH/$UBATCH   KV = $CACHE_TYPE_K/$CACHE_TYPE_V"
echo

LD_LIBRARY_PATH="$BIN_DIR" nohup "$BIN_DIR/llama-server" \
    -m "$MODEL_PATH" \
    --alias "$SERVED_NAME" --host "$HOST" --port "$PORT" \
    -c "$MAX_CTX" -ngl 999 --n-cpu-moe "$N_CPU_MOE" \
    -b "$BATCH" -ub "$UBATCH" -t "$THREADS" \
    -ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V" \
    -np 1 --jinja -fa on --no-op-offload \
    > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "Gestartet (PID $(cat "$PIDFILE")). Log: $LOG"
echo "Bereit, sobald im Log 'listening on' erscheint (~10 s bei warmem Page-Cache)."
