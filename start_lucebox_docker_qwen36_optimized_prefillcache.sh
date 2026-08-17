#!/usr/bin/env bash
#
# Startet Qwen3.6-35B-A3B UD-Q4_K_M mit dem PREBUILT Lucebox CUDA-Dockerimage.
#
# Kein lokales Kompilieren.
#
# Ziel:
#   - 16 GiB VRAM
#   - 64 GiB RAM
#   - hohe Single-Request Decode-TPS
#
# Aktiv:
#   - Luce Spark (MoE Expert Hot/Cold Placement + GPU Expert Cache)
#   - KVFlash
#   - qk KVFlash Policy fuer Qwen3.5/3.6
#   - GPU Sampler
#   - OpenAI-kompatible API
#
# Stand des Qwen3.6-35B-A3B Lucebox-Pfads:
#   - Spark: JA
#   - KVFlash: JA
#   - DFlash Speculative Decode: wird fuer diesen MoE-Pfad NICHT erzwungen
#   - PFlash Speculative Prefill: wird fuer diesen MoE-Pfad NICHT erzwungen
#
# Beispiele:
#
#   ./start_lucebox_docker_qwen36.sh
#   ./start_lucebox_docker_qwen36.sh --foreground
#
#   SPARK_VRAM_GIB=13.5 ./start_lucebox_docker_qwen36_optimized.sh
#   MAX_CTX=16384 ./start_lucebox_docker_qwen36_optimized.sh
#   PREFILL_CHUNK=1024 ./start_lucebox_docker_qwen36_optimized.sh
#   PREFILL_CHUNK= ./start_lucebox_docker_qwen36_optimized_prefillcache.sh
#   PREFIX_CACHE_SLOTS=64 ./start_lucebox_docker_qwen36_optimized.sh
#   PREFILL_CACHE_SLOTS=8 ./start_lucebox_docker_qwen36_optimized_prefillcache.sh
#   KVFLASH_POLICY=lru ./start_lucebox_docker_qwen36_optimized.sh
#
# Container-Logs:
#   docker logs -f lucebox-qwen36
#
# Stop:
#   docker stop lucebox-qwen36
#
# Fuer maximale TPS beim Client moeglichst temperature=0 verwenden.

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-ghcr.io/luce-org/lucebox-hub:cuda12}"

MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
MODELS_DIR="${MODELS_DIR:-$BASE_DIR/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/$MODEL_FILE}"

CONTAINER_MODELS_DIR="/opt/lucebox-hub/server/models"
CONTAINER_MODEL_PATH="$CONTAINER_MODELS_DIR/$MODEL_FILE"
SERVER_BIN="/opt/lucebox-hub/server/build/dflash_server"

CONTAINER_NAME="${CONTAINER_NAME:-lucebox-qwen36}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"

SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
TARGET_DEVICE="${TARGET_DEVICE:-cuda:0}"

# WICHTIG FUER TTFT:
# Leer lassen = Lucebox auto-fit. Ein festes, stark ueberdimensioniertes
# --max-ctx kann den Prefill deutlich bremsen.
#
# Nur setzen, wenn du bewusst einen festen Context erzwingen willst:
#   MAX_CTX=16384 ./start_lucebox_docker_qwen36_optimized.sh
MAX_CTX="${MAX_CTX:-}"

# 4096 ist fuer Coding meist ausreichend und vermeidet ein unnötig grosses
# Reply-Budget bei Auto-Fit. OpenCode kann pro Request weiterhin einen eigenen
# max_tokens-Wert schicken.
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-4096}"

# Prefix Cache: DER wichtigste Hebel fuer OpenCode, weil Systemprompt,
# Tool-Schemas, AGENTS.md und der bisherige Dialog bei Folgerequests meist
# einen grossen identischen Prefix bilden.
PREFIX_CACHE_SLOTS="${PREFIX_CACHE_SLOTS:-32}"

# Separater Lucebox Prefill-Cache.
# Default im Lucebox-Container ist normalerweise 0; fuer OpenCode testen wir
# bewusst 8 Slots. A/B-Test:
#   PREFILL_CACHE_SLOTS=0  -> aus
#   PREFILL_CACHE_SLOTS=8  -> Default hier
#   PREFILL_CACHE_SLOTS=16 -> groesserer Test
PREFILL_CACHE_SLOTS="${PREFILL_CACHE_SLOTS:-8}"

# Prefix Cache auch ueber Container-Neustarts erhalten.
PERSIST_PREFIX_CACHE="${PERSIST_PREFIX_CACHE:-1}"
PREFIX_CACHE_HOST_DIR="${PREFIX_CACHE_HOST_DIR:-$BASE_DIR/lucebox-prefix-cache}"
PREFIX_CACHE_CONTAINER_DIR="/opt/lucebox-hub/cache"

# Prefill-Ubatch nicht erzwingen; der Lucebox-Backend-Default ist der
# Ausgangspunkt. Fuer A/B-Tests z.B. PREFILL_CHUNK=1024 oder 2048 setzen.
PREFILL_CHUNK="${PREFILL_CHUNK:-}"

# PFlash ist fuer OpenCode standardmaessig AUS:
# Requests mit Tools werden von Lucebox bewusst nicht komprimiert.
# Fuer lange, tool-freie Prompts kann man es optional aktivieren.
ENABLE_PFLASH="${ENABLE_PFLASH:-0}"
PFLASH_THRESHOLD="${PFLASH_THRESHOLD:-16384}"
PFLASH_KEEP_RATIO="${PFLASH_KEEP_RATIO:-0.30}"
PFLASH_DRAFTER_FILE="${PFLASH_DRAFTER_FILE:-Qwen3-0.6B-BF16.gguf}"
PFLASH_DRAFTER_PATH="${PFLASH_DRAFTER_PATH:-$MODELS_DIR/$PFLASH_DRAFTER_FILE}"
PFLASH_DRAFTER_CONTAINER_PATH="$CONTAINER_MODELS_DIR/$PFLASH_DRAFTER_FILE"

ENABLE_SPARK="${ENABLE_SPARK:-1}"
SPARK_VRAM_GIB="${SPARK_VRAM_GIB:-}"
SPARK_VRAM_RESERVE_MIB="${SPARK_VRAM_RESERVE_MIB:-768}"

ENABLE_KVFLASH="${ENABLE_KVFLASH:-1}"
KVFLASH_SIZE="${KVFLASH_SIZE:-auto}"
KVFLASH_POLICY="${KVFLASH_POLICY:-qk}"

# GPU Sampling ist im CUDA-Server standardmaessig aktiv. Explizit setzen.
DFLASH_GPU_SAMPLE="${DFLASH_GPU_SAMPLE:-1}"
DFLASH_GPU_DRAFT_TOPK="${DFLASH_GPU_DRAFT_TOPK:-1}"
DFLASH_GPU_VERIFY_ARGMAX="${DFLASH_GPU_VERIFY_ARGMAX:-1}"

# Altes Hugging-Face-Modell wie bisher beim Start gezielt entfernen.
DELETE_OLD_MODEL="${DELETE_OLD_MODEL:-1}"
OLD_MODEL_ID="${OLD_MODEL_ID:-cyankiwi/Qwen3-Coder-Next-AWQ-4bit}"

# Optional vor jedem Start das neueste vorgebaute Image holen.
PULL_IMAGE_ON_START="${PULL_IMAGE_ON_START:-0}"

FOREGROUND=0
EXTRA_SERVER_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--foreground" ]]; then
        FOREGROUND=1
    else
        EXTRA_SERVER_ARGS+=("$arg")
    fi
done

die() {
    echo "FEHLER: $*" >&2
    exit 1
}

have() {
    command -v "$1" >/dev/null 2>&1
}

docker_cmd_init() {
    have docker || die "docker nicht gefunden. Erst install_lucebox_docker_qwen36.sh ausfuehren."

    if docker info >/dev/null 2>&1; then
        DOCKER=(docker)
        return
    fi

    if have sudo && sudo -n docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
        return
    fi

    if have sudo && sudo docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
        return
    fi

    die "Docker-Daemon ist nicht erreichbar."
}

delete_old_model() {
    [[ "$DELETE_OLD_MODEL" == "1" ]] || return 0

    local hf_hub_dir old_cache_name old_cache_dir

    if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
        hf_hub_dir="$HUGGINGFACE_HUB_CACHE"
    elif [[ -n "${HF_HOME:-}" ]]; then
        hf_hub_dir="$HF_HOME/hub"
    else
        hf_hub_dir="$HOME/.cache/huggingface/hub"
    fi

    old_cache_name="models--${OLD_MODEL_ID//\//--}"
    old_cache_dir="$hf_hub_dir/$old_cache_name"

    [[ -e "$old_cache_dir" ]] || return 0

    case "$old_cache_dir" in
        "$hf_hub_dir"/models--*) ;;
        *) die "Unsicherer Cache-Pfad; Loeschen abgebrochen: $old_cache_dir" ;;
    esac

    echo "Loesche altes Qwen3-Coder-Next aus HF-Cache:"
    echo "  $old_cache_dir"

    rm -rf -- "$old_cache_dir"
}

calculate_spark_vram() {
    local gpu_info gpu_name gpu_used gpu_free gpu_total

    gpu_info="$(
        nvidia-smi \
            -i "$CUDA_DEVICE" \
            --query-gpu=name,memory.used,memory.free,memory.total \
            --format=csv,noheader,nounits
    )"

    IFS=',' read -r gpu_name gpu_used gpu_free gpu_total <<<"$gpu_info"

    GPU_NAME="$(xargs <<<"$gpu_name")"
    GPU_USED_MIB="$(xargs <<<"$gpu_used")"
    GPU_FREE_MIB="$(xargs <<<"$gpu_free")"
    GPU_TOTAL_MIB="$(xargs <<<"$gpu_total")"

    if [[ -z "$SPARK_VRAM_GIB" ]]; then
        SPARK_VRAM_GIB="$(
            LC_ALL=C awk \
                -v free="$GPU_FREE_MIB" \
                -v total="$GPU_TOTAL_MIB" \
                -v reserve="$SPARK_VRAM_RESERVE_MIB" '
                BEGIN {
                    mib = free - reserve

                    # Nicht ueber 95 % der physischen Karte gehen.
                    max_mib = total * 0.95
                    if (mib > max_mib)
                        mib = max_mib

                    # Unter 8 GiB ergibt fuer dieses Modell keinen sinnvollen
                    # Spark-Start; wir lassen spaeter mit klarer Meldung abbrechen.
                    printf "%.1f", mib / 1024.0
                }
            '
        )"
    fi

    LC_ALL=C awk \
        -v cap="$SPARK_VRAM_GIB" \
        -v total="$GPU_TOTAL_MIB" '
        BEGIN {
            if (cap <= 0 || cap * 1024 > total + 1)
                exit 1
        }
    ' || die "Ungueltiges SPARK_VRAM_GIB=$SPARK_VRAM_GIB fuer ${GPU_TOTAL_MIB} MiB VRAM."

    LC_ALL=C awk \
        -v cap="$SPARK_VRAM_GIB" '
        BEGIN {
            if (cap < 8.0)
                exit 1
        }
    ' || die "Nur ${SPARK_VRAM_GIB} GiB Spark-Budget verfuegbar. Fuer Qwen3.6-35B-A3B ist das zu knapp."
}

cleanup_old_container() {
    if "${DOCKER[@]}" inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        RUNNING="$(
            "${DOCKER[@]}" inspect \
                -f '{{.State.Running}}' \
                "$CONTAINER_NAME" 2>/dev/null || echo false
        )"

        if [[ "$RUNNING" == "true" ]]; then
            die "Container '$CONTAINER_NAME' laeuft bereits."
        fi

        echo "Entferne alten gestoppten Container '$CONTAINER_NAME' ..."
        "${DOCKER[@]}" rm "$CONTAINER_NAME" >/dev/null
    fi
}

# ===========================================================================
# Checks
# ===========================================================================

[[ -f "$MODEL_PATH" ]] \
    || die "Modell nicht gefunden: $MODEL_PATH. Erst Installationsscript ausfuehren."

have nvidia-smi \
    || die "nvidia-smi nicht gefunden."

docker_cmd_init
delete_old_model
calculate_spark_vram
cleanup_old_container

if [[ "$PULL_IMAGE_ON_START" == "1" ]]; then
    echo "Aktualisiere Lucebox Docker-Image ..."
    "${DOCKER[@]}" pull "$IMAGE"
fi

AVAIL_GB="$(
    awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo
)"

if (( AVAIL_GB < 28 )); then
    echo "WARNUNG: nur ${AVAIL_GB} GiB Host-RAM frei." >&2
    echo "         Fuer Q4_K_M + kalte Experts sind >= 28 GiB frei empfehlenswert." >&2
fi

if [[ "$PERSIST_PREFIX_CACHE" == "1" ]]; then
    mkdir -p "$PREFIX_CACHE_HOST_DIR"
fi

if [[ "$ENABLE_PFLASH" == "1" && ! -f "$PFLASH_DRAFTER_PATH" ]]; then
    die "PFlash aktiviert, aber Drafter fehlt: $PFLASH_DRAFTER_PATH"
fi

# ===========================================================================
# Docker-Kommando
# ===========================================================================

DOCKER_ARGS=(
    run
    --name "$CONTAINER_NAME"
    --gpus all

    # Nur localhost auf dem Host exponieren.
    -p "${HOST}:${PORT}:${CONTAINER_PORT}"

    # Modellverzeichnis read/write mounten, damit Spark sein Placement-Profil
    # neben dem GGUF persistieren kann.
    -v "${MODELS_DIR}:${CONTAINER_MODELS_DIR}"

    -e "DFLASH_GPU_SAMPLE=$DFLASH_GPU_SAMPLE"
    -e "DFLASH_GPU_DRAFT_TOPK=$DFLASH_GPU_DRAFT_TOPK"
    -e "DFLASH_GPU_VERIFY_ARGMAX=$DFLASH_GPU_VERIFY_ARGMAX"

    "$IMAGE"

    # Das Image-Entrypoint reicht unbekannte Subcommands direkt durch.
    # Dadurch starten wir den nativen Server explizit mit GENAU unserem
    # 35B-A3B Target und muessen nicht auf Auto-Discovery vertrauen.
    "$SERVER_BIN"
    "$CONTAINER_MODEL_PATH"

    --host "0.0.0.0"
    --port "$CONTAINER_PORT"

    --target-device "$TARGET_DEVICE"

    --default-max-tokens "$DEFAULT_MAX_TOKENS"

    --model-name "$SERVED_NAME"

    --prefix-cache-slots "$PREFIX_CACHE_SLOTS"
    --prefill-cache-slots "$PREFILL_CACHE_SLOTS"
)

# Auto-fit Context: --max-ctx absichtlich nur setzen, wenn vom Benutzer
# explizit gewuenscht.
if [[ -n "$MAX_CTX" ]]; then
    DOCKER_ARGS+=(
        --max-ctx "$MAX_CTX"
    )
fi

if [[ "$PERSIST_PREFIX_CACHE" == "1" ]]; then
    # Volume muss VOR dem Image stehen. Wir fuegen es deshalb vor dem
    # Image-Element ein.
    for i in "${!DOCKER_ARGS[@]}"; do
        if [[ "${DOCKER_ARGS[$i]}" == "$IMAGE" ]]; then
            DOCKER_ARGS=(
                "${DOCKER_ARGS[@]:0:$i}"
                -v "${PREFIX_CACHE_HOST_DIR}:${PREFIX_CACHE_CONTAINER_DIR}"
                "${DOCKER_ARGS[@]:$i}"
            )
            break
        fi
    done

    DOCKER_ARGS+=(
        --kv-cache-dir "$PREFIX_CACHE_CONTAINER_DIR"
    )
fi

if [[ -n "$PREFILL_CHUNK" ]]; then
    DOCKER_ARGS+=(
        --chunk "$PREFILL_CHUNK"
    )
fi

if [[ "$ENABLE_PFLASH" == "1" ]]; then
    # Fuer code-lastige Prompts ist 0.30 absichtlich konservativ.
    # Tool-Requests werden von Lucebox ohnehin nicht komprimiert.
    DOCKER_ARGS+=(
        --prefill-compression auto
        --prefill-threshold "$PFLASH_THRESHOLD"
        --prefill-keep-ratio "$PFLASH_KEEP_RATIO"
        --prefill-drafter "$PFLASH_DRAFTER_CONTAINER_PATH"
    )
fi

if [[ "$ENABLE_SPARK" == "1" ]]; then
    DOCKER_ARGS+=(
        --spark
        --spark-vram "$SPARK_VRAM_GIB"
    )
fi

if [[ "$ENABLE_KVFLASH" == "1" ]]; then
    DOCKER_ARGS+=(
        --kvflash "$KVFLASH_SIZE"
        --kvflash-policy "$KVFLASH_POLICY"
    )
fi

if (( ${#EXTRA_SERVER_ARGS[@]} > 0 )); then
    DOCKER_ARGS+=("${EXTRA_SERVER_ARGS[@]}")
fi

# ===========================================================================
# Status
# ===========================================================================

echo
echo "======================================================================"
echo " Qwen3.6-35B-A3B / Lucebox Docker"
echo "======================================================================"
echo
echo "GPU:"
echo "  Name                  = $GPU_NAME"
echo "  VRAM total            = ${GPU_TOTAL_MIB} MiB"
echo "  VRAM benutzt          = ${GPU_USED_MIB} MiB"
echo "  VRAM frei             = ${GPU_FREE_MIB} MiB"
echo
echo "Lucebox:"
echo "  Image                 = $IMAGE"
echo "  Container             = $CONTAINER_NAME"
echo
echo "Modell:"
echo "  Host                  = $MODEL_PATH"
echo "  Container             = $CONTAINER_MODEL_PATH"
echo
echo "API:"
echo "  http://${HOST}:${PORT}/v1"
echo
echo "Kontext / Prefill:"
echo "  MAX_CTX               = ${MAX_CTX:-<auto-fit>}"
echo "  DEFAULT_MAX_TOKENS    = $DEFAULT_MAX_TOKENS"
echo "  PREFILL_CHUNK          = ${PREFILL_CHUNK:-<backend-default>}"
echo
echo "Prefix / Prefill Cache:"
echo "  PREFIX_CACHE_SLOTS    = $PREFIX_CACHE_SLOTS"
echo "  PREFILL_CACHE_SLOTS   = $PREFILL_CACHE_SLOTS"
echo "  PERSIST_PREFIX_CACHE  = $PERSIST_PREFIX_CACHE"
if [[ "$PERSIST_PREFIX_CACHE" == "1" ]]; then
    echo "  CACHE_DIR             = $PREFIX_CACHE_HOST_DIR"
fi
echo
echo "PFlash:"
echo "  ENABLE_PFLASH         = $ENABLE_PFLASH"
if [[ "$ENABLE_PFLASH" == "1" ]]; then
    echo "  THRESHOLD             = $PFLASH_THRESHOLD"
    echo "  KEEP_RATIO            = $PFLASH_KEEP_RATIO"
    echo "  DRAFTER               = $PFLASH_DRAFTER_PATH"
fi
echo
echo "Spark:"
echo "  ENABLE_SPARK          = $ENABLE_SPARK"
echo "  SPARK_VRAM_GIB        = $SPARK_VRAM_GIB"
echo "  Reserve vor Berechnung= ${SPARK_VRAM_RESERVE_MIB} MiB"
echo
echo "KVFlash:"
echo "  ENABLE_KVFLASH        = $ENABLE_KVFLASH"
echo "  KVFLASH_SIZE          = $KVFLASH_SIZE"
echo "  KVFLASH_POLICY        = $KVFLASH_POLICY"
echo
echo "Speculation:"
echo "  DFlash Spec Decode    = fuer 35B-A3B nicht erzwungen"
echo "  PFlash Spec Prefill   = fuer 35B-A3B nicht erzwungen"
echo
echo "Docker-Kommando:"
printf '  %q' "${DOCKER[@]}"
for arg in "${DOCKER_ARGS[@]}"; do
    printf ' %q' "$arg"
done
printf '\n\n'

# ===========================================================================
# Start
# ===========================================================================

if (( FOREGROUND )); then
    # Im Vordergrund Container nach Beenden automatisch entfernen.
    DOCKER_ARGS=(run --rm "${DOCKER_ARGS[@]:1}")
    exec "${DOCKER[@]}" "${DOCKER_ARGS[@]}"
else
    DOCKER_ARGS=(run -d "${DOCKER_ARGS[@]:1}")

    CONTAINER_ID="$("${DOCKER[@]}" "${DOCKER_ARGS[@]}")"

    echo "Lucebox gestartet."
    echo "  Container-ID: $CONTAINER_ID"
    echo
    echo "Logs:"
    echo "  ${DOCKER[*]} logs -f \"$CONTAINER_NAME\""
    echo
    echo "Stop:"
    echo "  ${DOCKER[*]} stop \"$CONTAINER_NAME\""
    echo
    echo "API-Test:"
    echo "  curl http://${HOST}:${PORT}/v1/models"
    echo
fi
