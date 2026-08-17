#!/usr/bin/env bash
#
# Installiert den PREBUILT Lucebox Docker-Pfad fuer:
#
#   Qwen3.6-35B-A3B UD-Q4_K_M
#   NVIDIA GPU
#   16 GiB VRAM / 64 GiB RAM
#
# Es wird Lucebox NICHT lokal kompiliert.
#
# Installiert/prueft:
#   - Docker Engine
#   - NVIDIA Container Toolkit
#   - ghcr.io/luce-org/lucebox-hub:cuda12
#   - Hugging-Face-Downloadtool in kleiner lokaler Venv
#   - Qwen3.6-35B-A3B UD-Q4_K_M GGUF
#
# Voraussetzungen:
#   - Linux
#   - NVIDIA-Treiber auf dem Host
#   - Ubuntu oder Debian fuer die automatische Docker/Toolkit-Installation
#
# Beispiele:
#
#   ./install_lucebox_docker_qwen36.sh
#
#   INSTALL_DOCKER=0 ./install_lucebox_docker_qwen36.sh
#
#   INSTALL_NVIDIA_TOOLKIT=0 ./install_lucebox_docker_qwen36.sh
#
#   DOWNLOAD_MODEL=0 ./install_lucebox_docker_qwen36.sh
#
#   PULL_IMAGE=0 ./install_lucebox_docker_qwen36.sh
#
# Hinweis:
# Docker-Gruppenmitgliedschaft wird NICHT automatisch veraendert.
# Falls dein Benutzer Docker nicht direkt nutzen darf, verwenden die Scripts
# automatisch sudo, sofern sudo verfuegbar ist.


BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-ghcr.io/luce-org/lucebox-hub:cuda12}"

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"

MODELS_DIR="${MODELS_DIR:-$BASE_DIR/models}"
MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

TOOLS_VENV="${TOOLS_VENV:-$BASE_DIR/.venv-lucebox-tools}"

INSTALL_DOCKER="${INSTALL_DOCKER:-1}"
INSTALL_NVIDIA_TOOLKIT="${INSTALL_NVIDIA_TOOLKIT:-1}"
PULL_IMAGE="${PULL_IMAGE:-1}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"

die() {
    echo "FEHLER: $*" >&2
    exit 1
}

have() {
    command -v "$1" >/dev/null 2>&1
}

run_root() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif have sudo; then
        sudo "$@"
    else
        die "Root-Rechte erforderlich, aber sudo wurde nicht gefunden: $*"
    fi
}

detect_os() {
    [[ -r /etc/os-release ]] || die "/etc/os-release nicht gefunden."
    # shellcheck disable=SC1091
    . /etc/os-release

    OS_ID="${ID:-}"
    OS_CODENAME="${VERSION_CODENAME:-}"

    if [[ "$OS_ID" == "ubuntu" ]]; then
        OS_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
    fi

}

docker_cmd_init() {
    if docker info >/dev/null 2>&1; then
        DOCKER=(docker)
        return
    fi

    if have sudo && sudo -n docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
        return
    fi

    # Bei gerade installierter Docker-Engine ist sudo eventuell mit Passwort
    # erforderlich. Testen, ohne -n.
    if have sudo && sudo docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
        return
    fi

    die "Docker ist installiert, aber der Docker-Daemon ist nicht erreichbar."
}

install_docker_engine() {
    if have docker; then
        echo "Docker bereits installiert:"
        docker --version || true
        return
    fi

    [[ "$INSTALL_DOCKER" == "1" ]] \
        || die "Docker fehlt und INSTALL_DOCKER=0 wurde gesetzt."

    detect_os

    echo
    echo "Installiere Docker Engine aus dem offiziellen Docker-Repository ..."

    run_root apt-get update
    run_root apt-get install -y ca-certificates curl

    run_root install -m 0755 -d /etc/apt/keyrings

    run_root curl -fsSL \
        "https://download.docker.com/linux/${OS_ID}/gpg" \
        -o /etc/apt/keyrings/docker.asc

    run_root chmod a+r /etc/apt/keyrings/docker.asc

    ARCH="$(dpkg --print-architecture)"

    TMP_SOURCE="$(mktemp)"
    trap 'rm -f "$TMP_SOURCE"' RETURN

    cat >"$TMP_SOURCE" <<EOF
Types: deb
URIs: https://download.docker.com/linux/${OS_ID}
Suites: ${OS_CODENAME}
Components: stable
Architectures: ${ARCH}
Signed-By: /etc/apt/keyrings/docker.asc
EOF

    run_root cp "$TMP_SOURCE" /etc/apt/sources.list.d/docker.sources
    rm -f "$TMP_SOURCE"
    trap - RETURN

    run_root apt-get update
    run_root apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin

    run_root systemctl enable --now docker
}

install_nvidia_toolkit() {
    if have nvidia-ctk; then
        echo "NVIDIA Container Toolkit bereits installiert:"
        nvidia-ctk --version || true
    else
        [[ "$INSTALL_NVIDIA_TOOLKIT" == "1" ]] \
            || die "nvidia-ctk fehlt und INSTALL_NVIDIA_TOOLKIT=0 wurde gesetzt."

        detect_os

        echo
        echo "Installiere NVIDIA Container Toolkit ..."

        run_root apt-get update
        run_root apt-get install -y curl gpg

        TMP_KEY="$(mktemp)"
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor >"$TMP_KEY"

        run_root install -m 0644 \
            "$TMP_KEY" \
            /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        rm -f "$TMP_KEY"

        TMP_LIST="$(mktemp)"
        curl -fsSL \
            https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed \
                's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            >"$TMP_LIST"

        run_root install -m 0644 \
            "$TMP_LIST" \
            /etc/apt/sources.list.d/nvidia-container-toolkit.list

        rm -f "$TMP_LIST"

        run_root apt-get update
        run_root apt-get install -y nvidia-container-toolkit
    fi

    echo
    echo "Konfiguriere NVIDIA Runtime fuer Docker ..."

    run_root nvidia-ctk runtime configure --runtime=docker
    run_root systemctl restart docker
}

install_hf_tool() {
    mkdir -p "$MODELS_DIR"

    if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
        have python3 || die "python3 nicht gefunden."

        echo
        echo "Erzeuge kleine Download-Venv:"
        echo "  $TOOLS_VENV"

        python3 -m venv "$TOOLS_VENV"
    fi

    "$TOOLS_VENV/bin/python" -m pip install -q --upgrade pip
    "$TOOLS_VENV/bin/python" -m pip install -q --upgrade huggingface_hub

    if [[ -x "$TOOLS_VENV/bin/hf" ]]; then
        HF=("$TOOLS_VENV/bin/hf")
    elif [[ -x "$TOOLS_VENV/bin/huggingface-cli" ]]; then
        HF=("$TOOLS_VENV/bin/huggingface-cli")
    else
        die "Hugging-Face CLI konnte nicht installiert werden."
    fi
}

download_model() {
    [[ "$DOWNLOAD_MODEL" == "1" ]] || {
        echo "Modelldownload uebersprungen (DOWNLOAD_MODEL=0)."
        return
    }

    if [[ -f "$MODEL_PATH" ]]; then
        echo
        echo "Modell bereits vorhanden:"
        echo "  $MODEL_PATH"
        return
    fi

    echo
    echo "Lade Qwen3.6-35B-A3B herunter ..."
    echo "  Repo:  $MODEL_REPO"
    echo "  Datei: $MODEL_FILE"
    echo

    "${HF[@]}" download \
        "$MODEL_REPO" \
        "$MODEL_FILE" \
        --local-dir "$MODELS_DIR"

    [[ -f "$MODEL_PATH" ]] \
        || die "Download beendet, aber Modell nicht gefunden: $MODEL_PATH"
}

verify_gpu_container() {
    echo
    echo "Pruefe NVIDIA-GPU-Zugriff aus Docker ..."

    # Das bereits benoetigte Lucebox-Image enthaelt nvidia-smi ueber die
    # injizierten Host-Treiber. Der EntryPoint reicht unbekannte Subcommands
    # direkt durch.
    "${DOCKER[@]}" run --rm --gpus all \
        "$IMAGE" \
        nvidia-smi \
        --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader
}

# ===========================================================================
# Start
# ===========================================================================

echo
echo "======================================================================"
echo " Lucebox Docker Installation - Qwen3.6-35B-A3B"
echo "======================================================================"
echo

[[ "$(uname -s)" == "Linux" ]] || die "Dieses Script ist fuer Linux vorgesehen."

have nvidia-smi || die "nvidia-smi nicht gefunden. NVIDIA-Treiber zuerst installieren."

echo "Host-GPU:"
nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader || true
echo

install_docker_engine
install_nvidia_toolkit

docker_cmd_init

if [[ "$PULL_IMAGE" == "1" ]]; then
    echo
    echo "Ziehe vorgebautes Lucebox-Image:"
    echo "  $IMAGE"
    "${DOCKER[@]}" pull "$IMAGE"
else
    echo "Docker-Image-Pull uebersprungen (PULL_IMAGE=0)."
fi

verify_gpu_container

install_hf_tool
download_model

echo
echo "======================================================================"
echo " Installation abgeschlossen"
echo "======================================================================"
echo
echo "Lucebox-Image:"
echo "  $IMAGE"
echo
echo "Modell:"
echo "  $MODEL_PATH"
echo
echo "Kein lokaler Lucebox-/CUDA-Build wurde durchgefuehrt."
echo
echo "Start:"
echo "  chmod +x \"$BASE_DIR/start_lucebox_docker_qwen36.sh\""
echo "  \"$BASE_DIR/start_lucebox_docker_qwen36.sh\" --foreground"
echo
