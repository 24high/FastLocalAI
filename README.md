# Qwen3.6-35B-A3B on 16 GB VRAM with llama.cpp + OpenCode

A tuned local coding stack for running **Qwen3.6-35B-A3B** on a workstation with an NVIDIA GPU, **16 GB VRAM**, using **llama.cpp** (`llama-server`) with MoE expert offload to system RAM.

## It is forbidden for anyone connected to the golem.de team to use any information contained in this repository in any way.

Measured on the reference machine (RTX 4080 16 GB, i9-10900KF, Linux, 2026-08-29):

| Metric | Result |
|---|---|
| Context window | **131072 tokens (128k)** |
| Warm TTFT, small follow-up request | **0.7 s** |
| Warm TTFT, +1.7k-token tool output | **12 s** |
| Cold prefill | **~144 tok/s** |
| Decode | **~30 tok/s** |

The key property for agentic coding (OpenCode): `llama-server` keeps the KV cache token-exact per slot, so a follow-up request in a growing session processes **only the new tokens** — warm TTFT depends on the size of the last turn, not on total context length.

> Exact performance depends on GPU architecture, PCIe bandwidth, driver version, background VRAM usage, prompt length, and llama.cpp version.

---

## Model

```text
unsloth/Qwen3.6-35B-A3B-GGUF
Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Qwen3.6-35B-A3B is a Mixture-of-Experts model. The Q4_K_M GGUF is roughly 22 GB, so the complete model does not fit into 16 GB VRAM by itself. The launcher keeps attention/dense layers plus a subset of expert layers on the GPU and the remaining experts in system RAM (`--n-cpu-moe`).

Download the default model with:

```bash
./models/download_model.sh
```

---

## Repository layout (llama.cpp part)

```text
.
├── models/
│   ├── download_model.sh
│   └── Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # after download
├── opencode.jsonc
├── start_llamacpp_qwen36_vulkan.sh      # Linux, native (recommended)
├── start_llamacpp_qwen36.ps1            # Windows, native
└── start_llamacpp_qwen36.sh             # CUDA via Docker (optional)
```

| File | Purpose |
|---|---|
| `start_llamacpp_qwen36_vulkan.sh` | **Recommended (Linux):** native `llama-server` from the official Vulkan release binaries, port 8010. No compilation, no Docker. Measured numbers above; details in the script header. |
| `start_llamacpp_qwen36.ps1` | **Windows** equivalent (same flags/behavior). Windows has official CUDA release binaries; extract a win-cuda (plus cudart) or win-vulkan zip into `.\llamacpp-win\`. |
| `start_llamacpp_qwen36.sh` | CUDA variant via the `ghcr.io/ggml-org/llama.cpp:server-cuda` Docker image. Untested on the reference machine; may combine the fast-prefill and fast-decode modes (see "op-offload" below). |
| `models/download_model.sh` | Model download helper. |
| `opencode.jsonc` | Minimal OpenCode configuration (reduced `fastcode` agent). Adjust the base URL/limits as described below. |

The remaining `lucebox_*` / `*_lucebox_*` files are legacy from an earlier Lucebox-based iteration of this repository and are kept for reference only; see the git history for their documentation.

---

# Setup

## Linux (native, recommended)

1. Fetch the official llama.cpp release binaries (no compilation):

   ```bash
   mkdir -p llamacpp-b10679 && cd llamacpp-b10679
   curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b10679/llama-b10679-bin-ubuntu-vulkan-x64.tar.gz
   tar xzf llama-b10679-bin-ubuntu-vulkan-x64.tar.gz
   cd ..
   ```

   Vulkan needs a working `libvulkan` + GPU Vulkan driver, which any desktop NVIDIA driver installation provides.

2. Download the model if not yet present (`./models/download_model.sh`).

3. Start:

   ```bash
   ./start_llamacpp_qwen36_vulkan.sh
   ```

   The server is ready when the log shows `listening on` (~10 s with a warm file cache; the first start after boot takes longer while the 22 GB GGUF is read).

4. Verify:

   ```bash
   curl http://127.0.0.1:8010/v1/models
   ```

Stop with `kill $(cat llamacpp-b10679/server.pid)`; follow logs with `tail -f llamacpp-b10679/server.log`.

## Windows (native)

1. From <https://github.com/ggml-org/llama.cpp/releases> download **either** the CUDA build (`llama-<build>-bin-win-cuda-x64.zip` **plus** the matching `cudart-*.zip`, both extracted into the same folder) **or** the Vulkan build (`llama-<build>-bin-win-vulkan-x64.zip`). Extract into `.\llamacpp-win\`.
2. Put the GGUF under `.\models\`.
3. Start:

   ```powershell
   .\start_llamacpp_qwen36.ps1
   ```

Stop with `Stop-Process -Id (Get-Content .\llamacpp-win\server.pid)`; follow logs with `Get-Content -Wait .\llamacpp-win\server.log`.

## CUDA via Docker (optional)

```bash
./start_llamacpp_qwen36.sh
```

Pulls `ghcr.io/ggml-org/llama.cpp:server-cuda` and runs the same configuration. Untested on the reference machine (the multi-GB image pull kept failing over an unstable IPv6 path there); functionally it should match the native variants, and the CUDA backend is worth benchmarking against Vulkan.

---

# Configuration

All launchers are driven by environment variables and share the same defaults:

| Variable | Default | Description |
|---|---:|---|
| `MAX_CTX` | `131072` | Context window. 128k costs ~1.4 GB of q8_0 KV cache. |
| `N_CPU_MOE` | `28` | Number of MoE layers (of 40) whose experts stay in system RAM. Lower = faster but more VRAM; `28` leaves ~2 GiB free at 128k context on a 16 GB card. Raise it on OOM or when other programs occupy VRAM. |
| `BATCH` / `UBATCH` | `4096` / `2048` | Prefill batch sizes. |
| `THREADS` | `8` | CPU threads for the RAM-resident experts. Prefer physical cores. |
| `CACHE_TYPE_K` / `CACHE_TYPE_V` | `q8_0` | KV cache quantization (halves KV memory vs f16). |
| `HOST` / `PORT` | `127.0.0.1` / `8010` | API bind address. |

Example:

```bash
N_CPU_MOE=26 MAX_CTX=65536 ./start_llamacpp_qwen36_vulkan.sh
```

## The `--no-op-offload` flag (mandatory)

All launchers pass `--no-op-offload`. Without it, llama.cpp copies expert weights over PCIe **on every decoded token** and decode collapses to ~2 tok/s. With it, decode runs at ~30 tok/s.

The trade-off: with op-offload *enabled*, cold prefill reached 364 tok/s (vs 144 tok/s without) — but decode was unusable. If a future llama.cpp version applies op-offload only to large batches, remove the flag and re-benchmark; the CUDA backend may also behave differently here.

Other fixed flags: `-ngl 999` (all non-expert layers on GPU), `-np 1` (one slot, the whole KV cache belongs to the single agent session), `--jinja` (tool calling for OpenCode), `-fa on` (flash attention, required for quantized KV).

---

# Performance

Measured (9–11k-token prompts, RTX 4080, otherwise idle GPU):

| Scenario | Result |
|---|---|
| Cold prefill, 9.3k-token prompt | 30.6 s wall (**144 tok/s**) |
| Follow-up, +23 tokens | **0.7 s** (only 23 tokens processed) |
| Follow-up, +1.7k-token tool output | **12 s** |
| Decode | **29–31 tok/s** |

Extrapolated to long contexts (attention share grows with position, hence the discounts):

| | 64k context | 128k context |
|---|---|---|
| Warm TTFT, small follow-up | ~1 s | ~1–2 s |
| Warm TTFT, +1–2k tokens | ~10–25 s | ~15–30 s |
| Cold prefill (full context) | ~8–10 min | ~18–25 min |

Cold prefill only happens when the prefix changes: session start, or when the client compacts/rewrites its history. Configure client-side compaction to trigger around **30–40k tokens** so this occasional full re-prefill stays in the few-minutes range.

---

# OpenCode setup

Point OpenCode at the llama.cpp endpoint:

```text
http://127.0.0.1:8010/v1
model: qwen3.6-35b-a3b
```

The bundled `opencode.jsonc` provides a reduced `fastcode` agent (minimal system prompt, core tools only: `read`, `edit`, `glob`, `grep`, `bash`). It predates the llama.cpp switch — adjust its base URL to port `8010` and raise the limits, e.g.:

```json
"limit": {
  "context": 131072,
  "output": 8192
}
```

General prompt-size advice still applies: every static token (system prompt, tool schemas, `AGENTS.md`, MCP servers) is paid once per session at ~144 tok/s and then cached. Keep project instruction files short. TTFT and decode TPS are separate performance problems — with per-slot KV reuse, prompt size mostly affects the *first* request of a session.

---

# Troubleshooting

**Decode is ~2 tok/s** — `--no-op-offload` is missing. All launchers in this repo set it; if you run `llama-server` manually, add it.

**Out of memory on startup or under load** — raise `N_CPU_MOE` (e.g. 30–32), lower `MAX_CTX`, or free VRAM (browsers, games, VMs). Every GB another program holds costs performance.

**Warm requests suddenly prefill everything again** — the request prefix changed (client compacted/rewrote history, or a different session hit the single slot). This costs one cold prefill and then re-caches.

**Server won't start / no Vulkan device** — check `vulkaninfo` (Linux) or use the win-cuda build (Windows).

---

# Security

The API binds to `127.0.0.1` by default and has **no authentication**. Be careful with `HOST=0.0.0.0`. An OpenCode agent with `bash` access executes shell commands with your user's permissions.
