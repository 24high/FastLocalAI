# Qwen3.6-35B-A3B on 16 GB VRAM with Lucebox + OpenCode

A tuned local coding stack for running **Qwen3.6-35B-A3B** on a Linux workstation with an NVIDIA GPU, **16 GB VRAM**, and **64 GB system RAM**.


## Warning: The context window in this setup is really low. You can extend it to 32k at the cost of processing speed.

## Qwen3.8 IQ3 M using llama.cpp will most likely perform better both quality- and speedwise, yet allowing a 64k context window. https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ3_M.gguf

The project uses the prebuilt Lucebox CUDA container and the `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` quantization. The main goal is to make a relatively large MoE coding model practical on consumer hardware while keeping OpenCode interactive enough for agentic coding.

The benchmarked configuration focuses on:

- high single-request decode throughput,
- aggressive MoE expert offload through **Luce Spark**,
- **KVFlash** for long-context KV management,
- small prefill chunks for better TTFT,
- prefix reuse,
- no separate Lucebox prefill cache,
- an aggressively reduced OpenCode system/tool prompt,
- an OpenAI-compatible local API on `127.0.0.1:8000`.

> **Reference hardware used for the measurements in this README:** 64 GB host RAM and a 16 GB NVIDIA GPU. Exact performance depends heavily on GPU architecture, PCIe bandwidth, driver version, background VRAM usage, prompt length, and Lucebox version.

---

## Repository layout

```text
.
├── install_lucebox_docker_qwen36.sh
├── models/
│   ├── download_model.sh
│   └── Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # after download
├── opencode.jsonc
├── run_lucebox.sh
└── start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

### Files

| File | Purpose |
|---|---|
| `install_lucebox_docker_qwen36.sh` | Installs/checks Docker, NVIDIA Container Toolkit, pulls the prebuilt Lucebox image, creates a small Hugging Face download environment, and can download the model. |
| `models/download_model.sh` | Dedicated model download helper. Use it when the model is not already present under `models/`. |
| `opencode.jsonc` | Minimal OpenCode configuration for the local Lucebox endpoint and a reduced `fastcode` agent. |
| `run_lucebox.sh` | Convenience launcher for the benchmarked/tuned runtime profile. |
| `start_lucebox_docker_qwen36_optimized_prefillcache.sh` | Full configurable Lucebox/Docker launcher. All important runtime knobs are exposed through environment variables. |

---

# What this setup does

## Model

The default model is:

```text
unsloth/Qwen3.6-35B-A3B-GGUF
Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Qwen3.6-35B-A3B is a Mixture-of-Experts model. The Q4_K_M GGUF is roughly 22 GB, so the complete model does not fit into 16 GB VRAM by itself.

Instead of trying to place all weights on the GPU, this project uses **Luce Spark** to keep useful/hot MoE experts in VRAM and cold experts in system RAM.

---

## Luce Spark

Spark is enabled by default:

```text
ENABLE_SPARK=1
```

The launcher calculates the Spark VRAM budget dynamically when `SPARK_VRAM_GIB` is not explicitly set.

The calculation is approximately:

```text
current free VRAM
- SPARK_VRAM_RESERVE_MIB
capped at 95% of physical VRAM
```

Default reserve:

```text
SPARK_VRAM_RESERVE_MIB=768
```

The launcher refuses an automatically calculated Spark budget below 8 GiB because that is too small for this model/profile.

This is the main mechanism that makes the ~35B-total-parameter MoE model practical on a 16 GB GPU.

---

## KVFlash

KVFlash is enabled by default:

```text
ENABLE_KVFLASH=1
KVFLASH_SIZE=auto
KVFLASH_POLICY=qk
```

The purpose is to avoid keeping the entire long-context KV cache permanently resident in VRAM. This is especially useful on a 16 GB card where model expert cache and KV cache compete for the same memory.

The `qk` policy is the tuned default for this Qwen3.5/Qwen3.6 path.

---

## Prefix cache

The normal prefix cache is enabled:

```text
PREFIX_CACHE_SLOTS=32
```

OpenCode requests often share a large common prefix:

- system instructions,
- tool definitions,
- project rules,
- earlier conversation turns.

Reusing these prefixes can reduce repeated work. On this setup the gain is workload-dependent; it helps some coding requests but is not a replacement for reducing the prompt itself.

Persistent prefix-cache storage is enabled by default:

```text
PERSIST_PREFIX_CACHE=1
```

Host directory:

```text
./lucebox-prefix-cache
```

Container directory:

```text
/opt/lucebox-hub/cache
```

The launcher passes it to Lucebox with:

```text
--kv-cache-dir /opt/lucebox-hub/cache
```

---

## Separate prefill cache

The full launcher currently has a conservative/test default of:

```text
PREFILL_CACHE_SLOTS=8
```

However, **the benchmarked production profile uses:**

```text
PREFILL_CACHE_SLOTS=0
```

A/B testing showed that the separate prefill cache did not consistently improve warm-request TTFT and introduced additional variance. Disabling it produced the best overall results in this setup.

This is intentionally different from the generic launcher's default.

---

## Prefill chunk size

The benchmarked profile uses:

```text
PREFILL_CHUNK=1024
```

This maps to:

```text
--chunk 1024
```

Tests with larger chunks, including `2048`, did not improve TTFT on the reference machine.

`1024` currently provides the best measured compromise between prompt ingestion speed and stable decode throughput.

---

## Context window

The benchmarked profile leaves:

```text
MAX_CTX=
```

empty.

That means Lucebox performs **auto-fit** instead of receiving a forced `--max-ctx`.

On the reference 16 GB setup, auto-fit selected an effective maximum context of:

```text
8192 tokens
```

This is intentional.

Forcing a substantially larger maximum context was observed to reduce prefill performance and increase TTFT, even before the additional context was fully used.

For the fastest interactive OpenCode experience, keep `MAX_CTX` empty.

If a larger window is required, it can be forced explicitly:

```bash
MAX_CTX=12288 ./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

or:

```bash
MAX_CTX=16384 ./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

Expect worse TTFT. Benchmark the actual workload before making a larger context the default.

---

## Output budget

The full launcher defaults to:

```text
DEFAULT_MAX_TOKENS=4096
```

The tuned OpenCode profile instead uses:

```text
DEFAULT_MAX_TOKENS=1024
```

and `opencode.jsonc` declares:

```json
"limit": {
  "context": 8192,
  "output": 1024
}
```

A smaller default output allowance leaves more usable room inside the 8K context and is sufficient for normal coding-agent turns.

---

## PFlash

PFlash support is present in the launcher but disabled by default:

```text
ENABLE_PFLASH=0
```

Optional parameters:

```text
PFLASH_THRESHOLD=16384
PFLASH_KEEP_RATIO=0.30
PFLASH_DRAFTER_FILE=Qwen3-0.6B-BF16.gguf
```

When enabled, the launcher adds:

```text
--prefill-compression auto
--prefill-threshold <threshold>
--prefill-keep-ratio <ratio>
--prefill-drafter <drafter.gguf>
```

This project does **not** enable PFlash for the normal OpenCode profile. Coding-agent requests contain tools and structured context where aggressive prompt compression can be undesirable.

---

## DFlash speculative decoding

The launcher does not force DFlash speculative decoding for this Qwen3.6-35B-A3B MoE path.

The CUDA sampling environment is still configured with:

```text
DFLASH_GPU_SAMPLE=1
DFLASH_GPU_DRAFT_TOPK=1
DFLASH_GPU_VERIFY_ARGMAX=1
```

These are passed into the container.

---

# Installation

## Requirements

The automatic installer is intended for:

- Linux,
- Ubuntu or Debian for automatic package installation,
- an NVIDIA GPU,
- a working NVIDIA host driver,
- internet access for Docker packages/image and model download.

Recommended reference system:

```text
GPU VRAM: 16 GB
System RAM: 64 GB
```

The launcher warns when less than roughly 28 GiB host RAM is currently available.

No local Lucebox/CUDA compilation is required.

The prebuilt image is:

```text
ghcr.io/luce-org/lucebox-hub:cuda12
```

---

## 1. Make the scripts executable

```bash
chmod +x \
  install_lucebox_docker_qwen36.sh \
  run_lucebox.sh \
  start_lucebox_docker_qwen36_optimized_prefillcache.sh \
  models/download_model.sh
```

---

## 2. Install Docker, NVIDIA Container Toolkit and Lucebox

The simplest installation path is:

```bash
./install_lucebox_docker_qwen36.sh
```

By default the installer:

1. checks for `nvidia-smi`,
2. installs Docker Engine if necessary,
3. installs NVIDIA Container Toolkit if necessary,
4. configures the NVIDIA Docker runtime,
5. pulls `ghcr.io/luce-org/lucebox-hub:cuda12`,
6. verifies GPU access from inside the container,
7. creates `.venv-lucebox-tools`,
8. installs `huggingface_hub`,
9. downloads the default GGUF model if it is missing.

The installer does not automatically add your user to the Docker group. If direct Docker access is unavailable, the scripts use `sudo docker` when possible.

---

## 3. Download the model

If the installer already downloaded the model, this step is not necessary.

The repository also contains a dedicated downloader:

```bash
./models/download_model.sh
```

The expected final path is:

```text
./models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

If you prefer the dedicated downloader rather than the installer's integrated download:

```bash
DOWNLOAD_MODEL=0 ./install_lucebox_docker_qwen36.sh
./models/download_model.sh
```

---

## 4. Start Lucebox

For the tuned profile, use:

```bash
./run_lucebox.sh
```

The benchmarked settings are equivalent to:

```bash
PREFILL_CHUNK=1024 \
PREFIX_CACHE_SLOTS=32 \
PREFILL_CACHE_SLOTS=0 \
MAX_CTX= \
DEFAULT_MAX_TOKENS=1024 \
./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

The API is exposed only on localhost by default:

```text
http://127.0.0.1:8000/v1
```

The served model name is:

```text
qwen3.6-35b-a3b
```

---

## 5. Verify the server

List models:

```bash
curl http://127.0.0.1:8000/v1/models
```

Minimal chat request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

Follow logs:

```bash
docker logs -f lucebox-qwen36
```

Stop:

```bash
docker stop lucebox-qwen36
```

---

# OpenCode setup

The repository contains a minimal `opencode.jsonc` designed specifically to reduce prompt overhead.

Install it with:

```bash
mkdir -p ~/.config/opencode

cp ~/.config/opencode/opencode.jsonc \
   ~/.config/opencode/opencode.jsonc.backup 2>/dev/null || true

cp ./opencode.jsonc ~/.config/opencode/opencode.jsonc
```

Then fully restart OpenCode and create a **new session**.

Do not benchmark an old session after changing the configuration; the existing conversation already contains its old accumulated context.

---

## Minimal OpenCode agent

The tuned configuration uses:

```text
default_agent = fastcode
```

with a deliberately small primary-agent prompt:

```text
You are a coding agent. Inspect files before editing. Use tools autonomously.
Make the smallest correct change, preserve unrelated work, run relevant tests,
and report results concisely. Never invent repository facts.
```

The agent denies everything by default and only allows the core coding tools:

```text
read
edit
glob
grep
bash
```

The configuration also disables or removes unnecessary prompt contributors:

```json
"mcp": {},
"instructions": [],
"lsp": false,
"compaction": {
  "auto": false,
  "prune": true,
  "reserved": 1024
}
```

This was a major TTFT improvement compared with using OpenCode's much larger default coding-agent prompt/tool set.

---

## AGENTS.md / CLAUDE.md

Large project instruction files directly increase prompt size and therefore TTFT.

Keep `AGENTS.md` short. A minimal example is:

```md
# Project rules

Keep changes minimal and consistent with the existing codebase.
Use the project's existing build, test, lint, and formatting commands.
```

Check for global/project rules with:

```bash
wc -c ~/.config/opencode/AGENTS.md 2>/dev/null

find . -maxdepth 3 \
  \( -name AGENTS.md -o -name CLAUDE.md \) \
  -print
```

If Claude Code compatibility is not needed, OpenCode can be started with:

```bash
OPENCODE_DISABLE_CLAUDE_CODE=1 opencode
```

The general rule is simple:

> Every static token that OpenCode sends on every agent step has a direct TTFT cost.

On this hardware, reducing the OpenCode prompt was more valuable than further micro-tuning of decode.

---

# Recommended runtime profile

For this specific hardware target, the current recommended profile is:

```text
Model                  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
Spark                  enabled
Spark VRAM             auto
Spark reserve          768 MiB
KVFlash                enabled
KVFlash size           auto
KVFlash policy         qk
MAX_CTX                auto-fit
Observed auto-fit      8192 tokens
DEFAULT_MAX_TOKENS     1024
PREFILL_CHUNK          1024
PREFIX_CACHE_SLOTS     32
PREFILL_CACHE_SLOTS    0
Persistent cache       enabled
PFlash                 disabled
DFlash spec decode     not forced
Temperature            0 recommended
API                    127.0.0.1:8000/v1
```

Launch command:

```bash
PREFILL_CHUNK=1024 \
PREFIX_CACHE_SLOTS=32 \
PREFILL_CACHE_SLOTS=0 \
MAX_CTX= \
DEFAULT_MAX_TOKENS=1024 \
./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

---

# Runtime parameter reference

## Installer parameters

These environment variables are supported by `install_lucebox_docker_qwen36.sh`.

| Variable | Default | Description |
|---|---|---|
| `IMAGE` | `ghcr.io/luce-org/lucebox-hub:cuda12` | Prebuilt Lucebox CUDA image. |
| `MODEL_REPO` | `unsloth/Qwen3.6-35B-A3B-GGUF` | Hugging Face model repository. |
| `MODEL_FILE` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | GGUF filename. |
| `MODELS_DIR` | `./models` | Local model directory. |
| `TOOLS_VENV` | `./.venv-lucebox-tools` | Small Python environment used for Hugging Face CLI. |
| `INSTALL_DOCKER` | `1` | Set to `0` to prevent automatic Docker installation. |
| `INSTALL_NVIDIA_TOOLKIT` | `1` | Set to `0` to prevent automatic NVIDIA Container Toolkit installation. |
| `PULL_IMAGE` | `1` | Pull the Lucebox image during installation. |
| `DOWNLOAD_MODEL` | `1` | Download the model during installation. |

Examples:

```bash
INSTALL_DOCKER=0 ./install_lucebox_docker_qwen36.sh
```

```bash
INSTALL_NVIDIA_TOOLKIT=0 ./install_lucebox_docker_qwen36.sh
```

```bash
DOWNLOAD_MODEL=0 ./install_lucebox_docker_qwen36.sh
```

```bash
PULL_IMAGE=0 ./install_lucebox_docker_qwen36.sh
```

---

## Server/network parameters

| Variable | Launcher default | Description |
|---|---:|---|
| `IMAGE` | `ghcr.io/luce-org/lucebox-hub:cuda12` | Lucebox image. |
| `CONTAINER_NAME` | `lucebox-qwen36` | Docker container name. |
| `HOST` | `127.0.0.1` | Host address exposed by Docker. |
| `PORT` | `8000` | Host API port. |
| `CONTAINER_PORT` | `8080` | Lucebox port inside the container. |
| `SERVED_NAME` | `qwen3.6-35b-a3b` | OpenAI API model name. |
| `CUDA_DEVICE` | `0` | GPU used by the launcher when inspecting VRAM with `nvidia-smi`. |
| `TARGET_DEVICE` | `cuda:0` | Device passed to Lucebox. |
| `PULL_IMAGE_ON_START` | `0` | Pull the latest configured image before each server start. |

The Docker launcher currently exposes all GPUs with:

```text
--gpus all
```

On a multi-GPU machine, note that `CUDA_DEVICE` controls the launcher's VRAM inspection while `TARGET_DEVICE` controls the Lucebox target device. Adjust the script/device selection if strict Docker GPU isolation is required.

---

## Model parameters

| Variable | Default | Description |
|---|---|---|
| `MODEL_FILE` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | Default GGUF model. |
| `MODELS_DIR` | `./models` | Host model directory. |
| `MODEL_PATH` | derived | Full host model path; can be overridden explicitly. |

The host model directory is mounted read/write into:

```text
/opt/lucebox-hub/server/models
```

Read/write access allows Lucebox/Spark to persist placement data next to the model when applicable.

---

## Context and generation parameters

| Variable | Launcher default | Tuned profile | Description |
|---|---:|---:|---|
| `MAX_CTX` | empty | empty | Empty means Lucebox auto-fit. |
| `DEFAULT_MAX_TOKENS` | `4096` | `1024` | Default generation limit when the client does not override it. |
| `PREFILL_CHUNK` | empty | `1024` | Optional prefill chunk/ubatch size. |

A fixed context is passed as:

```text
--max-ctx <MAX_CTX>
```

A prefill chunk is passed as:

```text
--chunk <PREFILL_CHUNK>
```

### Why auto-fit is preferred

On the reference machine, auto-fit produced an 8192-token maximum context and the best TTFT.

Larger fixed contexts worked but reduced prefill performance. If 8K is sufficient for the workload, leave `MAX_CTX` empty.

---

## Prefix/prefill cache parameters

| Variable | Launcher default | Tuned profile | Description |
|---|---:|---:|---|
| `PREFIX_CACHE_SLOTS` | `32` | `32` | Normal prefix-cache slots. |
| `PREFILL_CACHE_SLOTS` | `8` | `0` | Separate prefill cache. Disabled in the best measured profile. |
| `PERSIST_PREFIX_CACHE` | `1` | `1` | Persist cache across container restarts. |
| `PREFIX_CACHE_HOST_DIR` | `./lucebox-prefix-cache` | same | Host cache directory. |

---

## Spark parameters

| Variable | Default | Description |
|---|---:|---|
| `ENABLE_SPARK` | `1` | Enable Luce Spark. |
| `SPARK_VRAM_GIB` | empty/auto | Explicit Spark VRAM budget. Empty triggers dynamic calculation. |
| `SPARK_VRAM_RESERVE_MIB` | `768` | VRAM kept outside the automatically calculated Spark budget. |

Example:

```bash
SPARK_VRAM_GIB=13.5 ./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

Auto-calculation never intentionally exceeds approximately 95% of total physical VRAM.

---

## KVFlash parameters

| Variable | Default | Description |
|---|---|---|
| `ENABLE_KVFLASH` | `1` | Enable KVFlash. |
| `KVFLASH_SIZE` | `auto` | KVFlash size. |
| `KVFLASH_POLICY` | `qk` | KVFlash policy. |

Alternative experiment:

```bash
KVFLASH_POLICY=lru ./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

The benchmarked profile uses `qk`.

---

## PFlash parameters

| Variable | Default | Description |
|---|---:|---|
| `ENABLE_PFLASH` | `0` | Enable optional prefill compression. |
| `PFLASH_THRESHOLD` | `16384` | Prompt threshold before compression. |
| `PFLASH_KEEP_RATIO` | `0.30` | Fraction of prompt information retained by the configured compression path. |
| `PFLASH_DRAFTER_FILE` | `Qwen3-0.6B-BF16.gguf` | PFlash drafter model filename. |

The launcher refuses to start with PFlash enabled if the configured drafter model is missing.

---

## GPU sampling parameters

| Variable | Default |
|---|---:|
| `DFLASH_GPU_SAMPLE` | `1` |
| `DFLASH_GPU_DRAFT_TOPK` | `1` |
| `DFLASH_GPU_VERIFY_ARGMAX` | `1` |

These are exported into the Docker container.

---

## Cleanup parameters

| Variable | Default | Description |
|---|---|---|
| `DELETE_OLD_MODEL` | `1` | Remove a specifically configured old Hugging Face model cache at startup. |
| `OLD_MODEL_ID` | `cyankiwi/Qwen3-Coder-Next-AWQ-4bit` | Old model cache ID targeted by the cleanup. |

If you do not want the launcher to touch this old cache:

```bash
DELETE_OLD_MODEL=0 ./run_lucebox.sh
```

The script only deletes the matching Hugging Face cache path and contains a path-safety check before removal.

---

## CLI options

The server launcher recognizes:

```text
--foreground
```

This starts the container in the foreground and uses Docker `--rm`.

Example:

```bash
./start_lucebox_docker_qwen36_optimized_prefillcache.sh --foreground
```

Any other command-line arguments are appended to the native Lucebox server command, which allows additional Lucebox flags to be tested without editing the script.

---

# Benchmarks

## Benchmark definitions

The measurements below distinguish:

- **TTFT** — time to the first meaningful content/reasoning/tool-call delta,
- **prompt** — prompt tokens reported by the server,
- **effective prefill** — `prompt_tokens / TTFT`,
- **decode TPS** — completion tokens divided by decode time after TTFT,
- **total** — complete request duration.

The synthetic OpenCode benchmark uses realistic system instructions, coding context and tool definitions. It is useful for A/B testing, but the real OpenCode prompt can still be larger depending on project instructions and conversation history.

---

## Final tuned profile: coding and tool follow-up

Configuration:

```text
PREFILL_CHUNK=1024
PREFIX_CACHE_SLOTS=32
PREFILL_CACHE_SLOTS=0
MAX_CTX=auto-fit / observed 8192
DEFAULT_MAX_TOKENS=1024
```

### Coding

| Run | Cache | Prompt | TTFT | Effective prefill | Generated | Decode |
|---|---|---:|---:|---:|---:|---:|
| 1 | cold | 1734 | 8.704 s | 199.2 tok/s | 67 | 54.33 tok/s |
| 1 | warm | 1734 | 7.521 s | 230.6 tok/s | 66 | 53.02 tok/s |
| 2 | cold | 1735 | 8.325 s | 208.4 tok/s | 98 | 53.34 tok/s |
| 2 | warm | 1735 | 7.993 s | 217.1 tok/s | 97 | 53.12 tok/s |

Approximate coding averages:

```text
Cold TTFT:       8.51 s
Warm TTFT:       7.76 s
Cold prefill:  ~203.8 tok/s
Warm prefill:  ~223.9 tok/s
Decode:        ~53-54 tok/s
```

### Tool follow-up

| Run | Cache | Prompt | TTFT | Effective prefill | Generated | Decode |
|---|---|---:|---:|---:|---:|---:|
| 1 | cold | 2508 | 12.926 s | 194.0 tok/s | 75 | 52.66 tok/s |
| 1 | warm | 2508 | 12.962 s | 193.5 tok/s | 75 | 53.76 tok/s |
| 2 | cold | 2506 | 13.661 s | 183.4 tok/s | 75 | 52.85 tok/s |

The key result is that disabling the separate prefill cache preserved or improved TTFT while decode remained above roughly 52 tok/s.

---

## Larger OpenCode-like prompts

Benchmark command:

```bash
./benchmark_opencode_lucebox.py \
  --runs 2 \
  --scenarios tool_followup,long_session \
  --context-scale 2.5 \
  --max-tokens 256
```

### Tool follow-up

| Run | Cache | Prompt | TTFT | Effective prefill | Decode |
|---|---|---:|---:|---:|---:|
| 1 | cold | 2908 | 19.371 s | 150.1 tok/s | 53.85 tok/s |
| 1 | warm | 2908 | 17.117 s | 169.9 tok/s | 52.08 tok/s |
| 2 | cold | 2905 | 16.659 s | 174.4 tok/s | 54.24 tok/s |
| 2 | warm | 2905 | 16.472 s | 176.4 tok/s | 53.20 tok/s |

### Long session

| Run | Cache | Prompt | TTFT | Effective prefill | Decode |
|---|---|---:|---:|---:|---:|
| 1 | cold | 3378 | 20.581 s | 164.1 tok/s | 53.46 tok/s |
| 1 | warm | 3378 | 19.366 s | 174.4 tok/s | 51.97 tok/s |
| 2 | cold | 3377 | 19.305 s | 174.9 tok/s | 52.45 tok/s |

These results show the main characteristic of the setup:

> **Decode throughput remains excellent while TTFT is dominated by prompt length.**

---

## Earlier end-to-end Lucebox results

Before the final prompt/cache tuning, the same model already showed strong decode performance:

| Scenario | Prompt | TTFT | Effective prefill | Generated | Decode |
|---|---:|---:|---:|---:|---:|
| Short | 36 | 0.428 s | 84.1 tok/s | 15 | 60.73 tok/s |
| Coding | 80 | 0.763 s | 104.9 tok/s | 317 | 38.79 tok/s |
| Long prompt | 2262 | 22.626 s | 100.0 tok/s | 800 | 43.94 tok/s |
| Tool call | 323 | 3.557 s | 90.8 tok/s | 40 | 36.18 tok/s |

The later tuning increased coding/tool decode throughput into the ~52-54 tok/s range and substantially improved prefill throughput on moderate prompts.

---

# What the benchmarks taught us

## 1. Prompt size is the main OpenCode bottleneck

The original OpenCode setup could send roughly 7K+ prompt tokens for a simple agent step.

At approximately 160-220 prompt tokens/s, large static prompts naturally produce tens of seconds of TTFT.

Reducing the OpenCode system prompt, tool schema set, MCPs, project rules and other static context had a larger impact than further decode tuning.

---

## 2. Decode is no longer the main problem

With the tuned profile:

```text
~52-54 decode tok/s
```

is typical for the tested coding/tool workloads.

That is already above the original target of roughly 40 output tok/s.

---

## 3. `PREFILL_CHUNK=1024` is the current sweet spot

A larger `2048` chunk did not improve TTFT on the reference system.

The final profile therefore uses:

```text
PREFILL_CHUNK=1024
```

---

## 4. Separate prefill cache should stay off

Testing:

```text
PREFILL_CACHE_SLOTS=8
```

against:

```text
PREFILL_CACHE_SLOTS=0
```

showed no consistent warm-cache advantage for the separate prefill cache. The zero-slot configuration was faster and/or less variable in the important coding tests.

Keep:

```text
PREFILL_CACHE_SLOTS=0
```

unless a future Lucebox version changes the behavior.

---

## 5. The normal prefix cache is still useful

Keep:

```text
PREFIX_CACHE_SLOTS=32
```

Some repeated coding requests improved meaningfully, while others were nearly neutral. The memory/performance tradeoff is acceptable on the reference system.

---

## 6. Do not oversize context unless necessary

The fastest configuration uses auto-fit, resulting in an 8K context on the reference machine.

A larger fixed context can be useful when longer sessions are more important than latency, but it should be treated as a separate profile rather than a free upgrade.

---

# Suggested profiles

## Fast / interactive OpenCode

```bash
PREFILL_CHUNK=1024 \
PREFIX_CACHE_SLOTS=32 \
PREFILL_CACHE_SLOTS=0 \
MAX_CTX= \
DEFAULT_MAX_TOKENS=1024 \
./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

Recommended default.

---

## 12K context experiment

```bash
PREFILL_CHUNK=1024 \
PREFIX_CACHE_SLOTS=32 \
PREFILL_CACHE_SLOTS=0 \
MAX_CTX=12288 \
DEFAULT_MAX_TOKENS=1024 \
./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

Use only if 8K becomes restrictive. Expect higher TTFT.

This profile has **not** been included in the measured benchmark tables above unless you benchmark it separately on your machine.

---

## Generic launcher defaults

Running the full launcher without overrides:

```bash
./start_lucebox_docker_qwen36_optimized_prefillcache.sh
```

uses its built-in defaults, including:

```text
DEFAULT_MAX_TOKENS=4096
PREFIX_CACHE_SLOTS=32
PREFILL_CACHE_SLOTS=8
PREFILL_CHUNK=<backend default>
MAX_CTX=<auto-fit>
```

Those are **not** the final benchmarked production settings. Use `run_lucebox.sh` or the explicit tuned command for the optimized OpenCode profile.

---

# Troubleshooting

## Docker cannot access the GPU

Check the host first:

```bash
nvidia-smi
```

Then verify Docker:

```bash
docker run --rm --gpus all \
  ghcr.io/luce-org/lucebox-hub:cuda12 \
  nvidia-smi
```

If necessary, rerun:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Model not found

Expected file:

```text
models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Run:

```bash
./models/download_model.sh
```

or rerun the installer with model download enabled:

```bash
DOWNLOAD_MODEL=1 ./install_lucebox_docker_qwen36.sh
```

---

## Container already exists/runs

Check:

```bash
docker ps -a --filter name=lucebox-qwen36
```

Stop it:

```bash
docker stop lucebox-qwen36
```

The launcher automatically removes an old **stopped** container with the configured name, but deliberately refuses to replace one that is still running.

---

## OpenCode context-length errors

Make sure OpenCode and Lucebox agree.

For the fastest profile:

```json
"limit": {
  "context": 8192,
  "output": 1024
}
```

and leave:

```text
MAX_CTX=
```

for server auto-fit.

If a request plus output allowance exceeds the server context, reduce conversation/project prompt size, create a new session, reduce output allowance, or intentionally move to a larger server context profile.

---

## OpenCode is still slow on the first token

Check the actual prompt size.

The server can decode above 50 tok/s while still taking a long time before the first token if OpenCode sends several thousand prompt tokens.

Review:

- `AGENTS.md`,
- `CLAUDE.md`,
- global OpenCode instructions,
- enabled tools,
- MCP servers,
- old conversation history,
- large pasted files,
- unnecessary project rules.

TTFT and decode TPS are separate performance problems.

---

## Inspect cache behavior

```bash
docker logs lucebox-qwen36 2>&1 \
  | grep -Ei 'prefix|prefill|cache|hit|restore'
```

Do not assume a configured cache is helping. Measure cold/warm pairs.

---

# Security

By default, Docker publishes the API as:

```text
127.0.0.1:8000
```

so it is not exposed to other network hosts.

Be careful when changing:

```text
HOST=0.0.0.0
```

because the OpenAI-compatible endpoint has no authentication layer configured by these scripts.

The OpenCode agent also has `bash` access, so it can execute shell commands with the permissions of the user running OpenCode.

---

# Updating

To pull the currently configured Lucebox image manually:

```bash
docker pull ghcr.io/luce-org/lucebox-hub:cuda12
```

or start once with:

```bash
PULL_IMAGE_ON_START=1 ./run_lucebox.sh
```

Because Lucebox is under active development, rerun the benchmark suite after changing the container image. Cache behavior, context auto-fit, memory usage and throughput may change between versions.

---

# Design priorities

This repository intentionally optimizes for:

1. **agentic local coding** rather than maximum benchmark context,
2. **low TTFT** rather than unnecessarily large reserved context,
3. **stable ~50+ tok/s decode** on the reference setup,
4. **16 GB VRAM practicality** through MoE expert placement/offload,
5. **minimal OpenCode prompt overhead**,
6. reproducible, environment-variable-driven tuning.

The resulting stack is not intended to maximize every metric simultaneously. It is a practical latency/quality/memory compromise for running a capable Qwen3.6 coding agent locally on hardware that cannot keep the full model in VRAM.

---

# Quick reference

Install:

```bash
./install_lucebox_docker_qwen36.sh
```

Download model if needed:

```bash
./models/download_model.sh
```

Install OpenCode config:

```bash
mkdir -p ~/.config/opencode
cp ./opencode.jsonc ~/.config/opencode/opencode.jsonc
```

Start the tuned server:

```bash
./run_lucebox.sh
```

Check API:

```bash
curl http://127.0.0.1:8000/v1/models
```

Logs:

```bash
docker logs -f lucebox-qwen36
```

Stop:

```bash
docker stop lucebox-qwen36
```

Recommended core tuning:

```text
PREFILL_CHUNK=1024
PREFIX_CACHE_SLOTS=32
PREFILL_CACHE_SLOTS=0
MAX_CTX=auto-fit
DEFAULT_MAX_TOKENS=1024
KVFlash=auto/qk
Spark=enabled
OpenCode context=8192
OpenCode output=1024
```
