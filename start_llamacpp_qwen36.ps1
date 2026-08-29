# =====================================================================
# Startet Qwen3.6-35B-A3B UD-Q4_K_M mit llama.cpp unter WINDOWS (nativ).
#
# PowerShell-Pendant zu start_llamacpp_qwen36_vulkan.sh -- gleiche
# Konfiguration, gleiche Flags, gleiches Verhalten.
#
# Gemessen auf RTX 4080 16GB + i9-10900KF unter Linux (2026-08-29):
#   Cold-Prefill : ~144 tok/s   (Lucebox: ~90)
#   Warm-TTFT    : 0.7 s bei +23 Tokens, 12 s bei +1.7k Tokens
#                  -> es werden NUR die neuen Tokens verarbeitet
#   Decode       : ~30 tok/s    (Lucebox: ~27)
#
# WICHTIG: --no-op-offload ist Pflicht. Ohne das Flag kopiert llama.cpp
# beim Decode Expert-Weights pro Token ueber PCIe zur GPU -> ~2 tok/s.
#
# Binaries (einmalig) von https://github.com/ggml-org/llama.cpp/releases
# holen und nach .\llamacpp-win\ entpacken:
#   - CUDA-Build:   llama-<build>-bin-win-cuda-x64.zip
#                   PLUS das passende cudart-*-win-cuda-*.zip
#                   (beide in DENSELBEN Ordner entpacken)
#   - oder Vulkan:  llama-<build>-bin-win-vulkan-x64.zip
# Unter Windows ist der CUDA-Build verfuegbar und vermutlich die beste
# Wahl; die Vulkan-Variante funktioniert als Fallback genauso.
#
# Beispiele:
#   .\start_llamacpp_qwen36.ps1
#   $env:N_CPU_MOE=26; .\start_llamacpp_qwen36.ps1   # mehr Experten auf GPU
#   $env:MAX_CTX=65536; .\start_llamacpp_qwen36.ps1
#
# Logs:  Get-Content -Wait .\llamacpp-win\server.log
# Stop:  Stop-Process -Id (Get-Content .\llamacpp-win\server.pid)
# =====================================================================

$ErrorActionPreference = "Stop"

function Get-EnvOr([string]$Name, $Default) {
    $v = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($v)) { $Default } else { $v }
}

$BaseDir   = $PSScriptRoot
$BinDir    = Get-EnvOr "BIN_DIR"    (Join-Path $BaseDir "llamacpp-win")
$ModelFile = Get-EnvOr "MODEL_FILE" "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
$ModelPath = Get-EnvOr "MODEL_PATH" (Join-Path (Join-Path $BaseDir "models") $ModelFile)

$BindHost  = Get-EnvOr "HOST"        "127.0.0.1"
$Port      = Get-EnvOr "PORT"        "8010"
$ServedName= Get-EnvOr "SERVED_NAME" "qwen3.6-35b-a3b"

$MaxCtx    = Get-EnvOr "MAX_CTX"     "131072"

# MoE-Layer, deren Experten im RAM bleiben (Modell hat 40 Layer).
# 28 laesst bei 128k-Kontext ~2 GiB VRAM frei (16-GiB-Karte).
$NCpuMoe   = Get-EnvOr "N_CPU_MOE"   "28"

$Batch     = Get-EnvOr "BATCH"       "4096"
$Ubatch    = Get-EnvOr "UBATCH"      "2048"
$Threads   = Get-EnvOr "THREADS"     "8"      # physische Kerne bevorzugen
$CacheTypeK= Get-EnvOr "CACHE_TYPE_K" "q8_0"
$CacheTypeV= Get-EnvOr "CACHE_TYPE_V" "q8_0"

$Log       = Get-EnvOr "LOG" (Join-Path $BinDir "server.log")
$LogErr    = "$Log.err"
$PidFile   = Join-Path $BinDir "server.pid"

$ServerExe = Join-Path $BinDir "llama-server.exe"

if (-not (Test-Path $ServerExe)) {
    Write-Error @"
llama-server.exe nicht gefunden: $ServerExe
Bitte ein Windows-Release von https://github.com/ggml-org/llama.cpp/releases
nach '$BinDir' entpacken (CUDA-Build inkl. cudart-Zip, oder Vulkan-Build).
"@
    exit 1
}
if (-not (Test-Path $ModelPath)) {
    Write-Error "Modell nicht gefunden: $ModelPath"
    exit 1
}

if (Test-Path $PidFile) {
    $oldPid = Get-Content $PidFile
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
        Write-Error "llama-server laeuft bereits (PID $oldPid)."
        exit 1
    }
}

Write-Host "======================================================================"
Write-Host " Qwen3.6-35B-A3B / llama.cpp (Windows, nativ)"
Write-Host "======================================================================"
Write-Host "  API        = http://${BindHost}:${Port}/v1"
Write-Host "  MAX_CTX    = $MaxCtx   N_CPU_MOE = $NCpuMoe   THREADS = $Threads"
Write-Host "  BATCH/UB   = $Batch/$Ubatch   KV = $CacheTypeK/$CacheTypeV"
Write-Host ""

$ServerArgs = @(
    "-m", $ModelPath,
    "--alias", $ServedName, "--host", $BindHost, "--port", $Port,
    "-c", $MaxCtx, "-ngl", "999", "--n-cpu-moe", $NCpuMoe,
    "-b", $Batch, "-ub", $Ubatch, "-t", $Threads,
    "-ctk", $CacheTypeK, "-ctv", $CacheTypeV,
    "-np", "1", "--jinja", "-fa", "on", "--no-op-offload"
)

$proc = Start-Process -FilePath $ServerExe -ArgumentList $ServerArgs `
    -RedirectStandardOutput $Log -RedirectStandardError $LogErr `
    -WindowStyle Hidden -PassThru

$proc.Id | Out-File -Encoding ascii $PidFile

Write-Host "Gestartet (PID $($proc.Id)). Log: $Log"
Write-Host "Bereit, sobald im Log 'listening on' erscheint (~10 s bei warmem Datei-Cache)."
