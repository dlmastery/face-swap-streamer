# faceswap-mp launcher.
#
# Runs the multiprocessing Flask web app with the production-tuned config:
#   FACESWAP_WORKERS=6, FACESWAP_DET_SIZE=480, h264_nvenc encoder.
#
# Override any of these by setting the env var BEFORE running this script:
#   $env:FACESWAP_WORKERS = "4"; .\start.ps1
#
# Server URL: http://localhost:8082/

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot

# Locate conda
$condaExe = $null
foreach ($candidate in @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
    "$env:ProgramData\miniconda3\Scripts\conda.exe"
)) {
    if (Test-Path $candidate) { $condaExe = $candidate; break }
}
if (-not $condaExe) { throw "conda not found. Run install.ps1 first." }

# Locate ffmpeg (set FFMPEG_BIN so webapp_mp.py uses gyan.dev, not anaconda's broken build)
if (-not $env:FFMPEG_BIN) {
    $ff = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ff) {
        $env:FFMPEG_BIN = $ff.FullName
    }
}

# Default tuning (env vars override these)
if (-not $env:FACESWAP_PORT)            { $env:FACESWAP_PORT          = "8082" }
if (-not $env:FACESWAP_WORKERS)         { $env:FACESWAP_WORKERS       = "6"    }
if (-not $env:FACESWAP_DET_SIZE)        { $env:FACESWAP_DET_SIZE      = "480"  }
if (-not $env:FACESWAP_VIDEO_ENCODER)   { $env:FACESWAP_VIDEO_ENCODER = "h264_nvenc" }

Write-Host "Starting faceswap-mp on http://localhost:$env:FACESWAP_PORT/"
Write-Host "  workers      = $env:FACESWAP_WORKERS"
Write-Host "  det_size     = $env:FACESWAP_DET_SIZE"
Write-Host "  encoder      = $env:FACESWAP_VIDEO_ENCODER"
Write-Host "  ffmpeg       = $env:FFMPEG_BIN"
Write-Host ""
Write-Host "First-job warmup is ~45s (workers loading models). Subsequent jobs in"
Write-Host "the same server re-pay the warmup. Press Ctrl+C to stop."
Write-Host ""

Set-Location (Join-Path $ROOT "src")
& $condaExe run --no-capture-output -n faceswap-mp python webapp_mp.py
