# faceswap-mp installer for Windows.
#
# Run this ONCE on a fresh machine. It will:
#   1. Verify Windows 10/11 + NVIDIA driver (nvidia-smi) are present
#   2. Install Miniconda via winget if missing
#   3. Install ffmpeg (gyan.dev build) via winget if missing
#   4. Create the `faceswap-mp` conda env (Python 3.11) + pip-install deps
#   5. Copy buffalo_l + inswapper models to ~/.insightface/models/ and the
#      `<install>/deep-live-cam/models/` path that webapp_mp.py reads from
#   6. Verify ONNX Runtime loads with CUDA (test-cuda-dlc.py)
#
# After this runs successfully, use start.ps1 to launch the web app.
#
# Idempotent: re-running is safe - every step checks "is it already done" before
# acting. So if a step fails partway through, fix the issue and re-run.

$ErrorActionPreference = "Stop"
$script:ROOT = $PSScriptRoot
$script:ENV_NAME = "faceswap-mp"
$script:PY_VER   = "3.11"

function Section($name) {
    Write-Host ""
    Write-Host "===== $name =====" -ForegroundColor Cyan
}

function HaveCommand($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# ---- 1. Sanity ------------------------------------------------------------
Section "1/6  Verifying prerequisites"

$osver = [System.Environment]::OSVersion.Version
if ($osver.Major -lt 10) {
    throw "Windows 10 or newer required (detected $($osver.ToString()))"
}
Write-Host "  Windows $($osver.Build) - OK"

if (-not (HaveCommand nvidia-smi)) {
    Write-Host ""
    Write-Host "  NVIDIA driver not found." -ForegroundColor Red
    Write-Host "  This package requires an NVIDIA GPU with the R535+ driver installed."
    Write-Host "  Download: https://www.nvidia.com/Download/index.aspx"
    throw "nvidia-smi not on PATH"
}
$drvLine = (& nvidia-smi --query-gpu=driver_version,name --format=csv,noheader -i 0) -join " "
Write-Host "  GPU: $drvLine - OK"

if (-not (HaveCommand winget)) {
    Write-Host ""
    Write-Host "  winget not found." -ForegroundColor Red
    Write-Host "  winget ships with Windows 10 21H2+ and Windows 11."
    Write-Host "  Install from the Microsoft Store ('App Installer') and re-run this script."
    throw "winget not available"
}
Write-Host "  winget available - OK"

# ---- 2. Miniconda ---------------------------------------------------------
Section "2/6  Miniconda"

$condaExe = $null
foreach ($candidate in @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
    "$env:ProgramData\miniconda3\Scripts\conda.exe"
)) {
    if (Test-Path $candidate) { $condaExe = $candidate; break }
}

if (-not $condaExe) {
    Write-Host "  Installing Miniconda via winget (one-time, ~80 MB)..."
    winget install --id Anaconda.Miniconda3 -e --silent `
        --accept-source-agreements --accept-package-agreements | Out-Null
    foreach ($candidate in @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe"
    )) {
        if (Test-Path $candidate) { $condaExe = $candidate; break }
    }
}
if (-not $condaExe) { throw "Miniconda install failed (could not locate conda.exe)" }
Write-Host "  Miniconda at: $condaExe"

# ---- 3. ffmpeg -----------------------------------------------------------
Section "3/6  ffmpeg"

if (HaveCommand ffmpeg) {
    Write-Host "  ffmpeg already on PATH - OK"
} else {
    Write-Host "  Installing ffmpeg (gyan.dev build) via winget..."
    winget install --id Gyan.FFmpeg -e --silent `
        --accept-source-agreements --accept-package-agreements | Out-Null
    # winget puts ffmpeg under %LOCALAPPDATA%\Microsoft\WinGet\Packages\...
    $ffPath = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffPath) {
        Write-Host "  ffmpeg installed: $($ffPath.FullName)"
    } else {
        Write-Host "  ffmpeg installed (path will resolve after shell restart)"
    }
}

# ---- 4. Conda env + pip deps ---------------------------------------------
Section "4/6  Conda env '$ENV_NAME' (Python $PY_VER)"

$envList = & $condaExe env list 2>$null
if ($envList -match "^\s*$ENV_NAME\s") {
    Write-Host "  Env '$ENV_NAME' already exists - skipping creation"
} else {
    Write-Host "  Creating env (1-2 min)..."
    & $condaExe create -y -n $ENV_NAME "python=$PY_VER" pip
    if ($LASTEXITCODE -ne 0) { throw "conda create failed (exit $LASTEXITCODE)" }
}

Write-Host "  Installing pip deps (5-10 min depending on network)..."
$reqFile = Join-Path $ROOT "requirements-mp.txt"
& $condaExe run -n $ENV_NAME pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }

# ---- 5. Models ------------------------------------------------------------
Section "5/6  Models"

$srcBuffalo = Join-Path $ROOT "models\buffalo_l"
$dstBuffalo = Join-Path $env:USERPROFILE ".insightface\models\buffalo_l"
if (-not (Test-Path $dstBuffalo)) {
    New-Item -ItemType Directory -Force -Path $dstBuffalo | Out-Null
}
$expected = @("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")
foreach ($f in $expected) {
    $dst = Join-Path $dstBuffalo $f
    if (-not (Test-Path $dst)) {
        Copy-Item (Join-Path $srcBuffalo $f) $dst
        Write-Host "  Copied buffalo_l/$f"
    }
}

# webapp_mp.py looks for inswapper under deep-live-cam/models/ (legacy path
# from the early days of this repo). We mirror the same layout here so the
# code finds it without code changes.
$dstSwap = Join-Path $ROOT "deep-live-cam\models\inswapper_128_fp16.onnx"
if (-not (Test-Path $dstSwap)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $dstSwap) | Out-Null
    Copy-Item (Join-Path $ROOT "models\inswapper_128_fp16.onnx") $dstSwap
    Write-Host "  Copied inswapper_128_fp16.onnx -> deep-live-cam/models/"
}

# ---- 6. CUDA verification -----------------------------------------------
Section "6/6  Verify ONNX Runtime + CUDA"

$cudaTest = Join-Path $ROOT "tools\test-cuda-dlc.py"
& $condaExe run -n $ENV_NAME python $cudaTest
if ($LASTEXITCODE -ne 0) {
    throw "CUDA verification failed. ONNX Runtime did not load on the GPU. " +
          "Check that your driver version is R535+ and re-run."
}

# ---- Done ----------------------------------------------------------------
Write-Host ""
Write-Host "===== INSTALL COMPLETE =====" -ForegroundColor Green
Write-Host ""
Write-Host "Next: run start.ps1 to launch the web app."
Write-Host "  PS> .\start.ps1"
Write-Host ""
Write-Host "Browser: http://localhost:8082/"
Write-Host ""
