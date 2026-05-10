# cli/scripts/setup.ps1
# One-shot toolchain bootstrap for the C++ face-swap CLI on Windows.
#
# What it does (idempotent - safe to re-run):
#   1. Ensures CMake >= 3.26 is installed (winget Kitware.CMake).
#   2. Ensures MSVC Build Tools 2022 (C++ workload) is installed.
#   3. Downloads ONNX Runtime GPU 1.18.x release zip into cli/third_party/onnxruntime
#      (unless ORT_ROOT is already set in the environment).
#   4. Resolves OpenCV via the conda `dlc` env (we re-use the OpenCV from there,
#      since opencv_world is already present and version-matched to our wheel).
#   5. Copies the buffalo_l + inswapper models from the conda dlc env into
#      cli/models/, so the binary can find them at the default --models path.
#   6. Verifies ffmpeg is on PATH.
#
# Usage:  pwsh -File cli/scripts/setup.ps1

$ErrorActionPreference = "Stop"
$root        = Resolve-Path "$PSScriptRoot/.."
$thirdParty  = Join-Path $root "third_party"
$modelsDir   = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $thirdParty, $modelsDir | Out-Null

function Have-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# --- 1. CMake ---------------------------------------------------------------
if (-not (Have-Command cmake)) {
    Write-Host "[setup] installing cmake via winget..."
    winget install --id Kitware.CMake -e --silent --accept-source-agreements --accept-package-agreements
    $env:Path = "${env:Path};${env:ProgramFiles}\CMake\bin"
}
& cmake --version | Select-Object -First 1

# --- 2. MSVC Build Tools ---------------------------------------------------
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$haveMsvc = $false
if (Test-Path $vswhere) {
    $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsInstall) { $haveMsvc = $true; Write-Host "[setup] MSVC found at $vsInstall" }
}
if (-not $haveMsvc) {
    Write-Host "[setup] installing Visual Studio 2022 Build Tools (C++ workload)..."
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --silent `
        --accept-source-agreements --accept-package-agreements `
        --override "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
}

# --- 3. ONNX Runtime --------------------------------------------------------
if (-not $env:ORT_ROOT) {
    $ortVersion = "1.18.1"
    $ortDir = Join-Path $thirdParty "onnxruntime-win-x64-gpu-$ortVersion"
    if (-not (Test-Path $ortDir)) {
        $url = "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/onnxruntime-win-x64-gpu-$ortVersion.zip"
        $zip = Join-Path $thirdParty "ort.zip"
        Write-Host "[setup] downloading ONNX Runtime $ortVersion..."
        Invoke-WebRequest -Uri $url -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $thirdParty -Force
        Remove-Item $zip
    }
    $env:ORT_ROOT = $ortDir
    Write-Host "[setup] ORT_ROOT=$env:ORT_ROOT"
}

# --- 4. OpenCV (official Windows pack - pip cv2 is python-only) -----------
# The conda dlc env's `cv2` is the pip wheel; it ships .pyd only, no .lib or
# C++ headers. We download the official opencv-X.Y.Z-windows.exe self-extractor
# from the opencv GitHub release, which contains build/x64/vc16/{lib,bin} +
# build/include/opencv2 + an OpenCVConfig.cmake.
if (-not $env:OpenCV_DIR) {
    $cvVersion = "4.10.0"
    $cvDir     = Join-Path $thirdParty "opencv"
    $cvCfg     = Join-Path $cvDir "build"
    if (-not (Test-Path (Join-Path $cvCfg "OpenCVConfig.cmake"))) {
        $cvExe = Join-Path $thirdParty "opencv-$cvVersion-windows.exe"
        if (-not (Test-Path $cvExe)) {
            $url = "https://github.com/opencv/opencv/releases/download/$cvVersion/opencv-$cvVersion-windows.exe"
            Write-Host "[setup] downloading OpenCV $cvVersion (~250 MB)..."
            Invoke-WebRequest -Uri $url -OutFile $cvExe
        }
        Write-Host "[setup] extracting OpenCV (silent, ~1 min)..."
        # opencv-*-windows.exe is a 7-Zip SFX. -o<dir> -y extracts silently;
        # contents land at <dir>\opencv\.
        $tmpExtract = Join-Path $thirdParty "_cvtmp"
        if (Test-Path $tmpExtract) { Remove-Item -Recurse -Force $tmpExtract }
        New-Item -ItemType Directory -Force -Path $tmpExtract | Out-Null
        & $cvExe "-o$tmpExtract" -y | Out-Null
        $extracted = Join-Path $tmpExtract "opencv"
        if (-not (Test-Path $extracted)) {
            throw "OpenCV self-extractor failed; expected $extracted"
        }
        if (Test-Path $cvDir) { Remove-Item -Recurse -Force $cvDir }
        Move-Item $extracted $cvDir
        Remove-Item -Recurse -Force $tmpExtract
        Remove-Item $cvExe -ErrorAction SilentlyContinue
    }
    $env:OpenCV_DIR = $cvCfg
    Write-Host "[setup] OpenCV_DIR=$env:OpenCV_DIR"
}

# --- 5. Copy models from conda dlc env --------------------------------------
$ifaceModels = Join-Path $env:USERPROFILE ".insightface\models\buffalo_l"
$inswapper   = Join-Path $env:USERPROFILE ".insightface\models\inswapper_128_fp16.onnx"
if (Test-Path $ifaceModels) {
    $dst = Join-Path $modelsDir "buffalo_l"
    if (-not (Test-Path $dst)) {
        Write-Host "[setup] copying buffalo_l models..."
        Copy-Item -Path $ifaceModels -Destination $dst -Recurse
    }
}
if (Test-Path $inswapper) {
    $dst = Join-Path $modelsDir "inswapper_128_fp16.onnx"
    if (-not (Test-Path $dst)) {
        Write-Host "[setup] copying inswapper model..."
        Copy-Item -Path $inswapper -Destination $dst
    }
}

# --- 6. ffmpeg --------------------------------------------------------------
if (-not (Have-Command ffmpeg)) {
    Write-Warning "[setup] ffmpeg not on PATH. Install via:  winget install Gyan.FFmpeg"
}

Write-Host ""
Write-Host "[setup] complete. Next steps:"
Write-Host "  1. Open a 'Developer PowerShell for VS 2022' (or run scripts/build.ps1 - it loads vcvars64 automatically)."
Write-Host "  2. pwsh -File cli/scripts/build.ps1"
Write-Host ""
Write-Host "Environment exported to current shell:"
Write-Host "  ORT_ROOT  = $env:ORT_ROOT"
Write-Host "  OpenCV_DIR = $env:OpenCV_DIR"
