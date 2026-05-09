# One-shot setup for the faceswap repo on Windows.
# Creates the conda envs, clones upstream tools, downloads models, applies patches.
#
# Prerequisites:
#   - Anaconda or Miniconda on PATH
#   - Git on PATH
#   - NVIDIA GPU with recent driver (for CUDA 12 onnxruntime)
#
# Run from the repo root:
#   .\setup.ps1

[CmdletBinding()]
param(
    [switch] $SkipFaceFusion,
    [switch] $SkipDeepLiveCam,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) { throw "conda not found on PATH" }

Write-Host "==> Creating conda envs (faceswap=Py3.12, dlc=Py3.11)..." -ForegroundColor Cyan
& $conda create -n faceswap python=3.12 -y | Out-Null
& $conda create -n dlc      python=3.11 -y | Out-Null

# ---------- Path A: FaceFusion (high-quality offline render) -----------------
if (-not $SkipFaceFusion) {
    Write-Host "==> Cloning FaceFusion..." -ForegroundColor Cyan
    if (-not (Test-Path "$root\facefusion")) {
        git clone --depth 1 https://github.com/facefusion/facefusion.git "$root\facefusion"
    }
    Push-Location "$root\facefusion"
    Write-Host "==> Installing FaceFusion dependencies..." -ForegroundColor Cyan
    & $conda run -n faceswap --no-capture-output python install.py --onnxruntime cuda
    Pop-Location

    Write-Host "==> Installing CUDA runtime libs in faceswap env..." -ForegroundColor Cyan
    & $conda run -n faceswap --no-capture-output pip install -r "$root\requirements-facefusion.txt"

    Write-Host "==> Patching FaceFusion conda.py for cuDNN DLL discovery..." -ForegroundColor Cyan
    $condapy = "$root\facefusion\facefusion\conda.py"
    $patch = Get-Content $condapy -Raw
    if ($patch -notmatch 'add_dll_directory') {
        $needle  = "library_paths =\`n			[`n				os.path.join(conda_prefix, 'Lib'),`n				os.path.join(conda_prefix, 'Lib', 'site-packages', 'tensorrt_libs')`n			]"
        $replace = @"
site_packages = os.path.join(conda_prefix, 'Lib', 'site-packages')
			nvidia_bin_dirs = [ os.path.join(site_packages, 'nvidia', sub, 'bin')
				for sub in [ 'cudnn', 'cublas', 'cuda_runtime', 'cuda_nvrtc', 'curand', 'cufft' ] ]
			library_paths = [
				os.path.join(conda_prefix, 'Lib'),
				os.path.join(site_packages, 'tensorrt_libs'),
				*nvidia_bin_dirs,
			]
"@
        $patch = $patch.Replace($needle, $replace)
        # Inject add_dll_directory call right after library_paths filter
        $afterFilter = "library_paths = list(filter(os.path.exists, library_paths))"
        $injectAfter = "$afterFilter`n`n			for path in library_paths:`n				try: os.add_dll_directory(path)`n				except (OSError, AttributeError): pass"
        if ($patch.Contains($afterFilter) -and ($patch -notmatch 'add_dll_directory')) {
            $patch = $patch.Replace($afterFilter, $injectAfter)
        }
        Set-Content -Path $condapy -Value $patch -NoNewline
    } else {
        Write-Host "    (already patched)" -ForegroundColor DarkGray
    }
}

# ---------- Path B: Deep-Live-Cam (real-time GUI) ---------------------------
if (-not $SkipDeepLiveCam) {
    Write-Host "==> Cloning Deep-Live-Cam..." -ForegroundColor Cyan
    if (-not (Test-Path "$root\deep-live-cam")) {
        git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git "$root\deep-live-cam"
    }
    Write-Host "==> Installing DLC dependencies (incl. tensorflow ~ 380MB)..." -ForegroundColor Cyan
    Push-Location "$root\deep-live-cam"
    & $conda run -n dlc --no-capture-output pip install -r requirements.txt
    Pop-Location
    Write-Host "==> Installing webapp + CUDA runtime libs in dlc env..." -ForegroundColor Cyan
    & $conda run -n dlc --no-capture-output pip install -r "$root\requirements-webapp.txt"

    Write-Host "==> Downloading DLC models..." -ForegroundColor Cyan
    $models = "$root\deep-live-cam\models"
    New-Item -ItemType Directory -Force -Path $models | Out-Null
    if (-not (Test-Path "$models\inswapper_128_fp16.onnx") -or $Force) {
        curl.exe -L --fail --ssl-no-revoke --progress-bar `
            -o "$models\inswapper_128_fp16.onnx" `
            'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx'
    }
    if (-not (Test-Path "$models\GFPGANv1.4.pth") -or $Force) {
        curl.exe -L --fail --ssl-no-revoke --progress-bar `
            -o "$models\GFPGANv1.4.pth" `
            'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
    }

    Write-Host "==> Patching DLC run.py for cuDNN DLL discovery..." -ForegroundColor Cyan
    $runpy = "$root\deep-live-cam\run.py"
    $content = Get-Content $runpy -Raw
    if ($content -notmatch 'add_dll_directory') {
        # Insert _dll_cookies + add_dll_directory inside the existing nvidia loop
        $content = $content.Replace(
            '_nvidia_dir = os.path.join(_sp, "nvidia")',
            "_dll_cookies = []`n        _nvidia_dir = os.path.join(_sp, ""nvidia"")"
        )
        $content = $content.Replace(
            'os.environ["PATH"] = _bin_dir + os.pathsep + os.environ["PATH"]',
            "try: _dll_cookies.append(os.add_dll_directory(_bin_dir))`n                    except OSError: pass`n                    os.environ[""PATH""] = _bin_dir + os.pathsep + os.environ[""PATH""]"
        )
        Set-Content -Path $runpy -Value $content -NoNewline
    } else {
        Write-Host "    (already patched)" -ForegroundColor DarkGray
    }
}

# ---------- Working dirs ----------------------------------------------------
@("source","songs","out","webapp_jobs") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path "$root\$_" | Out-Null
}

Write-Host ""
Write-Host "==> Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1) Drop a face image in .\source\ and a video in .\songs\"
Write-Host "  2) Web app:        conda run -n dlc python webapp.py     (opens http://localhost:8080/)"
Write-Host "  3) FaceFusion CLI: .\swap-song.ps1 -Source .\source\me.jpg -Target .\songs\song.mp4"
Write-Host "  4) DLC GUI:        .\play-song.ps1"
