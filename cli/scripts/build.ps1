# cli/scripts/build.ps1
# Configure + build the face-swap CLI.
#
# Usage:
#   pwsh -File cli/scripts/build.ps1                # release build
#   pwsh -File cli/scripts/build.ps1 -Config Debug  # debug build
#   pwsh -File cli/scripts/build.ps1 -Clean         # wipe build/ first

param(
    [string]$Config = "Release",
    [switch]$Clean
)
$ErrorActionPreference = "Stop"

$root  = Resolve-Path "$PSScriptRoot/.."
$build = Join-Path $root "build"

# --- Locate vcvars64 and load it into this session -------------------------
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found - run cli/scripts/setup.ps1 first."
}
$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) { throw "VS 2022 with C++ tools not found - run cli/scripts/setup.ps1 first." }
$vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat missing in $vsInstall" }

# Import vcvars64.bat env into current PowerShell (only on first call).
if (-not $env:VSCMD_ARG_TGT_ARCH) {
    Write-Host "[build] loading vcvars64..."
    cmd /c "`"$vcvars`" && set" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] }
    }
}

if ($Clean -and (Test-Path $build)) {
    Write-Host "[build] cleaning $build"
    Remove-Item -Recurse -Force $build
}
New-Item -ItemType Directory -Force -Path $build | Out-Null

# --- Configure --------------------------------------------------------------
$cmakeArgs = @(
    "-S", "$root",
    "-B", "$build",
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$Config"
)
if ($env:ORT_ROOT)   { $cmakeArgs += "-DORT_ROOT=$env:ORT_ROOT" }
if ($env:OpenCV_DIR) { $cmakeArgs += "-DOpenCV_DIR=$env:OpenCV_DIR" }

# Fall back to "Visual Studio 17 2022" generator if Ninja isn't available.
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Host "[build] Ninja not found - falling back to VS 17 2022 generator"
    $cmakeArgs[$cmakeArgs.IndexOf("Ninja")] = "Visual Studio 17 2022"
    $cmakeArgs += "-A", "x64"
}

Write-Host "[build] configuring..."
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

# --- Build ------------------------------------------------------------------
Write-Host "[build] compiling ($Config)..."
& cmake --build $build --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

$exe = Get-ChildItem -Path $build -Recurse -Filter faceswap.exe | Select-Object -First 1
if ($exe) {
    Write-Host ""
    Write-Host "[build] success: $($exe.FullName)"
    Write-Host "[build] try:  $($exe.FullName) --help"
} else {
    throw "build finished but faceswap.exe not produced"
}
