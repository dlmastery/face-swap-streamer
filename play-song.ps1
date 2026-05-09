# Path B — launch Deep-Live-Cam for real-time playback of an MP4 with your face swapped on-the-fly.
#
# Two modes:
#   default            launch GUI — pick "Live" button after setting source + target video
#                      (this is the v2.7+ Real-time Playback feature)
#   -Render            run DLC's CLI to produce a swapped MP4 file (offline, like Path A)
#                      Note: FaceFusion (swap-song.ps1) is higher quality for offline render.

[CmdletBinding()]
param(
    [string] $Source,
    [string] $Target,
    [string] $Output,
    [switch] $Render,
    [switch] $ManyFaces,
    [switch] $MouthMask,
    [int]    $VideoQuality = 18,
    [ValidateSet('cuda', 'cpu', 'directml')] [string] $Provider = 'cuda',
    [int]    $Threads = 8
)

$ErrorActionPreference = 'Stop'
$root  = 'C:\Users\evija\faceswap'
$conda = 'C:\Users\evija\anaconda3\Scripts\conda.exe'
$dlc   = Join-Path $root 'deep-live-cam'
$dlcRun = Join-Path $dlc 'run.py'

if (-not (Test-Path $conda))   { throw "conda missing: $conda" }
if (-not (Test-Path $dlcRun))  { throw "deep-live-cam missing: $dlcRun" }

$models = Join-Path $dlc 'models'
$inswap = Join-Path $models 'inswapper_128_fp16.onnx'
$gfpgan = Join-Path $models 'GFPGANv1.4.pth'
foreach ($m in $inswap, $gfpgan) {
    if (-not (Test-Path $m)) {
        Write-Warning "model missing: $m — DLC will fail to start until this exists"
    }
}

Set-Location $dlc

if ($Render) {
    # Headless CLI render — saves swapped MP4 to disk
    if (-not $Source) { throw "-Source required when using -Render" }
    if (-not $Target) { throw "-Target required when using -Render" }
    if (-not (Test-Path $Source)) { throw "source not found: $Source" }
    if (-not (Test-Path $Target)) { throw "target not found: $Target" }
    if (-not $Output) {
        $name = [IO.Path]::GetFileNameWithoutExtension($Target)
        $Output = Join-Path $root "out\${name}_dlc.mp4"
    }
    $outDir = Split-Path -Parent $Output
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

    $argList = @(
        'run', '-n', 'dlc', '--no-capture-output',
        'python', $dlcRun,
        '-s', $Source, '-t', $Target, '-o', $Output,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--keep-fps', '--keep-audio',
        '--video-encoder', 'libx264',
        '--video-quality', $VideoQuality,
        '--execution-provider', $Provider,
        '--execution-threads', $Threads
    )
    if ($ManyFaces) { $argList += '--many-faces' }
    if ($MouthMask) { $argList += '--mouth-mask' }

    Write-Host "[play-song -Render] $Source -> $Target ($Provider)" -ForegroundColor Cyan
    & $conda @argList
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[play-song] rendered -> $Output" -ForegroundColor Green
    } else {
        Write-Host "[play-song] FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    # GUI mode — for v2.7+ real-time playback
    Write-Host "[play-song] launching Deep-Live-Cam GUI..." -ForegroundColor Cyan
    Write-Host "  - In the GUI: select source image, switch target to 'Video' and pick your MP4," -ForegroundColor DarkGray
    Write-Host "    then click 'Live' for real-time playback (or 'Start' for offline render)." -ForegroundColor DarkGray

    $argList = @(
        'run', '-n', 'dlc', '--no-capture-output',
        'python', $dlcRun,
        '--execution-provider', $Provider,
        '--execution-threads', $Threads
    )
    & $conda @argList
}
