[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Source,
    [Parameter(Mandatory = $true)] [string] $Target,
    [string] $Output,
    [ValidateSet('fast', 'balanced', 'cinema')] [string] $Quality = 'balanced',
    [ValidateSet('one', 'many', 'reference')] [string] $FaceMode = 'reference',
    [int] $ReferenceFacePosition = 0,
    [ValidateSet('cuda', 'tensorrt', 'directml', 'cpu')] [string] $Provider = 'cuda',
    [switch] $Upscale,
    [switch] $OpenWhenDone
)

$ErrorActionPreference = 'Stop'
$root = 'C:\Users\evija\faceswap'
$conda = 'C:\Users\evija\anaconda3\Scripts\conda.exe'
$facefusion = Join-Path $root 'facefusion\facefusion.py'

if (-not (Test-Path $conda))      { throw "conda missing: $conda" }
if (-not (Test-Path $facefusion)) { throw "facefusion missing: $facefusion" }
if (-not (Test-Path $Source))     { throw "source image not found: $Source" }
if (-not (Test-Path $Target))     { throw "target video not found: $Target" }

if (-not $Output) {
    $name = [IO.Path]::GetFileNameWithoutExtension($Target)
    $Output = Join-Path $root "out\${name}_swapped.mp4"
}
$outDir = Split-Path -Parent $Output
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

# Resolve to absolute paths — we cd into the facefusion repo so relatives would break.
$Source = (Resolve-Path -LiteralPath $Source).Path
$Target = (Resolve-Path -LiteralPath $Target).Path
$Output = [IO.Path]::GetFullPath($Output)

# Quality presets — pixel-boost trades VRAM/time for fidelity. RTX 4090 (16GB) handles 512 easily.
switch ($Quality) {
    'fast' {
        $swapperModel  = 'inswapper_128_fp16'
        $pixelBoost    = '256x256'
        $enhancerBlend = 60
        $processors    = @('face_swapper', 'face_enhancer')
    }
    'balanced' {
        $swapperModel  = 'inswapper_128_fp16'
        $pixelBoost    = '512x512'
        $enhancerBlend = 80
        $processors    = @('face_swapper', 'face_enhancer', 'expression_restorer')
    }
    'cinema' {
        $swapperModel  = 'inswapper_128'      # fp32 — slightly higher fidelity
        $pixelBoost    = '768x768'
        $enhancerBlend = 90
        $processors    = @('face_swapper', 'face_enhancer', 'expression_restorer')
    }
}
if ($Upscale) { $processors += 'frame_enhancer' }

# `conda run -n faceswap` sets CONDA_PREFIX + prepends env's Scripts dir to PATH,
# which FaceFusion's conda.setup() needs for TensorRT DLL discovery.
# `--cwd` runs in the facefusion repo root because FaceFusion uses
# relative paths (resolve_file_paths('facefusion/processors/modules')) for processor discovery.
$ffRoot = Split-Path -Parent $facefusion
$argList = @(
    'run', '-n', 'faceswap', '--cwd', $ffRoot, '--no-capture-output',
    'python', 'facefusion.py', 'headless-run',
    '--source-paths',                    $Source,
    '--target-path',                     $Target,
    '--output-path',                     $Output,
    '--processors')   + $processors + @(
    '--face-swapper-model',              $swapperModel,
    '--face-swapper-pixel-boost',        $pixelBoost,
    '--face-enhancer-model',             'gfpgan_1.4',
    '--face-enhancer-blend',             $enhancerBlend,
    '--face-selector-mode',              $FaceMode,
    '--reference-face-position',         $ReferenceFacePosition,
    '--execution-providers',             $Provider,
    '--execution-thread-count',          8,
    '--video-memory-strategy',           'tolerant',
    '--output-video-encoder',            'libx264',
    '--output-video-quality',            85,
    '--output-video-preset',             'medium',
    '--log-level',                       'info'
)
if ($Upscale) {
    $argList += @('--frame-enhancer-model', 'real_esrgan_x2_fp16',
                  '--frame-enhancer-blend', 80)
}

Write-Host "[swap-song] $Source -> $Target ($Quality, $Provider)" -ForegroundColor Cyan
$sw = [Diagnostics.Stopwatch]::StartNew()
& $conda @argList
$exit = $LASTEXITCODE
$sw.Stop()

if ($exit -eq 0) {
    Write-Host ("[swap-song] done in {0:n1}s -> {1}" -f $sw.Elapsed.TotalSeconds, $Output) -ForegroundColor Green
    if ($OpenWhenDone) { Invoke-Item $Output }
} else {
    Write-Host "[swap-song] FAILED (exit $exit)" -ForegroundColor Red
    exit $exit
}
