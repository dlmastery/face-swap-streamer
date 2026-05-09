[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Source,
    [string] $SongsDir = 'C:\Users\evija\faceswap\songs',
    [string] $OutDir   = 'C:\Users\evija\faceswap\out',
    [ValidateSet('fast', 'balanced', 'cinema')] [string] $Quality = 'balanced',
    [switch] $Upscale,
    [switch] $SkipExisting
)

$ErrorActionPreference = 'Stop'
$swap = Join-Path 'C:\Users\evija\faceswap' 'swap-song.ps1'
if (-not (Test-Path $swap))   { throw "swap-song.ps1 missing" }
if (-not (Test-Path $Source)) { throw "source image not found: $Source" }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$videos = Get-ChildItem -Path $SongsDir -File -Include *.mp4,*.mkv,*.mov,*.webm,*.m4v -Recurse
if (-not $videos) { Write-Warning "no videos in $SongsDir"; return }

$total  = $videos.Count
$ok     = 0
$failed = @()
$albumSw = [Diagnostics.Stopwatch]::StartNew()

for ($i = 0; $i -lt $total; $i++) {
    $v = $videos[$i]
    $out = Join-Path $OutDir ("{0}_swapped.mp4" -f $v.BaseName)
    Write-Host ""
    Write-Host ("=== [{0}/{1}] {2}" -f ($i + 1), $total, $v.Name) -ForegroundColor Yellow

    if ($SkipExisting -and (Test-Path $out)) {
        Write-Host "  -> exists, skipping"
        $ok++
        continue
    }

    try {
        & $swap -Source $Source -Target $v.FullName -Output $out -Quality $Quality -Upscale:$Upscale
        if ($LASTEXITCODE -eq 0) { $ok++ } else { $failed += $v.Name }
    } catch {
        Write-Warning $_.Exception.Message
        $failed += $v.Name
    }
}

$albumSw.Stop()
Write-Host ""
Write-Host ("=== album done: {0}/{1} ok in {2:n1}m" -f $ok, $total, $albumSw.Elapsed.TotalMinutes) -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
}
