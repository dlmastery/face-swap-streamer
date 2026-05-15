# Assembles faceswap-mp-installer.zip on this machine and stages it for
# upload to GitHub Releases.
#
#   pwsh -File tools/build-installer.ps1
#
# Output:
#   dist/faceswap-mp/                       (assembled tree)
#   dist/faceswap-mp-installer.zip          (~700 MB; bundle + scripts)
#
# Then:
#   gh release create v0.12 dist/faceswap-mp-installer.zip --title "..." --notes "..."

$ErrorActionPreference = "Stop"
$REPO = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DIST = Join-Path $REPO "dist"
$ROOT = Join-Path $DIST "faceswap-mp"
$ZIP  = Join-Path $DIST "faceswap-mp-installer.zip"

# Clean previous build
if (Test-Path $ROOT) { Remove-Item -Recurse -Force $ROOT }
if (Test-Path $ZIP)  { Remove-Item -Force $ZIP }
New-Item -ItemType Directory -Force -Path $ROOT | Out-Null

Write-Host "[build] assembling at $ROOT"

# ---- Scripts + docs ----
Copy-Item (Join-Path $REPO "installer\install.ps1")        (Join-Path $ROOT "install.ps1")
Copy-Item (Join-Path $REPO "installer\start.ps1")          (Join-Path $ROOT "start.ps1")
Copy-Item (Join-Path $REPO "installer\README.txt")         (Join-Path $ROOT "README.txt")
Copy-Item (Join-Path $REPO "installer\requirements-mp.txt") (Join-Path $ROOT "requirements-mp.txt")

# ---- Source ----
$srcDst = Join-Path $ROOT "src"
New-Item -ItemType Directory -Force -Path (Join-Path $srcDst "server") | Out-Null
Copy-Item (Join-Path $REPO "webapp_mp.py")             (Join-Path $srcDst "webapp_mp.py")
Copy-Item (Join-Path $REPO "server\swap_worker.py")    (Join-Path $srcDst "server\swap_worker.py")
Copy-Item (Join-Path $REPO "server\__init__.py")       (Join-Path $srcDst "server\__init__.py")

# ---- Tools ----
$toolsDst = Join-Path $ROOT "tools"
New-Item -ItemType Directory -Force -Path $toolsDst | Out-Null
Copy-Item (Join-Path $REPO "test-cuda-dlc.py")         (Join-Path $toolsDst "test-cuda-dlc.py")

# ---- Models ----
$modelsDst = Join-Path $ROOT "models"
$buffaloDst = Join-Path $modelsDst "buffalo_l"
New-Item -ItemType Directory -Force -Path $buffaloDst | Out-Null
$buffaloSrc = Join-Path $env:USERPROFILE ".insightface\models\buffalo_l"
if (-not (Test-Path $buffaloSrc)) {
    throw "buffalo_l models not found at $buffaloSrc - run a job in webapp.py once to download them, then re-run."
}
foreach ($f in "1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx") {
    Copy-Item (Join-Path $buffaloSrc $f) (Join-Path $buffaloDst $f)
}
$inswapperSrc = Join-Path $REPO "deep-live-cam\models\inswapper_128_fp16.onnx"
if (-not (Test-Path $inswapperSrc)) {
    throw "inswapper not found at $inswapperSrc"
}
Copy-Item $inswapperSrc (Join-Path $modelsDst "inswapper_128_fp16.onnx")

# ---- Show contents ----
Write-Host ""
Write-Host "[build] tree:"
Get-ChildItem -Recurse $ROOT | ForEach-Object {
    $rel = $_.FullName.Substring($ROOT.Length + 1)
    $size = if ($_.PSIsContainer) { "" } else { "  $([math]::Round($_.Length/1MB, 1)) MB" }
    Write-Host "  $rel$size"
}

# ---- Zip ----
Write-Host ""
Write-Host "[build] zipping to $ZIP (this takes ~30 s)..."
Compress-Archive -Path (Join-Path $ROOT "*") -DestinationPath $ZIP -CompressionLevel Optimal -Force

$zipSize = (Get-Item $ZIP).Length / 1MB
Write-Host ""
Write-Host "[build] DONE"
Write-Host "  zip:  $ZIP"
Write-Host "  size: $([math]::Round($zipSize, 1)) MB"
Write-Host ""
Write-Host "Upload to GitHub Releases with:"
Write-Host "  gh release create v0.12 $ZIP \\"
Write-Host "    --title 'faceswap-mp v0.12 installable bundle' \\"
Write-Host "    --notes 'See README.txt inside the zip.'"
