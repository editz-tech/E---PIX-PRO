$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
# Built by the repository's pull-request portable validation workflow when enabled.

function Invoke-Step([string]$Name, [scriptblock]$Action) {
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11 x64 is required. Install it from python.org, then run this script again."
}
$pyVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pyVersion -ne "3.11") {
    throw "This project build is pinned to Python 3.11. Detected Python $pyVersion."
}

Invoke-Step "Install build dependencies" { python -m pip install --upgrade pip; if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }; python -m pip install -r requirements.txt pyinstaller }
Invoke-Step "Download offline AI model" { python scripts/download_model.py }

$model = Join-Path $root "models\u2net.onnx"
if (-not (Test-Path $model -PathType Leaf) -or (Get-Item $model).Length -lt 50000000) {
    throw "The offline U2Net model is missing or incomplete."
}

Invoke-Step "Build E-PIX PRO" { python -m PyInstaller --noconfirm --clean "E-Pix.spec" }
$app = Join-Path $root "dist\E-PIX PRO"
$exe = Join-Path $app "E-PIX PRO.exe"
if (-not (Test-Path $exe -PathType Leaf)) { throw "PyInstaller did not produce E-PIX PRO.exe." }

$bundledModel = Join-Path $app "_internal\models\u2net.onnx"
if (-not (Test-Path $bundledModel -PathType Leaf)) {
    $bundledModel = Get-ChildItem $app -Filter "u2net.onnx" -File -Recurse | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $bundledModel) { throw "The portable build does not contain the offline U2Net model." }

@"
E-PIX PRO portable build
Windows x64
This build is unsigned. Windows may display publisher/reputation warnings.
Do not disable SmartScreen, Defender, or browser security controls.
"@ | Set-Content (Join-Path $app "DOWNLOAD-INFO.txt") -Encoding UTF8

$zip = Join-Path $root "E-PIX-PRO-Portable.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $app -DestinationPath $zip -CompressionLevel Optimal

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  E-PIX-PRO-Portable.zip" | Set-Content (Join-Path $root "SHA256SUMS.txt") -Encoding ASCII

Write-Host ""
Write-Host "BUILD COMPLETE" -ForegroundColor Green
Write-Host "Portable package: $zip"
Write-Host "SHA-256: $hash"
Write-Host "Copy the ZIP to your Windows PC and extract the entire folder before launching E-PIX PRO.exe."
