$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$TrainingPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $TrainingPython)) { throw 'Run setup-cuda.ps1 first.' }
$PreviousOfflineSetting = $env:HF_HUB_OFFLINE
try {
    $env:HF_HUB_OFFLINE = '1'
    & $TrainingPython -u distill/label_images.py --device cuda:0 --batch-size 8 --output work/dataset-production
    if ($LASTEXITCODE -ne 0) { throw 'Labeling stopped. Completed images are checkpointed; rerun to resume.' }
} finally {
    $env:HF_HUB_OFFLINE = $PreviousOfflineSetting
}
