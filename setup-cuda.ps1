$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if ($PSScriptRoot.StartsWith('\\')) {
    throw 'Copy/extract the project to a local Windows drive before running setup.'
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.11 with its Python launcher, then reopen PowerShell and rerun this script.'
}
& py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed.' }
$TrainingPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $TrainingPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
& $TrainingPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $TrainingPython -m pip install -r requirements-train.txt
if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
& $TrainingPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: check the NVIDIA driver'; x=torch.randn(256,256,device='cuda',requires_grad=True); (x@x).square().mean().backward(); torch.cuda.synchronize(); print('CUDA forward/backward passed:',torch.cuda.get_device_name(0),torch.__version__,torch.version.cuda)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA hardware validation failed.' }
& $TrainingPython distill/verify_transfer.py
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint verification failed.' }
Write-Host 'Setup and checkpoint verification passed. Run resume-labeling.ps1 next.'
