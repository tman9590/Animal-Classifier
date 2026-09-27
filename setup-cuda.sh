#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable: verify the NVIDIA driver and GPU visibility before continuing')
x = torch.randn(256, 256, device='cuda', requires_grad=True)
(x @ x).square().mean().backward()
torch.cuda.synchronize()
print('CUDA forward/backward passed:', torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY
