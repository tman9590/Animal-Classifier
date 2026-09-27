# GPU training handoff

The desktop RTX 5070 Ti should run teacher inference and student training with
PyTorch 2.8 / CUDA 12.8. Keep this Mac available for final CoreML runtime checks.
Use a fresh Python 3.11 environment on the destination; do not copy the Mac venv.
`setup-cuda.sh` prepares Linux/WSL and verifies an actual CUDA forward/backward pass.

## Checkpoint

- 8,175 downloaded public photos now have CC0/CC-BY attribution records.
- 635 private Scrypted frames remain local and Git-ignored.
- Public teacher pass: 812 processed photos, 365 retained images, 601 animal boxes.
- `work/dataset-production/public-audit.jsonl` makes public labeling resumable.
- `work/dataset-v3/` contains provisional private labels, **not approved training data**.
  Review found implausible species. The actual poultry species on the property
  must be confirmed and the private labels regenerated/reviewed before inclusion.
- No student has been trained and no deployment weights have been validated.

Transfer the repository and the ignored `work/source-images`, `work/models`,
`work/scrypted-events`, and `work/dataset-production` directories directly between
owned computers. Keep private images and review sheets out of GitHub and public
file-sharing services. Preserve the Hugging Face cache symlinks under `work/models`.
The transfer checksum manifest and detailed checkpoint are local under `work/transfer`.

## Windows 11 (native CUDA)

Use the Windows ZIP archive, which contains regular files instead of symlinks.
Copy it through the intended private SMB share, then extract it onto a local SSD
with at least 25 GB free. Do not train directly from the SMB network path.
Install Python 3.11 (including the Python launcher) and a current NVIDIA driver.
From PowerShell in the extracted project folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-cuda.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\resume-labeling.ps1
```

The execution-policy flag applies only to that process. Setup verifies an actual
CUDA forward/backward operation and every transferred checkpoint file. Native
Windows scripts have been prepared but must still be tested on the destination.
SMB transfers files; it does not provide a remote shell. Run the commands on the
Windows PC, or establish a remote shell to let the build be managed from the Mac.

## Resume public labeling

```sh
./setup-cuda.sh
HF_HUB_OFFLINE=1 .venv/bin/python -u distill/label_images.py \
  --device cuda:0 --batch-size 8 --output work/dataset-production
```

The command skips the 812 completed records, loads the fully recovered attribution
manifest, keeps observations in stable train/validation partitions, and regenerates
`dataset.yaml` with the destination path. It also excludes photos with conflicting
source species assignments. Increasing batch size is optional after measuring GPU
memory. Do not run separate large teacher processes concurrently on the GPU.

## Remaining work

1. Finish public labels and inspect class coverage, unknown rate, label quality,
   and duplicate/leakage risks. Collect more examples where support is inadequate.
2. Confirm the actual poultry candidate list; regenerate and review private labels.
   Temporal smoothing must stay within an object sequence, not a time bucket.
3. Train with `distill/train_student.py --device 0 --dataset work/dataset-production/dataset.yaml`.
   The default is YOLO11 Small, 640-pixel inputs, 60 epochs with early stopping.
4. Evaluate against independently reviewed held-out day/night camera events.
   Teacher agreement alone is not a production accuracy measurement.
5. Export CoreML and OpenVINO via `distill/export_scrypted.py --backends coreml openvino`.
   The checkpoint's class order must match the catalog before export is allowed.
6. On the Mac, run `distill/validate_exports.py WEIGHTS --images VALIDATION_IMAGES`.
   This checks declared files, actual runtime inference, raw YOLO output shapes,
   finite probabilities, class maps, and parity with the PyTorch checkpoint.
7. Check real Scrypted integration and latency on the deployment hosts, publish
   tested artifacts, and update the draft PR with measured accuracy and limitations.

Deployment labels remain common names only. Scientific names belong only in the
teacher prompts and catalog metadata. Do not call this production-ready until
accuracy, runtime, and installation checks pass.
