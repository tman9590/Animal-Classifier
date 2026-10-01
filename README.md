# SpeciesNet Animal Classifier for Scrypted

This repository packages Google's **SpeciesNet v4.0.3a** crop classifier for
every Scrypted backend that currently supports repository-based custom models:

| Scrypted backend | Typical hardware | Artifact |
| --- | --- | --- |
| ONNX | NVIDIA GPU, Windows/Linux CPU | ONNX |
| OpenVINO | Intel/AMD CPU, GPU, or NPU | OpenVINO IR |
| CoreML | Apple Silicon Neural Engine | Core ML package |
| NCNN | Vulkan-capable GPU | NCNN |

TensorFlow Lite is not listed because its Scrypted plugin does not currently
expose the custom GitHub-model loader used by these four plugins.

## Why SpeciesNet

The original request pointed to an animal re-identification dataset catalog.
Re-identification answers “is this the same individual?” and is a poor fit for
Scrypted's animal-classifier slot, which expects a fixed class label for an
already detected animal crop.

SpeciesNet is the better fit: Google trained it on more than 65 million
geographically diverse camera-trap images. The v4.0.3a “always crop” classifier
expects a tightly cropped 480 × 480 animal image, matching the crop Scrypted
passes to an animal classifier. It returns 2,498 taxonomy labels, including
2,066 species plus useful higher taxa, `blank`, `human`, and `vehicle`.

This is a second-stage classifier. Scrypted's selected object detector still
finds the animal and supplies the crop; SpeciesNet adds the species label used
for NVR search and notifications.

## Install in Scrypted

1. Install the detector plugin suited to the server: ONNX, OpenVINO, CoreML,
   or NCNN.
2. Open that plugin, find **Models**, and choose **Create Device**.
3. Name it `SpeciesNet Animals`.
4. Paste the configuration URL for the plugin you opened:

   - ONNX: `https://media.githubusercontent.com/media/tman9590/scrypted-animal-classifier/main/models/onnx/config.json`
   - OpenVINO: `https://media.githubusercontent.com/media/tman9590/scrypted-animal-classifier/main/models/openvino/config.json`
   - CoreML: `https://media.githubusercontent.com/media/tman9590/scrypted-animal-classifier/main/models/coreml/config.json`
   - NCNN: `https://media.githubusercontent.com/media/tman9590/scrypted-animal-classifier/main/models/ncnn/config.json`

5. Select the new device as the camera's **Animal Classifier** in Scrypted NVR.
6. Start with a classification threshold of `0.50`, then review real day and
   night events before using species labels in alerts.

Use the `media.githubusercontent.com` URL exactly as shown. The weights are in
Git LFS; GitHub's ordinary raw URLs return tiny LFS pointer files instead of the
model bytes. Each backend directory contains its own manifest and only the
files that its loader understands.

## Validation

The checked-in ONNX, OpenVINO, and NCNN artifacts passed source/export parity
on ten official SpeciesNet test images. The validator checks artifact hashes,
finite output, top-1 agreement at Scrypted's default threshold, top-5 overlap,
and maximum probability drift. CoreML package structure and model I/O are
checked cross-platform; CoreML inference validation requires macOS.

```sh
python distill/validate_speciesnet_scrypted.py \
  --images /path/to/cropped/validation/images \
  --samples 20

python -m unittest discover -s tests -v
python -m compileall -q distill tests
```

Conversion parity is not site-specific accuracy. Before automating alerts,
measure the model on reviewed crops from the installed cameras, including
infrared/night, rain, blur, partial animals, and common local confusion pairs.

## Rebuild all backends

Use Python 3.11. The build downloads the official checkpoint from
`google/speciesnet/pyTorch/v4.0.3a/1` and preserves the upstream output order.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-build.txt
.venv/bin/python distill/speciesnet_scrypted.py --no-compression
```

`--no-compression` keeps the validated FP16/FP32 conversions. Experimental
8-bit weight compression caused material confidence drift and is not used.
Large artifacts are stored with Git LFS.

## License and attribution

Repository code is Apache-2.0. The converted model comes from Google's
Apache-2.0 SpeciesNet project. See [MODEL_CARD.md](MODEL_CARD.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for intended use, limitations,
and upstream attribution.
