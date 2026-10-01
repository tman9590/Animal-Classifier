# Third-party notices

## Google SpeciesNet

The model artifacts under `models/` are conversions of Google SpeciesNet
v4.0.3a:

- Source: https://github.com/google/cameratrapai
- Checkpoint: `google/speciesnet/pyTorch/v4.0.3a/1`
- Copyright: Google and SpeciesNet contributors
- License: Apache License 2.0

The upstream license is available at
https://github.com/google/cameratrapai/blob/main/LICENSE.

The conversion adds an NCHW-to-NHWC input adapter and exports the same
classifier to ONNX, OpenVINO, CoreML, and NCNN. Taxonomy outputs are not removed
or reordered.
