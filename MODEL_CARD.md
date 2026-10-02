# Model card: North Carolina SpeciesNet v4.0.3a for Scrypted

## Intended use

This package classifies an animal crop supplied by Scrypted. It does not detect
or localize animals by itself. Labels are intended for NVR metadata, search,
and notification assistance.

## Source model and training domain

- Upstream: Google SpeciesNet v4.0.3a, always-crop variant
- Architecture: EfficientNet V2 M
- Input: one tightly cropped 480 × 480 RGB image
- Upstream output: 2,498 logits
- Exported output: 457 North Carolina-scoped logits
- Training domain: more than 65 million geographically diverse camera-trap
  images, including Wildlife Insights and public repositories

Google reports that the full SpeciesNet ensemble found 99.4% of held-out
images containing animals, produced species-level labels for 83%, and was
correct on 94.5% of those species-level predictions. Those figures describe
the upstream ensemble and test set, not this Scrypted conversion or a specific
home-camera deployment.

The exported vocabulary includes 298 locally plausible or explicitly
supported livestock species, their useful higher taxa, and human, vehicle,
and blank. It is derived from Google's official `USA`/`NC` allow/block rules,
plus an explicit override for common livestock. Other species without an
upstream range rule are excluded. Four rows without common names use
scientific names. Duplicate
common names are disambiguated without changing relative output order.

## Scrypted conversion

Scrypted converts RGB crops to NCHW float values in `[0, 1]`. A wrapper
permutes that tensor to the NHWC layout expected by SpeciesNet, then selects
the 457 retained logits before Scrypted applies softmax. The four
manifests use Scrypted's `resnet` classifier parser with identity
normalization. Artifacts are supplied for ONNX, OpenVINO, CoreML, and NCNN.

The geographic selection is fixed to North Carolina at build time. Scrypted
does not need camera coordinates, but the same artifacts should not be used
outside North Carolina. Seasonal range is not modeled.

## Validation status

ONNX, OpenVINO, and NCNN were compared with the source PyTorch checkpoint on
ten held-out animal crops. A build is rejected when a prediction at
or above Scrypted's default `0.50` threshold changes, top-five overlap falls
below four, probability drift exceeds `0.10`, output contains non-finite
values, or a configured artifact is absent. CoreML package integrity and model
I/O were validated on Windows; runtime inference validation requires macOS.

The saved report is build evidence, not a benchmark. Deployments should keep a
labeled holdout of their own Scrypted crops and report per-class precision and
recall across cameras and lighting conditions.

## Limitations

- Similar species, domestic breeds, juveniles, partial animals, and tiny crops
  can be confused.
- Residential cameras differ from camera traps in mounting, compression,
  infrared illumination, motion blur, and subject distance.
- The upstream geofence can lag range changes and does not guarantee presence
  at a particular North Carolina site.
- A result may be a family/order label rather than a species when the image is
  ambiguous.
- Softmax confidence is not a calibrated biological probability.
- The classifier cannot count or distinguish individual animals.

Use results as search metadata and alert assistance, not as scientific evidence
or for safety-critical wildlife decisions.
