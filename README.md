# North Carolina Species Detector for Scrypted

This repository builds a single-stage species detector that can be loaded by
Scrypted's **current Object Detection plugin**. The final deployed model uses
the existing Scrypted custom-model contract (`model: yolov9`); it does not
require a separate Scrypted plugin.

> **Build status:** the regional catalog and complete teacher, training, and
> export pipeline are ready. Exported Scrypted weights are intentionally not
> present on this development branch until representative camera clips have
> been pseudo-labeled and the student has passed held-out validation.

## Species coverage

The checked-in North Carolina catalog contains **529 animal taxa plus
`unknown`**. Every deployed result is a concise species-level common name, for
example:

```text
White-tailed Deer
Northern Cardinal
Domestic Chicken
```

The list includes 327 birds, 52 mammals, 61 reptiles, and 89 amphibians that
have at least 20 research-grade iNaturalist observations in North Carolina.
Poultry is explicitly retained even when it falls below that threshold:
chickens, turkeys, domestic and Muscovy ducks, domestic geese, guineafowl,
quail, peafowl, pheasants, domestic pigeons, emus, and ostriches.

The source catalog is [`species/north-carolina.json`](species/north-carolina.json).

## How the model is built

The requested multi-stage system is used as a teacher to create training labels:

```text
North Carolina images and camera clips
  -> MegaDetector V6 animal boxes
  -> BioCLIP 2 classification against the North Carolina species list
  -> IoU track association and temporal probability smoothing for video
  -> unknown when the smoothed score is below the threshold
  -> train a single YOLO species detector
  -> export ONNX, OpenVINO, CoreML, and NCNN for Scrypted
```

Scrypted's current custom-model loader accepts a single stateless YOLO or
ResNet graph. It cannot run network requests, change BioCLIP text embeddings,
or preserve probability history inside a model invocation. Distillation keeps
the full teacher pipeline in model creation and produces the one graph that the
current plugin can load. At runtime the trained model directly returns species
boxes, including an `unknown` class learned from uncertain teacher results.

## Build workflow

Use Python 3.11. Model weights and training data are written under ignored
`work/` directories.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-build.txt

# Rebuild the regional list from iNaturalist place 30 (North Carolina).
.venv/bin/python distill/build_catalog.py \
  --output species/north-carolina.json

# Download CC0/CC-BY observations and retain attribution metadata.
.venv/bin/python distill/download_inaturalist.py \
  --catalog species/north-carolina.json \
  --per-species 20

# Download MDV6-yolov9-c.pt from the official MegaDetector V6 model record
# into work/models/, then label public images.
.venv/bin/python distill/label_images.py \
  --catalog species/north-carolina.json

# Add representative camera clips. This path performs temporal smoothing.
.venv/bin/python distill/label_videos.py /path/to/camera-clips

# Scrypted NVR event-frame exports can be added directly. Consecutive frames in
# each event directory use the same temporal smoothing and unknown threshold.
.venv/bin/python distill/label_scrypted_frames.py /path/to/scrypted-event-frames

# Train and export the deployable single-stage model.
.venv/bin/python distill/train_student.py
.venv/bin/python distill/export_scrypted.py \
  work/training/north-carolina-wildlife/weights/best.pt
```

The downloader preserves existing image files and atomically checkpoints attribution
after each completed species. Rerun the same command after an interruption to
recover missing attribution and retry failed species requests. Public images
without recovered attribution are not used by the labeling pipeline.

For a bounded teacher smoke test, run `distill/label_images.py --limit 8
--output work/public-smoke-dataset` with the build environment's Python.

The default teacher settings are:

- MegaDetector V6 compact threshold: `0.25`
- BioCLIP 2 unknown threshold: `0.25`
- New-frame smoothing weight: `0.35`
- Track association IoU: `0.30`
- Student: YOLO11 Small at `640 × 640`

The Scrypted frame importer can restrict known coop feeds to the configured
poultry species. This prevents infrared chicken imagery from being assigned to
visually similar wild birds while leaving other cameras on the full regional
species list.

## Add the exported model to Scrypted

1. Install the Scrypted detection backend for the server: ONNX for NVIDIA,
   OpenVINO for Intel/AMD, CoreML for Apple Silicon, or NCNN where appropriate.
2. Open the backend plugin and choose **Create Device** under **Models**.
3. Set the model name to `North Carolina Species Detector`.
4. Enter this repository URL:
   `https://github.com/tman9590/scrypted-animal-classifier`.
5. Select the created model for the camera's Scrypted NVR object detection.

Scrypted resolves the repository URL to `models/<backend>/config.json`. Those
configs identify the output as YOLO-compatible and provide the complete class
map. The CoreML and OpenVINO configs are checked in and can be regenerated from
the catalog with `python3 distill/generate_configs.py`. Scientific names remain
in the catalog and BioCLIP prompts but are not shown in Scrypted detections.

## Validation

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q distill tests
```

Before using notifications, validate the exported student against held-out
day, night, infrared, rain, and motion-blurred clips from the installed
cameras. See [`MODEL_CARD.md`](MODEL_CARD.md) for decision limits.

## Licenses

Repository code is Apache-2.0. BioCLIP 2 weights are MIT licensed. The requested
MegaDetector V6 YOLOv10 compact weights are AGPL-3.0. Downloaded iNaturalist
photos are restricted to CC0 and CC-BY, and the downloader writes an
`attribution.csv` alongside them.
