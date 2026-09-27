from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from catalog import Species


MDV6_COMPACT_URL = "https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1"


class MegaDetector:
    def __init__(self, weights: Path, device: str = ""):
        from ultralytics import YOLO

        if not weights.exists():
            raise FileNotFoundError(
                f"Missing {weights}. Download the official MDV6-yolov10-c weights from {MDV6_COMPACT_URL}"
            )
        self.model = YOLO(weights)
        self.device = device or None

    def detect(self, image: Image.Image, threshold: float) -> list[tuple[tuple[float, float, float, float], float]]:
        return self.detect_many([image], threshold)[0]

    def detect_many(
        self, images: list[Image.Image], threshold: float
    ) -> list[list[tuple[tuple[float, float, float, float], float]]]:
        if not images:
            return []
        results = self.model.predict(
            [np.asarray(image.convert("RGB"))[:, :, ::-1].copy() for image in images],
            conf=threshold,
            imgsz=1280,
            device=self.device,
            verbose=False,
        )
        return [self._animal_boxes(result) for result in results]

    @staticmethod
    def _animal_boxes(result) -> list[tuple[tuple[float, float, float, float], float]]:
        ret = []
        for box, class_id, confidence in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu(), result.boxes.conf.cpu()):
            if int(class_id) == 0:
                ret.append((tuple(float(value) for value in box), float(confidence)))
        return ret


class BioClipClassifier:
    def __init__(self, species: list[Species], device: str = "cpu", cache_dir: Path | None = None):
        import open_clip
        import torch

        self.species = species
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.model = self.model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2", cache_dir=str(cache_dir) if cache_dir else None)
        prompts = [prompt for item in species for prompt in item.prompts]
        with torch.inference_mode():
            features = self.model.encode_text(tokenizer(prompts).to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.reshape(len(species), 2, -1).mean(dim=1)
            self.text_features = features / features.norm(dim=-1, keepdim=True)

    def classify(self, crops: list[Image.Image], batch_size: int = 64) -> list[dict[str, float]]:
        import torch

        if not crops:
            return []
        precision = torch.autocast("cuda", dtype=torch.float16) if self.device.startswith("cuda") else nullcontext()
        output = []
        for offset in range(0, len(crops), batch_size):
            inputs = torch.stack(
                [self.preprocess(crop.convert("RGB")) for crop in crops[offset : offset + batch_size]]
            ).to(self.device)
            with torch.inference_mode(), precision:
                features = self.model.encode_image(inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                scale = self.model.logit_scale.exp().clamp(max=100)
                probabilities = (scale * features @ self.text_features.T).softmax(dim=-1).float().cpu()
            output.extend(
                {item.label: float(row[index]) for index, item in enumerate(self.species)}
                for row in probabilities
            )
        return output


def crop_box(image: Image.Image, box: tuple[float, float, float, float], margin: float = 0.1) -> Image.Image:
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * margin, (y2 - y1) * margin
    return image.crop(
        (
            max(0, int(x1 - px)),
            max(0, int(y1 - py)),
            min(image.width, int(x2 + px)),
            min(image.height, int(y2 + py)),
        )
    )
