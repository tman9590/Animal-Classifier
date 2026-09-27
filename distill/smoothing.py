from __future__ import annotations

from dataclasses import dataclass, field


Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass
class Track:
    track_id: int
    box: Box
    last_frame: int
    probabilities: dict[str, float] = field(default_factory=dict)


class TemporalSmoother:
    def __init__(self, alpha: float, match_iou: float, ttl_frames: int, unknown_threshold: float):
        self.alpha = alpha
        self.match_iou = match_iou
        self.ttl_frames = ttl_frames
        self.unknown_threshold = unknown_threshold
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, frame: int, observations: list[tuple[Box, dict[str, float]]]) -> list[tuple[int, str, float]]:
        self.tracks = [track for track in self.tracks if frame - track.last_frame <= self.ttl_frames]
        available = set(range(len(self.tracks)))
        outputs = []
        for box, probabilities in observations:
            best = None
            best_overlap = self.match_iou
            for index in available:
                overlap = iou(box, self.tracks[index].box)
                if overlap >= best_overlap:
                    best, best_overlap = index, overlap
            if best is None:
                track = Track(self.next_id, box, frame, dict(probabilities))
                self.next_id += 1
                self.tracks.append(track)
            else:
                available.remove(best)
                track = self.tracks[best]
                labels = set(track.probabilities) | set(probabilities)
                track.probabilities = {
                    label: (1 - self.alpha) * track.probabilities.get(label, 0.0)
                    + self.alpha * probabilities.get(label, 0.0)
                    for label in labels
                }
                total = sum(track.probabilities.values())
                if total:
                    track.probabilities = {key: value / total for key, value in track.probabilities.items()}
                track.box = box
                track.last_frame = frame
            label, confidence = max(track.probabilities.items(), key=lambda item: item[1], default=("unknown", 0.0))
            if confidence < self.unknown_threshold:
                label = "unknown"
            outputs.append((track.track_id, label, confidence))
        return outputs
