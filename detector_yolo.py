from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image
from ultralytics import YOLO

from app.schemas import BoundingBox


PANTS_CLASS_NAMES = {
    "pants",
    "trousers",
    "jeans",
    "leggings",
}


@dataclass
class CandidateBox:
    bbox: BoundingBox
    score: float
    class_name: str

    @property
    def area(self) -> int:
        return max(0, self.bbox.x2 - self.bbox.x1) * max(0, self.bbox.y2 - self.bbox.y1)


class YoloDetector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model = YOLO(model_name)

    def detect(self, image_bytes: bytes) -> list[CandidateBox]:
        with Image.open(BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
        results = self.model.predict(rgb, verbose=False)
        if not results:
            return []

        res = results[0]
        names = res.names
        boxes = res.boxes
        if boxes is None:
            return []

        candidates: list[CandidateBox] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = str(names.get(cls_id, cls_id)).lower()
            score = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            candidate = CandidateBox(
                bbox=BoundingBox(x1=round(x1), y1=round(y1), x2=round(x2), y2=round(y2)),
                score=score,
                class_name=cls_name,
            )
            candidates.append(candidate)
        return candidates

    @staticmethod
    def select_primary_box(candidates: list[CandidateBox]) -> CandidateBox | None:
        pants = [c for c in candidates if c.class_name in PANTS_CLASS_NAMES]
        pool = pants if pants else candidates
        if not pool:
            return None
        return sorted(pool, key=lambda c: (c.area, c.score), reverse=True)[0]
