from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from app.schemas import BoundingBox
from app.settings import settings


TEXT_QUERIES = ["pants", "trousers", "jeans", "leggings"]


@dataclass
class CandidateBox:
    bbox: BoundingBox
    score: float
    label: str

    @property
    def area(self) -> int:
        return max(0, self.bbox.x2 - self.bbox.x1) * max(0, self.bbox.y2 - self.bbox.y1)


class GroundingDinoDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect(self, image_bytes: bytes) -> list[CandidateBox]:
        with Image.open(BytesIO(image_bytes)) as img:
            image = img.convert("RGB")

        text = ". ".join(TEXT_QUERIES) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=settings.dino_box_threshold,
            text_threshold=settings.dino_text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        candidates: list[CandidateBox] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"], strict=True):
            x1, y1, x2, y2 = [round(v) for v in box.tolist()]
            candidates.append(
                CandidateBox(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=float(score.item()),
                    label=str(label).lower(),
                )
            )
        return candidates

    @staticmethod
    def select_primary_box(candidates: list[CandidateBox], image_w: int, image_h: int) -> tuple[CandidateBox | None, dict]:
        debug = {
            "thresholds": {
                "box_threshold": settings.dino_box_threshold,
                "text_threshold": settings.dino_text_threshold,
                "min_area_ratio": settings.dino_min_area_ratio,
                "max_area_ratio": settings.dino_max_area_ratio,
                "min_aspect_ratio": settings.dino_min_aspect_ratio,
                "max_aspect_ratio": settings.dino_max_aspect_ratio,
                "center_max_offset_ratio": settings.dino_center_max_offset_ratio,
            },
            "candidate_count": len(candidates),
            "filtered_by_ratio": 0,
            "filtered_by_area": 0,
            "filtered_by_center": 0,
            "kept_count": 0,
        }
        if not candidates:
            return None, debug

        img_area = image_w * image_h
        cx0, cy0 = image_w / 2.0, image_h / 2.0
        kept: list[CandidateBox] = []
        for c in candidates:
            w = max(1, c.bbox.width())
            h = max(1, c.bbox.height())
            ar = w / h
            area_ratio = c.area / max(1, img_area)
            ccx = (c.bbox.x1 + c.bbox.x2) / 2.0
            ccy = (c.bbox.y1 + c.bbox.y2) / 2.0
            off = ((abs(ccx - cx0) / image_w) + (abs(ccy - cy0) / image_h)) / 2.0

            if ar < settings.dino_min_aspect_ratio or ar > settings.dino_max_aspect_ratio:
                debug["filtered_by_ratio"] += 1
                continue
            if area_ratio < settings.dino_min_area_ratio or area_ratio > settings.dino_max_area_ratio:
                debug["filtered_by_area"] += 1
                continue
            if off > settings.dino_center_max_offset_ratio:
                debug["filtered_by_center"] += 1
                continue
            kept.append(c)

        debug["kept_count"] = len(kept)
        pool = kept if kept else candidates
        selected = sorted(pool, key=lambda c: (c.score * (1 + 0.35 * min(1.0, c.area / max(1, img_area))), c.area), reverse=True)[0]
        return selected, debug
