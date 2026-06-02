from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.width() * self.height()


class DetectionResult(BaseModel):
    bbox: BoundingBox
    waist_y: int = Field(ge=0)
    hem_y: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class FusionResult(BaseModel):
    source: str
    fallback_reason: str | None = None
    bbox_detector: BoundingBox
    bbox_qwen: BoundingBox | None
    bbox_final: BoundingBox
    waist_y: int
    hem_y: int
    confidence: float
    detector_debug: dict | None = None
