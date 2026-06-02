from __future__ import annotations

import base64
import json
from io import BytesIO

from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

from app.schemas import BoundingBox, DetectionResult, FusionResult
from app.settings import settings


class QwenVisionClient:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)

    @staticmethod
    def _image_to_data_url(image_bytes: bytes) -> str:
        with Image.open(BytesIO(image_bytes)) as image:
            fmt = (image.format or "png").lower()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{fmt};base64,{encoded}"

    @staticmethod
    def _expand_box(box: BoundingBox, width: int, height: int, ratio: float = 0.08) -> BoundingBox:
        dx = round(box.width() * ratio)
        dy = round(box.height() * ratio)
        return BoundingBox(
            x1=max(0, box.x1 - dx),
            y1=max(0, box.y1 - dy),
            x2=min(width, box.x2 + dx),
            y2=min(height, box.y2 + dy),
        )

    def _detect_on_crop(self, crop_bytes: bytes) -> DetectionResult:
        prompt = """
只返回 JSON。你只看到裤子主体候选区域，请在这个局部图中输出紧贴裤子主体的检测结果。
输出字段：bbox{x1,y1,x2,y2}, waist_y, hem_y, confidence。
要求：bbox 不要贴背景，hem_y 在裤脚上方；不要输出解释。
""".strip()
        raw = self._request_detection(crop_bytes, prompt)
        return self._parse_detection(raw)

    def detect_full_image(self, image_bytes: bytes) -> DetectionResult:
        prompt = """
只返回 JSON。请在整张图里识别裤子主体，输出紧贴裤子主体的 bbox，并给出 waist_y、hem_y、confidence。
输出字段：bbox{x1,y1,x2,y2}, waist_y, hem_y, confidence。
约束：只框裤子，不要框背景、衣架、立柱、地面或上衣。
""".strip()
        raw = self._request_detection(image_bytes, prompt)
        return self._parse_detection(raw)

    def _request_detection(self, image_bytes: bytes, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=settings.qwen_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是视觉定位器，只返回有效 JSON。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_bytes)}},
                    ],
                },
            ],
        )
        return response.choices[0].message.content or "{}"

    def _parse_detection(self, raw: str) -> DetectionResult:
        try:
            parsed = self._normalize_bbox_payload(json.loads(raw))
            return DetectionResult.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Unexpected Qwen response: {raw}") from exc

    @staticmethod
    def _normalize_bbox_payload(payload: dict) -> dict:
        bbox = payload.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            payload["bbox"] = {
                "x1": round(bbox[0]),
                "y1": round(bbox[1]),
                "x2": round(bbox[2]),
                "y2": round(bbox[3]),
            }
        return payload

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = a.area() + b.area() - inter
        return inter / union if union > 0 else 0.0

    def refine_with_detector(self, image_bytes: bytes, detector_bbox: BoundingBox) -> FusionResult:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            roi = self._expand_box(detector_bbox, width, height)
            crop = rgb.crop((roi.x1, roi.y1, roi.x2, roi.y2))

        crop_buffer = BytesIO()
        crop.save(crop_buffer, format="PNG")
        qwen_detection = self._detect_on_crop(crop_buffer.getvalue())

        mapped_qwen = BoundingBox(
            x1=roi.x1 + qwen_detection.bbox.x1,
            y1=roi.y1 + qwen_detection.bbox.y1,
            x2=roi.x1 + qwen_detection.bbox.x2,
            y2=roi.y1 + qwen_detection.bbox.y2,
        )
        iou = self._iou(detector_bbox, mapped_qwen)
        area_ratio = mapped_qwen.area() / max(1, detector_bbox.area())

        if iou < 0.15 or area_ratio < 0.45 or area_ratio > 1.8:
            reason = "low_iou" if iou < 0.15 else "abnormal_area_ratio"
            return FusionResult(
                source="fallback",
                fallback_reason=reason,
                bbox_detector=detector_bbox,
                bbox_qwen=mapped_qwen,
                bbox_final=detector_bbox,
                waist_y=detector_bbox.y1,
                hem_y=detector_bbox.y2,
                confidence=min(0.5, qwen_detection.confidence),
            )

        return FusionResult(
            source="qwen_refined",
            fallback_reason=None,
            bbox_detector=detector_bbox,
            bbox_qwen=mapped_qwen,
            bbox_final=mapped_qwen,
            waist_y=roi.y1 + qwen_detection.waist_y,
            hem_y=roi.y1 + qwen_detection.hem_y,
            confidence=qwen_detection.confidence,
        )
