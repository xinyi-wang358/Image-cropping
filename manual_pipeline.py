from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from app.schemas import BoundingBox


DEFAULT_BBOX = BoundingBox(x1=231, y1=39, x2=739, y2=907)


def encode_png_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def fixed_detection_for_image(image_bytes: bytes) -> dict[str, object]:
    with Image.open(BytesIO(image_bytes)) as opened:
        width, height = opened.size

    bbox = clamp_bbox(DEFAULT_BBOX, width, height)
    return {
        "source": "manual_fixed",
        "fallback_reason": "auto_pipeline_disabled",
        "bbox_detector": bbox.model_dump(),
        "bbox_qwen": None,
        "bbox_final": bbox.model_dump(),
        "waist_y": bbox.y1,
        "hem_y": bbox.y2,
        "confidence": 0.99,
        "detector_debug": {
            "mode": "fixed_bbox",
            "note": "GroundingDINO and Qwen are disabled in this mode.",
        },
    }


def clamp_bbox(box: BoundingBox, width: int, height: int) -> BoundingBox:
    x1 = max(0, min(width - 1, box.x1))
    y1 = max(0, min(height - 1, box.y1))
    x2 = max(1, min(width, box.x2))
    y2 = max(1, min(height, box.y2))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def crop_with_bbox(image_bytes: bytes, box: BoundingBox) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        clamped = clamp_bbox(box, width, height)
        return image.crop((clamped.x1, clamped.y1, clamped.x2, clamped.y2))


def decode_data_url_image(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("Invalid data url.")

    raw = base64.b64decode(data_url.split(",", 1)[1])
    return Image.open(BytesIO(raw)).convert("RGB")


def stitch_images(images: list[Image.Image]) -> Image.Image:
    if len(images) != 3:
        raise ValueError("Need exactly 3 cropped images.")

    min_height = min(image.height for image in images)
    resized = [
        image.resize((round(image.width * (min_height / image.height)), min_height), Image.Resampling.LANCZOS)
        for image in images
    ]

    total_width = sum(image.width for image in resized)
    stitched = Image.new("RGB", (total_width, min_height), "white")

    x = 0
    for image in resized:
        stitched.paste(image, (x, 0))
        x += image.width

    return stitched
