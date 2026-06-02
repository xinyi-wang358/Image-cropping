from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from app.schemas import DetectionResult
from app.settings import settings


@dataclass
class ProcessedPanel:
    image: Image.Image
    detection: DetectionResult
    scale: float


def _resize_image(image: Image.Image, scale: float) -> Image.Image:
    new_width = max(1, round(image.width * scale))
    new_height = max(1, round(image.height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _fit_panel(image: Image.Image, detection: DetectionResult) -> ProcessedPanel:
    body_height = max(1, detection.hem_y - detection.waist_y)
    target_body_height = settings.target_hem_y - settings.target_waist_y
    scale = target_body_height / body_height

    resized = _resize_image(image, scale)

    scaled_bbox_center_x = ((detection.bbox.x1 + detection.bbox.x2) / 2.0) * scale
    scaled_waist_y = detection.waist_y * scale

    left = round(scaled_bbox_center_x - (settings.slot_width / 2))
    top = round(scaled_waist_y - settings.target_waist_y)

    crop_left = left
    crop_top = top
    crop_right = left + settings.slot_width
    crop_bottom = top + settings.slot_height

    crop_box = (
        max(0, crop_left),
        max(0, crop_top),
        min(resized.width, crop_right),
        min(resized.height, crop_bottom),
    )
    panel = resized.crop(crop_box)

    fixed = Image.new("RGB", (settings.slot_width, settings.slot_height), "white")
    paste_x = max(0, -crop_left)
    paste_y = max(0, -crop_top)
    fixed.paste(panel, (paste_x, paste_y))
    panel = fixed

    return ProcessedPanel(image=panel, detection=detection, scale=scale)


def build_composite(images: list[bytes], detections: list[DetectionResult]) -> bytes:
    if len(images) != 3 or len(detections) != 3:
        raise ValueError("Exactly 3 images and 3 detections are required.")

    panels: list[ProcessedPanel] = []
    for image_bytes, detection in zip(images, detections, strict=True):
        with Image.open(BytesIO(image_bytes)) as opened_image:
            image = opened_image.convert("RGB")
        panels.append(_fit_panel(image, detection))

    composite = Image.new("RGB", (settings.target_width, settings.target_height))
    for index, panel in enumerate(panels):
        composite.paste(panel.image, (index * settings.slot_width, 0))

    output = BytesIO()
    composite.save(output, format="PNG")
    return output.getvalue()


def draw_detection_box(image_bytes: bytes, detection: DetectionResult) -> bytes:
    with Image.open(BytesIO(image_bytes)) as opened_image:
        image = opened_image.convert("RGBA")

    canvas = image.copy()
    width, height = canvas.size

    left = max(0, min(width - 1, detection.bbox.x1))
    top = max(0, min(height - 1, detection.bbox.y1))
    right = max(0, min(width - 1, detection.bbox.x2))
    bottom = max(0, min(height - 1, detection.bbox.y2))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    # Semi-transparent red fill + clear red border for product-body highlighting.
    draw.rectangle([left, top, right, bottom], fill=(220, 38, 38, 70), outline=(239, 68, 68, 255), width=4)

    output = BytesIO()
    canvas.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def draw_detection_box_on_image(image: Image.Image, detection: DetectionResult) -> Image.Image:
    canvas = image.convert("RGBA").copy()
    width, height = canvas.size

    left = max(0, min(width - 1, detection.bbox.x1))
    top = max(0, min(height - 1, detection.bbox.y1))
    right = max(0, min(width - 1, detection.bbox.x2))
    bottom = max(0, min(height - 1, detection.bbox.y2))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([left, top, right, bottom], fill=(220, 38, 38, 70), outline=(239, 68, 68, 255), width=4)
    return canvas.convert("RGB")
