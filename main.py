from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.manual_pipeline import (
    crop_with_bbox,
    decode_data_url_image,
    encode_png_data_url,
    fixed_detection_for_image,
    stitch_images,
)
from app.schemas import BoundingBox


app = FastAPI(title="Pants 3-Image Composer")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/debug/annotate")
async def debug_annotate(image: UploadFile = File(...), bbox: str | None = Form(default=None)) -> dict[str, object]:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is required.")

    detection = fixed_detection_for_image(image_bytes)
    if bbox:
        detection["bbox_final"] = BoundingBox(**json.loads(bbox)).model_dump()
        detection["waist_y"] = detection["bbox_final"]["y1"]
        detection["hem_y"] = detection["bbox_final"]["y2"]

    return {"ok": True, "detection": detection}


@app.post("/debug/crop")
async def debug_crop(image: UploadFile = File(...), bbox: str = Form(...)) -> dict[str, object]:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is required.")

    try:
        box = BoundingBox(**json.loads(bbox))
        cropped = crop_with_bbox(image_bytes, box)
        encoded = encode_png_data_url(cropped)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Crop failed: {exc}") from exc

    return {"ok": True, "cropped_image_data_url": encoded}


@app.post("/debug/stitch")
async def debug_stitch(payload: dict[str, list[str]]) -> dict[str, object]:
    images = payload.get("images", [])
    if len(images) != 3:
        raise HTTPException(status_code=400, detail="Need exactly 3 cropped images.")

    try:
        decoded = [decode_data_url_image(item) for item in images]
        stitched = stitch_images(decoded)
        encoded = encode_png_data_url(stitched)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Stitch failed: {exc}") from exc

    return {"ok": True, "stitched_image_data_url": encoded}
