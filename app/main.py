from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from app.detector import detect_objects, capture_from_webcam
from app.database import (
    save_detection,
    get_latest,
    get_history,
    get_statistics,
    get_detection_by_sequence,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER    = "uploads"
DETECTION_FOLDER = "detections"
CAPTURE_INTERVAL = 20   # seconds

os.makedirs(UPLOAD_FOLDER,    exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Temporal smoothing globals
active_source = "webcam"
capture_task = None
last_countdown = None
last_color = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global capture_task
    capture_task = asyncio.create_task(periodic_capture())
    logger.info("Periodic capture started (every %ds)", CAPTURE_INTERVAL)
    yield
    if capture_task:
        capture_task.cancel()
        logger.info("Periodic capture stopped")


app = FastAPI(
    title="SafeV Camera System",
    description="Traffic Light Detection System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/detections", StaticFiles(directory="detections"), name="detections")
app.mount("/uploads",    StaticFiles(directory="uploads"),    name="uploads")


def _apply_temporal_smoothing(result):
    """
    Keep the last stable color for brief misses, but do not fabricate countdowns.
    Mutates result in-place and returns it.
    """
    global last_countdown, last_color

    tl = result["traffic_light"]
    new_cd = tl["countdown"]
    if new_cd is not None:
        last_countdown = new_cd
        last_color = tl["color"]
    else:
        result["traffic_light"]["countdown"] = None
        result["traffic_light"]["countdown_detected"] = False
        if last_color != "unknown":
            result["traffic_light"]["color"] = last_color

    if tl["color"] != "unknown":
        last_color = tl["color"]

    return result


async def periodic_capture():
    while True:
        try:
            logger.info("Capturing at %s", datetime.now())
            if active_source == "webcam":
                image_path, seq_num = capture_from_webcam()
                result = detect_objects(image_path)
                result = _apply_temporal_smoothing(result)
                save_detection(
                    result["id"],
                    result["image_path"],
                    result["original_image"],
                    result,
                )
                logger.info("Detection #%d complete – TL=%s cd=%s",
                            result['id'],
                            result['traffic_light']['color'],
                            result['traffic_light']['countdown'])
        except Exception as exc:
            logger.error("Periodic capture error: %s", exc)
        await asyncio.sleep(CAPTURE_INTERVAL)


@app.get("/")
async def root():
    return {
        "name":             "SafeV Traffic Light Detection",
        "version":          "2.0.0",
        "capture_interval": f"{CAPTURE_INTERVAL}s",
        "active_source":    active_source,
        "endpoints": {
            "GET /":                    "This info",
            "GET /webcam/capture":      "Manual capture",
            "GET /latest":              "Latest detection",
            "GET /history":             "Detection history",
            "GET /stats":               "Statistics",
            "GET /detection/{seq_num}": "Detection by sequence",
        },
    }


@app.get("/webcam/capture")
async def webcam_capture():
    global last_countdown, last_color
    try:
        image_path, seq_num = capture_from_webcam()
        result = detect_objects(image_path)
        result = _apply_temporal_smoothing(result)
        save_detection(result["id"], result["image_path"],
                       result["original_image"], result)
        result["image_url"]    = f"/detections/{result['id']}.jpg"
        result["original_url"] = f"/uploads/{result['id']}.jpg"
        return result
    except Exception as exc:
        logger.error("Manual capture failed: %s", exc)
        raise HTTPException(500, f"Capture failed: {exc}")


@app.get("/latest")
async def latest_detection():
    detection = get_latest()
    if detection is None:
        return {"message": "No detections yet"}
    detection["image_url"]    = f"/detections/{detection['sequence_number']}.jpg"
    detection["original_url"] = f"/uploads/{detection['sequence_number']}.jpg"
    return detection


@app.get("/history")
async def detection_history(limit: int = 50):
    history = get_history(limit)
    for item in history:
        item["image_url"]    = f"/detections/{item['sequence_number']}.jpg"
        item["original_url"] = f"/uploads/{item['sequence_number']}.jpg"
    return {"total": len(history), "detections": history}


@app.get("/detection/{seq_num}")
async def get_detection(seq_num: int):
    detection = get_detection_by_sequence(seq_num)
    if detection is None:
        raise HTTPException(404, f"Detection #{seq_num} not found")
    detection["image_url"]    = f"/detections/{detection['sequence_number']}.jpg"
    detection["original_url"] = f"/uploads/{detection['sequence_number']}.jpg"
    return detection


@app.get("/stats")
async def get_stats():
    return get_statistics()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
