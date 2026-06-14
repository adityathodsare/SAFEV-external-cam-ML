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
last_valid_detection = None  # Store last valid detection with high confidence


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
    Improved temporal smoothing that respects current detections
    and only uses history when confidence is low.
    """
    global last_countdown, last_color, last_valid_detection
    
    tl = result["traffic_light"]
    current_color = tl["color"]
    current_confidence = tl.get("confidence", "low")
    current_score = tl.get("score", 0.0)
    
    # If we have a valid detection with good confidence, use it
    if current_color != "unknown" and current_confidence in ["very_high", "high"]:
        last_color = current_color
        last_valid_detection = {
            "color": current_color,
            "countdown": tl["countdown"],
            "confidence": current_confidence,
            "score": current_score
        }
        logger.info(f"High confidence detection: {current_color} (score: {current_score})")
        return result
    
    # If current detection is unknown but we had a recent valid detection,
    # and the current frame has very low confidence, use last valid
    if current_color == "unknown" and last_valid_detection is not None:
        # Check if the last valid detection is still recent (within last 3 detections)
        # For now, we'll just use it if current score is very low
        if current_score < 0.1:
            logger.info(f"Using last valid detection: {last_valid_detection['color']} (current score too low)")
            result["traffic_light"]["color"] = last_valid_detection["color"]
            result["traffic_light"]["confidence"] = "medium"  # Downgrade confidence
            return result
    
    # If current detection is valid but low confidence, keep it
    if current_color != "unknown":
        last_color = current_color
        if current_confidence in ["very_high", "high"]:
            last_valid_detection = {
                "color": current_color,
                "countdown": tl["countdown"],
                "confidence": current_confidence,
                "score": current_score
            }
    
    # Handle countdown
    if tl["countdown"] is not None:
        last_countdown = tl["countdown"]
    
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
                logger.info("Detection #%d complete – TL=%s (conf=%s, score=%.3f) cd=%s",
                            result['id'],
                            result['traffic_light']['color'],
                            result['traffic_light']['confidence'],
                            result['traffic_light'].get('score', 0.0),
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
    global last_countdown, last_color, last_valid_detection
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