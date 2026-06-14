from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import os
import base64
import io
from PIL import Image
import time
from typing import Dict
import hashlib
import logging
import sys

# Add the current directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from app.detector import detect_objects
from app.database import (
    save_detection,
    get_latest,
    get_history,
    get_statistics,
    get_detection_by_sequence,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
DETECTION_FOLDER = "detections"
ALLOWED_API_KEYS = {"safev_local_2024"}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Rate limiting tracking
ESP32_TRACKING: Dict[str, float] = {}

# Temporal smoothing globals
last_countdown = None
last_color = "unknown"
last_valid_detection = None

# Create FastAPI app
app = FastAPI(
    title="SafeV ESP32-CAM Server",
    description="Traffic Light Detection System - ESP32 Camera Only",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folders for serving images
app.mount("/detections", StaticFiles(directory="detections"), name="detections")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def get_next_sequence_number(folder):
    """Get next sequence number for saved images"""
    existing = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    numbers = []
    for f in existing:
        try:
            numbers.append(int(os.path.splitext(f)[0]))
        except ValueError:
            pass
    return (max(numbers) + 1) if numbers else 1


def _apply_temporal_smoothing(result):
    """Apply temporal smoothing for consistent detections"""
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
        if current_score < 0.1:
            logger.info(f"Using last valid detection: {last_valid_detection['color']} (current score too low)")
            result["traffic_light"]["color"] = last_valid_detection["color"]
            result["traffic_light"]["confidence"] = "medium"
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


@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "name": "SafeV ESP32-CAM Traffic Light Detection",
        "version": "2.0.0",
        "mode": "ESP32-CAM Only",
        "status": "ready",
        "endpoints": {
            "POST /esp32/upload": "Upload image from ESP32-CAM",
            "GET /latest": "Get latest detection",
            "GET /history": "Get detection history",
            "GET /stats": "Get statistics",
            "GET /detection/{seq_num}": "Get specific detection",
            "GET /uploads/{filename}": "View uploaded images",
            "GET /detections/{filename}": "View processed images",
        },
    }


@app.post("/esp32/upload")
async def esp32_upload(
    api_key: str = Form(...),
    image: str = Form(...),
    timestamp: int = Form(0)
):
    """Endpoint for ESP32-CAM to upload images"""
    
    # Authentication
    if api_key not in ALLOWED_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Rate limiting (max 1 request per 3 seconds)
    device_id = hashlib.md5(f"{api_key}_{timestamp}".encode()).hexdigest()[:8]
    current_time = time.time()
    
    if device_id in ESP32_TRACKING:
        if current_time - ESP32_TRACKING[device_id] < 3:
            logger.warning(f"Rate limit exceeded for device {device_id}")
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Please wait 3 seconds."}
            )
    ESP32_TRACKING[device_id] = current_time
    
    try:
        # Decode base64 image
        if image.startswith('data:image'):
            image = image.split(',')[1]
        
        image_data = base64.b64decode(image)
        
        # Save to uploads folder
        seq_num = get_next_sequence_number(UPLOAD_FOLDER)
        temp_path = os.path.join(UPLOAD_FOLDER, f"{seq_num}.jpg")
        
        # Convert bytes to image and save
        img = Image.open(io.BytesIO(image_data))
        
        # Resize if needed (optional - to speed up processing)
        if img.size[0] > 1280:
            ratio = 1280 / img.size[0]
            new_size = (1280, int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        img.save(temp_path, 'JPEG', quality=85)
        
        logger.info(f"✓ Received ESP32 image #{seq_num} ({len(image_data)} bytes)")
        
        # Run detection using your existing detector
        result = detect_objects(temp_path)
        
        # Apply temporal smoothing
        result = _apply_temporal_smoothing(result)
        
        # Save to database
        save_detection(
            result["id"],
            result["image_path"],
            result["original_image"],
            result,
        )
        
        # Prepare response for ESP32
        response = {
            "status": "success",
            "sequence_number": result["id"],
            "detection": {
                "traffic_light_color": result["traffic_light"]["color"],
                "traffic_light_countdown": result["traffic_light"]["countdown"],
                "person_count": result["person_count"],
                "vehicle_count": result["vehicle_count"],
                "confidence": result["traffic_light"]["confidence"]
            }
        }
        
        logger.info(f"✓ ESP32 detection #{result['id']}: TL={result['traffic_light']['color']} "
                   f"(conf={result['traffic_light']['confidence']})")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"ESP32 upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/latest")
async def latest_detection():
    """Get the latest detection"""
    detection = get_latest()
    if detection is None:
        return {"message": "No detections yet"}
    detection["image_url"] = f"/detections/{detection['sequence_number']}.jpg"
    detection["original_url"] = f"/uploads/{detection['sequence_number']}.jpg"
    return detection


@app.get("/history")
async def detection_history(limit: int = 50):
    """Get detection history"""
    history = get_history(limit)
    for item in history:
        item["image_url"] = f"/detections/{item['sequence_number']}.jpg"
        item["original_url"] = f"/uploads/{item['sequence_number']}.jpg"
    return {"total": len(history), "detections": history}


@app.get("/detection/{seq_num}")
async def get_detection(seq_num: int):
    """Get specific detection by sequence number"""
    detection = get_detection_by_sequence(seq_num)
    if detection is None:
        raise HTTPException(404, f"Detection #{seq_num} not found")
    detection["image_url"] = f"/detections/{detection['sequence_number']}.jpg"
    detection["original_url"] = f"/uploads/{detection['sequence_number']}.jpg"
    return detection


@app.get("/stats")
async def get_stats():
    """Get statistics"""
    return get_statistics()


@app.get("/esp32/status")
async def esp32_status():
    """Check ESP32 server status"""
    latest = get_latest()
    return {
        "status": "active",
        "mode": "ESP32-CAM Only",
        "server_time": time.time(),
        "upload_endpoint": "/esp32/upload",
        "api_key_required": True,
        "rate_limit": "3 seconds between requests",
        "latest_detection": latest["sequence_number"] if latest else None,
        "total_detections": len(get_history(1000))
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   SafeV ESP32-CAM Traffic Light Detection Server")
    print("="*60)
    print("\n📡 Server Configuration:")
    print("   • Mode: ESP32-CAM Only (No webcam)")
    print("   • Server: http://0.0.0.0:8000")
    print("   • Upload endpoint: http://10.168.213.97:8000/esp32/upload")
    print("   • API Key: safev_local_2024")
    print("\n📸 ESP32-CAM Setup:")
    print("   • WiFi SSID: safev.vercel.app")
    print("   • WiFi Password: aditya45")
    print("   • Server URL: http://10.168.213.97:8000/esp32/upload")
    print("\n🔍 Endpoints:")
    print("   • GET  /              - Server info")
    print("   • POST /esp32/upload  - Upload images (ESP32-CAM)")
    print("   • GET  /latest        - Latest detection")
    print("   • GET  /history       - Detection history")
    print("   • GET  /stats         - Statistics")
    print("   • GET  /esp32/status  - Server status")
    print("\n" + "="*60)
    print("Starting ESP32-CAM server...")
    print("="*60 + "\n")
    
    # Run the server
    uvicorn.run(
        "esp32_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )