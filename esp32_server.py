from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi import Request
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
import socket

# mDNS / Zeroconf
from zeroconf import Zeroconf, ServiceInfo

# Add the current directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

UPLOAD_FOLDER    = "uploads"
DETECTION_FOLDER = "detections"
ALLOWED_API_KEYS = {"safev_local_2024"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

ESP32_TRACKING: Dict[str, float] = {}

last_countdown       = None
last_color           = "unknown"
last_valid_detection = None

app = FastAPI(
    title="SafeV ESP32-CAM Server",
    description="Traffic Light Detection System - ESP32 Camera Only",
    version="2.1.0",
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


# ─── mDNS registration ───────────────────────────────────────────────────────
_zeroconf = None

def start_mdns(port: int = 8000):
    """Register this server as safev.local on the local network."""
    global _zeroconf
    try:
        # Get the local IP on the active network interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        ip_bytes = socket.inet_aton(local_ip)
        hostname = socket.gethostname()

        info = ServiceInfo(
            "_http._tcp.local.",
            f"{hostname}._http._tcp.local.",
            addresses=[ip_bytes],
            port=port,
            properties={"path": "/esp32/upload"},
            server=f"{hostname}.local.",
        )
        _zeroconf = Zeroconf()
        _zeroconf.register_service(info)
        logger.info(f"✓ mDNS registered: {hostname}.local → {local_ip}:{port}")
    except Exception as e:
        logger.warning(f"mDNS registration failed (non-fatal): {e}")


def stop_mdns():
    global _zeroconf
    if _zeroconf:
        _zeroconf.close()
        _zeroconf = None


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_next_sequence_number(folder):
    existing = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    numbers  = []
    for f in existing:
        try:
            numbers.append(int(os.path.splitext(f)[0]))
        except ValueError:
            pass
    return (max(numbers) + 1) if numbers else 1


def _apply_temporal_smoothing(result):
    global last_countdown, last_color, last_valid_detection

    tl                 = result["traffic_light"]
    current_color      = tl["color"]
    current_confidence = tl.get("confidence", "low")
    current_score      = tl.get("score", 0.0)

    if current_color != "unknown" and current_confidence in ["very_high", "high"]:
        last_color           = current_color
        last_valid_detection = {
            "color":      current_color,
            "countdown":  tl["countdown"],
            "confidence": current_confidence,
            "score":      current_score,
        }
        return result

    if current_color == "unknown" and last_valid_detection and current_score < 0.1:
        result["traffic_light"]["color"]      = last_valid_detection["color"]
        result["traffic_light"]["confidence"] = "medium"
        return result

    if current_color != "unknown":
        last_color = current_color
        if current_confidence in ["very_high", "high"]:
            last_valid_detection = {
                "color":      current_color,
                "countdown":  tl["countdown"],
                "confidence": current_confidence,
                "score":      current_score,
            }

    if tl["countdown"] is not None:
        last_countdown = tl["countdown"]

    return result


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/discover")
async def discover(request_obj: Request):
    """Returns this server's IP as plain text so ESP32 can auto-discover it."""
    # Return the server's own local IP (not the client's IP)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "unknown"
    return PlainTextResponse(local_ip)


@app.get("/")
async def root():
    return {
        "name":    "SafeV ESP32-CAM Traffic Light Detection",
        "version": "2.1.0",
        "mode":    "ESP32-CAM Only",
        "status":  "ready",
        "mdns":    f"{socket.gethostname()}.local",
        "endpoints": {
            "POST /esp32/upload":       "Upload image from ESP32-CAM",
            "GET  /latest":             "Get latest detection",
            "GET  /history":            "Get detection history",
            "GET  /stats":              "Get statistics",
            "GET  /detection/{seq_num}":"Get specific detection",
            "GET  /uploads/{filename}": "View uploaded images",
            "GET  /detections/{filename}": "View processed images",
        },
    }


@app.post("/esp32/upload")
async def esp32_upload(
    api_key:   str = Form(...),
    image:     str = Form(...),
    timestamp: int = Form(0),
):
    # Auth
    if api_key not in ALLOWED_API_KEYS:
        logger.warning(f"Invalid API key: {api_key[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Rate limit (1 req / 3 s)
    device_id    = hashlib.md5(f"{api_key}_{timestamp}".encode()).hexdigest()[:8]
    current_time = time.time()
    if device_id in ESP32_TRACKING:
        if current_time - ESP32_TRACKING[device_id] < 3:
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Wait 3 seconds."},
            )
    ESP32_TRACKING[device_id] = current_time

    try:
        if image.startswith("data:image"):
            image = image.split(",")[1]

        image_data = base64.b64decode(image)

        seq_num   = get_next_sequence_number(UPLOAD_FOLDER)
        temp_path = os.path.join(UPLOAD_FOLDER, f"{seq_num}.jpg")

        img = Image.open(io.BytesIO(image_data))
        if img.size[0] > 1280:
            ratio    = 1280 / img.size[0]
            new_size = (1280, int(img.size[1] * ratio))
            img      = img.resize(new_size, Image.Resampling.LANCZOS)

        img.save(temp_path, "JPEG", quality=85)
        logger.info(f"✓ Received ESP32 image #{seq_num} ({len(image_data)} bytes)")

        result = detect_objects(temp_path)
        result = _apply_temporal_smoothing(result)

        save_detection(result["id"], result["image_path"], result["original_image"], result)

        response = {
            "status":          "success",
            "sequence_number": result["id"],
            "detection": {
                "traffic_light_color":     result["traffic_light"]["color"],
                "traffic_light_countdown": result["traffic_light"]["countdown"],
                "person_count":            result["person_count"],
                "vehicle_count":           result["vehicle_count"],
                "confidence":              result["traffic_light"]["confidence"],
            },
        }

        logger.info(
            f"✓ Detection #{result['id']}: TL={result['traffic_light']['color']} "
            f"(conf={result['traffic_light']['confidence']})"
        )
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"ESP32 upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


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


@app.get("/esp32/status")
async def esp32_status():
    latest = get_latest()
    return {
        "status":            "active",
        "mode":              "ESP32-CAM Only",
        "server_time":       time.time(),
        "mdns_hostname":     f"{socket.gethostname()}.local",
        "upload_endpoint":   "/esp32/upload",
        "api_key_required":  True,
        "rate_limit":        "3 seconds between requests",
        "latest_detection":  latest["sequence_number"] if latest else None,
        "total_detections":  len(get_history(1000)),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = 8000

    # Detect local IP for display
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "unknown"

    hostname = socket.gethostname()

    print("\n" + "=" * 60)
    print("   SafeV ESP32-CAM Traffic Light Detection Server")
    print("=" * 60)
    print(f"\n📡 Server Configuration:")
    print(f"   • Mode     : ESP32-CAM Only")
    print(f"   • Server   : http://0.0.0.0:{PORT}")
    print(f"   • Local IP : http://{local_ip}:{PORT}")
    print(f"   • mDNS     : http://{hostname}.local:{PORT}  ← ESP32 uses this")
    print(f"   • API Key  : safev_local_2024")
    print(f"\n📸 ESP32-CAM connects to:")
    print(f"   • WiFi SSID    : safev.vercel.app")
    print(f"   • Server (mDNS): {hostname}.local:{PORT}/esp32/upload")
    print(f"\n🔍 Endpoints:")
    print(f"   • GET  /              - Server info")
    print(f"   • POST /esp32/upload  - Upload images")
    print(f"   • GET  /latest        - Latest detection")
    print(f"   • GET  /history       - Detection history")
    print(f"   • GET  /stats         - Statistics")
    print(f"   • GET  /esp32/status  - Server status")
    print("\n" + "=" * 60)
    print("Starting server + mDNS...")
    print("=" * 60 + "\n")

    # Register mDNS so ESP32 can find us by hostname
    start_mdns(PORT)

    try:
        uvicorn.run("esp32_server:app", host="0.0.0.0", port=PORT, reload=False)
    finally:
        stop_mdns()