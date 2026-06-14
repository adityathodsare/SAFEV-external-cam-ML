# SafeV Camera System

Traffic scene analysis platform using YOLOv8 object detection and computer vision. Detects people, vehicles, and traffic lights from webcam or ESP32-CAM, analyzes traffic light color (red/yellow/green), reads 7-segment countdown timers, and stores results in SQLite.

## Features

- **Object Detection** — YOLOv8n detects persons, cars, trucks, buses, motorcycles, bicycles, and traffic lights
- **Traffic Light Analysis** — HSV/BGR color segmentation with positional boosting determines red/yellow/green status (confidence: low/medium/high/very_high)
- **Countdown OCR** — Reads 7-segment countdown digits near traffic light bounding boxes
- **Temporal Smoothing** — Stabilizes traffic light color/confidence across frames to avoid flicker
- **Dual Camera Support** — Local webcam (auto-capture every 20s) or ESP32-CAM (HTTP upload)
- **REST API** — FastAPI endpoints for capture, detection history, and statistics
- **SQLite Persistence** — All detections stored with per-object metadata
- **Annotated Output** — Bounding boxes, labels, traffic light status, countdown, and object counts overlaid on images

## Hardware Requirements

| Mode | Requirements |
|------|-------------|
| **Webcam** | Computer with Python 3.10+, any USB/built-in webcam (1920x1080 recommended) |
| **ESP32-CAM** | Same server computer + AI-Thinker ESP32-CAM with OV2640 on same WiFi network |
| **General** | 4GB+ RAM (8GB recommended), ~200MB disk, GPU optional (falls back to CPU) |

## Quick Start

### 1. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2a. Webcam Mode

```powershell
python -m app.main
# Server at http://localhost:8000
# Auto-captures from webcam every 20 seconds
```

### 2b. ESP32-CAM Mode

```powershell
python esp32_server.py
# Server at http://0.0.0.0:8000
# Waits for ESP32-CAM to POST images
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server info |
| GET | `/webcam/capture` | Manual webcam capture (webcam mode) |
| POST | `/esp32/upload` | ESP32 image upload (form: `api_key`, `image` [base64], `timestamp`) |
| GET | `/latest` | Most recent detection |
| GET | `/history?limit=50` | Detection history |
| GET | `/detection/{seq_num}` | Specific detection |
| GET | `/stats` | Aggregate statistics |
| GET | `/esp32/status` | ESP32 server status |
| GET | `/detections/{filename}` | View annotated images |
| GET | `/uploads/{filename}` | View original images |

## ESP32-CAM Setup

Flash Arduino firmware to the ESP32-CAM configured with:

- **WiFi SSID:** `safev.vercel.app`
- **WiFi Password:** `aditya45`
- **Server URL:** `http://<YOUR_SERVER_IP>:8000/esp32/upload`
- **API Key:** `safev_local_2024`

The ESP32 captures a JPEG, encodes to base64, and POSTs to the server. Rate limit: 1 request per 3 seconds.

## Project Structure

```
safev-camera-system-refine/
├── app/
│   ├── __init__.py              # Package marker
│   ├── main.py                  # Webcam FastAPI server
│   ├── detector.py              # YOLOv8 detection + annotation pipeline
│   ├── traffic_light_detector.py# Traffic light color analysis
│   ├── database.py              # SQLite persistence layer
│   ├── esp32_server.py          # ESP32 FastAPI server (duplicate)
│   └── test_countdown.py        # Countdown test script
├── esp32_server.py              # Root-level ESP32 server entry point
├── yolov8n.pt                   # YOLOv8 nano model weights
├── database.db                  # SQLite database (runtime)
├── detections/                  # Annotated output images
├── uploads/                     # Original uploaded images
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Configuration

All settings are currently hardcoded. Key values to change for your environment:

| Setting | Default | File |
|---------|---------|------|
| Server IP | `10.168.213.97` | `esp32_server.py` |
| API Key | `safev_local_2024` | `esp32_server.py` |
| WiFi SSID | `safev.vercel.app` | ESP32 firmware (not in repo) |
| WiFi Password | `aditya45` | ESP32 firmware (not in repo) |
| Capture Interval | 20s | `app/main.py:25` |
| YOLO Confidence | 0.25 | `app/detector.py:315` |

## Notes

- No frontend — pure backend API served via FastAPI + uvicorn
- No HTTPS — use a reverse proxy (nginx/caddy) for production
- GET endpoints have no authentication
- `app/esp32_server.py` and root `esp32_server.py` are duplicates; use the root version to run directly
