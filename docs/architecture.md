# SafeV Camera System — Architecture & Behavior

## System Overview

SafeV Camera System processes images from a camera (local webcam or ESP32-CAM) through a YOLOv8 object detection pipeline, analyzes traffic light color and countdown timers, annotates the image, and persists results to SQLite. All functionality is exposed via a FastAPI REST API.

```
Camera Image → YOLOv8 Detection → Object Classification →
  For each traffic light:
    → TrafficLightDetector.analyze_traffic_light() — determine color
    → detect_countdown_near_traffic_light() — OCR 7-segment digits
  → Select best traffic light candidate
  → Temporal smoothing (ESP32 mode)
  → Draw annotations → Save image → Store in DB → Return JSON
```

---

## File-by-File Breakdown

### `app/__init__.py`
Empty file that marks `app/` as a Python package.

---

### `app/main.py` — Webcam Server (216 lines)

**Purpose:** FastAPI application that captures frames from a local webcam on a periodic timer (every 20 seconds) and exposes REST endpoints.

| Function | Description |
|----------|-------------|
| `lifespan(app)` | Async context manager — starts the periodic capture loop on server startup, cancels it on shutdown |
| `periodic_capture()` | Async loop: calls `detector.capture_from_webcam()` then `detector.detect_objects()`, saves result to DB via `database.save_detection()`, sleeps 20s |
| `_apply_temporal_smoothing(result)` | Remembers last valid traffic light color/confidence. If current frame has lower confidence or unknown color, falls back to last high-confidence detection |
| `webcam_capture()` | **GET `/webcam/capture`** — triggers one capture/detection cycle on demand |
| `root()` | **GET `/`** — returns server info and endpoint list |
| `latest_detection()` | **GET `/latest`** — most recent detection from DB |
| `detection_history(limit)` | **GET `/history?limit=50`** — last N detections |
| `get_detection(seq_num)` | **GET `/detection/{seq_num}`** — specific detection by sequence number |
| `get_stats()` | **GET `/stats`** — aggregate stats (total counts, averages) |

**Behavior:** Starts capturing immediately on launch. Each capture grabs the sharpest frame from 5 webcam shots. The server also serves static files from `detections/` and `uploads/` directories.

---

### `app/detector.py` — Detection Engine (515 lines)

**Purpose:** Core detection pipeline — loads YOLOv8n model, runs inference, classifies objects, analyzes traffic lights, OCRs countdown digits, draws annotations.

| Function / Constant | Description |
|---------------------|-------------|
| `model = YOLO("yolov8n.pt")` | Loads YOLOv8 nano model at module import time |
| `TARGET_CLASSES` | `["person", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light"]` |
| `SEVEN_SEGMENT_DIGITS` | Dictionary mapping 7-segment on/off patterns (7-tuples) to digits 0-9 |
| `get_next_sequence_number(folder)` | Scans folder for existing numeric filenames, returns next available integer |
| `detect_objects(image_path)` | **Main function.** Reads image, runs YOLO predict (conf=0.25, imgsz=640), separates detections into people/vehicles/traffic lights. For each TL, calls `analyze_traffic_light()` and `detect_countdown_near_traffic_light()`. Selects best TL via `select_best_traffic_light()`. Draws bounding boxes, labels, counts, TL status, countdown. Saves annotated image to `detections/`. Returns dict with all results |
| `capture_from_webcam()` | Opens webcam (index 0), warms up for 10 frames, captures 5 frames, selects the sharpest via Laplacian variance. Saves to `uploads/`. Returns the file path |
| `draw_detection_info(image, info)` | Draws timestamp, person/vehicle/TL counts, traffic light color with confidence stars, countdown value onto the image |
| `select_best_traffic_light(candidates)` | Ranks TL candidates: valid colors (red/yellow/green) preferred over unknown, then sorted by composite score (color_score + area + confidence + countdown bonus) |
| `detect_countdown_near_traffic_light(image, bbox)` | Searches three regions around the TL bounding box (above, left, right) for 7-segment countdown digits. Calls `_read_countdown_from_roi()` on each region |
| `_read_countdown_from_roi(roi)` | Creates mask for red+amber colors, finds contours, merges nearby contours (within 5px), sorts left-to-right, classifies each merged contour as a digit via `_decode_digit()`, returns integer (None if <2 digits) |
| `_extract_countdown_mask(roi)` | Converts ROI to HSV, creates mask combining red (two ranges) and amber colors with brightness threshold (V > 60) |
| `_decode_digit(digit_mask)` | Splits digit mask into 7 segments (top, top-left, top-right, middle, bottom-left, bottom-right, bottom), checks fill ratio of each segment, matches against `SEVEN_SEGMENT_DIGITS` patterns. Returns digit 0-9 or None |
| `_clamp(v, lo, hi)` | Clamps integer value between lo and hi |

**Detection result dict structure:**
```python
{
    "image_path": "detections/5.jpg",
    "original_image": "uploads/5.jpg",
    "objects": ["person", "car", "traffic light", ...],
    "objects_detailed": [{"class": "person", "confidence": 0.92, "bbox": [x1,y1,x2,y2]}, ...],
    "person_count": 2,
    "vehicle_count": 3,
    "traffic_light_count": 1,
    "traffic_light": {
        "color": "red",
        "confidence": "very_high",
        "score": 0.65,
        "countdown": 15,
        "countdown_detected": True
    },
    "detection_count": 6
}
```

---

### `app/traffic_light_detector.py` — Traffic Light Color Detector (230 lines)

**Purpose:** Given a traffic light bounding box, determines which lamp is lit using multi-range HSV and BGR masking with positional boosting.

| Component | Description |
|-----------|-------------|
| `class TrafficLightDetector` | Main class with color range definitions and analysis methods |
| `self.hsv_ranges` | Per-color (red/yellow/green) list of 2-3 HSV range pairs. Red uses two separate ranges to wrap around the hue wheel |
| `self.bgr_ranges` | Per-color list of 2 BGR range pairs as a complementary filter |
| `analyze_traffic_light(image, bbox)` | Extracts ROI from image, determines layout (vertical/horizontal/single), scores each color via `_score_color()`. If best score < 0.12, returns "unknown" with low confidence. Returns dict with color, confidence label, score |
| `_build_mask_from_ranges(image, ranges)` | Creates binary mask by OR-ing all masks generated from each range pair |
| `_prepare_focus_roi(roi)` | Crops wide ROIs (>1.6x aspect) to focus on the lamp housing, excluding side areas (timers) |
| `_get_layout(roi)` | Determines layout: `"vertical"` if h > w*1.2, `"horizontal"` if w > h*1.2, else `"single"` |
| `_get_position_boost_mask(shape, color_name, layout)` | Creates float mask (1.18-1.50x) that boosts pixels at expected lamp positions. For vertical: top=red, middle=yellow, bottom=green. For horizontal: left=red, middle=yellow, right=green |
| `_score_color(roi, color_name, layout)` | Scores a single color candidate: combines HSV mask + BGR mask + brightness, finds contours, calculates score from area ratio, value, saturation, compactness, and positional boost |

**Confidence thresholds:**
| Score | Confidence |
|-------|------------|
| >= 0.42 and gap >= 0.08 from next | `very_high` |
| >= 0.24 | `high` |
| >= 0.12 | `medium` |
| < 0.12 | `low` (returns unknown) |

**Singleton:** Module creates `traffic_light_detector = TrafficLightDetector()` at import — all code uses this single instance.

---

### `app/database.py` — Database Layer (185 lines)

**Purpose:** SQLite wrapper for persisting detection results.

| Function | Description |
|----------|-------------|
| `save_detection(seq_num, img_path, orig_path, result)` | UPSERT — inserts or updates a row in the `detections` table with all fields from the result dict |
| `get_latest()` | Returns the most recent detection row as a dict |
| `get_history(limit=50)` | Returns last N rows ordered by sequence_number DESC |
| `get_detection_by_sequence(seq_num)` | Returns a single detection row by its sequence number |
| `get_statistics()` | Returns aggregate: total detections, person/vehicle/TL counts, color breakdown, countdown hits, avg objects per detection |
| `reset_database()` | Drops and recreates the detections table (development use only) |
| `_row_to_dict(row)` | Converts a sqlite3.Row to dict with JSON-deserialized object fields and proper defaults |

**Database schema (`detections` table):**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK AUTOINCREMENT | Internal row ID |
| `sequence_number` | INTEGER UNIQUE | Sequential image number |
| `image_path` | TEXT | Path to annotated detection image |
| `original_image` | TEXT | Path to original uploaded image |
| `objects` | TEXT (JSON) | List of detected class names |
| `objects_detailed` | TEXT (JSON) | Full per-object details |
| `person_count` | INTEGER | Number of people detected |
| `vehicle_count` | INTEGER | Number of vehicles detected |
| `traffic_light_count` | INTEGER | Number of TL boxes |
| `traffic_light_detected` | BOOLEAN | Whether a valid TL was found |
| `traffic_light_color` | TEXT | red/yellow/green/unknown |
| `traffic_light_countdown` | INTEGER | Countdown value or NULL |
| `traffic_light_countdown_detected` | BOOLEAN | Whether countdown was read |
| `traffic_light_confidence` | TEXT | low/medium/high/very_high |
| `detection_count` | INTEGER | Total objects detected |
| `timestamp` | DATETIME | Auto-set creation time |

---

### `esp32_server.py` (Root) and `app/esp32_server.py` — ESP32 Server (321 / 317 lines)

**Purpose:** FastAPI server optimized for receiving images from an ESP32-CAM over HTTP. Nearly identical files; root version uses `sys.path.insert(0, ...)` to allow direct `python esp32_server.py` execution.

| Function | Description |
|----------|-------------|
| `esp32_upload(api_key, image, timestamp)` | **POST `/esp32/upload`** — validates API key (`safev_local_2024`), rate-limits (3s between requests), decodes base64 image, resizes max dimension to 1280px, saves JPEG (quality 85) to `uploads/`, runs `detect_objects()`, applies temporal smoothing, saves to DB. Returns detection summary JSON |
| `esp32_status()` | **GET `/esp32/status`** — returns server status, rate limit info, and DB stats |
| `_apply_temporal_smoothing(result)` | Same logic as `main.py` — retains last valid TL color/confidence across frames |
| `get_next_sequence_number(folder)` | Same as `detector.py` |

**Rate limiting:** Uses `last_request_time` module variable. Returns 429 if request arrives within 3 seconds of the last one.

**API key check:** Returns 403 if `api_key` not in `ALLOWED_API_KEYS = {"safev_local_2024"}`.

**ESP32 form parameters:**
- `api_key` — string (required)
- `image` — base64-encoded JPEG (required, may include `data:image/...;base64,` prefix)
- `timestamp` — integer unix timestamp (optional)

---

### `app/test_countdown.py` — Test Script (51 lines)

**Purpose:** Standalone script to visually test traffic light countdown detection on a single image.

| Function | Description |
|----------|-------------|
| `test_countdown_detection(image_path)` | Loads image, runs `traffic_light_detector.analyze_traffic_light()`, prints results, displays image with OpenCV `imshow()` |

---

## Data Flow (Detailed)

### Webcam Mode Flow
```
1. Server starts → lifespan() starts periodic_capture() asyncio task
2. Every 20 seconds:
   a. capture_from_webcam() — open webcam, grab 5 frames, select sharpest, save to uploads/
   b. detect_objects() — YOLO inference on saved image
   c. For each detected traffic light:
      - TrafficLightDetector.analyze_traffic_light() → color + confidence
      - detect_countdown_near_traffic_light() → countdown integer or None
   d. select_best_traffic_light() — pick the best TL among candidates
   e. Draw all annotations → save to detections/
   f. save_detection() → UPSERT into SQLite
3. GET endpoints query the database for results
```

### ESP32-CAM Mode Flow
```
1. Server starts at http://0.0.0.0:8000
2. ESP32-CAM connects to WiFi, captures JPEG
3. ESP32-CAM POSTs to /esp32/upload with api_key + base64 image
4. Server validates API key, checks rate limit
5. Decodes base64 → resizes → saves to uploads/ → runs detect_objects()
6. _apply_temporal_smoothing() stabilizes TL color across frames
7. Saves to DB → returns JSON response to ESP32
```

---

## How to Run

### Prerequisites
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`
- Model file `yolov8n.pt` must be in project root

### Locally — Webcam Mode
```powershell
# Terminal 1: Start server
cd safev-camera-system-refine
python -m app.main

# Server starts at http://localhost:8000
# Auto-captures every 20 seconds from webcam
# Browse to http://localhost:8000/detections/ to see annotated images
```

### Locally — ESP32-CAM Mode
```powershell
# Start the ESP32 server
cd safev-camera-system-refine
python esp32_server.py

# Server waits for ESP32 uploads at http://0.0.0.0:8000
# Test with curl:
curl -X POST http://localhost:8000/esp32/upload ^
  -F "api_key=safev_local_2024" ^
  -F "image=@uploads/1.jpg;filename=image.jpg"
```

### On ESP32-CAM Device
No ESP32 firmware code is included in this repository. You must write Arduino code and flash it to the ESP32-CAM. The firmware must:

1. Connect to WiFi: SSID `safev.vercel.app`, password `aditya45`
2. Capture a JPEG using the ESP32 camera library
3. Convert the JPEG to base64
4. Send HTTP POST to `http://<SERVER_IP>:8000/esp32/upload` with:
   - `api_key = safev_local_2024`
   - `image = <base64_encoded_jpeg>`
   - `timestamp = <unix_timestamp>`
5. Respect the 3-second rate limit
6. Parse the JSON response for detection results

**Recommended ESP32-CAM board:** AI-Thinker ESP32-CAM with OV2640 camera, PSRAM enabled.

### Manual Single Image Test
```powershell
python -c "from app.detector import detect_objects; r = detect_objects('uploads/1.jpg'); print(r['traffic_light'])"
```

### Countdown Detection Test
```powershell
python -m app.test_countdown
```

---

## Configuration Reference

All configuration is hardcoded. See each file to modify:

| Setting | Value | Location |
|---------|-------|----------|
| YOLO model | `yolov8n.pt` | `detector.py:12` |
| YOLO confidence | `0.25` | `detector.py:315` |
| YOLO image size | `640` | `detector.py:315` |
| Capture interval | `20` seconds | `main.py:25` |
| Webcam resolution | `1920x1080` | `detector.py:488-489` |
| Webcam warmup | 10 frames | `detector.py:492` |
| Webcam capture | 5 frames (sharpest) | `detector.py:496` |
| Server host/port | `0.0.0.0:8000` | all entry points |
| CORS origins | `*` | all FastAPI apps |
| ESP32 API keys | `{"safev_local_2024"}` | `esp32_server.py` |
| ESP32 rate limit | 3 seconds | `esp32_server.py` |
| ESP32 max dimension | 1280px | `esp32_server.py:187` |
| ESP32 JPEG quality | 85 | `esp32_server.py:192` |
| DB file | `database.db` | `database.py:5` |

---

## Key Design Decisions

1. **No configuration files or environment variables** — all settings are hardcoded. Modify source files directly to change behavior.

2. **Duplicate `esp32_server.py`** — the root copy adds `sys.path.insert(0, ...)` to run standalone; `app/esp32_server.py` runs via `python -m app.esp32_server`. Both are functionally identical.

3. **Temporal smoothing** exists in both `main.py` and `esp32_server.py` to prevent traffic light color flicker between frames. The system remembers the last high-confidence detection and uses it when the current frame is uncertain.

4. **Traffic light selection** prioritizes valid colors (red/yellow/green) over unknown, then ranks by a composite score of color analysis + bounding box area + YOLO confidence + countdown detection bonus.

5. **Countdown OCR** uses classical CV (thresholding + contour detection + 7-segment decoding) rather than ML. This works well for red/amber displays but requires clear visibility of digits near the traffic light.
