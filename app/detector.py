from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np
import logging
from app.traffic_light_detector import traffic_light_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = YOLO("yolov8n.pt")

DETECTION_FOLDER = "detections"
UPLOAD_FOLDER = "uploads"
os.makedirs(DETECTION_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TARGET_CLASSES = [
    "person", "car", "truck", "bus",
    "motorcycle", "bicycle", "traffic light"
]

SEVEN_SEGMENT_DIGITS = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


def get_next_sequence_number(folder):
    existing = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    numbers = []
    for f in existing:
        try:
            numbers.append(int(os.path.splitext(f)[0]))
        except ValueError:
            pass
    return (max(numbers) + 1) if numbers else 1


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _extract_countdown_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 90, 130]), np.array([14, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 90, 130]), np.array([180, 255, 255]))
    amber = cv2.inRange(hsv, np.array([10, 90, 150]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.bitwise_or(mask, amber)

    value = hsv[:, :, 2]
    bright_threshold = max(150, int(np.percentile(value, 85)))
    bright_mask = cv2.inRange(value, bright_threshold, 255)
    mask = cv2.bitwise_and(mask, bright_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _decode_digit(digit_mask):
    h, w = digit_mask.shape[:2]
    if h < 10 or w < 5:
        return None, 0.0

    segments = [
        ((0.22, 0.05), (0.78, 0.18)),
        ((0.08, 0.16), (0.28, 0.48)),
        ((0.72, 0.16), (0.92, 0.48)),
        ((0.22, 0.42), (0.78, 0.58)),
        ((0.08, 0.54), (0.28, 0.86)),
        ((0.72, 0.54), (0.92, 0.86)),
        ((0.22, 0.82), (0.78, 0.95)),
    ]

    on = []
    confidences = []
    for (ax, ay), (bx, by) in segments:
        x1 = max(0, int(w * ax))
        y1 = max(0, int(h * ay))
        x2 = min(w, int(w * bx))
        y2 = min(h, int(h * by))
        if x2 <= x1 or y2 <= y1:
            on.append(0)
            confidences.append(0.0)
            continue

        patch = digit_mask[y1:y2, x1:x2]
        fill_ratio = cv2.countNonZero(patch) / max(patch.size, 1)
        confidences.append(fill_ratio)
        on.append(1 if fill_ratio > 0.28 else 0)

    pattern = tuple(on)
    if pattern in SEVEN_SEGMENT_DIGITS:
        return SEVEN_SEGMENT_DIGITS[pattern], float(np.mean(confidences))

    best_digit = None
    best_distance = 99
    for candidate_pattern, digit in SEVEN_SEGMENT_DIGITS.items():
        distance = sum(abs(a - b) for a, b in zip(pattern, candidate_pattern))
        if distance < best_distance:
            best_distance = distance
            best_digit = digit

    if best_distance <= 1:
        return best_digit, max(0.0, float(np.mean(confidences)) - 0.1)

    return None, 0.0


def _read_countdown_from_roi(roi):
    if roi is None or roi.size == 0:
        return None, 0.0

    mask = _extract_countdown_mask(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    roi_h, roi_w = mask.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 18:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)
        area_ratio = area / max(roi_h * roi_w, 1)
        if h < max(10, roi_h * 0.12):
            continue
        if not 0.15 <= aspect_ratio <= 0.95:
            continue
        if area_ratio > 0.25:
            continue
        candidates.append((x, y, w, h))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda b: b[0])
    merged = []
    for box in candidates:
        if not merged:
            merged.append(list(box))
            continue
        px, py, pw, ph = merged[-1]
        x, y, w, h = box
        if x <= px + pw + max(4, roi_w * 0.02):
            nx1 = min(px, x)
            ny1 = min(py, y)
            nx2 = max(px + pw, x + w)
            ny2 = max(py + ph, y + h)
            merged[-1] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
        else:
            merged.append(list(box))

    digit_boxes = []
    for x, y, w, h in merged:
        if w > h * 1.25:
            split = max(w // 2, 1)
            digit_boxes.append((x, y, split, h))
            digit_boxes.append((x + split, y, w - split, h))
        else:
            digit_boxes.append((x, y, w, h))

    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])[:2]
    digits = []
    score_parts = []
    for x, y, w, h in digit_boxes:
        pad = max(1, int(min(w, h) * 0.12))
        dx1 = max(0, x - pad)
        dy1 = max(0, y - pad)
        dx2 = min(roi_w, x + w + pad)
        dy2 = min(roi_h, y + h + pad)
        digit_mask = mask[dy1:dy2, dx1:dx2]
        digit, score = _decode_digit(digit_mask)
        if digit is None:
            return None, 0.0
        digits.append(str(digit))
        score_parts.append(score)

    if not digits:
        return None, 0.0

    value = int("".join(digits))
    if not 0 <= value <= 99:
        return None, 0.0
    return value, float(np.mean(score_parts))


def detect_countdown_near_traffic_light(image, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    search_regions = [
        [x2, max(0, y1 - bh // 5), min(W, x2 + int(bw * 2.2)), min(H, y2 + bh // 5)],
        [max(0, x1 + int(bw * 0.45)), y1, x2, y2],
        [max(0, x1 - bw // 4), y2, min(W, x2 + bw // 3), min(H, y2 + bh)],
    ]

    best_value = None
    best_score = 0.0
    for rx1, ry1, rx2, ry2 in search_regions:
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        roi = image[ry1:ry2, rx1:rx2]
        value, score = _read_countdown_from_roi(roi)
        if value is not None and score > best_score:
            best_value = value
            best_score = score

    if best_value is not None and best_value > 0 and best_score >= 0.18:
        return best_value
    return None


def draw_detection_info(image, detection_info):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, f"Time: {timestamp}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    counts = detection_info["counts"]
    cv2.putText(image,
                f"People: {counts['people']} | "
                f"Vehicles: {counts['vehicles']} | "
                f"Traffic Lights: {counts['traffic_lights']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    tl = detection_info["traffic_light"]
    if not tl["detected"] or tl["color"] == "unknown":
        return

    if tl["color"] == "red":
        bgr = (0, 0, 255)
        label = "RED LIGHT"
    elif tl["color"] == "yellow":
        bgr = (0, 255, 255)
        label = "YELLOW LIGHT"
    elif tl["color"] == "green":
        bgr = (0, 255, 0)
        label = "GREEN LIGHT"
    else:
        return

    conf_stars = {
        "very_high": "***",
        "high": "**",
        "medium": "*",
    }.get(tl["confidence"], "")

    cv2.putText(image, f"{label} {conf_stars}".strip(),
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2)

    if tl["countdown"] is not None:
        cv2.putText(image, f"Countdown: {tl['countdown']}s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def select_best_traffic_light(candidates):
    """
    Select the best traffic light from multiple candidates
    with improved logic to prefer valid detections over unknowns.
    """
    if not candidates:
        return None
    
    # First, separate valid and unknown detections
    valid_detections = [c for c in candidates if c["color"] != "unknown"]
    unknown_detections = [c for c in candidates if c["color"] == "unknown"]
    
    # Log all candidates for debugging
    for i, c in enumerate(candidates):
        logger.info(f"Candidate {i}: color={c['color']}, score={c['score']:.3f}, conf={c.get('model_confidence', 0):.2f}")
    
    # If we have valid detections, choose the best one based on score and confidence
    if valid_detections:
        # Sort by: score (highest first), then model confidence, then area
        valid_detections.sort(key=lambda x: (
            x["score"],
            x.get("model_confidence", 0),
            x.get("area", 0)
        ), reverse=True)
        
        best = valid_detections[0]
        logger.info(f"Selected valid detection: {best['color']} (score: {best['score']:.3f})")
        return best
    
    # If no valid detections, return the unknown with highest score (if any)
    if unknown_detections:
        unknown_detections.sort(key=lambda x: x["score"], reverse=True)
        best = unknown_detections[0]
        logger.info(f"No valid detections, using unknown with score: {best['score']:.3f}")
        return best
    
    return None


def detect_objects(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original = image.copy()
    results = model.predict(image, imgsz=640, conf=0.25)

    detected_objects, objects_detailed = [], []
    traffic_light_info = {
        "detected": False,
        "color": "unknown",
        "countdown": None,
        "bbox": None,
        "countdown_detected": False,
        "confidence": "low",
        "lit_region": None,
        "score": 0.0,
    }

    traffic_light_candidates = []
    person_count = vehicle_count = tl_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])

            if name not in TARGET_CLASSES or conf < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name == "person":
                person_count += 1
                color = (255, 255, 0)
                label = f"{name} {conf:.2f}"
            elif name in ("car", "truck", "bus", "motorcycle", "bicycle"):
                vehicle_count += 1
                color = (0, 255, 255)
                label = f"{name} {conf:.2f}"
            elif name == "traffic light":
                tl_count += 1
                tl_result = traffic_light_detector.analyze_traffic_light(image, (x1, y1, x2, y2))
                countdown = detect_countdown_near_traffic_light(image, (x1, y1, x2, y2))

                candidate = {
                    "detected": tl_result["color"] != "unknown",
                    "color": tl_result["color"],
                    "countdown": countdown,
                    "bbox": [x1, y1, x2, y2],
                    "countdown_detected": countdown is not None,
                    "confidence": tl_result["confidence"],
                    "lit_region": tl_result.get("lit_region"),
                    "score": tl_result.get("score", 0.0),
                    "model_confidence": conf,
                    "area": max((x2 - x1) * (y2 - y1), 1),
                }
                
                traffic_light_candidates.append(candidate)

                # Set color for bounding box
                if candidate["color"] == "red":
                    color = (0, 0, 255)
                elif candidate["color"] == "yellow":
                    color = (0, 255, 255)
                elif candidate["color"] == "green":
                    color = (0, 255, 0)
                else:
                    color = (128, 128, 128)

                # Create label
                label = "TL: UNKNOWN"
                if candidate["color"] != "unknown":
                    label = f"TL: {candidate['color'].upper()}"
                    if candidate["countdown"] is not None:
                        label += f" ({candidate['countdown']}s)"
                    if candidate["confidence"] == "very_high":
                        label += " ***"
                    elif candidate["confidence"] == "high":
                        label += " **"
                    elif candidate["confidence"] == "medium":
                        label += " *"

                logger.info(
                    "Traffic light bbox=%s color=%s score=%.3f countdown=%s conf=%s",
                    [x1, y1, x2, y2],
                    candidate["color"],
                    candidate["score"],
                    candidate["countdown"],
                    candidate["confidence"]
                )
            else:
                color = (128, 128, 128)
                label = f"{name} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw divider lines for traffic lights
            if name == "traffic light":
                height = y2 - y1
                region_h = max(height // 3, 1)
                for i in range(1, 3):
                    line_y = y1 + i * region_h
                    cv2.line(image, (x1, line_y), (x2, line_y), (200, 200, 200), 1)

            # Draw label background and text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            objects_detailed.append({
                "class": name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            detected_objects.append(name)

    # Select the best traffic light from all candidates
    best_candidate = select_best_traffic_light(traffic_light_candidates)
    if best_candidate:
        traffic_light_info = best_candidate
        logger.info(f"Final traffic light selection: {best_candidate['color']} (score: {best_candidate['score']:.3f})")

    detection_info = {
        "counts": {
            "people": person_count,
            "vehicles": vehicle_count,
            "traffic_lights": tl_count,
            "total": len(detected_objects),
        },
        "traffic_light": traffic_light_info,
    }

    draw_detection_info(image, detection_info)

    seq_num = get_next_sequence_number(DETECTION_FOLDER)
    filename = f"{seq_num}.jpg"
    save_path = os.path.join(DETECTION_FOLDER, filename)
    cv2.imwrite(save_path, image)

    orig_path = os.path.join(UPLOAD_FOLDER, filename)
    if image_path != orig_path:
        cv2.imwrite(orig_path, original)

    logger.info(f"Detection #{seq_num}: {person_count} people, "
                f"{vehicle_count} vehicles, {tl_count} traffic lights - Final TL: {traffic_light_info['color']}")

    return {
        "id": seq_num,
        "sequence_number": seq_num,
        "image_path": save_path,
        "original_image": orig_path,
        "image_url": f"/detections/{seq_num}.jpg",
        "original_url": f"/uploads/{seq_num}.jpg",
        "objects": detected_objects,
        "objects_detailed": objects_detailed,
        "traffic_light": traffic_light_info,
        "traffic_light_detected": traffic_light_info["detected"] and traffic_light_info["color"] != "unknown",
        "traffic_light_color": traffic_light_info["color"],
        "traffic_light_countdown": traffic_light_info["countdown"],
        "traffic_light_confidence": traffic_light_info["confidence"],
        "traffic_light_lit_region": traffic_light_info.get("lit_region"),
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "detection_count": len(detected_objects),
        "counts": detection_info["counts"],
        "timestamp": datetime.now().isoformat(),
    }


def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Warm up camera
    for _ in range(10):
        cap.read()

    best_frame, best_sharp = None, 0
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            sharp = cv2.Laplacian(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp > best_sharp:
                best_sharp, best_frame = sharp, frame

    cap.release()
    if best_frame is None:
        raise Exception("Failed to capture from webcam")

    seq_num = get_next_sequence_number(UPLOAD_FOLDER)
    filepath = os.path.join(UPLOAD_FOLDER, f"{seq_num}.jpg")
    cv2.imwrite(filepath, best_frame)
    logger.info(f"Captured #{seq_num} (sharpness={best_sharp:.1f})")
    return filepath, seq_num


__all__ = ["detect_objects", "capture_from_webcam"]