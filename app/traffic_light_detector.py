import cv2
import numpy as np
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficLightDetector:
    def __init__(self):
        # Tuned for illuminated lamps in low-quality dashcam frames.
        self.color_ranges = {
            "red": [
                (np.array([0, 130, 130]), np.array([12, 255, 255])),
                (np.array([168, 130, 130]), np.array([180, 255, 255])),
            ],
            "yellow": [
                (np.array([14, 110, 140]), np.array([40, 255, 255])),
            ],
            "green": [
                (np.array([38, 90, 110]), np.array([95, 255, 255])),
            ],
        }
        self.recent_colors = deque(maxlen=3)

    def _build_color_mask(self, hsv, color_name):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    def _crop_signal_housing(self, roi):
        """
        Remove side regions that often contain countdown digits instead of lamps.
        """
        h, w = roi.shape[:2]
        if h == 0 or w == 0:
            return roi

        if w / max(h, 1) > 0.75:
            return roi[:, :max(int(w * 0.62), 1)]

        pad_x = max(1, int(w * 0.08))
        pad_y = max(1, int(h * 0.04))
        end_y = max(h - pad_y, pad_y + 1)
        end_x = max(w - pad_x, pad_x + 1)
        return roi[pad_y:end_y, pad_x:end_x]

    def _score_region(self, region_roi, expected_color):
        if region_roi is None or region_roi.size == 0:
            return {
                "color": "unknown",
                "score": 0.0,
                "lit_ratio": 0.0,
                "brightness": 0.0,
                "circularity": 0.0,
            }

        hsv = cv2.cvtColor(region_roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)

        bright_threshold = max(160, int(np.percentile(gray, 88)))
        bright_mask = cv2.inRange(gray, bright_threshold, 255)
        kernel = np.ones((3, 3), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        color_mask = self._build_color_mask(hsv, expected_color)
        lit_mask = cv2.bitwise_and(bright_mask, color_mask)
        lit_pixels = cv2.countNonZero(lit_mask)
        region_pixels = region_roi.shape[0] * region_roi.shape[1]
        lit_ratio = lit_pixels / max(region_pixels, 1)

        if lit_pixels < 10:
            return {
                "color": "unknown",
                "score": 0.0,
                "lit_ratio": lit_ratio,
                "brightness": 0.0,
                "circularity": 0.0,
            }

        contours, _ = cv2.findContours(lit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_blob_score = 0.0
        best_circularity = 0.0
        best_brightness = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            if not 0.55 <= aspect_ratio <= 1.8:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < 0.32:
                continue

            component_mask = np.zeros(lit_mask.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [cnt], -1, 255, -1)
            mean_v = cv2.mean(hsv[:, :, 2], mask=component_mask)[0] / 255.0
            area_ratio = area / max(region_pixels, 1)
            score = (area_ratio * 2.8) + (mean_v * 0.9) + (circularity * 1.4)

            if score > best_blob_score:
                best_blob_score = score
                best_circularity = circularity
                best_brightness = mean_v

        if best_blob_score < 0.32:
            return {
                "color": "unknown",
                "score": best_blob_score,
                "lit_ratio": lit_ratio,
                "brightness": best_brightness,
                "circularity": best_circularity,
            }

        return {
            "color": expected_color,
            "score": best_blob_score,
            "lit_ratio": lit_ratio,
            "brightness": best_brightness,
            "circularity": best_circularity,
        }

    def analyze_traffic_light(self, image, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                "color": "unknown",
                "countdown": None,
                "countdown_detected": False,
                "confidence": "low",
                "lit_region": None,
                "score": 0.0,
            }

        roi = self._crop_signal_housing(roi)
        height = roi.shape[0]
        region_height = max(height // 3, 1)
        regions = {
            "top": roi[0:region_height, :],
            "middle": roi[region_height:min(2 * region_height, height), :],
            "bottom": roi[min(2 * region_height, height):height, :],
        }
        expected_colors = {
            "top": "red",
            "middle": "yellow",
            "bottom": "green",
        }

        region_scores = {}
        for position, region_roi in regions.items():
            if region_roi.size == 0:
                continue
            region_scores[position] = self._score_region(region_roi, expected_colors[position])

        detected_color = "unknown"
        lit_region = None
        confidence = "low"
        best_score = 0.0

        if region_scores:
            lit_region, best_result = max(region_scores.items(), key=lambda item: item[1]["score"])
            best_score = best_result["score"]
            ordered_scores = sorted(
                [result["score"] for result in region_scores.values()],
                reverse=True,
            )
            score_gap = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0]

            if best_result["color"] != "unknown" and best_score >= 0.42 and score_gap >= 0.08:
                detected_color = expected_colors[lit_region]
                if best_score >= 1.05:
                    confidence = "very_high"
                elif best_score >= 0.68:
                    confidence = "high"
                else:
                    confidence = "medium"
                logger.info(
                    "Traffic light %s in %s region (score=%.2f gap=%.2f)",
                    detected_color,
                    lit_region,
                    best_score,
                    score_gap,
                )
            else:
                lit_region = None

        return {
            "color": detected_color,
            "countdown": None,
            "countdown_detected": False,
            "confidence": confidence,
            "lit_region": lit_region,
            "score": round(float(best_score), 3),
        }


traffic_light_detector = TrafficLightDetector()
