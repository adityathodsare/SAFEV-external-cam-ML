import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficLightDetector:
    def __init__(self):
        # Multiple shades per color so dim, overexposed, and slightly shifted lamps still match.
        self.hsv_ranges = {
            "red": [
                (np.array([0, 70, 90]), np.array([12, 255, 255])),
                (np.array([165, 70, 90]), np.array([180, 255, 255])),
                (np.array([0, 40, 170]), np.array([15, 170, 255])),
            ],
            "yellow": [
                (np.array([12, 60, 120]), np.array([38, 255, 255])),
                (np.array([15, 35, 180]), np.array([45, 180, 255])),
            ],
            "green": [
                (np.array([35, 45, 70]), np.array([100, 255, 255])),
                (np.array([45, 20, 150]), np.array([105, 170, 255])),
            ],
        }

        self.bgr_ranges = {
            "red": [
                (np.array([0, 0, 100]), np.array([140, 140, 255])),
                (np.array([20, 20, 150]), np.array([180, 180, 255])),
            ],
            "yellow": [
                (np.array([0, 80, 120]), np.array([170, 255, 255])),
                (np.array([20, 120, 150]), np.array([200, 255, 255])),
            ],
            "green": [
                (np.array([20, 80, 20]), np.array([220, 255, 180])),
                (np.array([80, 140, 80]), np.array([255, 255, 255])),
            ],
        }

    def _build_mask_from_ranges(self, image, ranges):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(image, lower, upper))
        return mask

    def _prepare_focus_roi(self, roi):
        """
        Keep the lamp housing, but suppress timer-heavy side areas for wide detections.
        """
        h, w = roi.shape[:2]
        if h == 0 or w == 0:
            return roi

        if w > h * 1.15:
            return roi[:, :max(int(w * 0.68), 1)]

        return roi

    def _get_layout(self, roi):
        h, w = roi.shape[:2]
        if h > w * 1.2:
            return "vertical"
        if w > h * 1.2:
            return "horizontal"
        return "single"

    def _get_position_boost_mask(self, shape, color_name, layout):
        h, w = shape[:2]
        boost = np.ones((h, w), dtype=np.float32)

        if layout == "vertical":
            if color_name == "red":
                boost[: int(h * 0.42), :] = 1.25
            elif color_name == "yellow":
                boost[int(h * 0.22): int(h * 0.78), :] = 1.22
            elif color_name == "green":
                boost[int(h * 0.52):, :] = 1.25
        elif layout == "horizontal":
            if color_name == "red":
                boost[:, : int(w * 0.42)] = 1.18
            elif color_name == "yellow":
                boost[:, int(w * 0.22): int(w * 0.78)] = 1.18
            elif color_name == "green":
                boost[:, int(w * 0.52):] = 1.18

        return boost

    def _score_color(self, roi, color_name, layout):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        hsv_mask = self._build_mask_from_ranges(hsv, self.hsv_ranges[color_name])
        bgr_mask = self._build_mask_from_ranges(roi, self.bgr_ranges[color_name])
        color_mask = cv2.bitwise_and(hsv_mask, bgr_mask)

        bright_threshold = max(110, int(np.percentile(gray, 72)))
        bright_mask = cv2.inRange(gray, bright_threshold, 255)

        kernel = np.ones((3, 3), np.uint8)
        lit_mask = cv2.bitwise_and(color_mask, bright_mask)
        lit_mask = cv2.morphologyEx(lit_mask, cv2.MORPH_OPEN, kernel)
        lit_mask = cv2.morphologyEx(lit_mask, cv2.MORPH_CLOSE, kernel)

        lit_pixels = cv2.countNonZero(lit_mask)
        if lit_pixels < 6:
            return 0.0, None

        contours, _ = cv2.findContours(lit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None

        focus_mask = self._get_position_boost_mask(roi.shape, color_name, layout)
        best_score = 0.0
        best_center = None
        area_total = roi.shape[0] * roi.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            component_mask = np.zeros(lit_mask.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [cnt], -1, 255, -1)

            mean_v = cv2.mean(hsv[:, :, 2], mask=component_mask)[0] / 255.0
            mean_s = cv2.mean(hsv[:, :, 1], mask=component_mask)[0] / 255.0
            weighted_focus = float(np.mean(focus_mask[component_mask > 0])) if np.any(component_mask > 0) else 1.0
            compactness = min(w, h) / max(w, h)
            area_ratio = area / max(area_total, 1)

            score = (
                area_ratio * 3.4
                + mean_v * 0.9
                + mean_s * 0.9
                + compactness * 0.45
            ) * weighted_focus

            if score > best_score:
                best_score = score
                best_center = (x + w / 2.0, y + h / 2.0)

        return best_score, best_center

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

        roi = self._prepare_focus_roi(roi)
        layout = self._get_layout(roi)

        scores = {}
        centers = {}
        for color_name in ("red", "yellow", "green"):
            score, center = self._score_color(roi, color_name, layout)
            scores[color_name] = score
            centers[color_name] = center

        best_color = max(scores, key=scores.get)
        best_score = scores[best_color]
        second_score = max([v for k, v in scores.items() if k != best_color], default=0.0)
        score_gap = best_score - second_score

        if best_score < 0.12:
            detected_color = "unknown"
            confidence = "low"
            lit_region = None
        else:
            detected_color = best_color
            if layout == "vertical" and centers[best_color] is not None:
                cy = centers[best_color][1] / max(roi.shape[0], 1)
                if cy < 0.34:
                    lit_region = "top"
                elif cy < 0.67:
                    lit_region = "middle"
                else:
                    lit_region = "bottom"
            elif layout == "horizontal" and centers[best_color] is not None:
                cx = centers[best_color][0] / max(roi.shape[1], 1)
                if cx < 0.34:
                    lit_region = "left"
                elif cx < 0.67:
                    lit_region = "middle"
                else:
                    lit_region = "right"
            else:
                lit_region = "single"

            if best_score >= 0.42 and score_gap >= 0.08:
                confidence = "very_high"
            elif best_score >= 0.24:
                confidence = "high"
            else:
                confidence = "medium"

        logger.info(
            "Traffic light scores red=%.3f yellow=%.3f green=%.3f layout=%s result=%s",
            scores["red"],
            scores["yellow"],
            scores["green"],
            layout,
            detected_color,
        )

        return {
            "color": detected_color,
            "countdown": None,
            "countdown_detected": False,
            "confidence": confidence,
            "lit_region": lit_region,
            "score": round(float(best_score), 3),
        }


traffic_light_detector = TrafficLightDetector()
