import cv2
import numpy as np

from cctv_roi_ai_event_extractor.lpr import draw_plate_recognitions


LPR_LOCATION_IOU_THRESHOLD = 0.10
VEHICLE_TRACK_DUPLICATE_IOU_THRESHOLD = 0.35


def get_bottom_center(bbox):
    """Use the bottom-center point as the ROI anchor for person/vehicle detections."""
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_bottom = int(y2)
    return x_center, y_bottom


def point_in_polygon(point, polygon_np):
    result = cv2.pointPolygonTest(polygon_np, point, False)
    return result >= 0


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = bbox_area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def crop_bbox_from_frame(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def is_vehicle_detection(det):
    return det.get("class_name") in {"car", "motorcycle", "bus", "truck"}


def suppress_duplicate_vehicle_tracks(detections, iou_threshold=VEHICLE_TRACK_DUPLICATE_IOU_THRESHOLD):
    """Remove overlapping vehicle boxes, keeping the highest-confidence detection."""
    kept = []
    for det in sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        if any(bbox_iou(det["bbox"], existing["bbox"]) >= iou_threshold for existing in kept):
            continue
        kept.append(det)
    return kept


def plate_text_distance(a, b):
    """Small edit-distance helper tuned for plate strings that may differ by one OCR error."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return 2
    if len(a) == len(b):
        return sum(1 for left, right in zip(a, b) if left != right)

    if len(a) > len(b):
        a, b = b, a
    i = 0
    j = 0
    edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return edits
        j += 1
    return edits + (len(b) - j)


def plate_texts_similar(a, b):
    return bool(a and b and plate_text_distance(a, b) <= 1)


def plate_recognitions_match(left, right):
    """Treat recognitions as the same sighting by plate text similarity or box overlap."""
    if plate_texts_similar(left.text, right.text):
        return True
    return bbox_iou(left.bbox, right.bbox) >= LPR_LOCATION_IOU_THRESHOLD


def plate_recognition_quality(item):
    crop_quality = float(getattr(item, "crop_quality", 0.0) or 0.0)
    if crop_quality > 0:
        return crop_quality
    valid_bonus = 1.0 if item.valid_taiwan_format else 0.0
    return valid_bonus + float(item.confidence or 0.0) + (float(item.detector_score or 0.0) * 0.1)


def make_polygon_mask(frame_shape, polygon):
    """Rasterize a polygon so bbox intersection can be checked cheaply."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon and len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def bbox_intersects_mask(bbox, mask):
    if mask is None:
        return False
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 < x1 or y2 < y1:
        return False
    return bool(np.any(mask[y1:y2 + 1, x1:x2 + 1]))


class SimpleIouTracker:
    """Lightweight fallback tracker that links detections by class and IOU."""

    def __init__(self, iou_threshold=0.25, max_missed=8):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        for track in self.tracks.values():
            track["matched"] = False
            track["missed"] += 1

        tracked_detections = []
        for det in detections:
            best_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                if track["matched"]:
                    continue
                if track["class_name"] != det["class_name"]:
                    continue
                score = bbox_iou(track["bbox"], det["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_id is None or best_iou < self.iou_threshold:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = {
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "matched": True,
                "missed": 0,
            }
            det_with_track = dict(det)
            det_with_track["track_id"] = best_id
            tracked_detections.append(det_with_track)

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed
        ]
        for track_id in stale_ids:
            self.tracks.pop(track_id, None)

        return tracked_detections


def draw_detection(frame, det, inside=True):
    x1, y1, x2, y2 = det["bbox"]
    color = (0, 255, 0) if inside else (0, 165, 255)
    label = f'{det["class_name"]} {det["score"]:.2f}'
    if det.get("track_id") is not None:
        label += f' #{det["track_id"]}'
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_anchor_point(frame, point, inside):
    color = (0, 255, 0) if inside else (0, 0, 255)
    cv2.circle(frame, point, 5, color, -1)


def draw_polygon_overlay(frame, polygon, color=(255, 255, 0), fill_color=(0, 255, 255)):
    out = frame.copy()
    if len(polygon) >= 3:
        overlay = out.copy()
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], fill_color)
        out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
        cv2.polylines(out, [pts], True, color, 2)
    elif len(polygon) >= 2:
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(out, [pts], False, color, 2)
    return out


def polygon_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return x, y, w, h


def build_screenshot_frame(
    frame,
    detections,
    polygon,
    polygon_np,
    draw_roi_on_screenshot,
    plate_recognitions=None,
    touch_polygon=None,
    show_plate_debug_crops=False,
):
    if not draw_roi_on_screenshot:
        annotated = frame.copy()
        return draw_plate_recognitions(annotated, plate_recognitions, show_debug_crops=show_plate_debug_crops)

    annotated = draw_polygon_overlay(frame.copy(), polygon)
    if touch_polygon:
        annotated = draw_polygon_overlay(annotated, touch_polygon, color=(255, 0, 255), fill_color=(255, 0, 255))
    for det in detections:
        anchor = get_bottom_center(det["bbox"])
        inside = point_in_polygon(anchor, polygon_np)
        draw_detection(annotated, det, inside=inside)
        draw_anchor_point(annotated, anchor, inside=inside)
    draw_plate_recognitions(annotated, plate_recognitions, show_debug_crops=show_plate_debug_crops)
    return annotated
