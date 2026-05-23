import os

import cv2
from ultralytics import YOLO

from cctv_roi_ai_event_extractor.compute import get_auto_device_info


def _resolve_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_int_env(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _is_cuda_device(device: str | None) -> bool:
    if device is None:
        return False
    text = str(device).strip().lower()
    return text.startswith("cuda") or text.isdigit() or "," in text


def _resolve_yolo_half(device: str | None) -> bool:
    value = os.getenv("CCTV_ROI_YOLO_HALF")
    if value is None or value.strip().lower() in {"", "auto"}:
        return _is_cuda_device(device)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def default_tracker_path() -> str:
    """Return the app-local tracker profile tuned for CCTV vehicle streams."""
    return os.path.join(os.path.dirname(__file__), "trackers", "cctv_botsort.yaml")


class ObjectDetector:
    """YOLO object detector/tracker wrapper for people and vehicle classes."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.4,
        detect_width: int = 1280,
        device: str | None = None,
        tracker_path: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.detect_width = max(320, int(detect_width))
        self.device = device or get_auto_device_info()["device"]
        self.tracker_path = tracker_path or default_tracker_path()
        self.half = _resolve_yolo_half(self.device)
        self.imgsz = _resolve_int_env("CCTV_ROI_YOLO_IMGSZ")
        self.target_classes = {
            "person",
            "car",
            "motorcycle",
            "bus",
            "truck",
        }
        self.target_class_ids = self._resolve_target_class_ids()
        self._configure_runtime()

    def _configure_runtime(self):
        """Apply conservative runtime optimizations without changing outputs."""
        try:
            import torch

            torch_threads = _resolve_int_env("CCTV_ROI_TORCH_THREADS")
            if torch_threads and torch_threads > 0:
                torch.set_num_threads(torch_threads)
            if _is_cuda_device(self.device):
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        opencv_threads = _resolve_int_env("CCTV_ROI_OPENCV_THREADS")
        if opencv_threads and opencv_threads > 0:
            try:
                cv2.setNumThreads(opencv_threads)
            except Exception:
                pass

        if _resolve_bool_env("CCTV_ROI_YOLO_FUSE", True):
            try:
                self.model.fuse()
            except Exception:
                pass

    def _resolve_target_class_ids(self):
        names = getattr(self.model, "names", None) or {}
        if isinstance(names, dict):
            return [int(cls_id) for cls_id, name in names.items() if name in self.target_classes]
        return [idx for idx, name in enumerate(names) if name in self.target_classes]

    def _inference_kwargs(self):
        kwargs = {
            "conf": self.conf,
            "verbose": False,
            "device": self.device,
        }
        if self.half and _is_cuda_device(self.device):
            kwargs["half"] = True
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        if self.target_class_ids:
            kwargs["classes"] = self.target_class_ids
        return kwargs

    def _prepare_detect_frame(self, frame):
        """Resize large frames for inference and return factors for restoring boxes."""
        h, w = frame.shape[:2]
        if w <= self.detect_width:
            return frame, 1.0, 1.0

        scale = self.detect_width / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x = w / float(new_w)
        scale_y = h / float(new_h)
        return resized, scale_x, scale_y

    def detect(self, frame):
        detect_frame, scale_x, scale_y = self._prepare_detect_frame(frame)
        results = self.model(detect_frame, **self._inference_kwargs())
        detections = []

        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                score = float(box.conf[0].item())

                if cls_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(round(x1 * scale_x))
                y1 = int(round(y1 * scale_y))
                x2 = int(round(x2 * scale_x))
                y2 = int(round(y2 * scale_y))
                detections.append({
                    "class_name": cls_name,
                    "score": score,
                    "bbox": (x1, y1, x2, y2),
                })

        return detections

    def reset_trackers(self):
        """Reset Ultralytics tracker state between unrelated streams or large time gaps."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        for tracker in trackers or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def track(self, frame, persist=True):
        detect_frame, scale_x, scale_y = self._prepare_detect_frame(frame)
        track_kwargs = self._inference_kwargs()
        track_kwargs["persist"] = bool(persist)
        if self.tracker_path:
            track_kwargs["tracker"] = self.tracker_path
        results = self.model.track(source=detect_frame, **track_kwargs)
        detections = []

        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                score = float(box.conf[0].item())

                if cls_name not in self.target_classes:
                    continue

                track_id = None
                if getattr(box, "id", None) is not None:
                    try:
                        track_id = int(box.id[0].item())
                    except Exception:
                        track_id = None
                if track_id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(round(x1 * scale_x))
                y1 = int(round(y1 * scale_y))
                x2 = int(round(x2 * scale_x))
                y2 = int(round(y2 * scale_y))
                detections.append({
                    "class_name": cls_name,
                    "score": score,
                    "bbox": (x1, y1, x2, y2),
                    "track_id": track_id,
                })

        return detections
