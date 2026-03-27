import os
import sys
import csv
import json
import shutil
import threading
import queue
import urllib.request
from datetime import datetime

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None

from PySide6.QtCore import QObject, QPoint, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None


APP_VERSION = "4.4.0-roi-yolo26x-dragdrop-paste-paths"
LONG_STAY_SCREENSHOT_INTERVAL_SEC = 5.0


# ---------------------------
# 路徑工具：確保 EXE 放哪就跑哪
# ---------------------------
def get_app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_auto_device_info() -> dict:
    info = {
        "device": "cpu",
        "name": "CPU",
        "source": "cpu"
    }

    try:
        if torch is not None and torch.cuda.is_available():
            count = torch.cuda.device_count()

            if count > 0:
                for i in range(count):
                    name = torch.cuda.get_device_name(i)
                    if "nvidia" in name.lower():
                        return {
                            "device": f"cuda:{i}",
                            "name": name,
                            "source": "cuda"
                        }

                name = torch.cuda.get_device_name(0)
                return {
                    "device": "cuda:0",
                    "name": name,
                    "source": "cuda"
                }
    except Exception as e:
        info["name"] = f"CPU（CUDA偵測失敗：{e}）"

    return info


def resolve_default_model_path(app_dir: str) -> str:
    candidates = [
        os.path.join(app_dir, "models", "yolo26x.pt"),
        os.path.join(app_dir, "yolo26x.pt"),
        os.path.join(app_dir, "models", "yolo26n.pt"),
        os.path.join(app_dir, "yolo26n.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def _iter_model_download_urls(model_path: str):
    basename = os.path.basename(model_path)
    stem = os.path.splitext(basename)[0]
    env_names = [
        f"{stem.upper()}_MODEL_URL",
        "CCTV_ROI_MODEL_URL",
        "YOLO_MODEL_URL",
    ]
    seen = set()
    for env_name in env_names:
        url = (os.getenv(env_name) or "").strip()
        if url and url not in seen:
            seen.add(url)
            yield env_name, url


def _download_model_from_url(url: str, target_path: str, status_cb=None):
    ensure_dir(os.path.dirname(target_path))
    temp_path = target_path + ".download"

    if status_cb:
        status_cb(f"[MODEL] 下載模型中：{url}")

    with urllib.request.urlopen(url, timeout=120) as resp, open(temp_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    if not os.path.exists(temp_path) or os.path.getsize(temp_path) <= 0:
        raise RuntimeError("下載完成但檔案為空。")

    os.replace(temp_path, target_path)
    return target_path


def ensure_model_available(model_path: str, status_cb=None):
    model_path = os.path.abspath(model_path)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return True, model_path

    basename = os.path.basename(model_path)

    for env_name, url in _iter_model_download_urls(model_path):
        try:
            downloaded_path = _download_model_from_url(url, model_path, status_cb=status_cb)
            return True, downloaded_path
        except Exception as e:
            if status_cb:
                status_cb(f"[MODEL] {env_name} 下載失敗：{e}")

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        if status_cb:
            status_cb(f"[MODEL] 嘗試透過 Ultralytics 自動下載：{basename}")

        downloaded = attempt_download_asset(basename)
        if downloaded and os.path.exists(downloaded):
            ensure_dir(os.path.dirname(model_path))
            if norm_path(downloaded) != norm_path(model_path):
                shutil.copy2(downloaded, model_path)
            return True, model_path if os.path.exists(model_path) else downloaded
    except Exception as e:
        if status_cb:
            status_cb(f"[MODEL] Ultralytics 自動下載失敗：{e}")

    message = (
        f"找不到模型檔：\n{model_path}\n\n"
        "已嘗試：\n"
        "1. 本地 models 資料夾\n"
        "2. 環境變數 URL（YOLO26X_MODEL_URL / CCTV_ROI_MODEL_URL / YOLO_MODEL_URL）\n"
        "3. Ultralytics 資產自動下載\n"
    )
    return False, message


def norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def is_subpath(child_path: str, parent_path: str) -> bool:
    try:
        child = norm_path(child_path)
        parent = norm_path(parent_path)
        common = os.path.commonpath([child, parent])
        return common == parent
    except Exception:
        return False


def safe_relpath(full_path: str, base_dir: str) -> str:
    try:
        return os.path.relpath(full_path, base_dir)
    except Exception:
        return os.path.basename(full_path)


# ---------------------------
# ROI 設定存檔 / 載入（多邊形）
# ---------------------------
def get_roi_config_path(app_dir: str) -> str:
    return os.path.join(app_dir, "roi_config_polygon.json")


def load_roi_config(app_dir: str):
    path = get_roi_config_path(app_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        polygon = data.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            return None

        clean_points = []
        for pt in polygon:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                return None
            x, y = pt
            if not isinstance(x, int) or not isinstance(y, int):
                return None
            clean_points.append((x, y))
        return clean_points
    except Exception:
        return None


def save_roi_config(app_dir: str, polygon):
    path = get_roi_config_path(app_dir)
    data = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "polygon": [[int(x), int(y)] for x, y in polygon]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------
# 寫圖
# ---------------------------
def save_frame(out_dir, base_name, t_sec, frame_idx, frame_bgr):
    ensure_dir(out_dir)
    fn = f"{base_name}__t{t_sec:010.2f}s__f{frame_idx:09d}.jpg"
    out_path = os.path.abspath(os.path.join(out_dir, fn))

    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            with open(out_path, "wb") as f:
                f.write(buf.tobytes())
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return True, out_path
    except Exception:
        pass

    try:
        ok = cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True, out_path
    except Exception:
        pass

    return False, out_path


# ---------------------------
# 安全讀取影片資訊
# ---------------------------
def safe_get_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        return 25.0
    try:
        if np.isnan(fps) or fps <= 1:
            return 25.0
    except Exception:
        return 25.0
    return float(fps)


def safe_get_int_prop(cap, prop_id) -> int:
    try:
        val = int(cap.get(prop_id) or 0)
        return max(val, 0)
    except Exception:
        return 0


# ---------------------------
# Polygon ROI Picker
# ---------------------------
class PolygonROIPicker:
    def __init__(self, video_path: str, preset_polygon=None, display_width=1400):
        self.video_path = video_path
        self.points = list(preset_polygon) if preset_polygon else []
        self._img = None
        self._display_img = None
        self._scale = 1.0
        self._display_width = display_width
        self._win = "Polygon ROI Picker"

    def _resize_for_display(self, frame):
        h, w = frame.shape[:2]
        if w <= self._display_width:
            self._scale = 1.0
            return frame.copy()
        self._scale = self._display_width / w
        new_h = int(h * self._scale)
        return cv2.resize(frame, (self._display_width, new_h), interpolation=cv2.INTER_AREA)

    def _display_to_original(self, x, y):
        if self._img is None:
            return 0, 0

        h, w = self._img.shape[:2]
        ox = int(round(x / self._scale))
        oy = int(round(y / self._scale))

        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))
        return ox, oy

    def _draw_preview(self):
        frame = self._img.copy()

        if len(self.points) >= 3:
            overlay = frame.copy()
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        elif len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

        for idx, (x, y) in enumerate(self.points, start=1):
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                frame, str(idx), (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        info_lines = [
            f"目前點數: {len(self.points)}",
            "左鍵：新增點",
            "右鍵 / Backspace：刪除最後一點",
            "C：清空全部點",
            "Enter / Space：確認 ROI",
            "ESC：取消"
        ]

        y0 = 30
        for line in info_lines:
            cv2.putText(
                frame, line, (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
            )
            y0 += 32

        self._display_img = self._resize_for_display(frame)

    def _mouse_cb(self, event, x, y, flags, param):
        if self._img is None:
            return

        if event == cv2.EVENT_LBUTTONUP:
            ox, oy = self._display_to_original(x, y)
            self.points.append((ox, oy))
            self._draw_preview()
            cv2.imshow(self._win, self._display_img)

        elif event == cv2.EVENT_RBUTTONUP:
            if self.points:
                self.points.pop()
                self._draw_preview()
                cv2.imshow(self._win, self._display_img)

    def pick(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None

        ok, img = cap.read()
        cap.release()

        if not ok or img is None:
            return None

        self._img = img.copy()
        self._draw_preview()

        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win, self._display_img.shape[1], self._display_img.shape[0])

        cv2.imshow(self._win, self._display_img)
        cv2.waitKey(1)
        cv2.setMouseCallback(self._win, self._mouse_cb)

        try:
            cv2.setWindowProperty(self._win, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        while True:
            cv2.imshow(self._win, self._display_img)
            key = cv2.waitKeyEx(20)

            if key == 27:
                cv2.destroyWindow(self._win)
                return None
            elif key in (ord('c'), ord('C')):
                self.points = []
                self._draw_preview()
            elif key == 8:
                if self.points:
                    self.points.pop()
                    self._draw_preview()
            elif key in (13, 10, 32, 141):
                if len(self.points) >= 3:
                    cv2.destroyWindow(self._win)
                    return self.points


# ---------------------------
# 找第一支可成功開啟並可讀到第一幀的影片
# ---------------------------
def find_first_readable_video(video_paths):
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return vp
    return None


# ---------------------------
# AI Detector
# ---------------------------
class ObjectDetector:
    def __init__(self, model_path: str, conf: float = 0.4, detect_width: int = 1280, device: str | None = None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.detect_width = max(320, int(detect_width))
        self.device = device or get_auto_device_info()["device"]
        self.target_classes = {
            "person",
            "car",
            "motorcycle",
            "bus",
            "truck"
        }

    def _prepare_detect_frame(self, frame):
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
        results = self.model(detect_frame, conf=self.conf, verbose=False, device=self.device)
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
                    "bbox": (x1, y1, x2, y2)
                })

        return detections


# ---------------------------
# ROI / 繪圖工具
# ---------------------------
def get_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_bottom = int(y2)
    return x_center, y_bottom


def point_in_polygon(point, polygon_np):
    result = cv2.pointPolygonTest(polygon_np, point, False)
    return result >= 0


def draw_detection(frame, det, inside=True):
    x1, y1, x2, y2 = det["bbox"]
    color = (0, 255, 0) if inside else (0, 165, 255)
    label = f'{det["class_name"]} {det["score"]:.2f}'
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_anchor_point(frame, point, inside):
    color = (0, 255, 0) if inside else (0, 0, 255)
    cv2.circle(frame, point, 5, color, -1)


def draw_polygon_overlay(frame, polygon):
    out = frame.copy()
    if len(polygon) >= 3:
        overlay = out.copy()
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
        cv2.polylines(out, [pts], True, (255, 255, 0), 2)
    elif len(polygon) >= 2:
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(out, [pts], False, (255, 255, 0), 2)
    return out


def polygon_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return x, y, w, h


def build_screenshot_frame(frame, detections, polygon, polygon_np, draw_roi_on_screenshot):
    if not draw_roi_on_screenshot:
        return frame.copy()

    annotated = draw_polygon_overlay(frame.copy(), polygon)
    for det in detections:
        anchor = get_bottom_center(det["bbox"])
        inside = point_in_polygon(anchor, polygon_np)
        draw_detection(annotated, det, inside=inside)
        draw_anchor_point(annotated, anchor, inside=inside)
    return annotated


def try_save_screenshot(logs, screenshot_out_dir, base_name, rel_video_path, frame_idx, current_time_sec,
                        frame, detections, polygon, polygon_np, draw_roi_on_screenshot):
    screenshot_frame = build_screenshot_frame(
        frame=frame,
        detections=detections,
        polygon=polygon,
        polygon_np=polygon_np,
        draw_roi_on_screenshot=draw_roi_on_screenshot
    )

    ok_save, shot_path = save_frame(
        screenshot_out_dir,
        base_name,
        current_time_sec,
        frame_idx,
        screenshot_frame
    )

    logs.append({
        "type": "screenshot",
        "video_rel_path": rel_video_path,
        "event_time_sec": f"{current_time_sec:.2f}",
        "interval_start_sec": "",
        "interval_end_sec": "",
        "output_path": shot_path or "",
        "status": "OK" if ok_save else "SAVE_FAIL"
    })
    return ok_save, shot_path


# ---------------------------
# 輸出單一片段（原始影片，不畫框）
# ---------------------------
def export_interval_clip(
    video_path: str,
    clip_out_dir: str,
    base_name: str,
    start_t: float,
    end_t: float,
    clip_index: int,
    status_cb=None
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if status_cb:
            status_cb(f"[CLIP-SKIP] 無法開啟影片輸出片段：{os.path.basename(video_path)}")
        return False, None

    fps = safe_get_fps(cap)
    frame_w = safe_get_int_prop(cap, cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = safe_get_int_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = safe_get_int_prop(cap, cv2.CAP_PROP_FRAME_COUNT)

    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        if status_cb:
            status_cb(f"[CLIP-SKIP] 影片尺寸異常：{os.path.basename(video_path)}")
        return False, None

    start_t = max(0.0, start_t)
    if total_frames > 0:
        total_sec = total_frames / fps
        end_t = min(end_t, total_sec)

    start_frame = max(0, int(round(start_t * fps)))
    end_frame_exclusive = max(start_frame + 1, int(round(end_t * fps)))

    if total_frames > 0:
        end_frame_exclusive = min(end_frame_exclusive, total_frames)

    if end_frame_exclusive <= start_frame:
        cap.release()
        if status_cb:
            status_cb(f"[CLIP-SKIP] 合併片段範圍無效：{base_name}")
        return False, None

    ensure_dir(clip_out_dir)

    out_name = (
        f"{base_name}__clip{clip_index:03d}"
        f"__from{start_t:010.2f}s"
        f"__to{end_t:010.2f}s.mp4"
    )
    out_path = os.path.join(clip_out_dir, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        cap.release()
        if status_cb:
            status_cb(f"[CLIP-SKIP] 無法建立輸出影片：{out_name}")
        return False, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current = start_frame
    wrote_any = False

    while current < end_frame_exclusive:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        wrote_any = True
        current += 1

    writer.release()
    cap.release()

    if not wrote_any:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        if status_cb:
            status_cb(f"[CLIP-SKIP] 未成功寫入任何影格：{out_name}")
        return False, None

    if status_cb:
        status_cb(f"[CLIP-DONE] {os.path.basename(out_path)}")
    return True, out_path


# ---------------------------
# 主處理（AI 事件版）
# ---------------------------
def process_video(
    video_path: str,
    rel_video_path: str,
    screenshots_root: str,
    clips_root: str,
    polygon,
    detector,
    start_trigger_frames: int,
    end_hold_sec: float,
    pre_event_sec: float,
    post_event_sec: float,
    draw_roi_on_screenshot: bool,
    export_screenshots: bool,
    export_clips: bool,
    detect_every_n_frames: int,
    progress_cb=None,
    status_cb=None,
    stop_checker=None
):
    logs = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if status_cb:
            status_cb(f"[SKIP] 無法開啟：{os.path.basename(video_path)}")
        return {
            "status": "SKIP_OPEN_FAIL",
            "grabbed_count": 0,
            "clip_count": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "frames": 0,
            "logs": logs
        }

    fps = safe_get_fps(cap)
    frame_w = safe_get_int_prop(cap, cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = safe_get_int_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT)
    total = safe_get_int_prop(cap, cv2.CAP_PROP_FRAME_COUNT)

    if frame_w <= 0 or frame_h <= 0:
        if status_cb:
            status_cb(f"[SKIP] 影片尺寸異常：{os.path.basename(video_path)}")
        cap.release()
        return {
            "status": "SKIP_BAD_DIM",
            "grabbed_count": 0,
            "clip_count": 0,
            "fps": fps,
            "width": frame_w,
            "height": frame_h,
            "frames": total,
            "logs": logs
        }

    rel_dir = os.path.dirname(rel_video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    screenshot_out_dir = os.path.join(screenshots_root, rel_dir)
    clip_out_dir = os.path.join(clips_root, rel_dir)
    ensure_dir(screenshot_out_dir)
    ensure_dir(clip_out_dir)

    detect_every_n_frames = max(1, int(detect_every_n_frames))

    if status_cb:
        status_cb(
            f"[RUN] {rel_video_path} | FPS={fps:.2f} | {frame_w}x{frame_h} | "
            f"Frames={total} | detect_width={detector.detect_width} | stride={detect_every_n_frames}"
        )

    pre_event_frames = max(0, int(round(pre_event_sec * fps)))
    post_event_frames = max(0, int(round(post_event_sec * fps)))
    end_hold_frames = max(1, int(round(end_hold_sec * fps)))
    start_trigger_frames = max(1, int(start_trigger_frames))

    frame_idx = 0
    grabbed_count = 0
    clip_count = 0

    event_intervals = []
    polygon_np = np.array(polygon, dtype=np.int32)

    in_event = False
    event_start_frame = None
    last_inside_frame = None
    start_counter = 0

    cached_detections = []
    cached_inside_present = False
    next_long_stay_shot_frame = None

    while True:
        if stop_checker and stop_checker():
            cap.release()
            if status_cb:
                status_cb(f"[STOP] 已停止：{rel_video_path}")
            return {
                "status": "STOPPED",
                "grabbed_count": grabbed_count,
                "clip_count": clip_count,
                "fps": fps,
                "width": frame_w,
                "height": frame_h,
                "frames": total,
                "logs": logs
            }

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1

        if progress_cb and total > 0 and frame_idx % 10 == 0:
            progress_cb(frame_idx, total)

        # 關鍵原則：
        # 只有真正的偵測幀，才更新事件邏輯。
        # 非偵測幀仍可沿用快取結果供顯示 / 截圖參考，
        # 但不可拿來推進 start_counter / last_inside_frame，
        # 否則會產生事件起訖時間被灌水延伸的問題。
        should_detect = ((frame_idx - 1) % detect_every_n_frames == 0)

        if should_detect:
            cached_detections = detector.detect(frame)

            inside_count = 0
            for det in cached_detections:
                anchor = get_bottom_center(det["bbox"])
                if point_in_polygon(anchor, polygon_np):
                    inside_count += 1

            cached_inside_present = (inside_count > 0)

        detections = cached_detections
        inside_present = cached_inside_present
        current_time_sec = frame_idx / fps

        # ---------------------------------
        # 只有在真正偵測幀，才更新事件狀態
        # ---------------------------------
        if should_detect:
            if not in_event:
                if inside_present:
                    start_counter += 1
                else:
                    start_counter = 0

                if start_counter >= start_trigger_frames:
                    in_event = True

                    # 關鍵修正：
                    # 舊寫法：
                    #   trigger_frame = frame_idx - start_trigger_frames + 1
                    # 只適用於每一幀都 detect。
                    #
                    # 新寫法：
                    #   若 stride > 1，連續命中 start_trigger_frames 次，
                    #   代表跨越的是「偵測幀間距」，不是逐幀。
                    #   因此需依 detect_every_n_frames 回推真正較合理的事件起點。
                    trigger_frame = frame_idx - ((start_trigger_frames - 1) * detect_every_n_frames)

                    # 再往前保留 pre_event_sec 對應的 frame 數，
                    # 讓事件片段起點真正往前延伸。
                    event_start_frame = max(1, trigger_frame - pre_event_frames)

                    # 事件一成立，以當前偵測幀視為最後一次確認在 ROI 內的幀
                    last_inside_frame = frame_idx

                    if export_screenshots:
                        ok_save, shot_path = try_save_screenshot(
                            logs=logs,
                            screenshot_out_dir=screenshot_out_dir,
                            base_name=base_name,
                            rel_video_path=rel_video_path,
                            frame_idx=frame_idx,
                            current_time_sec=current_time_sec,
                            frame=frame,
                            detections=detections,
                            polygon=polygon,
                            polygon_np=polygon_np,
                            draw_roi_on_screenshot=draw_roi_on_screenshot
                        )
                        if ok_save:
                            grabbed_count += 1
                        if status_cb:
                            if ok_save:
                                status_cb(f"[SHOT] 已輸出截圖：{os.path.basename(shot_path)}")
                            else:
                                status_cb(f"[SHOT-FAIL] 截圖寫入失敗：{shot_path}")

                    next_long_stay_shot_frame = frame_idx + max(
                        1,
                        int(round(LONG_STAY_SCREENSHOT_INTERVAL_SEC * fps))
                    )

            else:
                if inside_present:
                    # 只有真正偵測到仍在 ROI 內，才更新 last_inside_frame
                    last_inside_frame = frame_idx

                    if (
                        export_screenshots
                        and next_long_stay_shot_frame is not None
                        and frame_idx >= next_long_stay_shot_frame
                    ):
                        ok_save, shot_path = try_save_screenshot(
                            logs=logs,
                            screenshot_out_dir=screenshot_out_dir,
                            base_name=base_name,
                            rel_video_path=rel_video_path,
                            frame_idx=frame_idx,
                            current_time_sec=current_time_sec,
                            frame=frame,
                            detections=detections,
                            polygon=polygon,
                            polygon_np=polygon_np,
                            draw_roi_on_screenshot=draw_roi_on_screenshot
                        )
                        if ok_save:
                            grabbed_count += 1
                        if status_cb:
                            if ok_save:
                                status_cb(f"[SHOT] 長時間停留補抓：{os.path.basename(shot_path)}")
                            else:
                                status_cb(f"[SHOT-FAIL] 長時間停留補抓失敗：{shot_path}")

                        next_long_stay_shot_frame = frame_idx + max(
                            1,
                            int(round(LONG_STAY_SCREENSHOT_INTERVAL_SEC * fps))
                        )

                # 注意：
                # 事件結束判定也只在偵測幀做，
                # 避免非偵測幀用舊快取把事件尾巴硬拖長。
                if last_inside_frame is not None:
                    frames_since_last_inside = frame_idx - last_inside_frame
                    if frames_since_last_inside >= end_hold_frames:
                        raw_end_frame = last_inside_frame
                        event_end_frame = raw_end_frame + post_event_frames
                        if total > 0:
                            event_end_frame = min(event_end_frame, total)

                        start_t = max(0.0, (event_start_frame - 1) / fps)
                        end_t = max(start_t, event_end_frame / fps)
                        event_intervals.append((start_t, end_t))

                        in_event = False
                        event_start_frame = None
                        last_inside_frame = None
                        start_counter = 0
                        next_long_stay_shot_frame = None

    if in_event and event_start_frame is not None:
        raw_end_frame = last_inside_frame if last_inside_frame is not None else frame_idx
        event_end_frame = raw_end_frame + post_event_frames
        if total > 0:
            event_end_frame = min(event_end_frame, total)

        start_t = max(0.0, (event_start_frame - 1) / fps)
        end_t = max(start_t, event_end_frame / fps)
        event_intervals.append((start_t, end_t))

    cap.release()

    if export_clips and event_intervals:
        if status_cb:
            status_cb(f"[CLIP] {rel_video_path} 事件 {len(event_intervals)} 個，準備輸出片段")

        for idx, (start_t, end_t) in enumerate(event_intervals, start=1):
            if stop_checker and stop_checker():
                if status_cb:
                    status_cb(f"[STOP] 片段輸出中止：{rel_video_path}")
                return {
                    "status": "STOPPED",
                    "grabbed_count": grabbed_count,
                    "clip_count": clip_count,
                    "fps": fps,
                    "width": frame_w,
                    "height": frame_h,
                    "frames": total,
                    "logs": logs
                }

            ok_clip, clip_path = export_interval_clip(
                video_path=video_path,
                clip_out_dir=clip_out_dir,
                base_name=base_name,
                start_t=start_t,
                end_t=end_t,
                clip_index=idx,
                status_cb=status_cb
            )
            if ok_clip:
                clip_count += 1
                logs.append({
                    "type": "clip",
                    "video_rel_path": rel_video_path,
                    "event_time_sec": "",
                    "interval_start_sec": f"{start_t:.2f}",
                    "interval_end_sec": f"{end_t:.2f}",
                    "output_path": clip_path,
                    "status": "OK"
                })

    if progress_cb and total > 0:
        progress_cb(total, total)

    if status_cb:
        status_cb(f"[DONE] {rel_video_path} 擷取 {grabbed_count} 張，輸出 {clip_count} 支事件片段")

    return {
        "status": "OK",
        "grabbed_count": grabbed_count,
        "clip_count": clip_count,
        "fps": fps,
        "width": frame_w,
        "height": frame_h,
        "frames": total,
        "logs": logs
    }


# ---------------------------
# 一次填完全部參數的視窗
# ---------------------------
class ParamsDialog(tk.Toplevel):
    def __init__(
        self,
        parent,
        confidence,
        start_trigger_frames,
        end_hold_sec,
        pre_event_sec,
        post_event_sec,
        detect_width,
        detect_every_n_frames
    ):
        super().__init__(parent)
        self.title("AI 參數設定")
        self.resizable(False, False)
        self.result = None

        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)

        ttk.Label(self, text="請一次輸入所有 AI 參數：").grid(
            row=0, column=0, columnspan=2, padx=12, pady=(12, 10), sticky="w"
        )

        ttk.Label(self, text="YOLO 偵測置信度：").grid(row=1, column=0, padx=12, pady=6, sticky="e")
        self.ent_conf = ttk.Entry(self, width=18)
        self.ent_conf.grid(row=1, column=1, padx=12, pady=6, sticky="w")
        self.ent_conf.insert(0, str(confidence))

        ttk.Label(self, text="事件開始連續幀數：").grid(row=2, column=0, padx=12, pady=6, sticky="e")
        self.ent_start = ttk.Entry(self, width=18)
        self.ent_start.grid(row=2, column=1, padx=12, pady=6, sticky="w")
        self.ent_start.insert(0, str(start_trigger_frames))

        ttk.Label(self, text="事件結束等待秒數：").grid(row=3, column=0, padx=12, pady=6, sticky="e")
        self.ent_end_hold = ttk.Entry(self, width=18)
        self.ent_end_hold.grid(row=3, column=1, padx=12, pady=6, sticky="w")
        self.ent_end_hold.insert(0, str(end_hold_sec))

        ttk.Label(self, text="事件前保留秒數：").grid(row=4, column=0, padx=12, pady=6, sticky="e")
        self.ent_pre = ttk.Entry(self, width=18)
        self.ent_pre.grid(row=4, column=1, padx=12, pady=6, sticky="w")
        self.ent_pre.insert(0, str(pre_event_sec))

        ttk.Label(self, text="事件後保留秒數：").grid(row=5, column=0, padx=12, pady=6, sticky="e")
        self.ent_post = ttk.Entry(self, width=18)
        self.ent_post.grid(row=5, column=1, padx=12, pady=6, sticky="w")
        self.ent_post.insert(0, str(post_event_sec))

        ttk.Label(self, text="偵測前縮圖寬度：").grid(row=6, column=0, padx=12, pady=6, sticky="e")
        self.ent_detect_width = ttk.Entry(self, width=18)
        self.ent_detect_width.grid(row=6, column=1, padx=12, pady=6, sticky="w")
        self.ent_detect_width.insert(0, str(detect_width))

        ttk.Label(self, text="每幾幀偵測一次：").grid(row=7, column=0, padx=12, pady=6, sticky="e")
        self.ent_detect_stride = ttk.Entry(self, width=18)
        self.ent_detect_stride.grid(row=7, column=1, padx=12, pady=6, sticky="w")
        self.ent_detect_stride.insert(0, str(detect_every_n_frames))

        tip = (
            "建議：縮圖寬度 960 或 1280；每 2 幀或 3 幀偵測一次，可大幅加速。\n"
            "事件片段仍輸出原始影片，不會縮小。"
        )
        ttk.Label(self, text=tip, foreground="#555").grid(
            row=8, column=0, columnspan=2, padx=12, pady=(4, 8), sticky="w"
        )

        btns = ttk.Frame(self)
        btns.grid(row=9, column=0, columnspan=2, pady=(8, 12))

        ttk.Button(btns, text="確定", command=self.on_ok).pack(side="left", padx=6)
        ttk.Button(btns, text="取消", command=self.on_cancel).pack(side="left", padx=6)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.on_cancel())

        self.update_idletasks()
        self.geometry(f"+{parent.winfo_rootx()+80}+{parent.winfo_rooty()+80}")

        self.lift()
        self.focus_force()
        self.ent_conf.focus_set()

    def on_ok(self):
        try:
            conf = float(self.ent_conf.get().strip())
            start_frames = int(self.ent_start.get().strip())
            end_hold = float(self.ent_end_hold.get().strip())
            pre_sec = float(self.ent_pre.get().strip())
            post_sec = float(self.ent_post.get().strip())
            detect_width = int(self.ent_detect_width.get().strip())
            detect_stride = int(self.ent_detect_stride.get().strip())

            if not (0.01 <= conf <= 1.0):
                raise ValueError("YOLO 偵測置信度需介於 0.01 ~ 1.0")
            if start_frames < 1:
                raise ValueError("事件開始連續幀數至少為 1")
            if end_hold < 0:
                raise ValueError("事件結束等待秒數不可小於 0")
            if pre_sec < 0:
                raise ValueError("事件前保留秒數不可小於 0")
            if post_sec < 0:
                raise ValueError("事件後保留秒數不可小於 0")
            if detect_width < 320:
                raise ValueError("偵測前縮圖寬度至少需為 320")
            if detect_stride < 1:
                raise ValueError("每幾幀偵測一次至少需為 1")

            self.result = {
                "confidence": conf,
                "start_trigger_frames": start_frames,
                "end_hold_sec": end_hold,
                "pre_event_sec": pre_sec,
                "post_event_sec": post_sec,
                "detect_width": detect_width,
                "detect_every_n_frames": detect_stride
            }
            self.destroy()
        except Exception as e:
            messagebox.showerror("輸入錯誤", str(e), parent=self)

    def on_cancel(self):
        self.result = None
        self.destroy()


class PastePathsDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("貼上多個來源路徑")
        self.geometry("760x420")
        self.transient(parent)
        self.grab_set()

        wrap = ttk.Frame(self, padding=12)
        wrap.pack(fill="both", expand=True)

        ttk.Label(
            wrap,
            text="每行一個路徑，可同時貼資料夾或影片檔。支援從檔案總管複製後直接貼上。",
            justify="left"
        ).pack(anchor="w")

        self.txt = tk.Text(wrap, wrap="none", height=16)
        self.txt.pack(fill="both", expand=True, pady=(8, 0))

        btns = ttk.Frame(wrap)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="貼上並加入", command=self.on_apply).pack(side="left")
        ttk.Button(btns, text="清空", command=lambda: self.txt.delete("1.0", tk.END)).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="取消", command=self.destroy).pack(side="right")

        self.txt.focus_set()

    def on_apply(self):
        raw = self.txt.get("1.0", tk.END)
        self.parent.apply_pasted_paths(raw)
        self.destroy()


# ---------------------------
# GUI App
# ---------------------------
class App(TkinterDnD.Tk if TkinterDnD is not None else tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CCTV ROI AI Event Extractor（Polygon ROI）")
        self.geometry("1120x700")
        self.resizable(True, True)
        print("目前執行的是這支：可縮放版")

        self.app_dir = get_app_dir()
        self.input_dir = self.app_dir
        self.selected_input_dirs = []
        self.selected_video_files = []
        self.input_mode = "folder"
        self.dragdrop_available = TkinterDnD is not None and DND_FILES is not None

        self.polygon = None
        self.total_videos = 0
        self.done_videos = 0
        self.stop_flag = False

        self.excluded_dir = None
        self.screenshots_root = None
        self.clips_root = None
        self.logs_root = None
        self.reports_root = None

        self.model_path = resolve_default_model_path(self.app_dir)
        self.device_info = get_auto_device_info()
        self.device = self.device_info["device"]
        self.detector = None

        self.confidence = 0.4
        self.start_trigger_frames = 4
        self.end_hold_sec = 1.5
        self.pre_event_sec = 10.0
        self.post_event_sec = 10.0
        self.detect_width = 1280
        self.detect_every_n_frames = 2

        self.video_exts = (".mp4", ".avi", ".mov", ".m4v", ".mkv", ".ts", ".264", ".265")

        self.export_screenshots_var = tk.BooleanVar(value=True)
        self.export_clips_var = tk.BooleanVar(value=True)
        self.draw_roi_on_screenshot_var = tk.BooleanVar(value=True)

        self.ui_queue = queue.Queue()
        self.worker_thread = None

        self._build_ui()
        self.after(80, self._poll_ui_queue)

    def _build_ui(self):
        pad = 12
        frm = ttk.Frame(self, padding=pad)
        frm.pack(fill="both", expand=True)

        ttk.Label(
            frm,
            text="執行方式：選輸出資料夾（自動排除）→ 掃描影片 → Polygon ROI（可載入舊設定）→ 一次輸入 AI 參數 → 勾選輸出類型 → 批次執行。"
        ).pack(anchor="w")

        src_wrap = ttk.LabelFrame(frm, text="影片來源", padding=8)
        src_wrap.pack(fill="x", pady=(8, 0))

        btn_row1 = ttk.Frame(src_wrap)
        btn_row1.pack(fill="x")
        self.btn_pick_input = ttk.Button(btn_row1, text="設為單一資料夾", command=self.pick_input_dir)
        self.btn_pick_input.pack(side="left")
        self.btn_add_input_dir = ttk.Button(btn_row1, text="加入資料夾", command=self.add_input_dir)
        self.btn_add_input_dir.pack(side="left", padx=(8, 0))
        self.btn_paste_dirs = ttk.Button(btn_row1, text="貼上多個資料夾路徑", command=self.open_paste_paths_dialog)
        self.btn_paste_dirs.pack(side="left", padx=(8, 0))
        self.btn_pick_file = ttk.Button(btn_row1, text="設為單一影片", command=self.pick_single_file)
        self.btn_pick_file.pack(side="left", padx=(8, 0))
        self.btn_pick_files = ttk.Button(btn_row1, text="設為多個影片", command=self.pick_input_files)
        self.btn_pick_files.pack(side="left", padx=(8, 0))

        btn_row2 = ttk.Frame(src_wrap)
        btn_row2.pack(fill="x", pady=(8, 0))
        self.btn_remove_selected_source = ttk.Button(btn_row2, text="移除選取來源", command=self.remove_selected_sources)
        self.btn_remove_selected_source.pack(side="left")
        self.btn_clear_input = ttk.Button(btn_row2, text="清空來源", command=self.clear_input_sources)
        self.btn_clear_input.pack(side="left", padx=(8, 0))
        drag_text = "可直接拖曳多個資料夾/影片到下方清單" if self.dragdrop_available else "拖曳功能需先安裝 tkinterdnd2：pip install tkinterdnd2"
        self.lbl_dragdrop = ttk.Label(btn_row2, text=drag_text, foreground="#555")
        self.lbl_dragdrop.pack(side="left", padx=(12, 0))

        self.lbl_folder = ttk.Label(src_wrap, text=f"影片來源資料夾：{self.input_dir}")
        self.lbl_folder.pack(anchor="w", pady=(8, 4))

        list_row = ttk.Frame(src_wrap)
        list_row.pack(fill="x")
        self.lst_sources = tk.Listbox(list_row, height=5, selectmode=tk.EXTENDED)
        self.lst_sources.pack(side="left", fill="x", expand=True)
        yscroll = ttk.Scrollbar(list_row, orient="vertical", command=self.lst_sources.yview)
        yscroll.pack(side="left", fill="y")
        self.lst_sources.config(yscrollcommand=yscroll.set)
        if self.dragdrop_available:
            try:
                self.lst_sources.drop_target_register(DND_FILES)
                self.lst_sources.dnd_bind('<<Drop>>', self.on_drop_sources)
            except Exception:
                self.dragdrop_available = False
                self.lbl_dragdrop.config(text="拖曳功能啟用失敗，仍可用貼上路徑功能", foreground="#a05a00")

        self.lbl_model = ttk.Label(frm, text=f"模型路徑：{self.model_path}")
        self.lbl_model.pack(anchor="w", pady=(6, 0))

        self.lbl_device = ttk.Label(frm, text=f"AI裝置：自動判斷（目前：{self.device_info['device']} | {self.device_info['name']}）")
        self.lbl_device.pack(anchor="w", pady=(6, 0))

        self.lbl_excluded = ttk.Label(frm, text="排除資料夾：尚未選擇")
        self.lbl_excluded.pack(anchor="w", pady=(6, 0))

        self.lbl_out = ttk.Label(frm, text="輸出結構：尚未建立")
        self.lbl_out.pack(anchor="w", pady=(6, 0))

        self.lbl_found = ttk.Label(frm, text="找到影片數：尚未掃描")
        self.lbl_found.pack(anchor="w", pady=(6, 0))

        self.lbl_roi = ttk.Label(frm, text="Polygon ROI：尚未選取")
        self.lbl_roi.pack(anchor="w", pady=(6, 0))

        self.lbl_ai = ttk.Label(
            frm,
            text=(
                f"AI參數：conf={self.confidence} | start_trigger_frames={self.start_trigger_frames} | "
                f"end_hold_sec={self.end_hold_sec} | pre={self.pre_event_sec} | post={self.post_event_sec} | "
                f"detect_width={self.detect_width} | stride={self.detect_every_n_frames}"
            )
        )
        self.lbl_ai.pack(anchor="w", pady=(6, 0))

        opts = ttk.Frame(frm)
        opts.pack(anchor="w", pady=(8, 0))

        ttk.Checkbutton(opts, text="輸出截圖", variable=self.export_screenshots_var).pack(side="left")
        ttk.Checkbutton(opts, text="輸出事件片段", variable=self.export_clips_var).pack(side="left", padx=(20, 0))
        ttk.Checkbutton(opts, text="截圖畫出 ROI / 框線", variable=self.draw_roi_on_screenshot_var).pack(side="left", padx=(20, 0))

        btns = ttk.Frame(frm)
        btns.pack(anchor="w", pady=(10, 0))

        self.btn_start = ttk.Button(btns, text="開始執行", command=self.start_flow)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(btns, text="停止", command=self.request_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(10, 0))

        self.lbl_progress = ttk.Label(frm, text="進度：0/0")
        self.lbl_progress.pack(anchor="w", pady=(10, 0))

        self.pbar = ttk.Progressbar(frm, orient="horizontal", mode="determinate")
        self.pbar.pack(fill="x", pady=(6, 0))

        self.lbl_frame_progress = ttk.Label(frm, text="目前影片進度：0/0")
        self.lbl_frame_progress.pack(anchor="w", pady=(6, 0))

        self.lbl_status = ttk.Label(frm, text="狀態：待命")
        self.lbl_status.pack(anchor="w", pady=(10, 0))

        help_text = (
            "來源選擇：可設為單一資料夾、累加多個資料夾、設為單一影片，或一次設為多個影片；也可在清單中多選後移除。\n"
            "Polygon ROI 操作：左鍵加點、右鍵刪點、C清空、Enter/Space確認。\n"
            "邏輯說明：偵測 person / car / motorcycle / bus / truck，只有當目標的底部中心點進入 Polygon ROI，且連續達到門檻幀數，才算事件開始。\n"
            "加速版：偵測前自動縮圖，可設定每幾幀偵測一次；事件片段仍輸出原始影片。\n"
            "製作人：家宏。"
        )
        ttk.Label(frm, text=help_text, foreground="#444", justify="left").pack(anchor="w", pady=(8, 0))

        self._refresh_source_listbox()

    def _post_ui(self, action, **kwargs):
        self.ui_queue.put((action, kwargs))

    def _poll_ui_queue(self):
        try:
            while True:
                action, kwargs = self.ui_queue.get_nowait()
                if action == "status":
                    self.lbl_status.config(text=f"狀態：{kwargs['text']}")
                elif action == "video_progress":
                    total = max(1, int(kwargs["total"]))
                    done = min(int(kwargs["done"]), total)
                    self.pbar["maximum"] = total
                    self.pbar["value"] = done
                    self.lbl_progress.config(text=f"進度：{done}/{total}")
                elif action == "frame_progress":
                    self.lbl_frame_progress.config(text=f"目前影片進度：{kwargs['current']}/{kwargs['total']}")
                elif action == "message_info":
                    messagebox.showinfo(kwargs["title"], kwargs["message"], parent=self)
                elif action == "message_error":
                    messagebox.showerror(kwargs["title"], kwargs["message"], parent=self)
                elif action == "set_buttons":
                    state = kwargs.get("pick_input_state", kwargs.get("start_state", "normal"))
                    self.btn_start.config(state=kwargs.get("start_state", "normal"))
                    self.btn_stop.config(state=kwargs.get("stop_state", "disabled"))
                    for name in ("btn_pick_input", "btn_add_input_dir", "btn_paste_dirs", "btn_pick_file", "btn_pick_files", "btn_remove_selected_source", "btn_clear_input"):
                        if hasattr(self, name):
                            getattr(self, name).config(state=state)
                    if hasattr(self, "lst_sources"):
                        self.lst_sources.config(state=state)
                self.update_idletasks()
        except queue.Empty:
            pass
        self.after(80, self._poll_ui_queue)

    def request_stop(self):
        self.stop_flag = True
        self.lbl_status.config(text="狀態：已要求停止（將盡快於影片處理中止）")

    def _refresh_source_listbox(self):
        if not hasattr(self, "lst_sources"):
            return
        self.lst_sources.delete(0, tk.END)
        items = []
        if self.input_mode == "files":
            items = [("file", p) for p in self.selected_video_files]
        elif self.input_mode == "folders":
            items = [("folder", p) for p in self.selected_input_dirs]
        else:
            items = [("folder", self.input_dir)]

        for kind, path in items:
            prefix = "[檔案]" if kind == "file" else "[資料夾]"
            self.lst_sources.insert(tk.END, f"{prefix} {path}")

    def _update_input_label(self):
        if self.input_mode == "files" and self.selected_video_files:
            if len(self.selected_video_files) == 1:
                label = f"影片來源檔案：{self.selected_video_files[0]}"
            else:
                label = f"影片來源檔案：共 {len(self.selected_video_files)} 支"
        elif self.input_mode == "folders" and self.selected_input_dirs:
            if len(self.selected_input_dirs) == 1:
                label = f"影片來源資料夾：{self.selected_input_dirs[0]}"
            else:
                label = f"影片來源資料夾：共 {len(self.selected_input_dirs)} 個"
        else:
            label = f"影片來源資料夾：{self.input_dir}"
        self.lbl_folder.config(text=label)
        self._refresh_source_listbox()

    def _split_dnd_items(self, raw: str):
        items = []
        token = ""
        in_brace = False
        for ch in raw:
            if ch == "{":
                if not in_brace:
                    in_brace = True
                    token = ""
                else:
                    token += ch
            elif ch == "}":
                if in_brace:
                    in_brace = False
                    items.append(token)
                    token = ""
                else:
                    token += ch
            elif ch.isspace() and not in_brace:
                if token:
                    items.append(token)
                    token = ""
            else:
                token += ch
        if token:
            items.append(token)
        return [norm_path(x) for x in items if str(x).strip()]

    def _apply_source_selection(self, folders=None, files=None, append=False):
        folders = [norm_path(x) for x in (folders or []) if str(x).strip()]
        files = [norm_path(x) for x in (files or []) if str(x).strip()]
        files = [x for x in files if x.lower().endswith(self.video_exts)]
        folders = [x for x in folders if os.path.isdir(x)]

        if files:
            if append and self.input_mode == "files":
                merged = self.selected_video_files + files
            else:
                merged = files
            uniq = []
            seen = set()
            for p in merged:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            self.input_mode = "files"
            self.selected_video_files = uniq
            self.selected_input_dirs = []
            self.input_dir = os.path.dirname(uniq[0]) if uniq else self.input_dir
            self._update_input_label()
            return len(uniq), "files"

        if folders:
            if append:
                base = []
                if self.input_mode == "folders":
                    base = list(self.selected_input_dirs)
                elif self.input_mode == "folder":
                    base = [self.input_dir]
                merged = base + folders
            else:
                merged = folders
            uniq = []
            seen = set()
            for p in merged:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            if len(uniq) == 1 and not append:
                self.input_mode = "folder"
                self.input_dir = uniq[0]
                self.selected_input_dirs = []
            else:
                self.input_mode = "folders"
                self.selected_input_dirs = uniq
                self.input_dir = uniq[0]
            self.selected_video_files = []
            self._update_input_label()
            return len(uniq), "folders"

        return 0, "none"

    def open_paste_paths_dialog(self):
        PastePathsDialog(self)

    def apply_pasted_paths(self, raw_text: str):
        lines = []
        normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized.split("\n"):
            line = line.strip().strip('"').strip("'")
            if line:
                lines.append(norm_path(line))

        folders = [p for p in lines if os.path.isdir(p)]
        files = [p for p in lines if os.path.isfile(p) and p.lower().endswith(self.video_exts)]
        invalid = [p for p in lines if p not in folders and p not in files]

        count = 0
        mode = "none"
        if files:
            count, mode = self._apply_source_selection(files=files, append=True)
            self.set_status(f"已加入影片來源，共 {count} 支")
        if folders:
            count, mode = self._apply_source_selection(folders=folders, append=True)
            self.set_status(f"已加入資料夾來源，共 {count} 個")
        if invalid:
            messagebox.showwarning("部分路徑無效", "以下路徑不存在或格式不支援：\n\n" + "\n".join(invalid[:20]), parent=self)
        if not folders and not files:
            messagebox.showwarning("未加入任何來源", "沒有偵測到有效的資料夾或支援影片檔。", parent=self)

    def on_drop_sources(self, event):
        try:
            items = self._split_dnd_items(event.data)
            folders = [p for p in items if os.path.isdir(p)]
            files = [p for p in items if os.path.isfile(p) and p.lower().endswith(self.video_exts)]
            invalid = [p for p in items if p not in folders and p not in files]

            if files:
                count, _ = self._apply_source_selection(files=files, append=True)
                self.set_status(f"已拖曳加入影片，共 {count} 支")
            if folders:
                count, _ = self._apply_source_selection(folders=folders, append=True)
                self.set_status(f"已拖曳加入資料夾，共 {count} 個")
            if invalid:
                messagebox.showwarning("部分拖曳來源未加入", "以下項目不存在或格式不支援：\n\n" + "\n".join(invalid[:20]), parent=self)
        except Exception as e:
            messagebox.showerror("拖曳加入失敗", str(e), parent=self)

    def pick_input_dir(self):
        initialdir = self.selected_input_dirs[0] if self.selected_input_dirs else self.input_dir
        selected = filedialog.askdirectory(title="設為單一來源資料夾", initialdir=initialdir)
        if not selected:
            return
        self.input_mode = "folder"
        self.dragdrop_available = TkinterDnD is not None and DND_FILES is not None
        self.input_dir = norm_path(selected)
        self.selected_input_dirs = []
        self.selected_video_files = []
        self._update_input_label()
        self.set_status(f"已設為單一資料夾：{self.input_dir}")

    def add_input_dir(self):
        initialdir = self.selected_input_dirs[-1] if self.selected_input_dirs else self.input_dir
        selected = filedialog.askdirectory(title="加入來源資料夾", initialdir=initialdir)
        if not selected:
            return
        count, _ = self._apply_source_selection(folders=[selected], append=True)
        if count:
            self.set_status(f"已加入來源資料夾，目前共 {count} 個")

    def pick_single_file(self):
        selected = filedialog.askopenfilename(
            title="設為單一影片檔",
            initialdir=self.input_dir,
            filetypes=[("影片檔", "*.mp4 *.avi *.mov *.m4v *.mkv *.ts *.264 *.265"), ("所有檔案", "*.*")],
        )
        if not selected:
            return
        count, _ = self._apply_source_selection(files=[selected], append=False)
        if count:
            self.set_status("已設為單一影片")

    def pick_input_files(self):
        selected = filedialog.askopenfilenames(
            title="設為多個影片檔（可 Ctrl / Shift 多選）",
            initialdir=self.input_dir,
            filetypes=[("影片檔", "*.mp4 *.avi *.mov *.m4v *.mkv *.ts *.264 *.265"), ("所有檔案", "*.*")],
        )
        if not selected:
            return
        count, _ = self._apply_source_selection(files=list(selected), append=False)
        if count:
            self.set_status(f"已設為多個影片，共 {count} 支")

    def remove_selected_sources(self):
        if not hasattr(self, "lst_sources"):
            return
        selected_idx = list(self.lst_sources.curselection())
        if not selected_idx:
            self.set_status("尚未選取要移除的來源")
            return
        if self.input_mode == "files":
            remain = [p for i, p in enumerate(self.selected_video_files) if i not in selected_idx]
            self.selected_video_files = remain
            if remain:
                self.input_dir = os.path.dirname(remain[0])
            else:
                self.input_mode = "folder"
                self.input_dir = self.app_dir
        elif self.input_mode == "folders":
            remain = [p for i, p in enumerate(self.selected_input_dirs) if i not in selected_idx]
            self.selected_input_dirs = remain
            if len(remain) == 1:
                self.input_mode = "folder"
                self.input_dir = remain[0]
                self.selected_input_dirs = []
            elif len(remain) > 1:
                self.input_mode = "folders"
                self.input_dir = remain[0]
            else:
                self.input_mode = "folder"
                self.input_dir = self.app_dir
        else:
            self.input_dir = self.app_dir
        self._update_input_label()
        self.set_status("已移除選取來源")

    def clear_input_sources(self):
        self.input_mode = "folder"
        self.input_dir = self.app_dir
        self.selected_input_dirs = []
        self.selected_video_files = []
        self._update_input_label()
        self.set_status("已清空來源，恢復為程式資料夾")

    def set_status(self, text: str):
        self.lbl_status.config(text=f"狀態：{text}")
        self.update_idletasks()

    def _find_videos(self, exclude_dir=None):
        exclude_dir_norm = norm_path(exclude_dir) if exclude_dir else None

        if self.input_mode == "files" and self.selected_video_files:
            videos = []
            seen = set()
            for full_path in self.selected_video_files:
                full_path = norm_path(full_path)
                if full_path in seen or not os.path.isfile(full_path):
                    continue
                if not full_path.lower().endswith(self.video_exts):
                    continue
                if exclude_dir_norm and is_subpath(full_path, exclude_dir_norm):
                    continue
                seen.add(full_path)
                videos.append(full_path)
            videos.sort()
            return videos

        if self.input_mode == "folders" and self.selected_input_dirs:
            folder_list = [norm_path(x) for x in self.selected_input_dirs if os.path.isdir(x)]
        else:
            folder_list = [norm_path(self.input_dir)] if os.path.isdir(self.input_dir) else []

        videos = []
        seen = set()
        for base_folder in folder_list:
            for root, dirs, files in os.walk(base_folder):
                root_norm = norm_path(root)
                if exclude_dir_norm and is_subpath(root_norm, exclude_dir_norm):
                    dirs[:] = []
                    continue
                if exclude_dir_norm:
                    kept_dirs = []
                    for d in dirs:
                        subdir_full = norm_path(os.path.join(root, d))
                        if is_subpath(subdir_full, exclude_dir_norm):
                            continue
                        kept_dirs.append(d)
                    dirs[:] = kept_dirs
                for name in files:
                    if not name.lower().endswith(self.video_exts):
                        continue
                    full_path = norm_path(os.path.join(root, name))
                    if exclude_dir_norm and is_subpath(full_path, exclude_dir_norm):
                        continue
                    if full_path in seen:
                        continue
                    seen.add(full_path)
                    videos.append(full_path)
        videos.sort()
        return videos

    def _prepare_output_dirs(self, out_dir):
        self.excluded_dir = out_dir
        self.screenshots_root = os.path.join(out_dir, "screenshots")
        self.clips_root = os.path.join(out_dir, "motion_clips")
        self.logs_root = os.path.join(out_dir, "logs")
        self.reports_root = os.path.join(out_dir, "reports")

        ensure_dir(self.excluded_dir)
        ensure_dir(self.screenshots_root)
        ensure_dir(self.clips_root)
        ensure_dir(self.logs_root)
        ensure_dir(self.reports_root)

        self.lbl_excluded.config(text=f"排除資料夾：{self.excluded_dir}")
        self.lbl_out.config(
            text=(
                f"輸出結構：{self.screenshots_root} | "
                f"{self.clips_root} | {self.logs_root} | {self.reports_root}"
            )
        )

    def _write_csv_log(self, rows):
        csv_path = os.path.join(self.logs_root, "detection_log.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_time",
                    "video_rel_path",
                    "record_type",
                    "event_time_sec",
                    "interval_start_sec",
                    "interval_end_sec",
                    "output_path",
                    "status"
                ]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return csv_path

    def _write_summary_report(self, summary_text: str):
        report_path = os.path.join(self.reports_root, "report_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        return report_path

    def _ask_all_params(self):
        dlg = ParamsDialog(
            self,
            confidence=self.confidence,
            start_trigger_frames=self.start_trigger_frames,
            end_hold_sec=self.end_hold_sec,
            pre_event_sec=self.pre_event_sec,
            post_event_sec=self.post_event_sec,
            detect_width=self.detect_width,
            detect_every_n_frames=self.detect_every_n_frames
        )
        self.wait_window(dlg)
        return dlg.result

    def start_flow(self):
        if not self.export_screenshots_var.get() and not self.export_clips_var.get():
            messagebox.showwarning("未選擇輸出類型", "請至少勾選一種輸出類型：截圖或事件片段。", parent=self)
            return

        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "模型不存在",
                f"找不到模型檔：\n{self.model_path}\n\n請先把 yolo26x.pt 放到 models 資料夾（或程式同層）。",
                parent=self
            )
            return

        out_dir = filedialog.askdirectory(title="選擇輸出資料夾（將自動排除不搜尋）", parent=self)
        if not out_dir:
            messagebox.showwarning("取消", "未選擇輸出資料夾，已取消。", parent=self)
            self.set_status("待命")
            return

        self._prepare_output_dirs(out_dir)

        self.set_status("掃描影片中...")
        videos = self._find_videos(exclude_dir=self.excluded_dir)
        self.lbl_found.config(text=f"找到影片數：{len(videos)} 支")

        if not videos:
            messagebox.showerror(
                "找不到影片",
                f"在以下資料夾及其子資料夾中，找不到支援的影片格式：\n{self.input_dir}\n\n"
                f"已自動排除輸出資料夾：\n{self.excluded_dir}\n\n"
                f"支援：{', '.join(self.video_exts)}",
                parent=self
            )
            self.set_status("待命")
            return

        readable_video = find_first_readable_video(videos)
        if not readable_video:
            messagebox.showerror(
                "無可用影片",
                "雖然有找到影片檔，但沒有任何一支影片可成功開啟並讀取第一幀。\n請確認影片格式或解碼器是否正常。",
                parent=self
            )
            self.set_status("待命")
            return

        saved_polygon = load_roi_config(self.app_dir)
        preset_polygon = None
        if saved_polygon:
            use_saved = messagebox.askyesno(
                "載入既有 Polygon ROI",
                f"偵測到先前已儲存 Polygon ROI，共 {len(saved_polygon)} 點。\n\n是否沿用並可再調整？",
                parent=self
            )
            if use_saved:
                preset_polygon = saved_polygon

        first_video_display = safe_relpath(readable_video, self.input_dir)
        self.set_status(f"Polygon ROI 框選中：{first_video_display}")
        self.update()

        picker = PolygonROIPicker(readable_video, preset_polygon=preset_polygon)
        polygon = picker.pick()
        if not polygon:
            messagebox.showwarning("取消", "已取消 Polygon ROI 選取。", parent=self)
            self.set_status("待命")
            return

        self.polygon = polygon
        bx, by, bw, bh = polygon_bbox(polygon)
        self.lbl_roi.config(text=f"Polygon ROI：{len(polygon)} 點 | 外接框 X={bx} Y={by} W={bw} H={bh}")
        save_roi_config(self.app_dir, polygon)

        self.lift()
        self.focus_force()
        self.update_idletasks()

        params = self._ask_all_params()
        if not params:
            messagebox.showwarning("取消", "未輸入 AI 參數，已取消。", parent=self)
            self.set_status("待命")
            return

        self.confidence = params["confidence"]
        self.start_trigger_frames = params["start_trigger_frames"]
        self.end_hold_sec = params["end_hold_sec"]
        self.pre_event_sec = params["pre_event_sec"]
        self.post_event_sec = params["post_event_sec"]
        self.detect_width = params["detect_width"]
        self.detect_every_n_frames = params["detect_every_n_frames"]

        self.lbl_ai.config(
            text=(
                f"AI參數：conf={self.confidence} | start_trigger_frames={self.start_trigger_frames} | "
                f"end_hold_sec={self.end_hold_sec} | pre={self.pre_event_sec} | post={self.post_event_sec} | "
                f"detect_width={self.detect_width} | stride={self.detect_every_n_frames}"
            )
        )

        try:
            self.set_status("載入 AI 模型中...")
            self.device_info = get_auto_device_info()
            self.device = self.device_info["device"]
            self.lbl_device.config(
                text=f"AI裝置：自動判斷（目前：{self.device_info['device']} | {self.device_info['name']}）"
            )
            self.detector = ObjectDetector(
                self.model_path,
                conf=self.confidence,
                detect_width=self.detect_width,
                device=self.device
            )
        except Exception as e:
            messagebox.showerror("模型載入失敗", str(e), parent=self)
            self.set_status("待命")
            return

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_pick_input.config(state="disabled")
        self.stop_flag = False
        self.lbl_frame_progress.config(text="目前影片進度：0/0")

        self.worker_thread = threading.Thread(
            target=self._run_batch,
            args=(videos,),
            daemon=True
        )
        self.worker_thread.start()

    def _run_batch(self, videos):
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.total_videos = len(videos)
        self.done_videos = 0

        self._post_ui("video_progress", done=0, total=self.total_videos)

        total_grabbed = 0
        total_clips = 0
        success_count = 0
        skipped_count = 0
        stopped_count = 0
        csv_rows = []

        for i, vp in enumerate(videos, start=1):
            if self.stop_flag:
                self._post_ui("status", text="已停止")
                break

            rel_name = safe_relpath(vp, self.input_dir)
            self._post_ui("status", text=f"處理中：{rel_name}（{i}/{self.total_videos}）")

            def status_cb(msg):
                self._post_ui("status", text=msg)

            def progress_cb(frame_idx, total_frames):
                self._post_ui("frame_progress", current=frame_idx, total=total_frames)

            result = process_video(
                video_path=vp,
                rel_video_path=rel_name,
                screenshots_root=self.screenshots_root,
                clips_root=self.clips_root,
                polygon=self.polygon,
                detector=self.detector,
                start_trigger_frames=self.start_trigger_frames,
                end_hold_sec=self.end_hold_sec,
                pre_event_sec=self.pre_event_sec,
                post_event_sec=self.post_event_sec,
                draw_roi_on_screenshot=self.draw_roi_on_screenshot_var.get(),
                export_screenshots=self.export_screenshots_var.get(),
                export_clips=self.export_clips_var.get(),
                detect_every_n_frames=self.detect_every_n_frames,
                progress_cb=progress_cb,
                status_cb=status_cb,
                stop_checker=lambda: self.stop_flag
            )

            if result["status"] == "OK":
                success_count += 1
            elif result["status"] == "STOPPED":
                stopped_count += 1
            else:
                skipped_count += 1

            total_grabbed += result["grabbed_count"]
            total_clips += result["clip_count"]

            for item in result["logs"]:
                csv_rows.append({
                    "run_time": run_time,
                    "video_rel_path": item["video_rel_path"],
                    "record_type": item["type"],
                    "event_time_sec": item["event_time_sec"],
                    "interval_start_sec": item["interval_start_sec"],
                    "interval_end_sec": item["interval_end_sec"],
                    "output_path": item["output_path"],
                    "status": item["status"]
                })

            self.done_videos = i
            self._post_ui("video_progress", done=i, total=self.total_videos)

            if self.stop_flag:
                self._post_ui("status", text="已停止")
                break

        csv_path = self._write_csv_log(csv_rows)

        bbox_text = "N/A"
        if self.polygon:
            bx, by, bw, bh = polygon_bbox(self.polygon)
            bbox_text = f"X={bx} Y={by} W={bw} H={bh}"

        summary_text = (
            f"Tool Name: CCTV ROI AI Event Extractor (Polygon ROI)\n"
            f"Tool Version: {APP_VERSION}\n"
            f"Run Time: {run_time}\n"
            f"Search Root: {self.input_dir}\n"
            f"Model Path: {self.model_path}\n"
            f"AI Device: {self.device}\n"
            f"AI Device Name: {self.device_info['name']}\n"
            f"Excluded Output Root: {self.excluded_dir}\n"
            f"Polygon Point Count: {len(self.polygon) if self.polygon else 0}\n"
            f"Polygon Bounding Box: {bbox_text}\n"
            f"Confidence: {self.confidence}\n"
            f"Start Trigger Frames: {self.start_trigger_frames}\n"
            f"End Hold Sec: {self.end_hold_sec}\n"
            f"Pre Event Sec: {self.pre_event_sec}\n"
            f"Post Event Sec: {self.post_event_sec}\n"
            f"Detect Width: {self.detect_width}\n"
            f"Detect Every N Frames: {self.detect_every_n_frames}\n"
            f"Export Screenshots: {self.export_screenshots_var.get()}\n"
            f"Export Clips: {self.export_clips_var.get()}\n"
            f"Draw ROI on Screenshot: {self.draw_roi_on_screenshot_var.get()}\n"
            f"Total Videos Found: {self.total_videos}\n"
            f"Success Videos: {success_count}\n"
            f"Skipped Videos: {skipped_count}\n"
            f"Stopped Videos: {stopped_count}\n"
            f"Total Screenshots: {total_grabbed}\n"
            f"Total Event Clips: {total_clips}\n"
            f"CSV Log Path: {csv_path}\n"
        )
        report_path = self._write_summary_report(summary_text)

        self._post_ui("set_buttons", start_state="normal", stop_state="disabled")

        if not self.stop_flag:
            self._post_ui("status", text=f"完成：共擷取 {total_grabbed} 張，輸出 {total_clips} 支事件片段")
            self._post_ui(
                "message_info",
                title="完成",
                message=(
                    f"已完成。\n"
                    f"找到影片：{self.total_videos} 支\n"
                    f"成功處理：{success_count} 支\n"
                    f"略過：{skipped_count} 支\n"
                    f"共擷取截圖：{total_grabbed} 張\n"
                    f"共輸出事件片段：{total_clips} 支\n\n"
                    f"CSV 日誌：\n{csv_path}\n\n"
                    f"摘要報表：\n{report_path}"
                )
            )


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
