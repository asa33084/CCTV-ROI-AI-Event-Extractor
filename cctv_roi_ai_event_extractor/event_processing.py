import os
import json
import shutil
import urllib.request
from datetime import datetime

import cv2
import numpy as np

from cctv_roi_ai_event_extractor.config import (
    APP_VERSION,
    ensure_runtime_environment,
    load_config,
)
from cctv_roi_ai_event_extractor.compute import (
    describe_available_compute_devices,
    get_auto_device_info,
    list_available_compute_devices,
)
from cctv_roi_ai_event_extractor.video_stream import VideoStreamServer
from cctv_roi_ai_event_extractor.vision_utils import (
    SimpleIouTracker,
    bbox_area,
    bbox_intersects_mask,
    bbox_iou,
    build_screenshot_frame,
    crop_bbox_from_frame,
    draw_anchor_point,
    draw_detection,
    draw_polygon_overlay,
    get_bottom_center,
    is_vehicle_detection,
    make_polygon_mask,
    plate_recognition_quality,
    plate_recognitions_match,
    plate_text_distance,
    plate_texts_similar,
    point_in_polygon,
    polygon_bbox,
    suppress_duplicate_vehicle_tracks,
)
from cctv_roi_ai_event_extractor.yolo_detector import ObjectDetector

ensure_runtime_environment()


LONG_STAY_SCREENSHOT_INTERVAL_SEC = load_config().long_stay_screenshot_interval_sec
# LPR uses short aggregation and suppression windows to avoid saving many screenshots
# for the same plate while the vehicle remains in view.
LPR_AGGREGATION_WINDOW_SEC = 20.0
LPR_DUPLICATE_SUPPRESS_SEC = 20.0
LPR_LOCATION_SUPPRESS_SEC = 60.0
LPR_LOCATION_IOU_THRESHOLD = 0.10
VEHICLE_TRACK_DUPLICATE_IOU_THRESHOLD = 0.35
STREAM_TRACK_MIN_SEEN_FRAMES = 8
STREAM_TRACK_MIN_DURATION_SEC = 0.5


# ---------------------------
# 路徑工具：確保 EXE 放哪就跑哪
# ---------------------------
def get_app_dir() -> str:
    return load_config().app_dir


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_default_model_path(app_dir: str) -> str:
    """Resolve the YOLO model path from config or common app-local locations."""
    configured_model_path = load_config().model_path
    if configured_model_path:
        return os.path.abspath(configured_model_path)

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
    """Ensure the detector model exists, trying configured URLs before Ultralytics assets."""
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
        rel_path = os.path.relpath(full_path, base_dir)
        if rel_path == os.pardir or rel_path.startswith(os.pardir + os.sep) or os.path.isabs(rel_path):
            return os.path.basename(full_path)
        return rel_path
    except Exception:
        return os.path.basename(full_path)


# ---------------------------
# ROI 設定存檔 / 載入（多邊形）
# ---------------------------
def get_roi_config_path(app_dir: str) -> str:
    configured_roi_path = load_config().roi_config_path
    if configured_roi_path:
        return os.path.abspath(configured_roi_path)
    return os.path.join(app_dir, "roi_config_polygon.json")


def load_roi_config(app_dir: str):
    regions = load_roi_regions(app_dir)
    return regions.get("detection_polygon")


def _clean_polygon(polygon):
    """Validate JSON-loaded polygon data and normalize points to integer tuples."""
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


def load_roi_regions(app_dir: str):
    path = get_roi_config_path(app_dir)
    if not os.path.exists(path):
        return {"detection_polygon": None, "touch_polygon": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        detection_polygon = _clean_polygon(data.get("detection_polygon"))
        if detection_polygon is None:
            detection_polygon = _clean_polygon(data.get("polygon"))
        touch_polygon = _clean_polygon(data.get("touch_polygon"))
        return {
            "detection_polygon": detection_polygon,
            "touch_polygon": touch_polygon,
        }
    except Exception:
        return {"detection_polygon": None, "touch_polygon": None}


def save_roi_config(app_dir: str, polygon):
    save_roi_regions(app_dir, polygon, None)


def save_roi_regions(app_dir: str, detection_polygon, touch_polygon=None):
    path = get_roi_config_path(app_dir)
    data = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "polygon": [[int(x), int(y)] for x, y in detection_polygon],
        "detection_polygon": [[int(x), int(y)] for x, y in detection_polygon],
        "touch_polygon": [[int(x), int(y)] for x, y in touch_polygon] if touch_polygon else None,
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
    """OpenCV-based polygon picker used by the legacy Tk interface."""

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
# 截圖與片段輸出
# ---------------------------
def try_save_screenshot(logs, screenshot_out_dir, base_name, rel_video_path, frame_idx, current_time_sec,
                        frame, detections, polygon, polygon_np, draw_roi_on_screenshot, lpr_pipeline=None,
                        touch_polygon=None, record_type="screenshot", plate_recognitions_override=None):
    """Save one screenshot and append the corresponding CSV/log row."""
    plate_recognitions = []
    if plate_recognitions_override is not None:
        plate_recognitions = plate_recognitions_override
    elif lpr_pipeline is not None:
        plate_recognitions = lpr_pipeline.recognize(frame, vehicle_detections=detections)

    screenshot_frame = build_screenshot_frame(
        frame=frame,
        detections=detections,
        polygon=polygon,
        polygon_np=polygon_np,
        draw_roi_on_screenshot=draw_roi_on_screenshot,
        plate_recognitions=[] if record_type == "lpr_touch" else plate_recognitions,
        touch_polygon=touch_polygon,
    )

    ok_save, shot_path = save_frame(
        screenshot_out_dir,
        base_name,
        current_time_sec,
        frame_idx,
        screenshot_frame
    )

    logs.append({
        "type": record_type,
        "video_rel_path": rel_video_path,
        "event_time_sec": f"{current_time_sec:.2f}",
        "interval_start_sec": "",
        "interval_end_sec": "",
        "output_path": shot_path or "",
        "status": "OK" if ok_save else "SAVE_FAIL",
        "plate_text": ";".join(item.text for item in plate_recognitions if item.text),
        "plate_raw_text": ";".join(item.raw_text for item in plate_recognitions if item.raw_text),
        "plate_confidence": ";".join(f"{item.confidence:.3f}" for item in plate_recognitions),
        "plate_bbox": ";".join(",".join(str(v) for v in item.bbox) for item in plate_recognitions),
        "plate_valid_taiwan_format": ";".join("Y" if item.valid_taiwan_format else "N" for item in plate_recognitions),
        "plate_ocr_engine": lpr_pipeline.engine_name if lpr_pipeline is not None else "",
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
    """Export a raw video segment for a detected event interval."""
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


def centered_event_interval(event_time_sec: float, pre_event_sec: float, post_event_sec: float, total_sec: float | None = None):
    """Return a clip interval centered on the event timestamp."""
    center = max(0.0, float(event_time_sec or 0.0))
    start_t = max(0.0, center - max(0.0, float(pre_event_sec or 0.0)))
    end_t = center + max(0.0, float(post_event_sec or 0.0))
    if total_sec is not None and total_sec > 0:
        end_t = min(end_t, total_sec)
    return center, start_t, max(start_t, end_t)


def _format_datetime(value):
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _track_row_key(camera_id, stream_id, track_id):
    """Build a stable internal key for one YOLO track inside one continuous stream."""
    return f"{camera_id}:{stream_id}:{track_id}"


def _stream_track_duration_sec(record):
    return max(0.0, float(record.get("end_stream_time_sec", 0.0)) - float(record.get("start_stream_time_sec", 0.0)))


def _is_reportable_stream_track(record):
    """Suppress one-frame YOLO fragments while keeping real but intermittently detected tracks."""
    seen_frames = int(record.get("seen_frames", 0) or 0)
    duration_sec = _stream_track_duration_sec(record)
    return seen_frames >= STREAM_TRACK_MIN_SEEN_FRAMES or duration_sec >= STREAM_TRACK_MIN_DURATION_SEC


def _empty_log_row(record_type, video_rel_path="", status="OK"):
    return {
        "type": record_type,
        "video_rel_path": video_rel_path,
        "event_time_sec": "",
        "interval_start_sec": "",
        "interval_end_sec": "",
        "output_path": "",
        "status": status,
        "camera_id": "",
        "stream_id": "",
        "track_id": "",
        "track_start_datetime": "",
        "track_end_datetime": "",
        "track_start_source": "",
        "track_end_source": "",
        "track_seen_frames": "",
        "track_duration_sec": "",
        "plate_text": "",
        "plate_raw_text": "",
        "plate_confidence": "",
        "plate_bbox": "",
        "plate_valid_taiwan_format": "",
        "plate_ocr_engine": "",
    }


def _build_stream_debug_frame(stream_item, detections, polygon, polygon_np, plate_recognitions=None):
    debug_frame = build_screenshot_frame(
        frame=stream_item.frame,
        detections=detections,
        polygon=polygon,
        polygon_np=polygon_np,
        draw_roi_on_screenshot=True,
        plate_recognitions=plate_recognitions,
    )
    header = (
        f"camera={stream_item.stream.camera_id} "
        f"stream={stream_item.stream.stream_id} "
        f"source={stream_item.segment.rel_path} "
        f"frame={stream_item.local_frame_idx} "
        f"time={_format_datetime(stream_item.absolute_datetime)}"
    )
    cv2.putText(debug_frame, header, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    return debug_frame


def process_video_stream(
    video_paths,
    input_dir: str,
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
    lpr_pipeline=None,
    touch_polygon=None,
    export_long_stay_screenshots: bool = True,
    debug_stream_preview: bool = False,
    debug_frame_cb=None,
    progress_cb=None,
    status_cb=None,
    stop_checker=None,
):
    """Process multiple video files as per-camera streams and summarize each tracked vehicle."""
    del start_trigger_frames, end_hold_sec
    del touch_polygon, export_long_stay_screenshots

    logs = []
    grabbed_count = 0
    clip_count = 0
    processed_frames = 0
    detect_every_n_frames = max(1, int(detect_every_n_frames))
    polygon_np = np.array(polygon, dtype=np.int32)
    detection_mask = None

    stream = VideoStreamServer.from_paths(video_paths, input_dir=input_dir, load_metadata=True)
    total_frames = stream.total_frames
    segments_by_rel_path = {segment.rel_path: segment for segment in stream.segments}
    debug_preview_closed = False

    if status_cb:
        cameras = ", ".join(stream.camera_ids) or "N/A"
        status_cb(
            f"[STREAM] 影片讀取器啟動 | cameras={cameras} | videos={len(stream.segments)} "
            f"| track_stride={detect_every_n_frames}"
        )
        if getattr(detector, "tracker_path", None):
            status_cb(f"[TRACKER] {detector.tracker_path}")
        if export_clips:
            status_cb("[STREAM] 事件片段會以 track 進入時間為中心，依前後保留秒數輸出。")
        if debug_stream_preview:
            status_cb("[DEBUG] Stream YOLO track 預覽已啟用；關閉 Qt 預覽視窗可停止顯示。")

    skipped_segments = 0
    success_segments = 0
    clip_index = 0

    for video_stream in stream.iter_streams():
        camera_id = video_stream.camera_id
        stream_id = video_stream.stream_id
        segments = list(video_stream.segments)
        detector.reset_trackers()
        records = {}

        if status_cb:
            status_cb(
                f"[STREAM] camera={camera_id} stream={stream_id} | "
                f"segments={len(segments)} | YOLO track reset"
            )

        current_segment = None
        for stream_item in stream.frames_for_stream(video_stream):
            if stop_checker and stop_checker():
                if status_cb:
                    status_cb("[STOP] 已停止 stream 處理")
                return {
                    "status": "STOPPED",
                    "grabbed_count": grabbed_count,
                    "clip_count": clip_count,
                    "fps": 0,
                    "width": 0,
                    "height": 0,
                    "frames": total_frames,
                    "logs": logs,
                    "total_videos": len(stream.segments),
                    "success_count": success_segments,
                    "skipped_count": skipped_segments,
                    "stopped_count": 1,
                }

            if stream_item.segment != current_segment:
                current_segment = stream_item.segment
                if status_cb:
                    status_cb(
                        f"[STREAM] camera={camera_id} stream={stream_id} "
                        f"segment={current_segment.rel_path} | track persist"
                    )

            processed_frames += 1
            if progress_cb and processed_frames % 10 == 0:
                progress_cb(processed_frames, total_frames)

            should_track = ((stream_item.stream_frame_idx - 1) % detect_every_n_frames == 0)
            if not should_track:
                continue

            detections = detector.track(stream_item.frame, persist=True)
            if detection_mask is None or detection_mask.shape[:2] != stream_item.frame.shape[:2]:
                detection_mask = make_polygon_mask(stream_item.frame.shape, polygon)
            # Track summaries are based on vehicles whose box intersects the detection ROI.
            vehicle_detections = suppress_duplicate_vehicle_tracks([
                det for det in detections
                if is_vehicle_detection(det)
            ])
            inside_vehicle_detections = [
                det for det in vehicle_detections
                if is_vehicle_detection(det) and bbox_intersects_mask(det["bbox"], detection_mask)
            ]
            should_run_lpr = True
            debug_detections = []
            debug_plate_recognitions = []

            for det in inside_vehicle_detections:
                track_id = det.get("track_id")
                if track_id is None:
                    continue
                key = _track_row_key(camera_id, stream_id, track_id)
                display_track_id = str(track_id)
                debug_det = dict(det)
                debug_det["track_id"] = display_track_id
                debug_detections.append(debug_det)

                record = records.get(key)
                if record is None:
                    # New track: store timing, source file, best plate, and best screenshot as it evolves.
                    record = {
                        "camera_id": camera_id,
                        "stream_id": stream_id,
                        "track_id": display_track_id,
                        "raw_track_id": track_id,
                        "raw_key": key,
                        "start_datetime": stream_item.absolute_datetime,
                        "end_datetime": stream_item.absolute_datetime,
                        "start_source": stream_item.segment.rel_path,
                        "end_source": stream_item.segment.rel_path,
                        "start_time_sec": stream_item.source_time_sec,
                        "end_time_sec": stream_item.source_time_sec,
                        "start_stream_time_sec": stream_item.stream_time_sec,
                        "end_stream_time_sec": stream_item.stream_time_sec,
                        "first_stream_frame_idx": stream_item.stream_frame_idx,
                        "last_stream_frame_idx": stream_item.stream_frame_idx,
                        "seen_frames": 0,
                        "video_rel_path": stream_item.segment.rel_path,
                        "best_quality": -1.0,
                        "best_plate": None,
                        "best_screenshot_path": "",
                    }
                    records[key] = record

                record["end_datetime"] = stream_item.absolute_datetime
                record["end_source"] = stream_item.segment.rel_path
                record["end_time_sec"] = stream_item.source_time_sec
                record["end_stream_time_sec"] = stream_item.stream_time_sec
                record["last_stream_frame_idx"] = stream_item.stream_frame_idx
                record["seen_frames"] = int(record.get("seen_frames", 0) or 0) + 1
                record["last_bbox"] = det["bbox"]
                if record.get("best_plate") is not None:
                    debug_plate_recognitions.append(record["best_plate"])

                if lpr_pipeline is None or not should_run_lpr:
                    continue

                # LPR is sampled by detect_every_n_frames and only run on the current vehicle crop.
                recognitions = [item for item in lpr_pipeline.recognize(stream_item.frame, vehicle_detections=[det]) if item.text]
                if not recognitions:
                    continue
                recognition = max(recognitions, key=plate_recognition_quality)
                quality = plate_recognition_quality(recognition)
                if quality <= record["best_quality"]:
                    continue

                record["best_quality"] = quality
                record["best_plate"] = recognition
                debug_plate_recognitions.append(recognition)

                if export_screenshots:
                    rel_dir = os.path.dirname(stream_item.segment.rel_path)
                    screenshot_out_dir = os.path.join(screenshots_root, rel_dir)
                    ensure_dir(screenshot_out_dir)
                    base_name = os.path.splitext(os.path.basename(stream_item.segment.path))[0]
                    screenshot_frame = build_screenshot_frame(
                        frame=stream_item.frame,
                        detections=detections,
                        polygon=polygon,
                        polygon_np=polygon_np,
                        draw_roi_on_screenshot=draw_roi_on_screenshot,
                        plate_recognitions=[recognition],
                    )
                    ok_save, shot_path = save_frame(
                        screenshot_out_dir,
                        base_name,
                        stream_item.source_time_sec,
                        stream_item.local_frame_idx,
                        screenshot_frame,
                    )
                    if ok_save:
                        grabbed_count += 1
                        record["best_screenshot_path"] = shot_path

                if status_cb:
                    status_cb(
                        f"[TRACK-LPR] camera={camera_id} stream={stream_id} "
                        f"track={display_track_id} plate={recognition.text}"
                    )

            if debug_stream_preview and not debug_preview_closed:
                try:
                    debug_frame = _build_stream_debug_frame(
                        stream_item,
                        debug_detections,
                        polygon,
                        polygon_np,
                        plate_recognitions=debug_plate_recognitions,
                    )
                    if debug_frame_cb is not None:
                        debug_frame_cb(debug_frame)
                except Exception as e:
                    debug_preview_closed = True
                    if status_cb:
                        status_cb(f"[DEBUG-SKIP] Stream 預覽無法顯示：{e}")

        success_segments += len(segments)

        for record in records.values():
            # Emit exactly one summary row per completed/active track at camera end.
            if not _is_reportable_stream_track(record):
                plate = record.get("best_plate")
                if status_cb and plate is not None:
                    status_cb(
                        f"[TRACK-SKIP] camera={record['camera_id']} stream={record['stream_id']} "
                        f"track={record['track_id']} plate={plate.text} "
                        f"frames={record.get('seen_frames', 0)} duration={_stream_track_duration_sec(record):.2f}s"
                    )
                continue

            plate = record.get("best_plate")
            row = _empty_log_row("track_summary", video_rel_path=record.get("video_rel_path", ""))
            row.update({
                "event_time_sec": f'{record.get("start_time_sec", 0.0):.2f}',
                "interval_start_sec": f'{record.get("start_time_sec", 0.0):.2f}',
                "interval_end_sec": f'{record.get("end_time_sec", 0.0):.2f}',
                "output_path": record.get("best_screenshot_path", ""),
                "camera_id": record["camera_id"],
                "stream_id": record["stream_id"],
                "track_id": str(record["track_id"]),
                "track_start_datetime": _format_datetime(record.get("start_datetime")),
                "track_end_datetime": _format_datetime(record.get("end_datetime")),
                "track_start_source": record.get("start_source", ""),
                "track_end_source": record.get("end_source", ""),
                "track_seen_frames": str(record.get("seen_frames", 0)),
                "track_duration_sec": f"{_stream_track_duration_sec(record):.2f}",
            })
            if plate is not None:
                row.update({
                    "plate_text": plate.text,
                    "plate_raw_text": plate.raw_text,
                    "plate_confidence": f"{plate.confidence:.3f}",
                    "plate_bbox": ",".join(str(v) for v in plate.bbox),
                    "plate_valid_taiwan_format": "Y" if plate.valid_taiwan_format else "N",
                    "plate_ocr_engine": lpr_pipeline.engine_name if lpr_pipeline is not None else "",
                })

            if export_clips:
                segment = segments_by_rel_path.get(record.get("start_source"))
                if segment is not None:
                    clip_index += 1
                    rel_dir = os.path.dirname(segment.rel_path)
                    clip_out_dir = os.path.join(clips_root, rel_dir)
                    base_name = os.path.splitext(os.path.basename(segment.path))[0]
                    event_time_sec, clip_start_t, clip_end_t = centered_event_interval(
                        record.get("start_time_sec", 0.0),
                        pre_event_sec,
                        post_event_sec,
                        (segment.frame_count / segment.fps) if segment.frame_count > 0 and segment.fps > 0 else None,
                    )
                    row["event_time_sec"] = f"{event_time_sec:.2f}"
                    row["interval_start_sec"] = f"{clip_start_t:.2f}"
                    row["interval_end_sec"] = f"{clip_end_t:.2f}"
                    ok_clip, clip_path = export_interval_clip(
                        video_path=segment.path,
                        clip_out_dir=clip_out_dir,
                        base_name=f"{base_name}__{record['stream_id']}__track{record['track_id']}",
                        start_t=clip_start_t,
                        end_t=clip_end_t,
                        clip_index=clip_index,
                        status_cb=status_cb,
                    )
                    if ok_clip:
                        clip_count += 1
                        if not row.get("output_path"):
                            row["output_path"] = clip_path or ""
                elif status_cb:
                    status_cb(f"[CLIP-SKIP] 找不到 track 來源影片：{record.get('start_source')}")
            logs.append(row)

    if progress_cb:
        progress_cb(total_frames if total_frames > 0 else processed_frames, total_frames)

    if status_cb:
        status_cb(
            f"[STREAM-DONE] tracks={sum(1 for item in logs if item.get('type') == 'track_summary')} "
            f"screenshots={grabbed_count} clips={clip_count}"
        )

    return {
        "status": "OK",
        "grabbed_count": grabbed_count,
        "clip_count": clip_count,
        "fps": 0,
        "width": 0,
        "height": 0,
        "frames": total_frames,
        "logs": logs,
        "total_videos": len(stream.segments),
        "success_count": success_segments,
        "skipped_count": skipped_segments,
        "stopped_count": 0,
    }


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
    lpr_pipeline=None,
    touch_polygon=None,
    export_long_stay_screenshots: bool = True,
    progress_cb=None,
    status_cb=None,
    stop_checker=None
):
    """Process one video file with ROI event intervals, screenshots, clips, and optional LPR."""
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

    end_hold_frames = max(1, int(round(end_hold_sec * fps)))
    start_trigger_frames = max(1, int(start_trigger_frames))

    frame_idx = 0
    grabbed_count = 0
    clip_count = 0

    event_intervals = []
    polygon_np = np.array(polygon, dtype=np.int32)
    detection_mask = None
    touch_mask = None
    use_touch_lpr = False

    in_event = False
    event_center_frame = None
    last_inside_frame = None
    start_counter = 0

    cached_detections = []
    cached_inside_present = False
    cached_inside_vehicle_detections = []
    next_long_stay_shot_frame = None
    tracker = SimpleIouTracker(max_missed=max(4, start_trigger_frames * 2)) if use_touch_lpr else None
    vehicle_candidates = {}
    lpr_completed_track_ids = set()
    use_detection_roi_lpr = lpr_pipeline is not None
    lpr_pending_groups = []
    lpr_suppressed_until = {}
    lpr_suppressed_locations = []

    def append_centered_clip_interval():
        if event_center_frame is None:
            return
        total_sec = (total / fps) if total > 0 else None
        center_t = max(0.0, (event_center_frame - 1) / fps)
        event_intervals.append(centered_event_interval(
            center_t,
            pre_event_sec,
            post_event_sec,
            total_sec,
        ))

    def flush_lpr_group(group):
        """Commit the best candidate in a pending LPR group to logs/screenshots."""
        nonlocal grabbed_count
        best_candidate = select_lpr_group_candidate(group)
        if best_candidate is None:
            return
        record = best_candidate["record"]
        recognition = best_candidate["recognition"]
        shot_path = ""
        ok_save = False
        if export_screenshots:
            ok_save, shot_path = try_save_screenshot(
                logs=logs,
                screenshot_out_dir=screenshot_out_dir,
                base_name=base_name,
                rel_video_path=rel_video_path,
                frame_idx=record["frame_idx"],
                current_time_sec=record["time_sec"],
                frame=record["frame"],
                detections=record["detections"],
                polygon=polygon,
                polygon_np=polygon_np,
                draw_roi_on_screenshot=draw_roi_on_screenshot,
                lpr_pipeline=None,
                touch_polygon=touch_polygon,
                record_type="lpr_detection",
                plate_recognitions_override=[recognition],
            )
            if ok_save:
                grabbed_count += 1
        else:
            logs.append({
                "type": "lpr_detection",
                "video_rel_path": rel_video_path,
                "event_time_sec": f'{record["time_sec"]:.2f}',
                "interval_start_sec": "",
                "interval_end_sec": "",
                "output_path": "",
                "status": "OK",
                "plate_text": recognition.text,
                "plate_raw_text": recognition.raw_text,
                "plate_confidence": f"{recognition.confidence:.3f}",
                "plate_bbox": ",".join(str(v) for v in recognition.bbox),
                "plate_valid_taiwan_format": "Y" if recognition.valid_taiwan_format else "N",
                "plate_ocr_engine": lpr_pipeline.engine_name,
            })

        lpr_suppressed_until[recognition.text] = record["time_sec"] + LPR_DUPLICATE_SUPPRESS_SEC
        lpr_suppressed_locations.append({
            "bbox": recognition.bbox,
            "until": record["time_sec"] + LPR_LOCATION_SUPPRESS_SEC,
        })
        if status_cb:
            if export_screenshots and ok_save:
                status_cb(f"[LPR] 偵測區車輛辨識：{recognition.text} | {os.path.basename(shot_path)}")
            else:
                status_cb(f"[LPR] 偵測區車輛辨識：{recognition.text}")

    def flush_due_lpr_groups(now_sec, force=False):
        """Flush LPR groups once the aggregation window expires or at end-of-video."""
        remaining = []
        for group in lpr_pending_groups:
            if force or (now_sec - group["first_seen_sec"]) >= LPR_AGGREGATION_WINDOW_SEC:
                flush_lpr_group(group)
            else:
                remaining.append(group)
        lpr_pending_groups[:] = remaining

    def is_lpr_text_suppressed(text, time_sec):
        """Skip plate text recently emitted from the same video."""
        expired_texts = [
            suppressed_text
            for suppressed_text, suppress_until in lpr_suppressed_until.items()
            if suppress_until <= time_sec
        ]
        for suppressed_text in expired_texts:
            lpr_suppressed_until.pop(suppressed_text, None)
        for suppressed_text, suppress_until in lpr_suppressed_until.items():
            if suppress_until > time_sec and plate_texts_similar(text, suppressed_text):
                return True
        return False

    def is_lpr_location_suppressed(bbox, time_sec):
        """Skip plate boxes from locations recently emitted from the same video."""
        active_locations = []
        suppressed = False
        for item in lpr_suppressed_locations:
            if item["until"] <= time_sec:
                continue
            active_locations.append(item)
            if bbox_iou(bbox, item["bbox"]) >= LPR_LOCATION_IOU_THRESHOLD:
                suppressed = True
        lpr_suppressed_locations[:] = active_locations
        return suppressed

    def select_lpr_group_candidate(group):
        """Choose the best candidate by majority-like text score and recognition quality."""
        candidates = group.get("candidates") or []
        if not candidates:
            return None

        text_scores = {}
        for candidate in candidates:
            text = candidate["recognition"].text
            text_scores[text] = text_scores.get(text, 0.0) + 2.0 + candidate["quality"]

        best_text = max(text_scores, key=text_scores.get)
        best_candidates = [
            candidate for candidate in candidates
            if candidate["recognition"].text == best_text
        ]
        return max(best_candidates, key=lambda candidate: candidate["quality"])

    def queue_lpr_recognitions(recognitions, frame, detections, frame_idx_value, time_sec):
        """Group nearby/repeated recognitions before writing a final LPR event."""
        for recognition in recognitions:
            if not recognition.text:
                continue
            if is_lpr_text_suppressed(recognition.text, time_sec):
                continue
            if is_lpr_location_suppressed(recognition.bbox, time_sec):
                continue

            matching_group = None
            for group in lpr_pending_groups:
                if plate_recognitions_match(recognition, group["recognition"]):
                    matching_group = group
                    break

            record = {
                "frame": frame.copy(),
                "detections": [dict(det) for det in detections],
                "frame_idx": frame_idx_value,
                "time_sec": time_sec,
            }
            quality = plate_recognition_quality(recognition)
            candidate = {
                "recognition": recognition,
                "quality": quality,
                "record": record,
            }
            if matching_group is None:
                lpr_pending_groups.append({
                    "first_seen_sec": time_sec,
                    "last_seen_sec": time_sec,
                    "recognition": recognition,
                    "quality": quality,
                    "record": record,
                    "candidates": [candidate],
                })
                continue

            matching_group["last_seen_sec"] = time_sec
            matching_group.setdefault("candidates", []).append(candidate)
            if quality > matching_group["quality"]:
                matching_group["recognition"] = recognition
                matching_group["quality"] = quality
                matching_group["record"] = record

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

        if progress_cb and frame_idx % 10 == 0:
            progress_cb(frame_idx, total)

        # 關鍵原則：
        # 只有真正的偵測幀，才更新事件邏輯。
        # 非偵測幀仍可沿用快取結果供顯示 / 截圖參考，
        # 但不可拿來推進 start_counter / last_inside_frame，
        # 否則會產生事件起訖時間被灌水延伸的問題。
        should_detect = ((frame_idx - 1) % detect_every_n_frames == 0)

        if should_detect:
            raw_detections = detector.detect(frame)
            if use_detection_roi_lpr and detection_mask is None:
                detection_mask = make_polygon_mask(frame.shape, polygon)
            if use_touch_lpr:
                vehicle_detections = [
                    det for det in raw_detections
                    if det["class_name"] in {"car", "motorcycle", "bus", "truck"}
                ]
                tracked_vehicles = tracker.update(vehicle_detections)
                tracked_by_bbox = {det["bbox"]: det for det in tracked_vehicles}
                active_track_ids = set(tracker.tracks)
                for stale_track_id in list(vehicle_candidates):
                    if stale_track_id not in active_track_ids:
                        vehicle_candidates.pop(stale_track_id, None)
                cached_detections = []
                for det in raw_detections:
                    tracked = tracked_by_bbox.get(det["bbox"])
                    cached_detections.append(tracked if tracked is not None else det)
            else:
                cached_detections = raw_detections

            inside_count = 0
            cached_inside_vehicle_detections = []
            for det in cached_detections:
                if (
                    use_detection_roi_lpr
                    and is_vehicle_detection(det)
                    and bbox_intersects_mask(det["bbox"], detection_mask)
                ):
                    cached_inside_vehicle_detections.append(det)
                anchor = get_bottom_center(det["bbox"])
                if point_in_polygon(anchor, polygon_np):
                    inside_count += 1
                    if (
                        use_touch_lpr
                        and det.get("track_id") is not None
                        and det.get("track_id") not in lpr_completed_track_ids
                    ):
                        crop = crop_bbox_from_frame(frame, det["bbox"])
                        if crop is not None:
                            track_id = det["track_id"]
                            area = bbox_area(det["bbox"])
                            current = vehicle_candidates.get(track_id)
                            if current is None or area > current["area"]:
                                vehicle_candidates[track_id] = {
                                    "area": area,
                                    "frame": crop,
                                    "bbox": det["bbox"],
                                    "frame_idx": frame_idx,
                                    "time_sec": frame_idx / fps,
                                }

            cached_inside_present = (inside_count > 0)

        detections = cached_detections
        inside_present = cached_inside_present
        current_time_sec = frame_idx / fps

        if should_detect and use_detection_roi_lpr and cached_inside_vehicle_detections:
            plate_recognitions = lpr_pipeline.recognize(frame, vehicle_detections=cached_inside_vehicle_detections)
            recognized_plate_recognitions = [item for item in plate_recognitions if item.text]
            if recognized_plate_recognitions:
                queue_lpr_recognitions(
                    recognized_plate_recognitions,
                    frame=frame,
                    detections=detections,
                    frame_idx_value=frame_idx,
                    time_sec=current_time_sec,
                )
        if use_detection_roi_lpr:
            flush_due_lpr_groups(current_time_sec)

        if should_detect and use_touch_lpr:
            if touch_mask is None:
                touch_mask = make_polygon_mask(frame.shape, touch_polygon)
            for det in detections:
                track_id = det.get("track_id")
                if track_id is None or track_id in lpr_completed_track_ids:
                    continue
                if det["class_name"] not in {"car", "motorcycle", "bus", "truck"}:
                    continue
                if not bbox_intersects_mask(det["bbox"], touch_mask):
                    continue

                candidate = vehicle_candidates.get(track_id)
                candidate_frame = candidate["frame"] if candidate is not None else crop_bbox_from_frame(frame, det["bbox"])
                if candidate_frame is None:
                    lpr_completed_track_ids.add(track_id)
                    continue

                plate_recognitions = lpr_pipeline.recognize(candidate_frame, vehicle_detections=[det])
                lpr_completed_track_ids.add(track_id)
                vehicle_candidates.pop(track_id, None)

                shot_path = ""
                ok_save = False
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
                        draw_roi_on_screenshot=draw_roi_on_screenshot,
                        lpr_pipeline=None,
                        touch_polygon=touch_polygon,
                        record_type="lpr_touch",
                        plate_recognitions_override=plate_recognitions,
                    )
                    if ok_save:
                        grabbed_count += 1
                else:
                    logs.append({
                        "type": "lpr_touch",
                        "video_rel_path": rel_video_path,
                        "event_time_sec": f"{current_time_sec:.2f}",
                        "interval_start_sec": "",
                        "interval_end_sec": "",
                        "output_path": "",
                        "status": "OK",
                        "plate_text": ";".join(item.text for item in plate_recognitions if item.text),
                        "plate_raw_text": ";".join(item.raw_text for item in plate_recognitions if item.raw_text),
                        "plate_confidence": ";".join(f"{item.confidence:.3f}" for item in plate_recognitions),
                        "plate_bbox": ";".join(",".join(str(v) for v in item.bbox) for item in plate_recognitions),
                        "plate_valid_taiwan_format": ";".join("Y" if item.valid_taiwan_format else "N" for item in plate_recognitions),
                        "plate_ocr_engine": lpr_pipeline.engine_name,
                    })

                if status_cb:
                    plate_text = ";".join(item.text for item in plate_recognitions if item.text) or "未辨識"
                    if export_screenshots and ok_save:
                        status_cb(f"[LPR] track #{track_id} 觸碰區觸發：{plate_text} | {os.path.basename(shot_path)}")
                    else:
                        status_cb(f"[LPR] track #{track_id} 觸碰區觸發：{plate_text}")

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

                    # 事件片段以事件成立時間點為中心，前後依設定秒數擷取。
                    event_center_frame = max(1, trigger_frame)

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
                            draw_roi_on_screenshot=draw_roi_on_screenshot,
                            lpr_pipeline=None if use_detection_roi_lpr else lpr_pipeline,
                            touch_polygon=touch_polygon if use_touch_lpr else None,
                        )
                        if ok_save:
                            grabbed_count += 1
                        if status_cb:
                            if ok_save:
                                status_cb(f"[SHOT] 已輸出截圖：{os.path.basename(shot_path)}")
                            else:
                                status_cb(f"[SHOT-FAIL] 截圖寫入失敗：{shot_path}")

                    if export_long_stay_screenshots:
                        next_long_stay_shot_frame = frame_idx + max(
                            1,
                            int(round(LONG_STAY_SCREENSHOT_INTERVAL_SEC * fps))
                        )
                    else:
                        next_long_stay_shot_frame = None

            else:
                if inside_present:
                    # 只有真正偵測到仍在 ROI 內，才更新 last_inside_frame
                    last_inside_frame = frame_idx

                    if (
                        export_screenshots
                        and export_long_stay_screenshots
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
                            draw_roi_on_screenshot=draw_roi_on_screenshot,
                            lpr_pipeline=None if use_detection_roi_lpr else lpr_pipeline,
                            touch_polygon=touch_polygon if use_touch_lpr else None,
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
                        append_centered_clip_interval()

                        in_event = False
                        event_center_frame = None
                        last_inside_frame = None
                        start_counter = 0
                        next_long_stay_shot_frame = None

    if in_event and event_center_frame is not None:
        append_centered_clip_interval()

    if use_detection_roi_lpr:
        flush_due_lpr_groups(frame_idx / fps if frame_idx > 0 else 0.0, force=True)

    cap.release()

    if export_clips and event_intervals:
        if status_cb:
            status_cb(f"[CLIP] {rel_video_path} 事件 {len(event_intervals)} 個，準備輸出片段")

        for idx, (event_time_sec, start_t, end_t) in enumerate(event_intervals, start=1):
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
                    "event_time_sec": f"{event_time_sec:.2f}",
                    "interval_start_sec": f"{start_t:.2f}",
                    "interval_end_sec": f"{end_t:.2f}",
                    "output_path": clip_path,
                    "status": "OK"
                })

    if progress_cb:
        progress_cb(total if total > 0 else frame_idx, total)

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
# Legacy Tk launcher
# ---------------------------
def main():
    from cctv_roi_ai_event_extractor.legacy_app import main as legacy_main

    legacy_main()
