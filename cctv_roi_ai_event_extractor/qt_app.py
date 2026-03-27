import csv
import os
import sys
from datetime import datetime

import cv2
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
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
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cctv_roi_ai_event_extractor.core import (
    APP_VERSION,
    ObjectDetector,
    ensure_model_available,
    ensure_dir,
    find_first_readable_video,
    get_app_dir,
    get_auto_device_info,
    is_subpath,
    load_roi_config,
    norm_path,
    polygon_bbox,
    process_video,
    resolve_default_model_path,
    safe_relpath,
    save_roi_config,
)


def write_csv_log(logs_root: str, rows):
    csv_path = os.path.join(logs_root, "detection_log.csv")
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
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def write_summary_report(reports_root: str, summary_text: str):
    report_path = os.path.join(reports_root, "report_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return report_path


def format_ai_params_text(confidence, start_trigger_frames, end_hold_sec, pre_event_sec, post_event_sec, detect_width, detect_every_n_frames):
    return (
        f"AI參數：conf={confidence} | start_trigger_frames={start_trigger_frames} | "
        f"end_hold_sec={end_hold_sec} | pre={pre_event_sec} | post={post_event_sec} | "
        f"detect_width={detect_width} | stride={detect_every_n_frames}"
    )


def cv_to_qpixmap(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class ParamsDialog(QDialog):
    def __init__(self, parent, confidence, start_trigger_frames, end_hold_sec, pre_event_sec, post_event_sec, detect_width, detect_every_n_frames):
        super().__init__(parent)
        self.setWindowTitle("AI 參數設定")
        self.setModal(True)
        self.result = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("請一次輸入所有 AI 參數："))

        form = QFormLayout()
        layout.addLayout(form)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.01)
        self.spin_conf.setValue(float(confidence))
        form.addRow("YOLO 偵測置信度：", self.spin_conf)

        self.spin_start = QSpinBox()
        self.spin_start.setRange(1, 999999)
        self.spin_start.setValue(int(start_trigger_frames))
        form.addRow("事件開始連續幀數：", self.spin_start)

        self.spin_end_hold = QDoubleSpinBox()
        self.spin_end_hold.setRange(0.0, 999999.0)
        self.spin_end_hold.setDecimals(2)
        self.spin_end_hold.setValue(float(end_hold_sec))
        form.addRow("事件結束等待秒數：", self.spin_end_hold)

        self.spin_pre = QDoubleSpinBox()
        self.spin_pre.setRange(0.0, 999999.0)
        self.spin_pre.setDecimals(2)
        self.spin_pre.setValue(float(pre_event_sec))
        form.addRow("事件前保留秒數：", self.spin_pre)

        self.spin_post = QDoubleSpinBox()
        self.spin_post.setRange(0.0, 999999.0)
        self.spin_post.setDecimals(2)
        self.spin_post.setValue(float(post_event_sec))
        form.addRow("事件後保留秒數：", self.spin_post)

        self.spin_detect_width = QSpinBox()
        self.spin_detect_width.setRange(320, 8192)
        self.spin_detect_width.setValue(int(detect_width))
        form.addRow("偵測前縮圖寬度：", self.spin_detect_width)

        self.spin_detect_stride = QSpinBox()
        self.spin_detect_stride.setRange(1, 999999)
        self.spin_detect_stride.setValue(int(detect_every_n_frames))
        form.addRow("每幾幀偵測一次：", self.spin_detect_stride)

        tip = QLabel("建議：縮圖寬度 960 或 1280；每 2 幀或 3 幀偵測一次，可大幅加速。\n事件片段仍輸出原始影片，不會縮小。")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_accept(self):
        self.result = {
            "confidence": float(self.spin_conf.value()),
            "start_trigger_frames": int(self.spin_start.value()),
            "end_hold_sec": float(self.spin_end_hold.value()),
            "pre_event_sec": float(self.spin_pre.value()),
            "post_event_sec": float(self.spin_post.value()),
            "detect_width": int(self.spin_detect_width.value()),
            "detect_every_n_frames": int(self.spin_detect_stride.value()),
        }
        self.accept()


class PastePathsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("貼上多個來源路徑")
        self.setModal(True)
        self.resize(760, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("每行一個路徑，可同時貼資料夾或影片檔。支援從檔案總管複製後直接貼上。"))
        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        row = QHBoxLayout()
        layout.addLayout(row)

        btn_apply = QPushButton("貼上並加入")
        btn_apply.clicked.connect(self.accept)
        row.addWidget(btn_apply)

        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self.text_edit.clear)
        row.addWidget(btn_clear)
        row.addStretch(1)

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)

    def get_text(self):
        return self.text_edit.toPlainText()


class SourceListWidget(QListWidget):
    paths_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local:
                    paths.append(norm_path(local))
            if paths:
                self.paths_dropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class PolygonRoiView(QGraphicsView):
    def __init__(self, frame_bgr, preset_polygon=None, parent=None):
        super().__init__(parent)
        self.frame_bgr = frame_bgr
        self.points = list(preset_polygon) if preset_polygon else []
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.pixmap_item = QGraphicsPixmapItem(cv_to_qpixmap(frame_bgr))
        self.scene.addItem(self.pixmap_item)
        self.polygon_item = None
        self.point_items = []
        self.label_items = []
        self.help_item = None
        self.refresh_overlay()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            x = int(round(pos.x()))
            y = int(round(pos.y()))
            h, w = self.frame_bgr.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                self.points.append((x, y))
                self.refresh_overlay()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.points:
                self.points.pop()
                self.refresh_overlay()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.points:
                self.points.pop()
                self.refresh_overlay()
            event.accept()
            return
        if event.key() == Qt.Key.Key_C:
            self.points = []
            self.refresh_overlay()
            event.accept()
            return
        super().keyPressEvent(event)

    def refresh_overlay(self):
        for item in self.point_items + self.label_items:
            self.scene.removeItem(item)
        self.point_items = []
        self.label_items = []

        if self.polygon_item is not None:
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None

        if self.help_item is not None:
            self.scene.removeItem(self.help_item)
            self.help_item = None

        if len(self.points) >= 2:
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in self.points])
            self.polygon_item = QGraphicsPolygonItem(polygon)
            self.polygon_item.setPen(QPen(QColor(220, 40, 40), 2))
            self.polygon_item.setBrush(QColor(255, 230, 0, 60) if len(self.points) >= 3 else QColor(0, 0, 0, 0))
            self.scene.addItem(self.polygon_item)

        for idx, (x, y) in enumerate(self.points, start=1):
            ellipse = self.scene.addEllipse(x - 5, y - 5, 10, 10, QPen(Qt.GlobalColor.black), QColor(0, 220, 90))
            label = self.scene.addText(str(idx))
            label.setDefaultTextColor(QColor(0, 220, 90))
            label.setPos(x + 6, y - 18)
            self.point_items.append(ellipse)
            self.label_items.append(label)

        self.help_item = QGraphicsTextItem(
            f"目前點數: {len(self.points)}\n左鍵：新增點\n右鍵 / Backspace：刪除最後一點\nC：清空全部點\n滑鼠滾輪：縮放\n拖曳：平移"
        )
        self.help_item.setDefaultTextColor(QColor(255, 255, 0))
        self.help_item.setPos(16, 16)
        self.scene.addItem(self.help_item)
        self.setSceneRect(QRectF(self.pixmap_item.boundingRect()))


class PolygonRoiDialog(QDialog):
    def __init__(self, video_path, preset_polygon=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Polygon ROI 選取")
        self.setModal(True)
        self.resize(1400, 900)
        self.result_polygon = None

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError("無法讀取 ROI 預覽影片的第一幀。")

        layout = QVBoxLayout(self)
        info = QLabel("請在畫面上點選 Polygon ROI 節點。至少需要 3 點。")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.view = PolygonRoiView(frame, preset_polygon=preset_polygon)
        layout.addWidget(self.view)

        row = QHBoxLayout()
        layout.addLayout(row)

        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self.clear_points)
        row.addWidget(btn_clear)

        btn_undo = QPushButton("刪除最後一點")
        btn_undo.clicked.connect(self.undo_point)
        row.addWidget(btn_undo)
        row.addStretch(1)

        btn_ok = QPushButton("確認 ROI")
        btn_ok.clicked.connect(self.accept_selection)
        row.addWidget(btn_ok)

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)

    def clear_points(self):
        self.view.points = []
        self.view.refresh_overlay()

    def undo_point(self):
        if self.view.points:
            self.view.points.pop()
            self.view.refresh_overlay()

    def accept_selection(self):
        if len(self.view.points) < 3:
            QMessageBox.warning(self, "ROI 點數不足", "Polygon ROI 至少需要 3 個點。")
            return
        self.result_polygon = list(self.view.points)
        self.accept()


class BatchWorker(QObject):
    status_changed = Signal(str)
    video_progress = Signal(int, int)
    frame_progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str, str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        try:
            run_time = self.config["run_time"]
            videos = self.config["videos"]
            total_videos = len(videos)
            total_grabbed = 0
            total_clips = 0
            success_count = 0
            skipped_count = 0
            stopped_count = 0
            csv_rows = []

            self.video_progress.emit(0, total_videos)

            for i, vp in enumerate(videos, start=1):
                if self.stop_requested:
                    self.status_changed.emit("已停止")
                    break

                rel_name = safe_relpath(vp, self.config["input_dir"])
                self.status_changed.emit(f"處理中：{rel_name}（{i}/{total_videos}）")

                result = process_video(
                    video_path=vp,
                    rel_video_path=rel_name,
                    screenshots_root=self.config["screenshots_root"],
                    clips_root=self.config["clips_root"],
                    polygon=self.config["polygon"],
                    detector=self.config["detector"],
                    start_trigger_frames=self.config["start_trigger_frames"],
                    end_hold_sec=self.config["end_hold_sec"],
                    pre_event_sec=self.config["pre_event_sec"],
                    post_event_sec=self.config["post_event_sec"],
                    draw_roi_on_screenshot=self.config["draw_roi_on_screenshot"],
                    export_screenshots=self.config["export_screenshots"],
                    export_clips=self.config["export_clips"],
                    detect_every_n_frames=self.config["detect_every_n_frames"],
                    progress_cb=lambda frame_idx, total_frames: self.frame_progress.emit(frame_idx, total_frames),
                    status_cb=self.status_changed.emit,
                    stop_checker=lambda: self.stop_requested,
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
                        "status": item["status"],
                    })

                self.video_progress.emit(i, total_videos)

                if self.stop_requested:
                    self.status_changed.emit("已停止")
                    break

            self.finished.emit({
                "run_time": run_time,
                "total_videos": total_videos,
                "success_count": success_count,
                "skipped_count": skipped_count,
                "stopped_count": stopped_count,
                "total_grabbed": total_grabbed,
                "total_clips": total_clips,
                "csv_rows": csv_rows,
                "stopped": self.stop_requested,
            })
        except Exception as e:
            self.failed.emit("批次處理失敗", str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV ROI AI Event Extractor (Qt)")
        self.resize(1180, 820)

        self.app_dir = get_app_dir()
        self.input_dir = self.app_dir
        self.selected_input_dirs = []
        self.selected_video_files = []
        self.input_mode = "folder"

        self.polygon = None
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
        self.worker_thread = None
        self.worker = None

        self.build_ui()
        self.refresh_source_list()
        self.update_device_label()
        self.update_ai_label()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        intro = QLabel("執行方式：選輸出資料夾（自動排除）→ 掃描影片 → Polygon ROI（可載入舊設定）→ 一次輸入 AI 參數 → 勾選輸出類型 → 批次執行。")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        source_group = QGroupBox("影片來源")
        layout.addWidget(source_group)
        source_layout = QVBoxLayout(source_group)

        row1 = QHBoxLayout()
        source_layout.addLayout(row1)

        self.btn_pick_input = QPushButton("設為單一資料夾")
        self.btn_pick_input.clicked.connect(self.pick_input_dir)
        row1.addWidget(self.btn_pick_input)

        self.btn_add_input_dir = QPushButton("加入資料夾")
        self.btn_add_input_dir.clicked.connect(self.add_input_dir)
        row1.addWidget(self.btn_add_input_dir)

        self.btn_paste_dirs = QPushButton("貼上多個來源路徑")
        self.btn_paste_dirs.clicked.connect(self.open_paste_paths_dialog)
        row1.addWidget(self.btn_paste_dirs)

        self.btn_pick_file = QPushButton("設為單一影片")
        self.btn_pick_file.clicked.connect(self.pick_single_file)
        row1.addWidget(self.btn_pick_file)

        self.btn_pick_files = QPushButton("設為多個影片")
        self.btn_pick_files.clicked.connect(self.pick_input_files)
        row1.addWidget(self.btn_pick_files)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        source_layout.addLayout(row2)

        self.btn_remove_selected_source = QPushButton("移除選取來源")
        self.btn_remove_selected_source.clicked.connect(self.remove_selected_sources)
        row2.addWidget(self.btn_remove_selected_source)

        self.btn_clear_input = QPushButton("清空來源")
        self.btn_clear_input.clicked.connect(self.clear_input_sources)
        row2.addWidget(self.btn_clear_input)

        row2.addWidget(QLabel("可直接拖曳多個資料夾或影片到下方清單"))
        row2.addStretch(1)

        self.lbl_folder = QLabel()
        self.lbl_folder.setWordWrap(True)
        source_layout.addWidget(self.lbl_folder)

        self.lst_sources = SourceListWidget()
        self.lst_sources.setMinimumHeight(120)
        self.lst_sources.paths_dropped.connect(self.handle_dropped_paths)
        source_layout.addWidget(self.lst_sources)

        self.lbl_model = QLabel(f"模型路徑：{self.model_path}")
        self.lbl_model.setWordWrap(True)
        layout.addWidget(self.lbl_model)

        self.lbl_device = QLabel()
        self.lbl_device.setWordWrap(True)
        layout.addWidget(self.lbl_device)

        self.lbl_excluded = QLabel("排除資料夾：尚未選擇")
        self.lbl_excluded.setWordWrap(True)
        layout.addWidget(self.lbl_excluded)

        self.lbl_out = QLabel("輸出結構：尚未建立")
        self.lbl_out.setWordWrap(True)
        layout.addWidget(self.lbl_out)

        self.lbl_found = QLabel("找到影片數：尚未掃描")
        layout.addWidget(self.lbl_found)

        self.lbl_roi = QLabel("Polygon ROI：尚未選取")
        self.lbl_roi.setWordWrap(True)
        layout.addWidget(self.lbl_roi)

        self.lbl_ai = QLabel()
        self.lbl_ai.setWordWrap(True)
        layout.addWidget(self.lbl_ai)

        option_row = QHBoxLayout()
        layout.addLayout(option_row)

        self.chk_export_screenshots = QCheckBox("輸出截圖")
        self.chk_export_screenshots.setChecked(True)
        option_row.addWidget(self.chk_export_screenshots)

        self.chk_export_clips = QCheckBox("輸出事件片段")
        self.chk_export_clips.setChecked(True)
        option_row.addWidget(self.chk_export_clips)

        self.chk_draw_roi = QCheckBox("截圖畫出 ROI / 框線")
        self.chk_draw_roi.setChecked(True)
        option_row.addWidget(self.chk_draw_roi)
        option_row.addStretch(1)

        action_row = QHBoxLayout()
        layout.addLayout(action_row)

        self.btn_start = QPushButton("開始執行")
        self.btn_start.clicked.connect(self.start_flow)
        action_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_stop.setEnabled(False)
        action_row.addWidget(self.btn_stop)
        action_row.addStretch(1)

        self.lbl_progress = QLabel("進度：0/0")
        layout.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.lbl_frame_progress = QLabel("目前影片進度：0/0")
        layout.addWidget(self.lbl_frame_progress)

        self.lbl_status = QLabel("狀態：待命")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        help_text = QPlainTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(140)
        help_text.setPlainText(
            "來源選擇：可設為單一資料夾、累加多個資料夾、設為單一影片，或一次設為多個影片；也可在清單中多選後移除。\n"
            "Polygon ROI 操作：左鍵加點、右鍵刪點、清空、確認。\n"
            "邏輯說明：偵測 person / car / motorcycle / bus / truck，只有當目標的底部中心點進入 Polygon ROI，且連續達到門檻幀數，才算事件開始。\n"
            "加速版：偵測前自動縮圖，可設定每幾幀偵測一次；事件片段仍輸出原始影片。"
        )
        layout.addWidget(help_text)

    def update_device_label(self):
        self.lbl_device.setText(f"AI裝置：自動判斷（目前：{self.device_info['device']} | {self.device_info['name']}）")

    def update_ai_label(self):
        self.lbl_ai.setText(
            format_ai_params_text(
                self.confidence,
                self.start_trigger_frames,
                self.end_hold_sec,
                self.pre_event_sec,
                self.post_event_sec,
                self.detect_width,
                self.detect_every_n_frames,
            )
        )

    def set_status(self, text):
        self.lbl_status.setText(f"狀態：{text}")

    def refresh_source_list(self):
        self.lst_sources.clear()
        if self.input_mode == "files":
            items = [("檔案", p) for p in self.selected_video_files]
        elif self.input_mode == "folders":
            items = [("資料夾", p) for p in self.selected_input_dirs]
        else:
            items = [("資料夾", self.input_dir)]

        for kind, path in items:
            self.lst_sources.addItem(QListWidgetItem(f"[{kind}] {path}"))

        if self.input_mode == "files" and self.selected_video_files:
            label = f"影片來源檔案：{self.selected_video_files[0]}" if len(self.selected_video_files) == 1 else f"影片來源檔案：共 {len(self.selected_video_files)} 支"
        elif self.input_mode == "folders" and self.selected_input_dirs:
            label = f"影片來源資料夾：{self.selected_input_dirs[0]}" if len(self.selected_input_dirs) == 1 else f"影片來源資料夾：共 {len(self.selected_input_dirs)} 個"
        else:
            label = f"影片來源資料夾：{self.input_dir}"
        self.lbl_folder.setText(label)

    def apply_source_selection(self, folders=None, files=None, append=False):
        folders = [norm_path(x) for x in (folders or []) if str(x).strip()]
        files = [norm_path(x) for x in (files or []) if str(x).strip()]
        files = [x for x in files if os.path.isfile(x) and x.lower().endswith(self.video_exts)]
        folders = [x for x in folders if os.path.isdir(x)]

        if files:
            merged = (self.selected_video_files + files) if append and self.input_mode == "files" else files
            uniq = []
            seen = set()
            for path in merged:
                if path not in seen:
                    seen.add(path)
                    uniq.append(path)
            self.input_mode = "files"
            self.selected_video_files = uniq
            self.selected_input_dirs = []
            self.input_dir = os.path.dirname(uniq[0]) if uniq else self.input_dir
            self.refresh_source_list()
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
            for path in merged:
                if path not in seen:
                    seen.add(path)
                    uniq.append(path)

            if len(uniq) == 1 and not append:
                self.input_mode = "folder"
                self.input_dir = uniq[0]
                self.selected_input_dirs = []
            else:
                self.input_mode = "folders"
                self.selected_input_dirs = uniq
                self.input_dir = uniq[0]
            self.selected_video_files = []
            self.refresh_source_list()
            return len(uniq), "folders"

        return 0, "none"

    def open_paste_paths_dialog(self):
        dialog = PastePathsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_pasted_paths(dialog.get_text())

    def apply_pasted_paths(self, raw_text):
        lines = []
        normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized.split("\n"):
            line = line.strip().strip('"').strip("'")
            if line:
                lines.append(norm_path(line))

        folders = [p for p in lines if os.path.isdir(p)]
        files = [p for p in lines if os.path.isfile(p) and p.lower().endswith(self.video_exts)]
        invalid = [p for p in lines if p not in folders and p not in files]

        if files:
            count, _ = self.apply_source_selection(files=files, append=True)
            self.set_status(f"已加入影片來源，共 {count} 支")
        if folders:
            count, _ = self.apply_source_selection(folders=folders, append=True)
            self.set_status(f"已加入資料夾來源，共 {count} 個")
        if invalid:
            QMessageBox.warning(self, "部分路徑無效", "以下路徑不存在或格式不支援：\n\n" + "\n".join(invalid[:20]))
        if not folders and not files:
            QMessageBox.warning(self, "未加入任何來源", "沒有偵測到有效的資料夾或支援影片檔。")

    def handle_dropped_paths(self, paths):
        folders = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if os.path.isfile(p) and p.lower().endswith(self.video_exts)]
        invalid = [p for p in paths if p not in folders and p not in files]
        if files:
            count, _ = self.apply_source_selection(files=files, append=True)
            self.set_status(f"已拖曳加入影片，共 {count} 支")
        if folders:
            count, _ = self.apply_source_selection(folders=folders, append=True)
            self.set_status(f"已拖曳加入資料夾，共 {count} 個")
        if invalid:
            QMessageBox.warning(self, "部分拖曳來源未加入", "以下項目不存在或格式不支援：\n\n" + "\n".join(invalid[:20]))

    def pick_input_dir(self):
        initialdir = self.selected_input_dirs[0] if self.selected_input_dirs else self.input_dir
        selected = QFileDialog.getExistingDirectory(self, "設為單一來源資料夾", initialdir)
        if not selected:
            return
        self.input_mode = "folder"
        self.input_dir = norm_path(selected)
        self.selected_input_dirs = []
        self.selected_video_files = []
        self.refresh_source_list()
        self.set_status(f"已設為單一資料夾：{self.input_dir}")

    def add_input_dir(self):
        initialdir = self.selected_input_dirs[-1] if self.selected_input_dirs else self.input_dir
        selected = QFileDialog.getExistingDirectory(self, "加入來源資料夾", initialdir)
        if not selected:
            return
        count, _ = self.apply_source_selection(folders=[selected], append=True)
        if count:
            self.set_status(f"已加入來源資料夾，目前共 {count} 個")

    def pick_single_file(self):
        selected, _ = QFileDialog.getOpenFileName(self, "設為單一影片檔", self.input_dir, "影片檔 (*.mp4 *.avi *.mov *.m4v *.mkv *.ts *.264 *.265);;所有檔案 (*.*)")
        if not selected:
            return
        count, _ = self.apply_source_selection(files=[selected], append=False)
        if count:
            self.set_status("已設為單一影片")

    def pick_input_files(self):
        selected, _ = QFileDialog.getOpenFileNames(self, "設為多個影片檔（可 Ctrl / Shift 多選）", self.input_dir, "影片檔 (*.mp4 *.avi *.mov *.m4v *.mkv *.ts *.264 *.265);;所有檔案 (*.*)")
        if not selected:
            return
        count, _ = self.apply_source_selection(files=list(selected), append=False)
        if count:
            self.set_status(f"已設為多個影片，共 {count} 支")

    def remove_selected_sources(self):
        rows = sorted({self.lst_sources.row(item) for item in self.lst_sources.selectedItems()})
        if not rows:
            self.set_status("尚未選取要移除的來源")
            return

        if self.input_mode == "files":
            self.selected_video_files = [p for i, p in enumerate(self.selected_video_files) if i not in rows]
            if self.selected_video_files:
                self.input_dir = os.path.dirname(self.selected_video_files[0])
            else:
                self.input_mode = "folder"
                self.input_dir = self.app_dir
        elif self.input_mode == "folders":
            self.selected_input_dirs = [p for i, p in enumerate(self.selected_input_dirs) if i not in rows]
            if len(self.selected_input_dirs) == 1:
                self.input_mode = "folder"
                self.input_dir = self.selected_input_dirs[0]
                self.selected_input_dirs = []
            elif len(self.selected_input_dirs) > 1:
                self.input_mode = "folders"
                self.input_dir = self.selected_input_dirs[0]
            else:
                self.input_mode = "folder"
                self.input_dir = self.app_dir
        else:
            self.input_dir = self.app_dir

        self.refresh_source_list()
        self.set_status("已移除選取來源")

    def clear_input_sources(self):
        self.input_mode = "folder"
        self.input_dir = self.app_dir
        self.selected_input_dirs = []
        self.selected_video_files = []
        self.refresh_source_list()
        self.set_status("已清空來源，恢復為程式資料夾")

    def find_videos(self, exclude_dir=None):
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
                    dirs[:] = [d for d in dirs if not is_subpath(norm_path(os.path.join(root, d)), exclude_dir_norm)]
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

    def prepare_output_dirs(self, out_dir):
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

        self.lbl_excluded.setText(f"排除資料夾：{self.excluded_dir}")
        self.lbl_out.setText(f"輸出結構：{self.screenshots_root} | {self.clips_root} | {self.logs_root} | {self.reports_root}")

    def set_controls_enabled(self, enabled):
        for widget in (
            self.btn_pick_input,
            self.btn_add_input_dir,
            self.btn_paste_dirs,
            self.btn_pick_file,
            self.btn_pick_files,
            self.btn_remove_selected_source,
            self.btn_clear_input,
            self.lst_sources,
            self.btn_start,
        ):
            widget.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    def request_stop(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.set_status("已要求停止（將盡快於影片處理中止）")

    def ask_all_params(self):
        dialog = ParamsDialog(
            self,
            self.confidence,
            self.start_trigger_frames,
            self.end_hold_sec,
            self.pre_event_sec,
            self.post_event_sec,
            self.detect_width,
            self.detect_every_n_frames,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.result
        return None

    def start_flow(self):
        if not self.chk_export_screenshots.isChecked() and not self.chk_export_clips.isChecked():
            QMessageBox.warning(self, "未選擇輸出類型", "請至少勾選一種輸出類型：截圖或事件片段。")
            return

        self.set_status("檢查模型中...")
        model_ok, model_result = ensure_model_available(self.model_path, status_cb=self.set_status)
        if not model_ok:
            QMessageBox.critical(self, "模型不存在", model_result)
            return
        self.model_path = model_result
        self.lbl_model.setText(f"模型路徑：{self.model_path}")

        out_dir = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾（將自動排除不搜尋）", self.input_dir)
        if not out_dir:
            self.set_status("待命")
            return

        self.prepare_output_dirs(out_dir)
        self.set_status("掃描影片中...")
        videos = self.find_videos(exclude_dir=self.excluded_dir)
        self.lbl_found.setText(f"找到影片數：{len(videos)} 支")

        if not videos:
            QMessageBox.critical(
                self,
                "找不到影片",
                f"在以下資料夾及其子資料夾中，找不到支援的影片格式：\n{self.input_dir}\n\n已自動排除輸出資料夾：\n{self.excluded_dir}\n\n支援：{', '.join(self.video_exts)}",
            )
            self.set_status("待命")
            return

        readable_video = find_first_readable_video(videos)
        if not readable_video:
            QMessageBox.critical(self, "無可用影片", "雖然有找到影片檔，但沒有任何一支影片可成功開啟並讀取第一幀。\n請確認影片格式或解碼器是否正常。")
            self.set_status("待命")
            return

        saved_polygon = load_roi_config(self.app_dir)
        preset_polygon = None
        if saved_polygon:
            result = QMessageBox.question(self, "載入既有 Polygon ROI", f"偵測到先前已儲存 Polygon ROI，共 {len(saved_polygon)} 點。\n\n是否沿用並可再調整？")
            if result == QMessageBox.StandardButton.Yes:
                preset_polygon = saved_polygon

        self.set_status(f"Polygon ROI 框選中：{safe_relpath(readable_video, self.input_dir)}")

        try:
            picker = PolygonRoiDialog(readable_video, preset_polygon=preset_polygon, parent=self)
        except Exception as e:
            QMessageBox.critical(self, "ROI 預覽失敗", str(e))
            self.set_status("待命")
            return

        if picker.exec() != QDialog.DialogCode.Accepted or not picker.result_polygon:
            self.set_status("待命")
            return

        self.polygon = picker.result_polygon
        bx, by, bw, bh = polygon_bbox(self.polygon)
        self.lbl_roi.setText(f"Polygon ROI：{len(self.polygon)} 點 | 外接框 X={bx} Y={by} W={bw} H={bh}")
        save_roi_config(self.app_dir, self.polygon)

        params = self.ask_all_params()
        if not params:
            self.set_status("待命")
            return

        self.confidence = params["confidence"]
        self.start_trigger_frames = params["start_trigger_frames"]
        self.end_hold_sec = params["end_hold_sec"]
        self.pre_event_sec = params["pre_event_sec"]
        self.post_event_sec = params["post_event_sec"]
        self.detect_width = params["detect_width"]
        self.detect_every_n_frames = params["detect_every_n_frames"]
        self.update_ai_label()

        try:
            self.set_status("載入 AI 模型中...")
            self.device_info = get_auto_device_info()
            self.device = self.device_info["device"]
            self.update_device_label()
            self.detector = ObjectDetector(self.model_path, conf=self.confidence, detect_width=self.detect_width, device=self.device)
        except Exception as e:
            QMessageBox.critical(self, "模型載入失敗", str(e))
            self.set_status("待命")
            return

        self.set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.lbl_progress.setText("進度：0/0")
        self.lbl_frame_progress.setText("目前影片進度：0/0")

        config = {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "videos": videos,
            "input_dir": self.input_dir,
            "screenshots_root": self.screenshots_root,
            "clips_root": self.clips_root,
            "polygon": self.polygon,
            "detector": self.detector,
            "start_trigger_frames": self.start_trigger_frames,
            "end_hold_sec": self.end_hold_sec,
            "pre_event_sec": self.pre_event_sec,
            "post_event_sec": self.post_event_sec,
            "draw_roi_on_screenshot": self.chk_draw_roi.isChecked(),
            "export_screenshots": self.chk_export_screenshots.isChecked(),
            "export_clips": self.chk_export_clips.isChecked(),
            "detect_every_n_frames": self.detect_every_n_frames,
        }

        self.worker_thread = QThread(self)
        self.worker = BatchWorker(config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.status_changed.connect(self.set_status)
        self.worker.video_progress.connect(self.on_worker_video_progress)
        self.worker.frame_progress.connect(self.on_worker_frame_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(lambda _title, _message: self.worker_thread.quit())
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()

    def on_worker_video_progress(self, done, total):
        total = max(1, int(total))
        done = min(int(done), total)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.lbl_progress.setText(f"進度：{done}/{total}")

    def on_worker_frame_progress(self, current, total):
        self.lbl_frame_progress.setText(f"目前影片進度：{current}/{total}")

    def on_worker_failed(self, title, message):
        self.set_controls_enabled(True)
        QMessageBox.critical(self, title, message)
        self.set_status("待命")

    def on_worker_finished(self, result):
        csv_path = write_csv_log(self.logs_root, result["csv_rows"])

        bbox_text = "N/A"
        if self.polygon:
            bx, by, bw, bh = polygon_bbox(self.polygon)
            bbox_text = f"X={bx} Y={by} W={bw} H={bh}"

        summary_text = (
            f"Tool Name: CCTV ROI AI Event Extractor (Polygon ROI, Qt)\n"
            f"Tool Version: {APP_VERSION}\n"
            f"Run Time: {result['run_time']}\n"
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
            f"Export Screenshots: {self.chk_export_screenshots.isChecked()}\n"
            f"Export Clips: {self.chk_export_clips.isChecked()}\n"
            f"Draw ROI on Screenshot: {self.chk_draw_roi.isChecked()}\n"
            f"Total Videos Found: {result['total_videos']}\n"
            f"Success Videos: {result['success_count']}\n"
            f"Skipped Videos: {result['skipped_count']}\n"
            f"Stopped Videos: {result['stopped_count']}\n"
            f"Total Screenshots: {result['total_grabbed']}\n"
            f"Total Event Clips: {result['total_clips']}\n"
            f"CSV Log Path: {csv_path}\n"
        )
        report_path = write_summary_report(self.reports_root, summary_text)

        self.set_controls_enabled(True)
        if result["stopped"]:
            self.set_status("已停止")
            QMessageBox.information(self, "已停止", f"批次處理已停止。\n\nCSV 日誌：\n{csv_path}\n\n摘要報表：\n{report_path}")
            return

        self.set_status(f"完成：共擷取 {result['total_grabbed']} 張，輸出 {result['total_clips']} 支事件片段")
        QMessageBox.information(
            self,
            "完成",
            f"已完成。\n找到影片：{result['total_videos']} 支\n成功處理：{result['success_count']} 支\n略過：{result['skipped_count']} 支\n共擷取截圖：{result['total_grabbed']} 張\n共輸出事件片段：{result['total_clips']} 支\n\nCSV 日誌：\n{csv_path}\n\n摘要報表：\n{report_path}",
        )

    def cleanup_worker(self):
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def closeEvent(self, event):
        if self.worker is not None:
            result = QMessageBox.question(self, "正在處理中", "目前仍在處理影片。是否要求停止並關閉視窗？")
            if result != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.request_stop()
            event.ignore()
            return
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
