"""Legacy Tk GUI backend kept separate from the processing pipeline."""

import csv
import os
import queue
import threading
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None

from cctv_roi_ai_event_extractor.event_processing import *  # noqa: F401,F403

# ---------------------------
# 一次填完全部參數的視窗
# ---------------------------
class ParamsDialog(tk.Toplevel):
    """Legacy Tk dialog for collecting detection timing and inference parameters."""

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

        ttk.Label(self, text="事件中心前保留秒數：").grid(row=4, column=0, padx=12, pady=6, sticky="e")
        self.ent_pre = ttk.Entry(self, width=18)
        self.ent_pre.grid(row=4, column=1, padx=12, pady=6, sticky="w")
        self.ent_pre.insert(0, str(pre_event_sec))

        ttk.Label(self, text="事件中心後保留秒數：").grid(row=5, column=0, padx=12, pady=6, sticky="e")
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
            "事件片段以事件時間點為中心，仍輸出原始影片且不會縮小。"
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
        """Validate user-entered numeric parameters before closing the dialog."""
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
    """Legacy Tk dialog for pasting multiple source file/folder paths."""

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
    """Legacy Tk application retained for compatibility with older entry points."""

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
        """Build the legacy Tk controls and wire button commands."""
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
            "加速版：偵測前自動縮圖，可設定每幾幀偵測一次；事件片段以事件時間點為中心輸出原始影片。\n"
            "製作人：家宏。"
        )
        ttk.Label(frm, text=help_text, foreground="#444", justify="left").pack(anchor="w", pady=(8, 0))

        self._refresh_source_listbox()

    def _post_ui(self, action, **kwargs):
        """Queue UI work from background threads so Tk updates stay on the main thread."""
        self.ui_queue.put((action, kwargs))

    def _poll_ui_queue(self):
        """Apply queued UI updates and reschedule polling."""
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
                    if int(kwargs["total"]) > 0:
                        self.lbl_frame_progress.config(text=f"目前影片進度：{kwargs['current']}/{kwargs['total']}")
                    else:
                        self.lbl_frame_progress.config(text=f"目前影片進度：已處理 {kwargs['current']} 幀（總幀數未知）")
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
        """Split tkinterdnd2 drop payloads while preserving paths wrapped in braces."""
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
        """Apply source selections and keep file/folder modes mutually exclusive."""
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
        """Parse pasted paths and add any valid supported sources."""
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
        """Handle drag-and-drop source additions in the legacy Tk UI."""
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
        """Find supported videos from current sources while excluding output folders."""
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
        """Create output directories and update labels in the legacy UI."""
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
        """Write the legacy UI batch CSV log."""
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
                    "status",
                    "camera_id",
                    "stream_id",
                    "track_id",
                    "track_start_datetime",
                    "track_end_datetime",
                    "track_start_source",
                    "track_end_source",
                    "track_seen_frames",
                    "track_duration_sec",
                    "plate_text",
                    "plate_raw_text",
                    "plate_confidence",
                    "plate_bbox",
                    "plate_crop_path",
                    "plate_crop_quality",
                    "plate_valid_taiwan_format",
                    "plate_ocr_engine",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return csv_path

    def _write_summary_report(self, summary_text: str):
        """Write the legacy UI run summary report."""
        report_path = os.path.join(self.reports_root, "report_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        return report_path

    def _ask_all_params(self):
        """Open the parameter dialog and return its validated result."""
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
        """Collect user choices, load model/ROI, and start the legacy worker thread."""
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
            self.set_status(f"車輛 YOLO tracker：{self.detector.tracker_path}")
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
        """Background worker for the legacy UI; posts progress/results through a queue."""
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
                    "status": item["status"],
                    "camera_id": item.get("camera_id", ""),
                    "stream_id": item.get("stream_id", ""),
                    "track_id": item.get("track_id", ""),
                    "track_start_datetime": item.get("track_start_datetime", ""),
                    "track_end_datetime": item.get("track_end_datetime", ""),
                    "track_start_source": item.get("track_start_source", ""),
                    "track_end_source": item.get("track_end_source", ""),
                    "track_seen_frames": item.get("track_seen_frames", ""),
                    "track_duration_sec": item.get("track_duration_sec", ""),
                    "plate_text": item.get("plate_text", ""),
                    "plate_raw_text": item.get("plate_raw_text", ""),
                    "plate_confidence": item.get("plate_confidence", ""),
                    "plate_bbox": item.get("plate_bbox", ""),
                    "plate_crop_path": item.get("plate_crop_path", ""),
                    "plate_crop_quality": item.get("plate_crop_quality", ""),
                    "plate_valid_taiwan_format": item.get("plate_valid_taiwan_format", ""),
                    "plate_ocr_engine": item.get("plate_ocr_engine", ""),
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
            f"Tracker Path: {getattr(self.detector, 'tracker_path', '') if self.detector else ''}\n"
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
    """Launch the legacy Tk GUI application."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
