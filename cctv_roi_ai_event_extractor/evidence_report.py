import os
import re
from copy import copy
from datetime import datetime, timedelta

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


VIDEO_DATETIME_PATTERNS = (
    re.compile(r"(?P<date>20\d{6})[_-](?P<time>\d{6})"),
    re.compile(r"p(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})[_-](?P<time>\d{6})", re.IGNORECASE),
)


def parse_video_start_datetime(video_rel_path: str):
    name = os.path.basename(video_rel_path or "")
    for pattern in VIDEO_DATETIME_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        groups = match.groupdict()
        if "date" in groups and groups.get("date"):
            raw_date = groups["date"]
            raw_time = groups["time"]
            return datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")
        year = 2000 + int(groups["yy"])
        raw = f"{year:04d}{groups['mm']}{groups['dd']}{groups['time']}"
        return datetime.strptime(raw, "%Y%m%d%H%M%S")
    return None


def format_event_datetime(video_rel_path: str, seconds_text: str):
    if seconds_text is None or str(seconds_text).strip() == "":
        return ""
    start = parse_video_start_datetime(video_rel_path)
    if start is None:
        return ""
    try:
        seconds = float(seconds_text or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return (start + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


def _first_existing_path(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def build_evidence_rows(csv_rows):
    rows = []
    for item in csv_rows:
        if item.get("record_type") != "screenshot":
            continue
        screenshot_path = _first_existing_path([item.get("output_path")])
        if not screenshot_path:
            continue
        rows.append({
            "entry_datetime": format_event_datetime(item.get("video_rel_path", ""), item.get("event_time_sec", "")),
            "exit_datetime": format_event_datetime(item.get("video_rel_path", ""), item.get("interval_end_sec", "")),
            "plate_text": item.get("plate_text", ""),
            "screenshot_path": screenshot_path,
        })
    return rows


def write_evidence_workbook(output_path: str, csv_rows):
    rows = build_evidence_rows(csv_rows)

    wb = Workbook()
    ws = wb.active
    ws.title = "蒐證資料"

    headers = ["編號", "進入日期", "出去日期", "車號", "車輛截圖"]
    ws.append(headers)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    ws.freeze_panes = "A2"
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 22
    ws.column_dimensions["D"].width = 16
    ws.column_dimensions["E"].width = 34

    for idx, item in enumerate(rows, start=1):
        row_idx = idx + 1
        ws.cell(row=row_idx, column=1, value=idx)
        ws.cell(row=row_idx, column=2, value=item["entry_datetime"])
        ws.cell(row=row_idx, column=3, value=item["exit_datetime"])
        ws.cell(row=row_idx, column=4, value=item["plate_text"])
        ws.row_dimensions[row_idx].height = 82

        for col_idx in range(1, 5):
            ws.cell(row=row_idx, column=col_idx).alignment = Alignment(horizontal="center", vertical="center")

        try:
            img = ExcelImage(item["screenshot_path"])
            img.width = 160
            img.height = 90
            ws.add_image(img, f"E{row_idx}")
        except Exception:
            ws.cell(row=row_idx, column=5, value=item["screenshot_path"])

    for row in ws.iter_rows(min_row=1, max_row=max(1, len(rows) + 1), min_col=1, max_col=5):
        for cell in row:
            alignment = copy(cell.alignment)
            alignment.wrap_text = True
            alignment.vertical = "center"
            cell.alignment = alignment

    ensure_dir = os.path.dirname(output_path)
    if ensure_dir:
        os.makedirs(ensure_dir, exist_ok=True)
    wb.save(output_path)
    return output_path, len(rows)
