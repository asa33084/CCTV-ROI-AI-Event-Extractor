import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass

from cctv_roi_ai_event_extractor.config import ensure_runtime_environment, load_config

ensure_runtime_environment()


# 台灣常見車牌格式。OCR 輸出會先被正規化，再用這些格式挑出最可信的候選字串。
TAIWAN_PLATE_PATTERNS = (
    re.compile(r"^[A-Z]{3}[0-9]{4}$"),
    re.compile(r"^[A-Z]{2}[0-9]{4}$"),
    re.compile(r"^[0-9]{4}[A-Z]{2}$"),
    re.compile(r"^[A-Z]{2}[0-9]{3}$"),
)
PLATE_TEXT_CLEAN_RE = re.compile(r"[^A-Z0-9]")
YOLO_LABEL_CLEAN_RE = re.compile(r"[^A-Z0-9_ -]")
YOLO_LABEL_SPLIT_RE = re.compile(r"[_ -]+")
PLATE_TEXT_MIN_LEN = 4
PLATE_TEXT_MAX_LEN = 8
OCR_CROP_MIN_HEIGHT = 96
OCR_CROP_MIN_WIDTH = 240
OCR_CROP_MAX_SCALE = 4.0

STRIP_MAP = str.maketrans({
    " ": "",
    "-": "",
    "_": "",
    ".": "",
    ":": "",
})

LETTER_SLOT_MAP = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
})

DIGIT_SLOT_MAP = str.maketrans({
    "O": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
})


@dataclass(frozen=True)
class PlateCandidate:
    """A detected plate region before OCR is applied."""

    bbox: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class PlateRecognition:
    """Final plate recognition result with detector and OCR metadata."""

    text: str
    raw_text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    detector_score: float
    valid_taiwan_format: bool
    engine: str
    debug_crop_bgr: object | None = None
    crop_quality: float = 0.0
    sharpness_score: float = 0.0
    crop_size_score: float = 0.0
    exposure_score: float = 0.0


def normalize_taiwan_plate_text(raw_text: str) -> str:
    """Normalize OCR text and prefer candidates matching Taiwan plate layouts."""
    text = (raw_text or "").upper()
    text = PLATE_TEXT_CLEAN_RE.sub("", text)
    if not text:
        return ""

    text = text.translate(STRIP_MAP)
    candidates = _plate_text_candidates(text)

    for candidate in candidates:
        if is_valid_taiwan_plate(candidate):
            return candidate
    return candidates[0] if candidates else ""


def _plate_text_candidates(text: str) -> list[str]:
    """Generate full-string and sliding-window candidates for noisy OCR output."""
    candidates = []

    def add(value):
        if PLATE_TEXT_MIN_LEN <= len(value) <= PLATE_TEXT_MAX_LEN and value not in candidates:
            candidates.append(value)
            for positioned in _taiwan_plate_position_candidates(value):
                if positioned not in candidates:
                    candidates.append(positioned)

    add(text)
    if len(text) > PLATE_TEXT_MAX_LEN:
        for size in range(PLATE_TEXT_MAX_LEN, PLATE_TEXT_MIN_LEN - 1, -1):
            for start in range(0, len(text) - size + 1):
                add(text[start:start + size])
    return candidates


def _letters(value: str) -> str:
    return value.translate(LETTER_SLOT_MAP)


def _digits(value: str) -> str:
    return value.translate(DIGIT_SLOT_MAP)


def _taiwan_plate_position_candidates(text: str) -> list[str]:
    """Correct common OCR confusions according to expected letter/digit slots."""
    candidates = []
    if len(text) == 7:
        candidates.append(_letters(text[:3]) + _digits(text[3:]))
    if len(text) == 6:
        candidates.append(_letters(text[:2]) + _digits(text[2:]))
        candidates.append(_digits(text[:4]) + _letters(text[4:]))
    if len(text) == 5:
        candidates.append(_letters(text[:2]) + _digits(text[2:]))
    return [candidate for candidate in candidates if candidate != text]


def is_valid_taiwan_plate(text: str) -> bool:
    clean = PLATE_TEXT_CLEAN_RE.sub("", (text or "").upper())
    return any(pattern.match(clean) for pattern in TAIWAN_PLATE_PATTERNS)


def crop_bbox(frame, bbox, padding_ratio: float = 0.06):
    result = crop_bbox_with_offset(frame, bbox, padding_ratio=padding_ratio)
    if result is None:
        return None
    crop, _offset_x, _offset_y = result
    return crop


def crop_bbox_with_offset(frame, bbox, padding_ratio: float = 0.06):
    """Crop a bounding box and return the crop plus its offset in the source frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * padding_ratio))
    pad_y = int(round(bh * padding_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy(), x1, y1


def _plate_like_aspect_ratio(width: float, height: float) -> bool:
    short_side = max(1.0, min(float(width), float(height)))
    long_side = max(float(width), float(height))
    ratio = long_side / short_side
    return 1.8 <= ratio <= 8.0


def rectify_plate_crop(plate_bgr):
    import cv2
    import numpy as np

    # Use the strongest contour to deskew oblique plate crops before OCR.
    if plate_bgr is None or plate_bgr.size == 0:
        return plate_bgr

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_bgr

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < max(20, plate_bgr.shape[0] * plate_bgr.shape[1] * 0.12):
        return plate_bgr

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype("float32")
    width = int(max(rect[1]))
    height = int(min(rect[1]))
    if width <= 0 or height <= 0:
        return plate_bgr
    if not _plate_like_aspect_ratio(width, height):
        return plate_bgr
    rect_area = float(width * height)
    crop_area = float(plate_bgr.shape[0] * plate_bgr.shape[1])
    if rect_area < crop_area * 0.25:
        return plate_bgr
    if height > width:
        width, height = height, width

    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    ordered = np.array([
        box[np.argmin(s)],
        box[np.argmin(diff)],
        box[np.argmax(s)],
        box[np.argmax(diff)],
    ], dtype="float32")
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(plate_bgr, matrix, (width, height))


def enhance_plate_crop_for_ocr(plate_bgr):
    """Upscale and enhance a plate crop while preserving a natural BGR image for OCR engines."""
    import cv2

    if plate_bgr is None or plate_bgr.size == 0:
        return plate_bgr

    height, width = plate_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return plate_bgr

    scale = max(OCR_CROP_MIN_HEIGHT / height, OCR_CROP_MIN_WIDTH / width, 1.0)
    scale = min(scale, OCR_CROP_MAX_SCALE)
    enhanced = plate_bgr
    if scale > 1.01:
        enhanced = cv2.resize(
            enhanced,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    tile_size = max(4, min(8, min(l_channel.shape[:2]) // 8 or 4))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    enhanced = cv2.bilateralFilter(enhanced, 5, 35, 35)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)

    border_y = max(2, int(round(enhanced.shape[0] * 0.04)))
    border_x = max(4, int(round(enhanced.shape[1] * 0.03)))
    return cv2.copyMakeBorder(enhanced, border_y, border_y, border_x, border_x, cv2.BORDER_REPLICATE)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def plate_crop_quality_scores(plate_bgr) -> tuple[float, float, float]:
    """Score OCR crop sharpness, usable size, and exposure on normalized 0..1 scales."""
    import cv2

    if plate_bgr is None or plate_bgr.size == 0:
        return 0.0, 0.0, 0.0

    height, width = plate_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return 0.0, 0.0, 0.0

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = _clamp_score(laplacian_var / 500.0)

    crop_size_score = _clamp_score(min(width / OCR_CROP_MIN_WIDTH, height / OCR_CROP_MIN_HEIGHT))

    mean_value = float(gray.mean())
    std_value = float(gray.std())
    mean_score = 1.0 - (abs(mean_value - 128.0) / 128.0)
    contrast_score = _clamp_score(std_value / 50.0)
    exposure_score = _clamp_score(mean_score) * contrast_score
    return sharpness_score, crop_size_score, exposure_score


def plate_recognition_quality_score(
    detector_score: float,
    sharpness_score: float,
    crop_size_score: float,
    exposure_score: float,
    ocr_confidence: float,
    valid_taiwan_format: bool,
) -> float:
    """Combine detector, crop, and OCR signals so the best crop wins across a track."""
    valid_bonus = 0.25 if valid_taiwan_format else 0.0
    return (
        0.35 * _clamp_score(detector_score)
        + 0.20 * _clamp_score(sharpness_score)
        + 0.15 * _clamp_score(crop_size_score)
        + 0.10 * _clamp_score(exposure_score)
        + 0.20 * _clamp_score(ocr_confidence)
        + valid_bonus
    )


class YoloPlateDetector:
    """YOLO wrapper that returns plate boxes in a small project-specific shape."""

    def __init__(self, model_path: str, conf: float = 0.50, device: str | None = None):
        from ultralytics import YOLO

        self.model_path = os.path.abspath(model_path)
        self.model = YOLO(self.model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame) -> list[PlateCandidate]:
        results = self.model(frame, conf=self.conf, verbose=False, device=self.device)
        plates = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                plates.append(PlateCandidate((x1, y1, x2, y2), score))
        return plates


class OcrEngine:
    """Null OCR engine used when LPR detection is enabled without text recognition."""

    name = "none"

    def recognize(self, image_bgr) -> tuple[str, float]:
        return "", 0.0


class EasyOcrEngine(OcrEngine):
    name = "easyocr"

    def __init__(self):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=False)

    def recognize(self, image_bgr) -> tuple[str, float]:
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        if not results:
            return "", 0.0
        _bbox, text, confidence = max(results, key=lambda item: float(item[2]))
        return str(text), float(confidence)


def _tesseract_executable_name() -> str:
    return "tesseract.exe" if os.name == "nt" else "tesseract"


def _candidate_tesseract_commands(path: str) -> list[str]:
    """Expand either a tesseract executable path or a containing directory."""
    if not path:
        return []

    clean = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    executable = _tesseract_executable_name()
    candidates = []
    if os.path.isdir(clean):
        candidates.append(os.path.join(clean, executable))
        if os.name == "nt":
            candidates.append(os.path.join(clean, "tesseract.exe"))
    else:
        candidates.append(clean)
        root, ext = os.path.splitext(clean)
        if os.name == "nt" and not ext:
            candidates.append(f"{clean}.exe")
    return candidates


def resolve_tesseract_cmd(config=None) -> str | None:
    """Find Tesseract from explicit config, PATH, bundled app paths, and common installs."""
    config = config or load_config()
    configured = (config.lpr_tesseract_cmd or "").strip()
    candidates = _candidate_tesseract_commands(configured)

    path_command = shutil.which("tesseract")
    if path_command:
        candidates.append(path_command)

    executable = _tesseract_executable_name()
    app_dir = os.path.abspath(config.app_dir)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend([
        os.path.join(app_dir, executable),
        os.path.join(app_dir, "tesseract", executable),
        os.path.join(app_dir, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(project_root, ".venv", "bin", "tesseract"),
        os.path.join(project_root, ".venv", "Scripts", "tesseract.exe"),
    ])

    if os.name == "nt":
        program_files = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ]
        for base_dir in [item for item in program_files if item]:
            candidates.append(os.path.join(base_dir, "Tesseract-OCR", "tesseract.exe"))
    else:
        candidates.extend([
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            "/opt/local/bin/tesseract",
        ])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


class TesseractOcrEngine(OcrEngine):
    name = "tesseract"

    def __init__(self):
        try:
            import pytesseract
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "使用 tesseract OCR 時必須先在目前 Python 環境安裝 pytesseract。"
            ) from exc

        self.pytesseract = pytesseract
        command = resolve_tesseract_cmd()
        if not command:
            raise FileNotFoundError(
                "找不到 Tesseract 執行檔。請安裝 tesseract-ocr，或設定 "
                "CCTV_ROI_LPR_TESSERACT_CMD 為 tesseract 執行檔或安裝資料夾。"
            )
        self.tesseract_cmd = command
        self.pytesseract.pytesseract.tesseract_cmd = command

    def _preprocess_variants(self, image_bgr):
        import cv2

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        scale = max(2, int(round(96 / max(1, gray.shape[0]))))
        scale = min(scale, 4)
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        _threshold, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        return (resized, otsu, adaptive)

    def _recognize_variant(self, image, config: str) -> tuple[str, float]:
        text = self.pytesseract.image_to_string(image, config=config).strip()
        confidences = []
        try:
            data = self.pytesseract.image_to_data(
                image,
                config=config,
                output_type=self.pytesseract.Output.DICT,
            )
        except Exception:
            data = None
        if data:
            for value in data.get("conf", []):
                try:
                    confidence = float(value)
                except (TypeError, ValueError):
                    continue
                if confidence >= 0:
                    confidences.append(confidence / 100.0)
        return text, _mean_confidence(confidences)

    def recognize(self, image_bgr) -> tuple[str, float]:
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        candidates = [
            self._recognize_variant(image, config)
            for image in self._preprocess_variants(image_bgr)
        ]
        candidates = [item for item in candidates if item[0].strip()]
        if not candidates:
            return "", 0.0
        scored = []
        for text, confidence in candidates:
            normalized = normalize_taiwan_plate_text(text)
            scored.append((is_valid_taiwan_plate(normalized), confidence, len(normalized), text, confidence))
        _valid, _score, _length, text, confidence = max(scored, key=lambda item: item[:3])
        return text, confidence


def _mean_confidence(scores: list[float]) -> float:
    valid_scores = [float(score) for score in scores if score is not None]
    if not valid_scores:
        return 0.0
    return sum(valid_scores) / len(valid_scores)


def _unwrap_paddle_result(result) -> Mapping | None:
    if isinstance(result, Mapping):
        data = result
    elif hasattr(result, "res") and isinstance(result.res, Mapping):
        data = result.res
    else:
        try:
            data = dict(result)
        except (TypeError, ValueError):
            return None

    nested = data.get("res")
    if isinstance(nested, Mapping):
        return nested
    return data


def _extract_paddle_v3_text(result) -> tuple[str, float]:
    """Read PaddleOCR 3.x result objects and keep text ordered from left to right."""
    data = _unwrap_paddle_result(result)
    if not data:
        return "", 0.0

    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []
    if not texts:
        return "", 0.0

    indexes = list(range(len(texts)))
    boxes = data.get("rec_boxes")
    if boxes is not None:
        try:
            indexes.sort(key=lambda idx: float(boxes[idx][0]))
        except (IndexError, TypeError, ValueError):
            pass

    ordered_texts = [str(texts[idx]).strip() for idx in indexes if str(texts[idx]).strip()]
    ordered_scores = []
    for idx in indexes:
        try:
            ordered_scores.append(float(scores[idx]))
        except (IndexError, TypeError, ValueError):
            continue
    return "".join(ordered_texts), _mean_confidence(ordered_scores)


def _extract_paddle_legacy_text(results) -> tuple[str, float]:
    """Read older PaddleOCR nested list output."""
    candidates = []
    for item in results or []:
        if not isinstance(item, (list, tuple)):
            continue
        for line in item:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            text_score = line[1]
            if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
                continue
            text = str(text_score[0]).strip()
            try:
                score = float(text_score[1])
            except (TypeError, ValueError):
                score = 0.0
            if text:
                candidates.append((text, score))
    if not candidates:
        return "", 0.0
    text = "".join(item[0] for item in candidates)
    return text, _mean_confidence([item[1] for item in candidates])


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"

    def __init__(self):
        from paddleocr import PaddleOCR

        config = load_config()
        kwargs = {
            "text_detection_model_name": config.lpr_paddle_det_model_name,
            "text_recognition_model_name": config.lpr_paddle_rec_model_name,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if not config.lpr_paddle_det_model_name and not config.lpr_paddle_rec_model_name:
            kwargs["ocr_version"] = config.lpr_paddle_ocr_version
        if config.lpr_paddle_device:
            kwargs["device"] = config.lpr_paddle_device

        self.ocr = PaddleOCR(**kwargs)

    def recognize(self, image_bgr) -> tuple[str, float]:
        results = self.ocr.predict(image_bgr)
        texts = []
        confidences = []
        for result in results or []:
            text, confidence = _extract_paddle_v3_text(result)
            if text:
                texts.append(text)
                confidences.append(confidence)
        if texts:
            return "".join(texts), _mean_confidence(confidences)
        return _extract_paddle_legacy_text(results)


def _parse_input_size(value: str) -> tuple[int, int]:
    clean = (value or "48x160").lower().replace(",", "x").replace(" ", "")
    parts = clean.split("x")
    if len(parts) != 2:
        return 48, 160
    try:
        height = int(parts[0])
        width = int(parts[1])
    except ValueError:
        return 48, 160
    return max(16, height), max(32, width)


def _load_charset(charset: str, charset_path: str | None) -> list[str]:
    if charset_path and os.path.exists(charset_path):
        with open(charset_path, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f if line.strip()]
        if chars:
            return chars
    return list(charset or "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _softmax(values, axis=-1):
    import numpy as np

    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def ctc_decode(logits, charset: list[str], blank_index: int = 0) -> tuple[str, float]:
    """Decode CTC logits by removing blanks and repeated consecutive classes."""
    import numpy as np

    arr = np.asarray(logits)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return "", 0.0

    class_count = len(charset) + 1
    if arr.shape[0] == class_count and arr.shape[1] != class_count:
        arr = arr.transpose(1, 0)

    probs = _softmax(arr, axis=-1)
    indexes = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)

    chars = []
    kept_confidences = []
    prev_idx = None
    for idx, conf in zip(indexes.tolist(), confidences.tolist()):
        if idx == blank_index:
            prev_idx = idx
            continue
        if idx == prev_idx:
            continue
        char_idx = idx if blank_index != 0 else idx - 1
        if 0 <= char_idx < len(charset):
            chars.append(charset[char_idx])
            kept_confidences.append(float(conf))
        prev_idx = idx

    if not chars:
        return "", 0.0
    return "".join(chars), float(np.mean(kept_confidences))


def index_sequence_decode(indexes, confidences, charset: list[str], blank_index: int = 0) -> tuple[str, float]:
    """Decode ONNX models that already output class indexes and per-position confidence."""
    import numpy as np

    idx_arr = np.asarray(indexes)
    conf_arr = np.asarray(confidences)
    if idx_arr.ndim >= 2:
        idx_arr = idx_arr[0]
    if conf_arr.ndim >= 2:
        conf_arr = conf_arr[0]

    chars = []
    kept_confidences = []
    prev_idx = None
    for raw_idx, raw_conf in zip(idx_arr.tolist(), conf_arr.tolist()):
        idx = int(raw_idx)
        if idx == blank_index:
            prev_idx = idx
            continue
        if idx == prev_idx:
            continue
        char_idx = idx if blank_index != 0 else idx - 1
        if 0 <= char_idx < len(charset):
            chars.append(charset[char_idx])
            kept_confidences.append(float(raw_conf))
        prev_idx = idx

    if not chars:
        return "", 0.0
    return "".join(chars), float(np.mean(kept_confidences))


class SvtrOcrEngine(OcrEngine):
    """ONNX SVTR/Transformer OCR engine for plate text recognition."""

    name = "svtr"

    def __init__(self):
        import onnxruntime as ort

        config = load_config()
        if not config.lpr_svtr_model_path:
            raise ValueError("使用 svtr OCR 時必須設定 CCTV_ROI_LPR_SVTR_MODEL_PATH。")

        self.model_path = os.path.abspath(config.lpr_svtr_model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到 SVTR/Transformer OCR 模型：{self.model_path}")

        self.input_height, self.input_width = _parse_input_size(config.lpr_svtr_input_size)
        self.charset = _load_charset(config.lpr_svtr_charset, config.lpr_svtr_charset_path)
        self.blank_index = int(config.lpr_svtr_blank_index)

        providers = self._resolve_providers(config.lpr_svtr_providers, ort)
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _resolve_providers(self, configured: str, ort):
        available = ort.get_available_providers()
        if configured and configured.lower() != "auto":
            requested = [item.strip() for item in configured.split(",") if item.strip()]
            selected = [item for item in requested if item in available]
            if selected:
                return selected
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        return [item for item in preferred if item in available] or available

    def _preprocess(self, image_bgr):
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype("float32") / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    def recognize(self, image_bgr) -> tuple[str, float]:
        import numpy as np

        tensor = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) >= 2 and np.asarray(outputs[0]).dtype.kind in {"i", "u"}:
            return index_sequence_decode(outputs[0], outputs[1], self.charset, blank_index=self.blank_index)
        return ctc_decode(outputs[0], self.charset, blank_index=self.blank_index)


YOLO_OCR_FALLBACK_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _normalize_yolo_ocr_label(label: object, class_id: int | None = None) -> str:
    text = str(label or "").strip().upper()
    text = YOLO_LABEL_CLEAN_RE.sub("", text)
    for token in reversed(YOLO_LABEL_SPLIT_RE.split(text)):
        if len(token) == 1 and token.isalnum():
            return token
    if class_id is not None and 0 <= class_id < len(YOLO_OCR_FALLBACK_CHARSET):
        return YOLO_OCR_FALLBACK_CHARSET[class_id]
    return ""


class PlateNumberYolo26xOcrEngine(OcrEngine):
    """OCR engine for YOLO models that detect one alphanumeric class per plate character."""

    name = "plate_number_yolo26x"

    def __init__(self, device: str | None = None):
        from ultralytics import YOLO

        config = load_config()
        model_path = config.lpr_yolo_ocr_model_path or os.path.join(
            config.app_dir,
            "models",
            "plate_number_yolo26x.pt",
        )
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "找不到 YOLO26x 車牌號碼 OCR 模型："
                f"{self.model_path}\n請設定 CCTV_ROI_LPR_YOLO_OCR_MODEL_PATH，"
                "或放置 models/plate_number_yolo26x.pt。"
            )
        self.model = YOLO(self.model_path)
        self.conf = float(config.lpr_yolo_ocr_confidence)
        self.device = device

    def _class_label(self, result, class_id: int) -> str:
        names = getattr(result, "names", None) or getattr(self.model, "names", None) or {}
        if isinstance(names, Mapping):
            label = names.get(class_id, names.get(str(class_id), ""))
        else:
            try:
                label = names[class_id]
            except (IndexError, TypeError):
                label = ""
        return _normalize_yolo_ocr_label(label, class_id=class_id)

    def recognize(self, image_bgr) -> tuple[str, float]:
        results = self.model(image_bgr, conf=self.conf, verbose=False, device=self.device)
        chars = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                char = self._class_label(result, class_id)
                if not char:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf[0].item())
                chars.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, char, confidence))

        if not chars:
            return "", 0.0

        chars.sort(key=lambda item: (item[0], item[1]))
        text = "".join(item[2] for item in chars)
        confidence = _mean_confidence([item[3] for item in chars])
        return text, confidence


def create_ocr_engine(engine_name: str | None, device: str | None = None) -> OcrEngine:
    name = (engine_name or "none").strip().lower()
    if name in ("svtr", "transformer"):
        return SvtrOcrEngine()
    if name in ("plate_number_yolo26x", "yolo26x", "yolo_ocr", "yolo-char", "yolo_char"):
        return PlateNumberYolo26xOcrEngine(device=device)
    if name in ("paddleocr", "paddle", "ppocr"):
        return PaddleOcrEngine()
    if name == "easyocr":
        return EasyOcrEngine()
    if name in ("tesseract", "pytesseract"):
        return TesseractOcrEngine()
    return OcrEngine()


class LicensePlateRecognizer:
    """Run plate detection, crop normalization, OCR, and Taiwan-format validation."""

    def __init__(
        self,
        plate_model_path: str,
        ocr_engine: str = "none",
        conf: float = 0.50,
        device: str | None = None,
    ):
        self.detector = YoloPlateDetector(plate_model_path, conf=conf, device=device)
        self.ocr = create_ocr_engine(ocr_engine, device=device)

    @property
    def engine_name(self):
        return self.ocr.name

    def recognize(self, frame, vehicle_detections=None) -> list[PlateRecognition]:
        recognitions = []
        vehicle_bboxes = []
        for det in vehicle_detections or []:
            bbox = det.get("bbox") if isinstance(det, dict) else None
            if bbox is not None and bbox not in vehicle_bboxes:
                vehicle_bboxes.append(bbox)

        if vehicle_bboxes:
            plate_sources = []
            for vehicle_bbox in vehicle_bboxes:
                # Detect plates inside each vehicle crop first, then map boxes back to the full frame.
                crop_result = crop_bbox_with_offset(frame, vehicle_bbox, padding_ratio=0.02)
                if crop_result is None:
                    continue
                vehicle_crop, offset_x, offset_y = crop_result
                for plate in self.detector.detect(vehicle_crop):
                    x1, y1, x2, y2 = plate.bbox
                    plate_sources.append(PlateCandidate(
                        bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                        score=plate.score,
                    ))
        else:
            plate_sources = self.detector.detect(frame)

        for plate in plate_sources:
            crop = crop_bbox(frame, plate.bbox, padding_ratio=0.10)
            if crop is None:
                continue
            # Score the natural crop before enhancement; OCR still receives the enhanced crop.
            rectified_crop = rectify_plate_crop(crop)
            sharpness_score, crop_size_score, exposure_score = plate_crop_quality_scores(rectified_crop)
            ocr_crop = enhance_plate_crop_for_ocr(rectified_crop)
            raw_text, ocr_conf = self.ocr.recognize(ocr_crop)
            text = normalize_taiwan_plate_text(raw_text)
            valid_taiwan_format = is_valid_taiwan_plate(text)
            crop_quality = plate_recognition_quality_score(
                detector_score=plate.score,
                sharpness_score=sharpness_score,
                crop_size_score=crop_size_score,
                exposure_score=exposure_score,
                ocr_confidence=ocr_conf,
                valid_taiwan_format=valid_taiwan_format,
            )
            recognitions.append(PlateRecognition(
                text=text,
                raw_text=raw_text,
                confidence=ocr_conf,
                bbox=plate.bbox,
                detector_score=plate.score,
                valid_taiwan_format=valid_taiwan_format,
                engine=self.engine_name,
                debug_crop_bgr=ocr_crop.copy() if ocr_crop is not None else None,
                crop_quality=crop_quality,
                sharpness_score=sharpness_score,
                crop_size_score=crop_size_score,
                exposure_score=exposure_score,
            ))
        return recognitions


def _draw_plate_debug_crop(frame, item, index):
    import cv2

    crop = getattr(item, "debug_crop_bgr", None)
    if crop is None or getattr(crop, "size", 0) == 0:
        return

    frame_h, frame_w = frame.shape[:2]
    crop_h, crop_w = crop.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return

    target_w = min(220, max(96, int(frame_w * 0.18)))
    target_w = min(target_w, max(24, frame_w - 12))
    scale = target_w / crop_w
    target_h = max(24, int(round(crop_h * scale)))
    if target_h > 96:
        target_h = 96
        target_w = max(48, int(round(crop_w * (target_h / crop_h))))
    target_h = min(target_h, max(16, frame_h - 32))
    target_w = min(target_w, max(24, frame_w - 12))

    x1, y1, x2, y2 = item.bbox
    pad = 6
    label_h = 20
    panel_w = target_w + pad * 2
    panel_h = target_h + label_h + pad * 2
    px = min(max(0, x1), max(0, frame_w - panel_w))
    below_y = y2 + 8
    above_y = y1 - panel_h - 8
    py = below_y if below_y + panel_h <= frame_h else max(0, above_y)

    color = (255, 0, 255) if item.valid_taiwan_format else (255, 180, 0)
    roi = frame[py:py + panel_h, px:px + panel_w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w - 1, panel_h - 1), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.78, roi, 0.22, 0, dst=roi)
    cv2.rectangle(frame, (px, py), (px + panel_w - 1, py + panel_h - 1), color, 2)

    thumb = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    tx = px + pad
    ty = py + label_h + pad
    frame[ty:ty + target_h, tx:tx + target_w] = thumb
    label = f"OCR crop {index}"
    cv2.putText(frame, label, (px + pad, py + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_plate_recognitions(frame, recognitions, show_debug_crops=False):
    import cv2

    crop_index = 1
    for item in recognitions or []:
        x1, y1, x2, y2 = item.bbox
        color = (255, 0, 255) if item.valid_taiwan_format else (255, 180, 0)
        label = item.text or "plate"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if show_debug_crops:
            _draw_plate_debug_crop(frame, item, crop_index)
            crop_index += 1
    return frame
