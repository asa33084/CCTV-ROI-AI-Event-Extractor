import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from cctv_roi_ai_event_extractor.config import ensure_runtime_environment, load_config

ensure_runtime_environment()


TAIWAN_PLATE_PATTERNS = (
    re.compile(r"^[A-Z]{3}[0-9]{4}$"),
    re.compile(r"^[A-Z]{2}[0-9]{4}$"),
    re.compile(r"^[0-9]{4}[A-Z]{2}$"),
    re.compile(r"^[A-Z]{2}[0-9]{3}$"),
)

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
    bbox: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class PlateRecognition:
    text: str
    raw_text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    detector_score: float
    valid_taiwan_format: bool
    engine: str


def normalize_taiwan_plate_text(raw_text: str) -> str:
    text = (raw_text or "").upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    if not text:
        return ""

    text = text.translate(STRIP_MAP)
    candidates = [text]
    candidates.extend(_taiwan_plate_position_candidates(text))

    for candidate in candidates:
        if is_valid_taiwan_plate(candidate):
            return candidate
    return candidates[-1] if candidates else text


def _letters(value: str) -> str:
    return value.translate(LETTER_SLOT_MAP)


def _digits(value: str) -> str:
    return value.translate(DIGIT_SLOT_MAP)


def _taiwan_plate_position_candidates(text: str) -> list[str]:
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
    clean = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return any(pattern.match(clean) for pattern in TAIWAN_PLATE_PATTERNS)


def crop_bbox(frame, bbox, padding_ratio: float = 0.06):
    result = crop_bbox_with_offset(frame, bbox, padding_ratio=padding_ratio)
    if result is None:
        return None
    crop, _offset_x, _offset_y = result
    return crop


def crop_bbox_with_offset(frame, bbox, padding_ratio: float = 0.06):
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


def rectify_plate_crop(plate_bgr):
    import cv2
    import numpy as np

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


class YoloPlateDetector:
    def __init__(self, model_path: str, conf: float = 0.35, device: str | None = None):
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


class TesseractOcrEngine(OcrEngine):
    name = "tesseract"

    def __init__(self):
        import pytesseract

        self.pytesseract = pytesseract

    def recognize(self, image_bgr) -> tuple[str, float]:
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        text = self.pytesseract.image_to_string(image_bgr, config=config)
        return text, 0.0


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
            "ocr_version": config.lpr_paddle_ocr_version,
            "text_detection_model_name": config.lpr_paddle_det_model_name,
            "text_recognition_model_name": config.lpr_paddle_rec_model_name,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
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


class SvtrOcrEngine(OcrEngine):
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
        tensor = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        return ctc_decode(outputs[0], self.charset, blank_index=self.blank_index)


def create_ocr_engine(engine_name: str | None) -> OcrEngine:
    name = (engine_name or "none").strip().lower()
    if name in ("svtr", "transformer"):
        return SvtrOcrEngine()
    if name in ("paddleocr", "paddle", "ppocr"):
        return PaddleOcrEngine()
    if name == "easyocr":
        return EasyOcrEngine()
    if name in ("tesseract", "pytesseract"):
        return TesseractOcrEngine()
    return OcrEngine()


class LicensePlateRecognizer:
    def __init__(
        self,
        plate_model_path: str,
        ocr_engine: str = "none",
        conf: float = 0.35,
        device: str | None = None,
    ):
        self.detector = YoloPlateDetector(plate_model_path, conf=conf, device=device)
        self.ocr = create_ocr_engine(ocr_engine)

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
            crop = crop_bbox(frame, plate.bbox)
            if crop is None:
                continue
            crop = rectify_plate_crop(crop)
            raw_text, ocr_conf = self.ocr.recognize(crop)
            text = normalize_taiwan_plate_text(raw_text)
            recognitions.append(PlateRecognition(
                text=text,
                raw_text=raw_text,
                confidence=ocr_conf,
                bbox=plate.bbox,
                detector_score=plate.score,
                valid_taiwan_format=is_valid_taiwan_plate(text),
                engine=self.engine_name,
            ))
        return recognitions


def draw_plate_recognitions(frame, recognitions):
    import cv2

    for item in recognitions or []:
        x1, y1, x2, y2 = item.bbox
        color = (255, 0, 255) if item.valid_taiwan_format else (255, 180, 0)
        label = item.text or "plate"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
