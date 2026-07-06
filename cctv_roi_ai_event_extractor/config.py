import os
import sys
from dataclasses import dataclass


# 所有設定都先從環境變數讀取，讓打包後的 EXE 與開發環境可以共用同一套邏輯。
APP_VERSION = "4.4.0-roi-yolo26x-dragdrop-paste-paths"

ENV_APP_DIR = "CCTV_ROI_APP_DIR"
ENV_MODEL_PATH = "CCTV_ROI_MODEL_PATH"
ENV_ROI_CONFIG_PATH = "CCTV_ROI_CONFIG_PATH"
ENV_LONG_STAY_SCREENSHOT_INTERVAL_SEC = "CCTV_ROI_LONG_STAY_SCREENSHOT_INTERVAL_SEC"
ENV_LPR_ENABLED = "CCTV_ROI_LPR_ENABLED"
ENV_LPR_PLATE_MODEL_PATH = "CCTV_ROI_LPR_PLATE_MODEL_PATH"
ENV_LPR_OCR_ENGINE = "CCTV_ROI_LPR_OCR_ENGINE"
ENV_LPR_CONFIDENCE = "CCTV_ROI_LPR_CONFIDENCE"
ENV_LPR_SVTR_MODEL_PATH = "CCTV_ROI_LPR_SVTR_MODEL_PATH"
ENV_LPR_SVTR_CHARSET = "CCTV_ROI_LPR_SVTR_CHARSET"
ENV_LPR_SVTR_CHARSET_PATH = "CCTV_ROI_LPR_SVTR_CHARSET_PATH"
ENV_LPR_SVTR_INPUT_SIZE = "CCTV_ROI_LPR_SVTR_INPUT_SIZE"
ENV_LPR_SVTR_BLANK_INDEX = "CCTV_ROI_LPR_SVTR_BLANK_INDEX"
ENV_LPR_SVTR_PROVIDERS = "CCTV_ROI_LPR_SVTR_PROVIDERS"
ENV_LPR_YOLO_OCR_MODEL_PATH = "CCTV_ROI_LPR_YOLO_OCR_MODEL_PATH"
ENV_LPR_YOLO_OCR_CONFIDENCE = "CCTV_ROI_LPR_YOLO_OCR_CONFIDENCE"
ENV_LPR_PADDLE_DEVICE = "CCTV_ROI_LPR_PADDLE_DEVICE"
ENV_LPR_PADDLE_OCR_VERSION = "CCTV_ROI_LPR_PADDLE_OCR_VERSION"
ENV_LPR_PADDLE_DET_MODEL_NAME = "CCTV_ROI_LPR_PADDLE_DET_MODEL_NAME"
ENV_LPR_PADDLE_REC_MODEL_NAME = "CCTV_ROI_LPR_PADDLE_REC_MODEL_NAME"
ENV_LPR_TESSERACT_CMD = "CCTV_ROI_LPR_TESSERACT_CMD"
ENV_ULTRALYTICS_CONFIG_DIR = "YOLO_CONFIG_DIR"
_DOTENV_LOADED = False


def _project_root() -> str:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(package_dir)


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv_file(path: str | None = None) -> None:
    """Load app-local .env values without overriding real environment variables."""
    env_path = path or os.path.join(_project_root(), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = _unquote_env_value(value)


def ensure_dotenv_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv_file()
    _DOTENV_LOADED = True


def resolve_app_dir() -> str:
    ensure_dotenv_loaded()
    """Resolve the directory used for models, ROI config, and runtime files."""
    env_value = os.getenv(ENV_APP_DIR)
    if env_value:
        return os.path.abspath(env_value)
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return _project_root()


def ensure_runtime_environment(app_dir: str | None = None) -> None:
    """Keep third-party caches inside the app folder instead of the user profile."""
    base_dir = app_dir or resolve_app_dir()
    if not os.getenv(ENV_ULTRALYTICS_CONFIG_DIR):
        runtime_dir = os.path.join(base_dir, ".runtime", "ultralytics")
        os.makedirs(runtime_dir, exist_ok=True)
        os.environ[ENV_ULTRALYTICS_CONFIG_DIR] = runtime_dir


def resolve_long_stay_screenshot_interval(default: float = 5.0) -> float:
    env_value = os.getenv(ENV_LONG_STAY_SCREENSHOT_INTERVAL_SEC)
    if not env_value:
        return default
    try:
        value = float(env_value)
    except ValueError:
        return default
    return value if value > 0 else default


def resolve_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def resolve_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration collected from environment variables."""

    app_dir: str
    model_path: str | None
    roi_config_path: str | None
    long_stay_screenshot_interval_sec: float
    lpr_enabled: bool
    lpr_plate_model_path: str | None
    lpr_ocr_engine: str
    lpr_confidence: float
    lpr_svtr_model_path: str | None
    lpr_svtr_charset: str
    lpr_svtr_charset_path: str | None
    lpr_svtr_input_size: str
    lpr_svtr_blank_index: int
    lpr_svtr_providers: str
    lpr_yolo_ocr_model_path: str | None
    lpr_yolo_ocr_confidence: float
    lpr_paddle_device: str
    lpr_paddle_ocr_version: str
    lpr_paddle_det_model_name: str
    lpr_paddle_rec_model_name: str
    lpr_tesseract_cmd: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        ensure_dotenv_loaded()
        return cls(
            app_dir=resolve_app_dir(),
            model_path=os.getenv(ENV_MODEL_PATH),
            roi_config_path=os.getenv(ENV_ROI_CONFIG_PATH),
            long_stay_screenshot_interval_sec=resolve_long_stay_screenshot_interval(),
            lpr_enabled=resolve_bool_env(ENV_LPR_ENABLED),
            lpr_plate_model_path=os.getenv(ENV_LPR_PLATE_MODEL_PATH),
            lpr_ocr_engine=os.getenv(ENV_LPR_OCR_ENGINE, "none"),
            lpr_confidence=resolve_float_env(ENV_LPR_CONFIDENCE, 0.50),
            lpr_svtr_model_path=os.getenv(ENV_LPR_SVTR_MODEL_PATH),
            lpr_svtr_charset=os.getenv(ENV_LPR_SVTR_CHARSET, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            lpr_svtr_charset_path=os.getenv(ENV_LPR_SVTR_CHARSET_PATH),
            lpr_svtr_input_size=os.getenv(ENV_LPR_SVTR_INPUT_SIZE, "48x160"),
            lpr_svtr_blank_index=resolve_int_env(ENV_LPR_SVTR_BLANK_INDEX, 0),
            lpr_svtr_providers=os.getenv(ENV_LPR_SVTR_PROVIDERS, "auto"),
            lpr_yolo_ocr_model_path=os.getenv(ENV_LPR_YOLO_OCR_MODEL_PATH),
            lpr_yolo_ocr_confidence=resolve_float_env(ENV_LPR_YOLO_OCR_CONFIDENCE, 0.35),
            lpr_paddle_device=os.getenv(ENV_LPR_PADDLE_DEVICE, ""),
            lpr_paddle_ocr_version=os.getenv(ENV_LPR_PADDLE_OCR_VERSION, "PP-OCRv5"),
            lpr_paddle_det_model_name=os.getenv(ENV_LPR_PADDLE_DET_MODEL_NAME, "PP-OCRv5_mobile_det"),
            lpr_paddle_rec_model_name=os.getenv(ENV_LPR_PADDLE_REC_MODEL_NAME, "PP-OCRv5_mobile_rec"),
            lpr_tesseract_cmd=os.getenv(ENV_LPR_TESSERACT_CMD, ""),
        )


def load_config() -> AppConfig:
    ensure_dotenv_loaded()
    return AppConfig.from_env()
