# CCTV ROI AI Event Extractor

以 Polygon ROI 搭配 YOLO 物件偵測分析 CCTV 影片，擷取進入指定區域的人車事件，並輸出截圖、事件片段與處理報表。GUI 已改為 Qt（PySide6）。

## 功能重點

- 支援單一資料夾、多資料夾、單一影片、多影片批次處理
- 以 Polygon ROI 框選偵測區與車牌辨識觸碰區
- 偵測 `person`、`car`、`motorcycle`、`bus`、`truck`
- 只有當目標底部中心點進入偵測區並連續達到門檻幀數時，才判定事件開始
- 啟用車牌辨識時，車輛 bbox 碰到觸碰區後，才會使用偵測區內暫存的較佳候選影像做 LPR
- 可輸出事件截圖、事件片段、CSV 日誌與文字摘要報表
- 長時間停留補抓截圖可在 GUI 中獨立開關
- 可輸出蒐證 Excel，欄位為 `編號`、`進入日期`、`出去日期`、`車號`、`車輛截圖`
- 支援 CPU / CUDA 自動判斷
- Qt GUI，內建拖放來源清單
- UI 內建即時 log 視窗
- 可手動指定 CPU、單張 GPU 或多張 GPU 裝置字串
- 可選配車牌辨識：YOLO 車牌偵測、透視校正、OCR 引擎介面、臺灣牌照格式校正

## 專案結構

```text
.
|- cctv_roi_ai_event_extractor/
|  |- __init__.py
|  |- __main__.py
|  |- config.py
|  |- core.py
|  |- event_processing.py
|  |- evidence_report.py
|  |- gui.py
|  |- lpr.py
|  `- qt_app.py
|- .env.example
|- cctv_roi_ai_event_extractor_legacy_backend.py
|- cctv_roi_ai_event_extractor_qt.py
|- cctv_roi_ai_event_extractor_v4_new.py
|- README.md
`- requirements.txt
```

說明：

- `cctv_roi_ai_event_extractor/core.py`：核心 API 對外入口
- `cctv_roi_ai_event_extractor/config.py`：環境變數與執行設定
- `cctv_roi_ai_event_extractor/event_processing.py`：偵測、ROI、截圖、片段輸出等處理流程
- `cctv_roi_ai_event_extractor/evidence_report.py`：蒐證 Excel 輸出與截圖嵌入
- `cctv_roi_ai_event_extractor/gui.py`：Qt GUI
- `cctv_roi_ai_event_extractor/lpr.py`：車牌偵測、校正、OCR 介面、臺灣格式校正
- `cctv_roi_ai_event_extractor/qt_app.py`：舊 package 入口相容檔
- `cctv_roi_ai_event_extractor_qt.py`：相容啟動檔
- `cctv_roi_ai_event_extractor_v4_new.py`：相容 re-export 檔
- `cctv_roi_ai_event_extractor_legacy_backend.py`：舊版匯入/執行相容檔
- `.env.example`：環境變數範例

## 12-Factor 對應

- Codebase：主程式已集中在 `cctv_roi_ai_event_extractor/` package，根目錄只保留相容入口。
- Dependencies：Python 套件依賴仍由 `requirements.txt` 與 `requirements-gpu.txt` 明確宣告。
- Config：可變設定改由環境變數注入，不需修改原始碼。
- Backing services：模型下載來源以 URL 環境變數設定。
- Build / release / run：安裝依賴、設定環境變數、執行 GUI 三步分離。
- Processes：建議以 `python -m cctv_roi_ai_event_extractor` 啟動單一前景程序。
- Logs：GUI 內顯示即時 log，批次結果輸出到 `logs/detection_log.csv` 與 `reports/report_summary.txt`。

## 執行需求

- Python 3.10 以上
- Windows 桌面環境
- 需可執行 Qt 視窗程式
- 建議使用虛擬環境

## 安裝

### 1. 建立虛擬環境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. 安裝套件

```powershell
pip install -r requirements.txt
```

若要啟用 `PaddleOCR` 車牌 OCR，還需要先安裝 PaddlePaddle framework。若同一個虛擬環境也要安裝 PyTorch GPU，建議 PaddlePaddle 使用 CPU 版，讓 YOLO / PyTorch 使用 GPU，避免兩套框架在同一個 venv 內 pin 不同版本的 `nvidia-*` CUDA runtime 套件：

```bash
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt
```

若一定要讓 PaddleOCR 也使用 GPU，建議另建獨立虛擬環境，只安裝 PaddlePaddle GPU 與 PaddleOCR，不要和本專案的 PyTorch GPU 環境混用。PaddlePaddle GPU 版需依 driver 選擇官方 wheel index：

```bash
# Driver >= 450.80.02
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Driver >= 550.54.14
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

pip install -r requirements.txt
```

若接受不受官方相依宣告保證的組合，也可在同一個 venv 內先安裝本專案 PyTorch GPU 依賴，再用 `--no-deps` 安裝 PaddlePaddle GPU，讓它不要覆蓋 PyTorch 已安裝的 `nvidia-*` CUDA runtime 套件：

```bash
pip install -r requirements-gpu.txt

# Driver >= 550.54.14 / CUDA 12.6 wheel
python -m pip install paddlepaddle-gpu==3.2.0 --no-deps -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install opt_einsum

export CCTV_ROI_LPR_PADDLE_DEVICE="gpu:0"
```

這個方式只是繞過 pip 相依檢查；若 PaddlePaddle import 或推論時出現 `undefined symbol`、CUDA kernel error、segmentation fault，需改回 PaddlePaddle CPU 或分離到獨立 venv。

也可不啟用 PowerShell 環境，直接使用專案虛擬環境：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m cctv_roi_ai_event_extractor
```

### 3. 安裝 PyTorch

本專案會用到 `torch` 做 CUDA 自動判斷與 GPU 推論。

如果你要讓程式自動使用 NVIDIA GPU，直接安裝 GPU 依賴檔：

```powershell
pip install -r requirements-gpu.txt
```

目前 `requirements-gpu.txt` 使用 PyTorch 官方 CUDA 12.6 wheel。若要在同一個 venv 內安裝 `paddlepaddle-gpu==3.2.0`，需使用 `--no-deps`，否則兩者會 pin 不同版本的 `nvidia-cudnn-cu12`、`nvidia-cusparselt-cu12`、`nvidia-nccl-cu12`。

如果你只打算跑 CPU，則維持：

```powershell
pip install -r requirements.txt
```

程式本身會自動偵測：

- 偵測到可用 CUDA 時使用 GPU
- 否則自動退回 CPU

如果你的顯示卡或驅動不適合 CUDA 12.6，需改成對應版本的 PyTorch wheel 索引，且 PaddlePaddle GPU 也要使用相同 CUDA 系列。

## 模型檔放置

程式會依序尋找以下模型路徑：

```text
models\yolo26x.pt
yolo26x.pt
models\yolo26n.pt
yolo26n.pt
```

若找不到模型檔，程式會自動嘗試補齊：

1. 先檢查本地 `models/` 與程式同層
2. 再嘗試 Ultralytics 資產自動下載
3. 若你使用自家模型來源，可透過環境變數指定下載網址

可用環境變數：

```powershell
$env:CCTV_ROI_MODEL_PATH="C:\models\yolo26x.pt"
$env:YOLO26X_MODEL_URL="https://your-server/path/yolo26x.pt"
```

也支援：

```powershell
$env:CCTV_ROI_MODEL_URL="https://your-server/path/yolo26x.pt"
$env:YOLO_MODEL_URL="https://your-server/path/yolo26x.pt"
```

下載成功後，模型會落到預設模型路徑，例如 `models\yolo26x.pt`。

其他可用設定：

```powershell
$env:CCTV_ROI_APP_DIR="C:\cctv-roi-runtime"
$env:CCTV_ROI_CONFIG_PATH="C:\cctv-roi-runtime\roi_config_polygon.json"
$env:CCTV_ROI_LONG_STAY_SCREENSHOT_INTERVAL_SEC="5"
$env:YOLO_CONFIG_DIR="C:\cctv-roi-runtime\.runtime\ultralytics"
```

若未指定 `YOLO_CONFIG_DIR`，程式會自動使用專案內 `.runtime\ultralytics`，避免 Ultralytics 嘗試寫入受限的使用者設定目錄。

車牌辨識可用設定：

```powershell
$env:CCTV_ROI_LPR_ENABLED="true"
$env:CCTV_ROI_LPR_PLATE_MODEL_PATH="C:\models\license_plate_yolo.pt"
$env:CCTV_ROI_LPR_OCR_ENGINE="paddleocr"
$env:CCTV_ROI_LPR_CONFIDENCE="0.35"
$env:CCTV_ROI_LPR_PADDLE_OCR_VERSION="PP-OCRv5"
$env:CCTV_ROI_LPR_PADDLE_DET_MODEL_NAME="PP-OCRv5_mobile_det"
$env:CCTV_ROI_LPR_PADDLE_REC_MODEL_NAME="PP-OCRv5_mobile_rec"
$env:CCTV_ROI_LPR_PADDLE_DEVICE="cpu"
$env:CCTV_ROI_LPR_SVTR_MODEL_PATH="C:\models\taiwan_plate_svtr.onnx"
$env:CCTV_ROI_LPR_SVTR_CHARSET="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
$env:CCTV_ROI_LPR_SVTR_INPUT_SIZE="48x160"
$env:CCTV_ROI_LPR_SVTR_BLANK_INDEX="0"
$env:CCTV_ROI_LPR_SVTR_PROVIDERS="auto"
```

說明：

- `CCTV_ROI_LPR_PLATE_MODEL_PATH`：車牌偵測 YOLO 模型路徑。
- `CCTV_ROI_LPR_OCR_ENGINE`：GUI 會提供 `paddleocr` 與 `svtr` 二選一；環境變數可用來指定初始選項。建議先用 `paddleocr` 作為通用 OCR，若有自訓臺灣車牌模型再使用 `svtr`。
- `CCTV_ROI_LPR_PADDLE_OCR_VERSION`：PaddleOCR 版本，預設 `PP-OCRv5`。
- `CCTV_ROI_LPR_PADDLE_DET_MODEL_NAME` / `CCTV_ROI_LPR_PADDLE_REC_MODEL_NAME`：PaddleOCR 偵測與辨識模型名稱，預設使用 PP-OCRv5 mobile 模型。
- `CCTV_ROI_LPR_PADDLE_DEVICE`：PaddleOCR 推論裝置，例如 `cpu`、`gpu:0`；留空則使用 PaddleOCR 預設值。
- `CCTV_ROI_LPR_SVTR_MODEL_PATH`：SVTR / Transformer-style recognizer 的 ONNX 模型路徑。
- `CCTV_ROI_LPR_SVTR_CHARSET`：模型輸出的字元集，預設為臺灣車牌常用英數字元。
- `CCTV_ROI_LPR_SVTR_CHARSET_PATH`：若字元集較複雜，可改用一行一字元的文字檔。
- `CCTV_ROI_LPR_SVTR_INPUT_SIZE`：OCR 模型輸入尺寸，格式為 `高度x寬度`。
- `CCTV_ROI_LPR_SVTR_BLANK_INDEX`：CTC blank index，常見為 `0`。
- `CCTV_ROI_LPR_SVTR_PROVIDERS`：ONNX Runtime provider，`auto` 會優先使用 CUDA / DirectML / CPU 中可用者。
- PaddleOCR 是可選依賴，啟用前需依 PaddleOCR 官方安裝方式安裝 `paddleocr` 與 PaddlePaddle；專案使用 PaddleOCR v3 的 `PaddleOCR(...).predict(...)` API，並預設關閉文件方向分類、文件展平與文字行方向分類。
- CRNN / LPRNet / 其他 Transformer OCR 也可接到 `lpr.py` 的 `OcrEngine` 介面；輸出會先經臺灣牌照格式校正後寫入 CSV。

## 啟動方式

```powershell
python -m cctv_roi_ai_event_extractor
```

相容舊方式：

```powershell
python cctv_roi_ai_event_extractor_qt.py
```

## 使用流程

1. 啟動程式
2. 選擇影片來源
3. 選擇輸出資料夾
4. 以第一支可讀影片依序框選偵測區 Polygon ROI 與觸碰區 Polygon ROI
5. 輸入 AI 參數
6. 開始批次分析

## 支援影片格式

```text
.mp4 .avi .mov .m4v .mkv .ts .264 .265
```

## 輸出結果

選定輸出資料夾後，程式會建立以下結構：

```text
output_root/
|- screenshots/
|- motion_clips/
|- logs/
|  `- detection_log.csv
`- reports/
   |- evidence_report.xlsx
   `- report_summary.txt
```

若啟用車牌辨識，`detection_log.csv` 會增加：

```text
plate_text, plate_raw_text, plate_confidence, plate_bbox, plate_valid_taiwan_format, plate_ocr_engine
```

另外 ROI 設定會儲存在程式同層。新版設定包含 `detection_polygon` 與 `touch_polygon`，仍相容舊版 `polygon`：

```text
roi_config_polygon.json
```

若設定 `CCTV_ROI_CONFIG_PATH`，ROI 設定會改存到指定檔案。

截圖、事件片段、CSV 與報表都會輸出在 GUI 選擇的輸出資料夾底下。若來源影片不在目前來源根目錄下，輸出時會使用影片檔名，避免相對路徑逃出輸出資料夾。

## 主要依賴說明

- `ultralytics`：YOLO 模型推論
- `opencv-python-headless`：影片讀寫與影像處理；Qt GUI 由 `PySide6` 提供，避免 OpenCV 內建 Qt plugin 和 PySide6 衝突
- `numpy`：數值運算
- `PySide6`：Qt GUI
- `torch`：GPU 自動偵測與模型執行環境
- `openpyxl` / `Pillow`：蒐證 Excel 與截圖嵌入
- `onnxruntime`：SVTR / Transformer-style 車牌 OCR 推論
- `paddleocr` / `easyocr` / `pytesseract`：可選 OCR 後端，只有啟用對應車牌 OCR 時才需要安裝

## 注意事項

- 這是 Qt GUI 工具，不是命令列批次腳本
- Ubuntu 若出現 `Could not load the Qt platform plugin "xcb"`，請先安裝 Qt xcb 系統依賴，並移除非 headless OpenCV：

```bash
sudo apt update
sudo apt install libxcb-cursor0 libxcb-xinerama0 libxcb-keysyms1 libxcb-render-util0 libxcb-icccm4 libxcb-image0 libxkbcommon-x11-0 libgl1
pip uninstall -y opencv-python opencv-contrib-python
pip install -r requirements.txt
```

- 若要使用 GPU，需正確安裝 CUDA 版 PyTorch
- 若輸出資料夾位於掃描根目錄內，程式會自動排除該輸出資料夾，避免重複掃描
- 事件片段輸出仍以原始影片解析度與時間軸為準

## 後續可補強

- 補上版本發布流程
- 視部署需求補 `requirements-cpu.txt`
- 補實際畫面截圖與範例輸出
