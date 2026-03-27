# CCTV ROI AI Event Extractor

以 Polygon ROI 搭配 YOLO 物件偵測分析 CCTV 影片，擷取進入指定區域的人車事件，並輸出截圖、事件片段與處理報表。GUI 已改為 Qt（PySide6）。

## 功能重點

- 支援單一資料夾、多資料夾、單一影片、多影片批次處理
- 以 Polygon ROI 框選監控區域
- 偵測 `person`、`car`、`motorcycle`、`bus`、`truck`
- 只有當目標底部中心點進入 ROI 並連續達到門檻幀數時，才判定事件開始
- 可輸出事件截圖、事件片段、CSV 日誌與文字摘要報表
- 支援 CPU / CUDA 自動判斷
- Qt GUI，內建拖放來源清單

## 專案結構

```text
.
|- cctv_roi_ai_event_extractor/
|  |- __init__.py
|  |- __main__.py
|  |- core.py
|  `- qt_app.py
|- cctv_roi_ai_event_extractor_legacy_backend.py
|- cctv_roi_ai_event_extractor_qt.py
|- cctv_roi_ai_event_extractor_v4_new.py
|- README.md
`- requirements.txt
```

說明：

- `cctv_roi_ai_event_extractor/core.py`：核心 API 對外入口
- `cctv_roi_ai_event_extractor/qt_app.py`：Qt GUI
- `cctv_roi_ai_event_extractor_qt.py`：相容啟動檔
- `cctv_roi_ai_event_extractor_v4_new.py`：相容 re-export 檔
- `cctv_roi_ai_event_extractor_legacy_backend.py`：舊版 backend 實作，供過渡期沿用

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

### 3. 安裝 PyTorch

本專案會用到 `torch` 做 CUDA 自動判斷與 GPU 推論。

如果你要讓程式自動使用 NVIDIA GPU，直接安裝 GPU 依賴檔：

```powershell
pip install -r requirements-gpu.txt
```

目前 `requirements-gpu.txt` 使用 PyTorch 官方 CUDA 12.8 wheel。

如果你只打算跑 CPU，則維持：

```powershell
pip install -r requirements.txt
```

程式本身會自動偵測：

- 偵測到可用 CUDA 時使用 GPU
- 否則自動退回 CPU

如果你的顯示卡或驅動不適合 CUDA 12.8，需改成對應版本的 PyTorch wheel 索引。

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
$env:YOLO26X_MODEL_URL="https://your-server/path/yolo26x.pt"
```

也支援：

```powershell
$env:CCTV_ROI_MODEL_URL="https://your-server/path/yolo26x.pt"
$env:YOLO_MODEL_URL="https://your-server/path/yolo26x.pt"
```

下載成功後，模型會落到預設模型路徑，例如 `models\yolo26x.pt`。

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
4. 以第一支可讀影片框選 Polygon ROI
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
   `- report_summary.txt
```

另外 ROI 設定會儲存在程式同層：

```text
roi_config_polygon.json
```

## 主要依賴說明

- `ultralytics`：YOLO 模型推論
- `opencv-python`：影片讀寫與影像處理
- `numpy`：數值運算
- `PySide6`：Qt GUI
- `torch`：GPU 自動偵測與模型執行環境

## 注意事項

- 這是 Qt GUI 工具，不是命令列批次腳本
- 若要使用 GPU，需正確安裝 CUDA 版 PyTorch
- 若輸出資料夾位於掃描根目錄內，程式會自動排除該輸出資料夾，避免重複掃描
- 事件片段輸出仍以原始影片解析度與時間軸為準

## 後續可補強

- 將主程式檔名縮短
- 補上版本發布流程
- 增加 `requirements-gpu.txt` / `requirements-cpu.txt`
- 補實際畫面截圖與範例輸出
