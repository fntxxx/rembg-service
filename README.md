# rembg-service

去背服務（Background Removal Service），基於 FastAPI + rembg，部署於 Hugging Face Spaces。

提供單張圖片去背 API，支援品質模式、尺寸控制，以及低信心與邊界品質拒絕機制。

---

## 🔗 文件與入口

服務啟動後：

- Swagger UI（API 文件）：`/`
- OpenAPI schema：`/openapi.json`

👉 在 Hugging Face Spaces 首頁即為 Swagger UI

---

## 🚀 本機啟動

### 1. 建立環境並安裝套件

```bash
cd /d/Projects/node/rembg-service
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. 啟動服務

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### 3. 開啟

- Swagger UI：http://localhost:7860/
- Health：http://localhost:7860/healthz

---

## 🐳 Docker 啟動

```bash
docker build -t rembg-service .
docker run --rm -p 7860:7860 rembg-service
```

---

## 📌 API 一覽

### 1. Service Info

```
GET /service-info
```

回傳服務名稱、預設模型與 API 路徑。

---

### 2. Health

```
GET /health
GET /healthz
```

- `/health`：完整狀態
- `/healthz`：輕量健康檢查（給 container 用）

---

### 3. Warmup

```
GET /warmup
```

預先初始化模型 session，避免第一筆請求延遲。

---

### 4. Remove Background（核心 API）

```
POST /remove-bg
```

### Request

- Content-Type：`multipart/form-data`
- 欄位：

| 名稱 | 型別 | 說明 |
|------|------|------|
| file | file | 要去背的圖片 |

### Query 參數

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| max_side | int | 512 | 圖片最長邊縮放尺寸 |
| quality | string | fast | `fast` / `high` |
| model | string | None | 指定 rembg 模型 |
| reject_low_confidence | bool | true | 是否啟用低信心拒絕 |
| reject_edge_quality | bool | true | 是否啟用邊界品質拒絕 |

---

## ✅ 成功回應

- Status: `200 OK`
- Content-Type: `image/png`

回傳透明背景 PNG

### Response Headers

| Header | 說明 |
|--------|------|
| X-RemoveBg-Model | 使用模型 |
| X-RemoveBg-Fallback-Used | 是否 fallback |
| X-Edge-Quality-Candidate | 邊界品質標記 |
| X-Edge-Band-Ratio | 邊界比例 |
| X-Edge-Band-Mid-Ratio | 中透明比例 |
| X-Edge-Band-Low-Ratio | 低透明比例 |

---

## ❌ 拒絕回應（422）

當圖片品質不足時：

```json
{
  "ok": false,
  "code": "LOW_CONFIDENCE_MASK",
  "message": "背景過於複雜或主體邊界不清楚",
  "reason": "low_confidence_mask",
  "metrics": {
    "foreground_ratio": 0.28,
    "alpha_mean": 62.18
  }
}
```

---

## ⚠️ 錯誤回應

### 400 Bad Request

```json
{
  "detail": "Invalid image"
}
```

### 502 Internal Error

```json
{
  "error": "rembg failed",
  "detail": "session initialization failed",
  "model": "isnet-general-use",
  "quality": "fast"
}
```

---

## 🧠 設計說明

### 為什麼 Swagger UI 掛在 `/`

- Hugging Face Spaces 預設首頁為 `/`
- 直接顯示 Swagger UI 可讓 API 可用性最大化
- 不需要再額外記 `/docs`

---

### 為什麼有 reject 機制

避免：

- 去背失敗但仍回傳 PNG
- 前端顯示品質差圖片

改為：

- 直接回傳 JSON
- 由前端提示使用者重新上傳

---

### 為什麼使用 headers 傳 metrics

原因：

- 成功回傳是 binary（PNG）
- 無法同時帶 JSON body
- 所以用 HTTP headers 傳輔助資訊

---

## 📦 技術棧

- FastAPI
- rembg
- Pillow
- Uvicorn
- Docker
- Hugging Face Spaces

---

## 📌 注意事項

- 僅支援單張圖片
- 圖片格式需為 Pillow 可解析格式
- 高品質模式（`high`）會增加處理時間
- 若部署於 CPU-only 環境，建議使用 `fast`

---

## 🧪 測試建議

建議使用：

- 純色背景
- 主體清晰
- 邊界明確的衣物圖片

避免：

- 強烈陰影
- 多重主體
- 與背景顏色接近的衣物