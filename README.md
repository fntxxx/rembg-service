---
title: rembg-service
emoji: 🧼
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# rembg-service

去背服務（Background Removal Service），基於 FastAPI + rembg，部署於 Hugging Face Spaces。

提供單張圖片去背 API。成功時直接回傳透明背景 PNG，失敗時維持統一 JSON error contract，並將固定處理策略收斂到伺服器端。

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

### 2. 設定共用 API Token

```bash
export INTERNAL_API_TOKEN="replace-with-shared-token"
```

### 3. 啟動服務

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### 4. 開啟

- Swagger UI：http://localhost:7860/
- Health：http://localhost:7860/healthz

---

## 🐳 Docker 啟動

```bash
docker build -t rembg-service .
docker run --rm -p 7860:7860 -e INTERNAL_API_TOKEN=replace-with-shared-token rembg-service
```

---


## 🗂️ 專案結構

```text
app/
  api/        # FastAPI 路由
  core/       # 設定、驗證、例外與 session 管理
  domain/     # 去背後處理與評估邏輯
  schemas/    # API 文件與 response schema
  services/   # 去背服務主流程
scripts/      # 本機回歸與分析腳本（不納入 pytest）
tests/        # 正式 API contract 測試
```

## 📌 API 一覽

### 1. Service Info

```http
GET /service-info
```

回傳服務名稱、預設模型、主要 API 路徑，以及目前伺服器端固定處理策略。

### 成功回應

```json
{
  "ok": true,
  "data": {
    "service": "rembg-service",
    "model": "isnet-general-use",
    "endpoints": {
      "service_info": "/service-info",
      "health": "/health",
      "healthz": "/healthz",
      "warmup": "/warmup",
      "remove_bg": "POST /remove-bg"
    },
    "processing_defaults": {
      "max_side": 512,
      "quality": "fast",
      "reject_low_confidence": true,
      "reject_edge_quality": true
    }
  }
}
```

---

### 2. Health

```http
GET /health
GET /healthz
```

- `/health`：完整狀態
- `/healthz`：輕量健康檢查（給 container 用）

### 成功回應

```json
{
  "ok": true,
  "data": {
    "service": "rembg-service",
    "model": "isnet-general-use",
    "model_warmed": true
  }
}
```

---

### 3. Warmup

```http
GET /warmup
```

預先初始化模型 session，避免第一筆請求延遲。

### 成功回應

```json
{
  "ok": true,
  "data": {
    "service": "rembg-service",
    "model": "isnet-general-use",
    "model_warmed": true,
    "warmed_by": "warmup_endpoint"
  }
}
```

---

### 4. Remove Background（核心 API）

```http
POST /remove-bg
```

### Request

- Content-Type：`multipart/form-data`
- 欄位：

| 名稱 | 型別 | 說明 |
|------|------|------|
| file | file | 要去背的圖片 |

### 圖檔白名單規則

#### 允許的副檔名（必填）

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.avif`
- `.heic`
- `.heif`

#### 允許的 Content-Type（request 有帶時必須符合）

- `image/jpeg`
- `image/png`
- `image/webp`
- `image/avif`
- `image/heic`
- `image/heif`

#### 驗證規則

- 副檔名必須在白名單內，否則直接拒絕。
- 若 request 有帶 `content_type`，則 `content_type` 也必須在白名單內。
- `content_type` 若缺失，前置驗證會退回只檢查副檔名。
- `svg` / `image/svg+xml` 明確禁止，不可進入去背流程。
- 服務不以 Pillow 是否剛好能開啟某個格式來決定是否放行。

### 固定處理策略

以下策略已由伺服器端內建，不再作為公開 query 參數：

| 策略 | 固定值 |
|------|--------|
| max_side | 512 |
| quality | fast |
| model | isnet-general-use |
| reject_low_confidence | true |
| reject_edge_quality | true |

呼叫端不需要再傳這些 query string，Swagger UI 也不會再顯示它們。

### curl 範例

將結果直接寫成 PNG 檔案：

```bash
curl -X POST \
  'http://localhost:7860/remove-bg' \
  -H 'accept: image/png' \
  -H 'Authorization: Bearer $INTERNAL_API_TOKEN' \
  -F 'file=@dog_pet.jpg;type=image/jpeg' \
  --output removed_bg.png
```

若只想查看回應 header：

```bash
curl -X POST \
  'http://localhost:7860/remove-bg' \
  -H 'accept: image/png' \
  -H 'Authorization: Bearer $INTERNAL_API_TOKEN' \
  -F 'file=@dog_pet.jpg;type=image/jpeg' \
  -D - \
  --output removed_bg.png
```

---

## ✅ 成功回應

- Status: `200 OK`
- Content-Type: `image/png`
- Body: 透明背景 PNG binary

### Response Headers

| Header | 說明 |
|--------|------|
| X-RemoveBg-Model | 實際使用模型 |
| X-RemoveBg-Fallback-Used | 是否使用 fallback |
| X-Edge-Quality-Candidate | 是否被標記為邊界品質偏低候選 |
| X-Edge-Band-Ratio | 邊界帶整體比例 |
| X-Edge-Band-Mid-Ratio | 邊界帶中間透明度比例 |
| X-Edge-Band-Low-Ratio | 邊界帶低透明度比例 |

> 成功回應不再使用 JSON 或 base64 包裝圖片內容。

---

## ❌ 錯誤回應 contract

所有錯誤統一為：

```json
{
  "ok": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "可讀訊息",
    "details": {}
  }
}
```

### 422 業務拒絕（低信心 / 邊界品質不足）

```json
{
  "ok": false,
  "error": {
    "code": "LOW_CONFIDENCE_MASK",
    "message": "背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。",
    "details": {
      "reason": "complex_background_low_confidence",
      "metrics": {
        "foreground_ratio": 0.286412,
        "alpha_mean": 62.1843,
        "mid_alpha_ratio": 0.038211,
        "high_alpha_ratio": 0.241933,
        "bbox_width_ratio": 0.624512,
        "bbox_height_ratio": 0.918274,
        "edge_band_ratio": 0.017223,
        "edge_band_mid_ratio": 0.009121,
        "edge_band_low_ratio": 0.004388,
        "edge_quality_low_candidate": true
      }
    }
  }
}
```

### 422 請求驗證錯誤

```json
{
  "ok": false,
  "error": {
    "code": "REQUEST_VALIDATION_ERROR",
    "message": "請求參數驗證失敗。",
    "details": {
      "errors": [
        {
          "loc": ["body", "file"],
          "msg": "Field required",
          "type": "missing"
        }
      ]
    }
  }
}
```

### 400 一般輸入錯誤

```json
{
  "ok": false,
  "error": {
    "code": "BAD_REQUEST",
    "message": "Invalid image",
    "details": null
  }
}
```

### 502 引擎或後處理錯誤

```json
{
  "ok": false,
  "error": {
    "code": "REMBG_EXECUTION_FAILED",
    "message": "去背引擎執行失敗。",
    "details": {
      "cause": "session initialization failed",
      "model": "isnet-general-use",
      "quality": "fast"
    }
  }
}
```

---

## 🧠 設計說明

### 為什麼 Swagger UI 掛在 `/`

- Hugging Face Spaces 預設首頁為 `/`
- 直接顯示 Swagger UI 可讓 API 可用性最大化
- 不需要再額外記 `/docs`

### 為什麼成功回應改回 PNG binary

原因：

- 去背 API 的核心產出本來就是圖片檔，不需要再多一層 JSON/base64 包裝
- 避免 base64 放大 payload，降低傳輸與解析成本
- 呼叫端可直接將 response body 當成圖片檔案儲存或轉送
- 錯誤情境仍保留既有 JSON envelope，方便機器判斷與除錯

### 為什麼固定參數要內建

原因：

- `max_side=512`
- `quality=fast`
- `model=isnet-general-use`
- `reject_low_confidence=true`
- `reject_edge_quality=true`

這些目前屬於服務固定運行策略，不是呼叫端自由調校的公開 contract。
將它們內建後，可以避免 Swagger UI 暗示使用者必須帶一長串 query string，也能減少外部整合的維護成本。

---

## 📦 技術棧

- FastAPI
- rembg
- Pillow
- pillow-heif
- Uvicorn
- Docker
- Hugging Face Spaces

---

## 📌 注意事項

- 僅支援單張圖片
- 僅接受白名單中的點陣圖格式：`.jpg`、`.jpeg`、`.png`、`.webp`、`.avif`、`.heic`、`.heif`
- request 若有帶 `content_type`，也必須落在白名單：`image/jpeg`、`image/png`、`image/webp`、`image/avif`、`image/heic`、`image/heif`
- `svg` / `image/svg+xml` 明確禁止
- 成功回應直接是 PNG binary，不是 JSON
- 若部署於 CPU-only 環境，目前固定使用 `fast` 品質策略
- HEIC / HEIF / AVIF 透過 `pillow-heif` 與 `register_heif_opener()` 接入 Pillow 解碼流程

---

## 🧪 測試建議

```bash
pytest -q
```

本機分析腳本已收斂到 `scripts/`：

- `python scripts/benchmark_api.py`
- `python scripts/color_compare_local.py`
- `python scripts/color_diff_report.py`

建議至少確認以下情境：

- `.jpg` / `.jpeg` / `.png` / `.webp` / `.avif` / `.heic` / `.heif` 可通過前置驗證
- `.svg` 與任何不在白名單中的副檔名會被 `400` 拒絕
- `content_type` 若有帶值但不在白名單中，會被 `400` 拒絕
- `content_type` 若缺失，會退回只檢查副檔名

若要手動驗證 OpenAPI：

```bash
python - <<'PY'
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
schema = client.get('/openapi.json').json()
remove_bg = schema['paths']['/remove-bg']['post']
print(remove_bg.get('parameters', []))
print(remove_bg['responses']['200']['content'].keys())
PY
```

預期 `/remove-bg` 的 `parameters` 為空陣列或不存在，成功回應內容型別應為 `image/png`。
