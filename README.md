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

提供單張圖片去背 API，成功與錯誤皆採用統一 JSON contract，並將固定處理策略收斂到伺服器端。

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

### 精簡 curl 範例

```bash
curl -X POST \
  'http://localhost:7860/remove-bg' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@dog_pet.jpg;type=image/jpeg'
```

---

## ✅ 成功回應

- Status: `200 OK`
- Content-Type: `application/json`

```json
{
  "ok": true,
  "data": {
    "image": {
      "filename": "removed_bg.png",
      "mime_type": "image/png",
      "base64": "iVBORw0KGgoAAAANSUhEUgAA...",
      "width": 768,
      "height": 1024
    },
    "model": "isnet-general-use",
    "fallback_used": false,
    "edge_quality_low_candidate": false,
    "metrics": {
      "edge_band_ratio": 0.017223,
      "edge_band_mid_ratio": 0.009121,
      "edge_band_low_ratio": 0.004388
    },
    "processing": {
      "max_side": 512,
      "quality": "fast",
      "reject_low_confidence": true,
      "reject_edge_quality": true
    }
  }
}
```

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

### 為什麼改成統一 JSON contract

原因：

- 呼叫端不需要再同時處理 binary 與多套 JSON 錯誤格式
- Swagger UI / OpenAPI 可以完整描述 success 與 error schema
- 成功與錯誤可用 `ok` 明確區分
- `error.code` 與 `error.details` 可同時支援人類閱讀與機器判斷

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
- Uvicorn
- Docker
- Hugging Face Spaces

---

## 📌 注意事項

- 僅支援單張圖片
- 圖片格式需為 Pillow 可解析格式
- 成功回傳中的圖片內容以 base64 放在 `data.image.base64`
- 若部署於 CPU-only 環境，目前固定使用 `fast` 品質策略

---

## 🧪 測試建議

```bash
pytest -q
```

若要手動驗證 OpenAPI：

```bash
python - <<'PY'
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
schema = client.get('/openapi.json').json()
remove_bg = schema['paths']['/remove-bg']['post']
print(remove_bg.get('parameters', []))
PY
```

預期 `/remove-bg` 的 `parameters` 為空陣列或不存在，只保留 multipart `file` 上傳欄位。
