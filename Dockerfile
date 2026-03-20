FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 固定 rembg 模型目錄，避免路徑依賴使用者 home 行為
ENV U2NET_HOME=/root/.u2net

# 固定服務預設模型
ENV REMBG_MODEL=isnet-general-use

# 降低 ONNX / BLAS 執行緒數，避免 CPU container 記憶體尖峰
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 預先下載 isnet-general-use，避免啟動時才抓模型
RUN mkdir -p /root/.u2net \
  && curl -L --fail -o /root/.u2net/isnet-general-use.onnx \
     https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx \
  && test -s /root/.u2net/isnet-general-use.onnx

COPY app ./app

EXPOSE 7860

# 健康檢查只打輕量 endpoint，不碰模型推論
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-7860}/healthz || exit 1

# 明確單 worker，避免多 worker 重複載入大模型吃掉記憶體
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 30"]