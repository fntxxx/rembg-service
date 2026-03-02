FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Pillow / rembg 可能需要的系統相依（保守加）
# 另外加 curl + ca-certificates，讓 build 階段可下載模型
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# build 階段先把 u2netp 模型放進 image，避免 runtime 才下載造成啟動變慢 + 更容易 OOM
RUN mkdir -p /root/.u2net \
  && curl -L --fail -o /root/.u2net/u2netp.onnx \
     https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx

COPY main.py .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]