from fastapi import FastAPI

from app.api.routes import router
from app.core.session import warmup_models

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
def startup_event():
    # 啟動時先完成模型預熱；如果模型壞掉或下載檔有問題，
    # 讓服務直接啟動失敗，比啟動成功後第一筆 request 才爆更容易發現。
    warmup_models()