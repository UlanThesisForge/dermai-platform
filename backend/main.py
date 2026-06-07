"""
main.py — точка входа FastAPI приложения
Запуск: uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from config import settings
from database import engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models.db_models import Base
from routes import auth, diagnosis, users


# ── Создаём таблицы при старте (альтернатива Alembic для разработки) ──────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.GRADCAM_DIR, exist_ok=True)
    print("✓ База данных инициализирована")
    print("✓ Папки для загрузок созданы")
    yield


app = FastAPI(
    title="DermAI Diagnostic API",
    description="API для диагностики кожных заболеваний на основе нейросети MobileNetV2",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — разрешаем фронтенд ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Роуты ─────────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(diagnosis.router)
app.include_router(users.router)


# ── Статика (загруженные изображения и Grad-CAM через API роуты) ──────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
