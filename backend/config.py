"""
config.py — настройки приложения из переменных окружения
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # База данных
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5430/dermai"

    # JWT
    SECRET_KEY: str = "замени-на-случайную-строку-минимум-32-символа"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Пути к файлам
    UPLOAD_DIR: str = "uploads"
    GRADCAM_DIR: str = "uploads/gradcam"
    MODEL_PATH: str = "../models/best_model.h5"

    # CORS
    FRONTEND_URL: str = "http://localhost:5173"

    class Config:
        env_file = ".env"


settings = Settings()
