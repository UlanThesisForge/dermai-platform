"""
auth.py — JWT токены, хеширование паролей, текущий пользователь
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional

# Workaround: passlib 1.7.4 не знает об атрибуте __about__ в bcrypt 4.x
import bcrypt as _bcrypt

if not hasattr(_bcrypt, "__about__"):

    class _FakeAbout:
        __version__ = _bcrypt.__version__

    _bcrypt.__about__ = _FakeAbout()

from config import settings
from database import get_db
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from models.db_models import Doctor
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


# ── Пароли ────────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT токены ────────────────────────────────────────────────────────────────
def create_access_token(doctor_id: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": doctor_id, "role": role, "exp": expire, "type": "access"},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def create_refresh_token(doctor_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": doctor_id, "exp": expire, "type": "refresh", "jti": str(uuid.uuid4())},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный или истёкший токен",
        )


# ── Dependency: текущий пользователь ──────────────────────────────────────────
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Doctor:
    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Неверный тип токена")

    doctor_id = payload.get("sub")
    result = await db.execute(select(Doctor).where(Doctor.doctor_id == doctor_id))
    doctor = result.scalar_one_or_none()

    if not doctor or not doctor.is_active:
        raise HTTPException(
            status_code=401, detail="Пользователь не найден или деактивирован"
        )

    return doctor


def require_admin(current_user: Doctor = Depends(get_current_user)) -> Doctor:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Требуются права администратора")
    return current_user
