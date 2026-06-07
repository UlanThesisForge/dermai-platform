"""
routes/auth.py — регистрация, вход, выход, обновление токена
"""

from datetime import datetime, timedelta

from auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from config import settings
from database import get_db
from fastapi import APIRouter, Depends, HTTPException
from models.db_models import Doctor, UserSession
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/auth", tags=["Авторизация"])


# ── Схемы ─────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str = "doctor"


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    doctor_id: str
    full_name: str
    email: str
    role: str


# ── Вход ──────────────────────────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Doctor).where(Doctor.email == data.email))
    doctor = result.scalar_one_or_none()

    if not doctor or not verify_password(data.password, doctor.password_hash):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    if not doctor.is_active:
        raise HTTPException(status_code=403, detail="Аккаунт деактивирован")

    # Обновляем время последнего входа
    doctor.last_login = datetime.utcnow()

    access = create_access_token(str(doctor.doctor_id), doctor.role)
    refresh = create_refresh_token(str(doctor.doctor_id))

    # Сохраняем refresh токен в БД
    session = UserSession(
        doctor_id=doctor.doctor_id,
        refresh_token=refresh,
        expires_at=datetime.utcnow()
        + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(session)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        doctor_id=str(doctor.doctor_id),
        full_name=doctor.full_name,
        email=doctor.email,
        role=doctor.role,
    )


# ── Регистрация ───────────────────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Проверяем уникальность email
    existing = await db.execute(select(Doctor).where(Doctor.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email уже используется")

    doctor = Doctor(
        full_name=data.full_name,
        email=data.email,
        password_hash=hash_password(data.password),
        role=data.role if data.role in ("doctor", "viewer") else "doctor",
    )
    db.add(doctor)
    await db.flush()  # получаем doctor_id

    access = create_access_token(str(doctor.doctor_id), doctor.role)
    refresh = create_refresh_token(str(doctor.doctor_id))

    session = UserSession(
        doctor_id=doctor.doctor_id,
        refresh_token=refresh,
        expires_at=datetime.utcnow()
        + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(session)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        doctor_id=str(doctor.doctor_id),
        full_name=doctor.full_name,
        email=doctor.email,
        role=doctor.role,
    )


# ── Обновление токена ─────────────────────────────────────────────────────────
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(data.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Неверный тип токена")

    # Проверяем есть ли токен в БД
    result = await db.execute(
        select(UserSession).where(UserSession.refresh_token == data.refresh_token)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=401, detail="Токен не найден или отозван")

    # Ротация: удаляем старый, создаём новый
    await db.delete(session)

    doctor_result = await db.execute(
        select(Doctor).where(Doctor.doctor_id == payload["sub"])
    )
    doctor = doctor_result.scalar_one_or_none()
    if not doctor or not doctor.is_active:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    new_access = create_access_token(str(doctor.doctor_id), doctor.role)
    new_refresh = create_refresh_token(str(doctor.doctor_id))

    new_session = UserSession(
        doctor_id=doctor.doctor_id,
        refresh_token=new_refresh,
        expires_at=datetime.utcnow()
        + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(new_session)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        doctor_id=str(doctor.doctor_id),
        full_name=doctor.full_name,
        email=doctor.email,
        role=doctor.role,
    )


# ── Выход ─────────────────────────────────────────────────────────────────────
@router.post("/logout")
async def logout(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(UserSession).where(UserSession.refresh_token == data.refresh_token)
    )
    session = result.scalar_one_or_none()
    if session:
        await db.delete(session)
    return {"message": "Выход выполнен"}


# ── Профиль ───────────────────────────────────────────────────────────────────
@router.get("/me")
async def me(current_user: Doctor = Depends(get_current_user)):
    return {
        "doctor_id": str(current_user.doctor_id),
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role,
        "created_at": current_user.created_at,
    }
