"""
routes/users.py — управление пользователями (только для admin)
"""

from datetime import datetime
from typing import Optional

from auth import get_current_user, hash_password, require_admin
from database import get_db
from fastapi import APIRouter, Depends, HTTPException
from models.db_models import Doctor
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/users", tags=["Пользователи"])


# ── Схемы ─────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str = "doctor"


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    doctor_id: str
    full_name: str
    email: str
    role: str
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Список пользователей ──────────────────────────────────────────────────────
@router.get("/", response_model=list[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_db),
    admin: Doctor = Depends(require_admin),
):
    result = await db.execute(select(Doctor).order_by(Doctor.created_at.desc()))
    users = result.scalars().all()
    return [
        UserResponse(
            doctor_id=str(u.doctor_id),
            full_name=u.full_name,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            last_login=u.last_login,
            created_at=u.created_at,
        )
        for u in users
    ]


# ── Создать пользователя ──────────────────────────────────────────────────────
@router.post("/", response_model=UserResponse)
async def create_user(
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
    admin: Doctor = Depends(require_admin),
):
    existing = await db.execute(select(Doctor).where(Doctor.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email уже занят")

    if data.role not in ("admin", "doctor", "viewer"):
        raise HTTPException(400, "Недопустимая роль")

    user = Doctor(
        full_name=data.full_name,
        email=data.email,
        password_hash=hash_password(data.password),
        role=data.role,
    )
    db.add(user)
    await db.flush()

    return UserResponse(
        doctor_id=str(user.doctor_id),
        full_name=user.full_name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        last_login=user.last_login,
        created_at=user.created_at,
    )


# ── Обновить пользователя ─────────────────────────────────────────────────────
@router.patch("/{doctor_id}", response_model=UserResponse)
async def update_user(
    doctor_id: str,
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    admin: Doctor = Depends(require_admin),
):
    result = await db.execute(select(Doctor).where(Doctor.doctor_id == doctor_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "Пользователь не найден")

    if data.full_name is not None:
        user.full_name = data.full_name
    if data.email is not None:
        user.email = data.email
    if data.role is not None:
        user.role = data.role
    if data.is_active is not None:
        user.is_active = data.is_active
    if data.password is not None:
        user.password_hash = hash_password(data.password)

    return UserResponse(
        doctor_id=str(user.doctor_id),
        full_name=user.full_name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        last_login=user.last_login,
        created_at=user.created_at,
    )


# ── Удалить пользователя ──────────────────────────────────────────────────────
@router.delete("/{doctor_id}")
async def delete_user(
    doctor_id: str,
    db: AsyncSession = Depends(get_db),
    admin: Doctor = Depends(require_admin),
):
    if doctor_id == str(admin.doctor_id):
        raise HTTPException(400, "Нельзя удалить самого себя")

    result = await db.execute(select(Doctor).where(Doctor.doctor_id == doctor_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "Пользователь не найден")

    await db.delete(user)
    return {"message": f"Пользователь {user.full_name} удалён"}


# ── Своя страница ─────────────────────────────────────────────────────────────
@router.get("/me/profile")
async def my_profile(user: Doctor = Depends(get_current_user)):
    return UserResponse(
        doctor_id=str(user.doctor_id),
        full_name=user.full_name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        last_login=user.last_login,
        created_at=user.created_at,
    )
