"""
models/db_models.py — SQLAlchemy ORM модели
Используем declarative_base() — совместимо с SQLAlchemy 1.4 и 2.0
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ModelVersion(Base):
    __tablename__ = "model_version"

    model_version_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    architecture = Column(String(100), nullable=False)
    version_number = Column(String(20), nullable=False)
    accuracy = Column(Numeric(5, 4))
    summary = Column(Text)
    deployed_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    reports = relationship("DiagnosisReport", back_populates="model_version")


class Doctor(Base):
    __tablename__ = "doctor"

    doctor_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(200), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="doctor")
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patients = relationship("Patient", back_populates="doctor")
    images = relationship("SkinLesionImage", back_populates="doctor")
    sessions = relationship("UserSession", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patient"

    patient_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(200), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String(10), nullable=True)
    medical_record_id = Column(String(50), unique=True, nullable=True)
    doctor_id = Column(
        UUID(as_uuid=True), ForeignKey("doctor.doctor_id"), nullable=True
    )
    created_at = Column(DateTime, default=datetime.utcnow)

    doctor = relationship("Doctor", back_populates="patients")
    images = relationship("SkinLesionImage", back_populates="patient")


class SkinLesionImage(Base):
    __tablename__ = "skin_lesion_image"

    image_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(
        UUID(as_uuid=True), ForeignKey("patient.patient_id"), nullable=True
    )
    doctor_id = Column(
        UUID(as_uuid=True), ForeignKey("doctor.doctor_id"), nullable=True
    )
    file_path = Column(String(500), nullable=False)
    file_format = Column(String(10), default="jpg")
    file_size_kb = Column(Integer, nullable=True)
    original_filename = Column(String(255), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    preprocessing_status = Column(String(20), default="pending")

    doctor = relationship("Doctor", back_populates="images")
    patient = relationship("Patient", back_populates="images")
    report = relationship("DiagnosisReport", back_populates="image", uselist=False)


class DiagnosisReport(Base):
    __tablename__ = "diagnosis_report"

    report_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("skin_lesion_image.image_id"))
    model_version_id = Column(
        UUID(as_uuid=True), ForeignKey("model_version.model_version_id"), nullable=True
    )
    prediction_class = Column(String(10), nullable=False)
    confidence_score = Column(Numeric(5, 4), nullable=False)
    probability_distribution = Column(JSON, nullable=False)
    gradcam_map_path = Column(String(500), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    image = relationship("SkinLesionImage", back_populates="report")
    model_version = relationship("ModelVersion", back_populates="reports")


class UserSession(Base):
    __tablename__ = "user_session"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctor.doctor_id"))
    refresh_token = Column(String(500), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    doctor = relationship("Doctor", back_populates="sessions")
