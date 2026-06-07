"""
routes/diagnosis.py — загрузка фото и диагностика
"""
import os
import time
import uuid
import numpy as np
from io import BytesIO
from pathlib import Path

import cv2
import tensorflow as tf
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from PIL import Image

from database import get_db
from models.db_models import Doctor, SkinLesionImage, DiagnosisReport, ModelVersion
from auth import get_current_user
from config import settings

router = APIRouter(prefix="/diagnosis", tags=["Диагностика"])

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_INFO = {
    'akiec': {'name': 'Актинический кератоз',          'risk': 'medium'},
    'bcc':   {'name': 'Базальноклеточная карцинома',   'risk': 'high'},
    'bkl':   {'name': 'Доброкачественный кератоз',     'risk': 'low'},
    'df':    {'name': 'Дерматофиброма',                 'risk': 'low'},
    'mel':   {'name': 'Меланома',                       'risk': 'high'},
    'nv':    {'name': 'Меланоцитарный невус (родинка)', 'risk': 'low'},
    'vasc':  {'name': 'Сосудистое поражение',           'risk': 'medium'},
}

# Модель загружается один раз при старте
_model = None

def get_model():
    global _model
    if _model is None and os.path.exists(settings.MODEL_PATH):
        _model = tf.keras.models.load_model(settings.MODEL_PATH)
    return _model

def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Предобработка: resize + MobileNetV2 нормализация."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, 0)

def generate_gradcam(model, img_array: np.ndarray, class_idx: int, save_path: str):
    """Генерирует Grad-CAM и сохраняет наложение на оригинал."""
    try:
        last_conv = next(
            l.name for l in reversed(model.layers)
            if 'conv' in l.name.lower() and len(l.output.shape) == 4
        )
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            score = preds[:, class_idx]

        grads       = tape.gradient(score, conv_out)
        pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap     = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap     = tf.squeeze(heatmap)
        heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np  = heatmap.numpy()

        # Восстанавливаем оригинал из preprocess
        orig = ((img_array[0] + 1) / 2 * 255).astype(np.uint8)
        h, w = orig.shape[:2]
        hmap = cv2.resize(heatmap_np, (w, h))
        hmap = np.uint8(255 * hmap)
        hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig[:, :, ::-1], 0.6, hmap, 0.4, 0)
        cv2.imwrite(save_path, overlay)
    except Exception as e:
        print(f"Grad-CAM ошибка: {e}")

# ── Загрузка и диагностика ────────────────────────────────────────────────────
@router.post("/analyse")
async def analyse(
    file:       UploadFile = File(...),
    patient_id: str        = Form(None),
    db:         AsyncSession = Depends(get_db),
    user:       Doctor       = Depends(get_current_user),
):
    # Валидация файла
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Поддерживаются только JPEG и PNG")

    img_bytes = await file.read()
    if len(img_bytes) > 16 * 1024 * 1024:
        raise HTTPException(400, "Файл слишком большой (максимум 16 МБ)")

    # Сохраняем оригинал
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.GRADCAM_DIR, exist_ok=True)
    image_id  = str(uuid.uuid4())
    ext       = Path(file.filename or "image.jpg").suffix or ".jpg"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{image_id}{ext}")
    with open(file_path, "wb") as f:
        f.write(img_bytes)

    # Запись в БД
    img_record = SkinLesionImage(
        image_id          = image_id,
        doctor_id         = user.doctor_id,
        patient_id        = patient_id,
        file_path         = file_path,
        file_format       = ext.lstrip("."),
        file_size_kb      = len(img_bytes) // 1024,
        original_filename = file.filename,
        preprocessing_status = "processing",
    )
    db.add(img_record)
    await db.flush()

    # Диагностика
    model = get_model()
    if model is None:
        raise HTTPException(503, "Модель не загружена. Положите best_model.h5 в папку models/")

    start_ms = time.time()
    x        = preprocess_image(img_bytes)
    probs    = model.predict(x, verbose=0)[0]
    top_idx  = int(probs.argmax())
    top_cls  = CLASS_LABELS[top_idx]
    ms       = int((time.time() - start_ms) * 1000)

    # Grad-CAM
    gradcam_path = os.path.join(settings.GRADCAM_DIR, f"gradcam_{image_id}.jpg")
    generate_gradcam(model, x, top_idx, gradcam_path)

    # Получаем активную версию модели
    mv_result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)
    )
    model_version = mv_result.scalar_one_or_none()

    # Сохраняем отчёт
    prob_dict = {cls: float(p) for cls, p in zip(CLASS_LABELS, probs)}
    report = DiagnosisReport(
        image_id                 = image_id,
        model_version_id         = model_version.model_version_id if model_version else None,
        prediction_class         = top_cls,
        confidence_score         = float(probs[top_idx]),
        probability_distribution = prob_dict,
        gradcam_map_path         = gradcam_path,
        processing_time_ms       = ms,
    )
    db.add(report)
    img_record.preprocessing_status = "done"

    return {
        "report_id":    str(report.report_id),
        "image_id":     image_id,
        "prediction":   top_cls,
        "class_name":   CLASS_INFO[top_cls]["name"],
        "risk":         CLASS_INFO[top_cls]["risk"],
        "confidence":   float(probs[top_idx]),
        "probabilities": prob_dict,
        "gradcam_url":  f"/diagnosis/gradcam/{image_id}",
        "image_url":    f"/diagnosis/image/{image_id}",
        "processing_ms": ms,
        "class_info":   CLASS_INFO,
    }

# ── Получить Grad-CAM ─────────────────────────────────────────────────────────
@router.get("/gradcam/{image_id}")
async def get_gradcam(image_id: str):
    """Публичный — изображения доступны только внутри Docker сети."""
    path = os.path.join(settings.GRADCAM_DIR, f"gradcam_{image_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(404, "Grad-CAM не найден")
    return FileResponse(path, media_type="image/jpeg")

@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Публичный — изображения доступны только внутри Docker сети."""
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(settings.UPLOAD_DIR, f"{image_id}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "Изображение не найдено")

# ── История диагностик ────────────────────────────────────────────────────────
@router.get("/history")
async def history(
    limit:  int = 20,
    offset: int = 0,
    db:     AsyncSession = Depends(get_db),
    user:   Doctor       = Depends(get_current_user),
):
    result = await db.execute(
        select(DiagnosisReport, SkinLesionImage)
        .join(SkinLesionImage, DiagnosisReport.image_id == SkinLesionImage.image_id)
        .where(SkinLesionImage.doctor_id == user.doctor_id)
        .order_by(desc(DiagnosisReport.created_at))
        .limit(limit).offset(offset)
    )
    rows = result.all()

    return [
        {
            "report_id":    str(r.report_id),
            "image_id":     str(r.image_id),
            "prediction":   r.prediction_class,
            "class_name":   CLASS_INFO.get(r.prediction_class, {}).get("name", r.prediction_class),
            "risk":         CLASS_INFO.get(r.prediction_class, {}).get("risk", "unknown"),
            "confidence":   float(r.confidence_score),
            "gradcam_url":  f"/diagnosis/gradcam/{str(r.image_id)}",
            "image_url":    f"/diagnosis/image/{str(r.image_id)}",
            "created_at":   r.created_at.isoformat(),
            "patient_id":   str(img.patient_id) if img.patient_id else None,
        }
        for r, img in rows
    ]

# ── Статистика ────────────────────────────────────────────────────────────────
@router.get("/stats")
async def stats(
    db:   AsyncSession = Depends(get_db),
    user: Doctor       = Depends(get_current_user),
):
    from sqlalchemy import func

    total = await db.execute(
        select(func.count(DiagnosisReport.report_id))
        .join(SkinLesionImage)
        .where(SkinLesionImage.doctor_id == user.doctor_id)
    )

    by_class = await db.execute(
        select(DiagnosisReport.prediction_class, func.count())
        .join(SkinLesionImage)
        .where(SkinLesionImage.doctor_id == user.doctor_id)
        .group_by(DiagnosisReport.prediction_class)
    )

    avg_ms = await db.execute(
        select(func.avg(DiagnosisReport.processing_time_ms))
        .join(SkinLesionImage)
        .where(SkinLesionImage.doctor_id == user.doctor_id)
    )

    mv_result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)
    )
    mv = mv_result.scalar_one_or_none()

    return {
        "total_diagnoses": total.scalar() or 0,
        "by_class": {row[0]: row[1] for row in by_class.all()},
        "avg_processing_ms": int(avg_ms.scalar() or 0),
        "model": {
            "architecture": mv.architecture if mv else "N/A",
            "version":      mv.version_number if mv else "N/A",
            "accuracy":     float(mv.accuracy) if mv else 0,
        } if mv else None,
    }


# ── Тестовые примеры из HAM10000 ─────────────────────────────────────────────
@router.get("/test_samples")
async def get_test_samples(user: Doctor = Depends(get_current_user)):
    """
    Возвращает по одному случайному изображению каждого класса из HAM10000.
    Используется на вкладке тестирования в UI.
    """
    import pandas as pd
    import random

    CLASS_LABELS_LOCAL = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    data_dir = settings.HAM10000_DIR

    csv_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(503, f"Датасет HAM10000 не найден. Смонтируйте его в {data_dir}")

    df = pd.read_csv(csv_path)

    def find_ham_image(image_id: str) -> str | None:
        for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']:
            p = os.path.join(data_dir, folder, f"{image_id}.jpg")
            if os.path.exists(p):
                return p
        return None

    samples = []
    for cls in CLASS_LABELS_LOCAL:
        cls_rows = df[df['dx'] == cls]
        if cls_rows.empty:
            continue
        # Пробуем найти изображение (перемешиваем пока не найдём)
        shuffled = cls_rows.sample(frac=1)
        for _, row in shuffled.iterrows():
            path = find_ham_image(row['image_id'])
            if path:
                samples.append({
                    "image_id":   row['image_id'],
                    "real_class": cls,
                    "age":        int(row['age']) if pd.notna(row.get('age')) else None,
                    "sex":        row.get('sex', ''),
                    "image_url":  f"/diagnosis/ham_image/{row['image_id']}",
                })
                break

    return samples


@router.get("/ham_image/{image_id}")
async def get_ham_image(image_id: str):
    """Отдаёт изображение из датасета HAM10000 (публичный, только по UUID)."""
    data_dir = settings.HAM10000_DIR
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']:
        path = os.path.join(data_dir, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "Изображение не найдено в датасете HAM10000")


@router.post("/test_predict")
async def test_predict(
    body: dict,
    user: Doctor = Depends(get_current_user),
):
    """
    Предсказание для изображения из HAM10000 по его image_id.
    Используется на вкладке тестирования.
    """
    image_id = body.get("image_id")
    data_dir = settings.HAM10000_DIR

    path = None
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']:
        p = os.path.join(data_dir, folder, f"{image_id}.jpg")
        if os.path.exists(p):
            path = p
            break

    if not path:
        raise HTTPException(404, f"Изображение {image_id} не найдено")

    model = get_model()
    if model is None:
        raise HTTPException(503, "Модель не загружена")

    start = time.time()
    x     = preprocess_image(open(path, 'rb').read())
    probs = model.predict(x, verbose=0)[0]
    ms    = int((time.time() - start) * 1000)
    top_i = int(probs.argmax())
    top   = CLASS_LABELS[top_i]

    return {
        "prediction":    top,
        "class_name":    CLASS_INFO[top]["name"],
        "risk":          CLASS_INFO[top]["risk"],
        "confidence":    float(probs[top_i]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASS_LABELS, probs)},
        "processing_ms": ms,
    }
