"""
generate_demo_data.py — Генерация демо-данных для тестирования без реального датасета
Запуск: python generate_demo_data.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import random

print("Генерация демо-данных для тестирования...")

DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# Реальное соотношение HAM10000 (невус доминирует)
COUNTS      = [327, 514, 1099, 115, 1113, 6705, 142]

records = []
total   = 0

for cls, count in zip(CLASS_NAMES, COUNTS):
    # Для демо — берём 10% от реальных данных (иначе долго генерировать)
    n = max(20, count // 10)
    for i in range(n):
        image_id = f"{cls}_{i:05d}"
        path     = os.path.join(IMAGES_DIR, f"{image_id}.jpg")

        # Рисуем цветные квадраты — уникальный цвет для каждого класса
        base_colors = {
            'akiec': (200, 150, 100),
            'bcc':   (180, 100, 80),
            'bkl':   (150, 130, 110),
            'df':    (160, 100, 90),
            'mel':   (50,  30,  20),   # тёмная — похожа на меланому
            'nv':    (200, 170, 140),
            'vasc':  (210, 80,  80),
        }
        r, g, b = base_colors[cls]
        noise   = np.random.randint(-30, 30, (224, 224, 3))
        arr     = np.clip(
            np.full((224, 224, 3), [r, g, b], dtype=np.int32) + noise,
            0, 255
        ).astype(np.uint8)

        # Добавляем пятно в центре
        cx, cy = 112 + random.randint(-20, 20), 112 + random.randint(-20, 20)
        radius = random.randint(30, 60)
        for px in range(arr.shape[0]):
            for py in range(arr.shape[1]):
                if (px - cx)**2 + (py - cy)**2 < radius**2:
                    arr[px, py] = np.clip(
                        np.array([r//2, g//2, b//2]) + np.random.randint(-10, 10, 3),
                        0, 255
                    )

        Image.fromarray(arr).save(path, quality=85)
        records.append({
            'image_id': image_id,
            'dx':       cls,
            'dx_type':  'consensus',
            'age':      random.randint(20, 80),
            'sex':      random.choice(['male', 'female']),
            'localization': random.choice(['back', 'lower extremity', 'trunk'])
        })
        total += 1

    print(f"   {cls}: {n} изображений")

df = pd.DataFrame(records)
df.to_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"), index=False)

print(f"\n✓ Создано {total} демо-изображений")
print(f"✓ data/HAM10000_metadata.csv")
print(f"\nТеперь запусти: python train.py")
