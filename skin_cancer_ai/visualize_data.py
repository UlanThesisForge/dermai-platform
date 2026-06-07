"""
visualize_data.py
-----------------
Визуализация датасета HAM10000 перед началом обучения модели.

Что делает этот скрипт:
  1. Строит график распределения классов (столбчатый + круговой)
  2. Показывает примеры изображений по каждому из 7 классов
  3. Демонстрирует работу аугментации на одном изображении

Все графики сохраняются в папку results/

Запуск:
  python visualize_data.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # режим без GUI — работает на любой машине
import matplotlib.pyplot as plt

# Подавляем лишние предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import (
    load_img, img_to_array, ImageDataGenerator
)

# ── Пути и настройки ──────────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Все 7 классов датасета HAM10000
CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Русскоязычные описания для подписей на графиках
CLASS_DESCRIPTIONS = {
    'akiec': 'Актинический\nкератоз',
    'bcc':   'Базальноклеточная\nкарцинома',
    'bkl':   'Доброкачественный\nкератоз',
    'df':    'Дерматофиброма',
    'mel':   'Меланома',
    'nv':    'Невус\n(родинка)',
    'vasc':  'Сосудистое\nпоражение'
}

# Уникальный цвет для каждого класса
COLOR_MAP = {
    'akiec': '#e74c3c',
    'bcc':   '#3498db',
    'bkl':   '#2ecc71',
    'df':    '#f39c12',
    'mel':   '#9b59b6',
    'nv':    '#1abc9c',
    'vasc':  '#e67e22'
}

print("=" * 55)
print("  ВИЗУАЛИЗАЦИЯ ДАТАСЕТА HAM10000")
print("=" * 55)

# ── Загрузка метаданных ───────────────────────────────────────────────────────
csv_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

if not os.path.exists(csv_path):
    print(f"\n❌ Файл не найден: {csv_path}")
    print("   Скачайте датасет и поместите его в папку data/")
    exit(1)

df = pd.read_csv(csv_path)

def find_image(image_id):
    """Ищем изображение в обеих частях датасета."""
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(DATA_DIR, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df['path'] = df['image_id'].apply(find_image)
df = df.dropna(subset=['path'])  # убираем записи без изображений

print(f"\nНайдено изображений: {len(df)}")
print(f"Классов:             {df['dx'].nunique()}")

# ── График 1: Распределение классов ──────────────────────────────────────────
print("\n[1/3] Строим распределение классов...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df['dx'].value_counts()

# Столбчатая диаграмма
bars = axes[0].bar(
    counts.index,
    counts.values,
    color=[COLOR_MAP.get(cls, '#95a5a6') for cls in counts.index]
)
axes[0].set_title('Распределение классов в датасете', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Класс кожного заболевания')
axes[0].set_ylabel('Количество изображений')
axes[0].grid(axis='y', alpha=0.3)

# Подписи значений над столбцами
for bar, val in zip(bars, counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        str(val),
        ha='center', va='bottom',
        fontsize=11, fontweight='bold'
    )

# Круговая диаграмма
axes[1].pie(
    counts.values,
    labels=counts.index,
    colors=[COLOR_MAP.get(cls, '#95a5a6') for cls in counts.index],
    autopct='%1.1f%%',
    startangle=90
)
axes[1].set_title('Процентное соотношение классов', fontsize=13, fontweight='bold')

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'data_distribution.png')
plt.savefig(out_path, dpi=150)
print(f"   ✓ {out_path}")

# ── График 2: Примеры изображений по каждому классу ──────────────────────────
print("\n[2/3] Загружаем примеры изображений...")

fig, axes = plt.subplots(3, 7, figsize=(20, 9))
fig.suptitle('Примеры дерматоскопических изображений по каждому классу',
             fontsize=14, fontweight='bold')

for col, cls in enumerate(CLASS_LABELS):
    # Берём 3 случайных фото данного класса
    cls_samples = df[df['dx'] == cls].sample(
        min(3, len(df[df['dx'] == cls])),
        random_state=42
    )

    for row, (_, sample) in enumerate(cls_samples.iterrows()):
        try:
            img = load_img(sample['path'], target_size=(224, 224))
            arr = img_to_array(img).astype(np.uint8)
            axes[row, col].imshow(arr)
            axes[row, col].axis('off')

            # Заголовок только для первой строки
            if row == 0:
                axes[row, col].set_title(
                    f"{cls}\n{CLASS_DESCRIPTIONS[cls]}",
                    fontsize=9,
                    fontweight='bold',
                    color=COLOR_MAP[cls]
                )
        except Exception:
            axes[row, col].axis('off')

    # Если фото меньше 3 — скрываем пустые ячейки
    for row in range(len(cls_samples), 3):
        axes[row, col].axis('off')

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'sample_images.png')
plt.savefig(out_path, dpi=120)
print(f"   ✓ {out_path}")

# ── График 3: Демонстрация аугментации ───────────────────────────────────────
print("\n[3/3] Демонстрация аугментации данных...")

# Берём случайное изображение из датасета
sample_path = df.sample(1, random_state=42)['path'].values[0]

# Загружаем в формате uint8 [0, 255] — нужно для корректной работы генератора
orig_uint8 = img_to_array(load_img(sample_path, target_size=(224, 224)))

# Параметры аугментации — те же что используются при обучении
augmentor = ImageDataGenerator(
    rotation_range=20,        # поворот до 20 градусов
    width_shift_range=0.1,    # горизонтальный сдвиг до 10%
    height_shift_range=0.1,   # вертикальный сдвиг до 10%
    horizontal_flip=True,     # горизонтальное отражение
    vertical_flip=True,       # вертикальное отражение
    zoom_range=0.1,            # масштабирование до 10%
    fill_mode='nearest'        # заполнение пустых пикселей
)

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle(
    'Аугментация данных: из одного изображения создаём много вариаций',
    fontsize=13, fontweight='bold'
)

# Оригинал в первой ячейке
axes[0, 0].imshow(orig_uint8.astype(np.uint8))
axes[0, 0].set_title('Оригинал', fontweight='bold', fontsize=11)
axes[0, 0].axis('off')

# Генерируем 9 аугментированных вариаций
batch = np.expand_dims(orig_uint8, axis=0)
gen   = augmentor.flow(batch, batch_size=1, seed=42)

for i, ax in enumerate(axes.flatten()[1:]):
    aug_img = next(gen)[0]
    ax.imshow(np.clip(aug_img / 255.0, 0, 1))
    ax.set_title(f'Вариация {i + 1}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'augmentation_demo.png')
plt.savefig(out_path, dpi=150)
print(f"   ✓ {out_path}")

# ── Статистика в консоль ──────────────────────────────────────────────────────
print(f"\n{'─' * 55}")
print(f"  СТАТИСТИКА ДАТАСЕТА")
print(f"{'─' * 55}")
print(f"  Всего изображений : {len(df)}")
print(f"  Количество классов: {df['dx'].nunique()}")

if 'sex' in df.columns:
    sex_counts = df['sex'].value_counts().to_dict()
    print(f"  Пол пациентов     : {sex_counts}")

if 'age' in df.columns:
    print(f"  Возраст пациентов : "
          f"мин={df['age'].min():.0f}, "
          f"среднее={df['age'].mean():.1f}, "
          f"макс={df['age'].max():.0f}")

print(f"\n  Классы по частоте (от редкого к частому):")
for cls, cnt in df['dx'].value_counts().sort_values().items():
    pct = cnt / len(df) * 100
    bar = '█' * max(1, int(pct / 2))
    print(f"  {cls:6s} : {cnt:5d} ({pct:5.1f}%) {bar}")

print(f"\n{'─' * 55}")
print(f"  Графики сохранены в: {RESULTS_DIR}/")
print(f"  Следующий шаг      : python train_v4.py")
