"""
train.py
--------
Первая версия обучения модели — базовый подход.

Архитектура: EfficientNetB0 + Transfer Learning
Проблема этой версии: модель предсказывала только невус (nv) потому что
его 67% в датасете. Точность была 66.9%, но recall меланомы = 0%.
Причина: обычные веса классов не справились с таким дисбалансом,
и нормализация /255.0 не подходит для EfficientNet.

Эта проблема решена в финальной версии train_v4.py.

Запуск:
  python train.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("  ОБУЧЕНИЕ v1 — EfficientNetB0 (базовая версия)")
print("  Финальная версия: train_v4.py")
print("=" * 60)

# ── Настройки ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
DATA_DIR    = "data"
RESULTS_DIR = "results"
MODELS_DIR  = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

CLASS_NAMES = {
    'akiec': 0,  # Актинический кератоз
    'bcc':   1,  # Базальноклеточная карцинома
    'bkl':   2,  # Доброкачественный кератоз
    'df':    3,  # Дерматофиброма
    'mel':   4,  # Меланома
    'nv':    5,  # Меланоцитарный невус (родинка)
    'vasc':  6   # Сосудистое поражение
}
CLASS_LABELS = list(CLASS_NAMES.keys())

# ── Шаг 1: Загрузка метаданных ────────────────────────────────────────────────
print("\n[1/6] Загрузка метаданных датасета...")

csv_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

if not os.path.exists(csv_path):
    print(f"\n❌ Файл не найден: {csv_path}")
    print("   Скачайте датасет с Kaggle и распакуйте в папку data/")
    exit(1)

df = pd.read_csv(csv_path)
print(f"   Загружено записей: {len(df)}")
print(f"\n   Распределение классов:")
for cls, cnt in df['dx'].value_counts().items():
    bar = '█' * (cnt // 100)
    print(f"   {cls:6s}: {cnt:5d} {bar}")

# ── Шаг 2: Поиск изображений ─────────────────────────────────────────────────
print("\n[2/6] Поиск изображений...")

def find_image(image_id):
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(DATA_DIR, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df['path']  = df['image_id'].apply(find_image)
df['label'] = df['dx'].map(CLASS_NAMES)
df = df.dropna(subset=['path'])
print(f"   Найдено изображений: {len(df)}")

# ── Шаг 3: Загрузка изображений ───────────────────────────────────────────────
print("\n[3/6] Загрузка изображений...")

def load_images(dataframe, img_size=224):
    images, labels = [], []
    total = len(dataframe)
    for i, (_, row) in enumerate(dataframe.iterrows()):
        if i % 500 == 0:
            print(f"   Загружено {i}/{total}...")
        try:
            img = load_img(row['path'], target_size=(img_size, img_size))
            arr = img_to_array(img) / 255.0  # нормализация [0, 1]
            images.append(arr)
            labels.append(row['label'])
        except Exception as e:
            print(f"   Ошибка: {row['path']}: {e}")
    return np.array(images), np.array(labels)

X, y = load_images(df)
print(f"   Загружено: X={X.shape}")

# ── Шаг 4: Разбивка Train / Val / Test ────────────────────────────────────────
print("\n[4/6] Разбивка данных...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)
print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

y_train_cat = to_categorical(y_train, num_classes=7)
y_val_cat   = to_categorical(y_val,   num_classes=7)

# ── Шаг 5: Веса классов ───────────────────────────────────────────────────────
print("\n[5/6] Вычисление весов классов...")

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_arr))
for i, (cls, w) in enumerate(zip(CLASS_LABELS, class_weights_arr)):
    print(f"   {cls:6s}: {w:.2f}")

# ── Шаг 6: Аугментация ───────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1,
    height_shift_range=0.1, horizontal_flip=True,
    vertical_flip=True, zoom_range=0.1, fill_mode='nearest'
).flow(X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True)

val_gen = ImageDataGenerator().flow(
    X_val, y_val_cat, batch_size=BATCH_SIZE, shuffle=False
)

# ── Построение модели ─────────────────────────────────────────────────────────
print("\n[6/6] Построение модели EfficientNetB0...")

base = EfficientNetB0(weights='imagenet', include_top=False,
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x      = GlobalAveragePooling2D()(base.output)
x      = BatchNormalization()(x)
x      = Dropout(0.3)(x)
x      = Dense(256, activation='relu')(x)
x      = Dropout(0.2)(x)
output = Dense(7, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
print(f"   Параметров: {model.count_params():,}")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(MODELS_DIR, 'best_model.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ── Фаза 1: обучение ─────────────────────────────────────────────────────────
print("\n== Обучение ==")
h1 = model.fit(
    train_gen, steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS, validation_data=val_gen,
    validation_steps=len(X_val) // BATCH_SIZE,
    class_weight=class_weights, callbacks=callbacks, verbose=1
)

# ── Фаза 2: Fine-tuning ───────────────────────────────────────────────────────
print("\n== Fine-tuning (последние 20 слоёв) ==")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(LR / 10), loss='categorical_crossentropy', metrics=['accuracy'])

h2 = model.fit(
    train_gen, steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=10, validation_data=val_gen,
    validation_steps=len(X_val) // BATCH_SIZE,
    class_weight=class_weights, callbacks=callbacks, verbose=1
)

# ── Результаты ────────────────────────────────────────────────────────────────
print("\n== РЕЗУЛЬТАТЫ ==")
y_pred = model.predict(X_test, batch_size=BATCH_SIZE).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

# Графики
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
a  = h1.history['accuracy']     + h2.history['accuracy']
v  = h1.history['val_accuracy'] + h2.history['val_accuracy']
ft = len(h1.history['accuracy'])
ep = range(1, len(a) + 1)

axes[0].plot(ep, a, label='Train', color='steelblue', lw=2)
axes[0].plot(ep, v, label='Val',   color='darkorange', lw=2)
axes[0].axvline(ft, color='red', ls='--', alpha=0.5, label='Fine-tuning')
axes[0].set_title('Accuracy (v1)'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, h1.history['loss'] + h2.history['loss'], label='Train', color='steelblue', lw=2)
axes[1].plot(ep, h1.history['val_loss'] + h2.history['val_loss'], label='Val', color='darkorange', lw=2)
axes[1].axvline(ft, color='red', ls='--', alpha=0.5)
axes[1].set_title('Loss (v1)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
ax.set_title('Confusion Matrix v1')
ax.set_xlabel('Предсказано'); ax.set_ylabel('Реально')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)

model.save(os.path.join(MODELS_DIR, 'final_model.h5'))

print(f"\n  Модель сохранена: models/final_model.h5")
print(f"  Для лучшего результата используй: python train_v4.py")
