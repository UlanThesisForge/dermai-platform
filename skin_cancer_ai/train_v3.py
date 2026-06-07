"""
train_v3.py
-----------
Третья версия обучения — эксперимент с Focal Loss.

Проблема v2: агрессивный oversampling + ограниченные веса дали перекос в BCC.
Гипотеза v3: Focal Loss сам справится с дисбалансом без ручной настройки весов.

Focal Loss — функция потерь специально разработанная для несбалансированных
медицинских датасетов. Основная идея: модель получает большой штраф за
уверенные ошибки на редких классах и малый штраф за правильно классифицированные
примеры. Параметр gamma=2.0 — стандартное значение для медицинских задач.

Результат: модель снова вернулась к коллапсу в NV (accuracy 66.9%).
Причина: EfficientNet требует специальной нормализации (не /255.0),
а не обычного CrossEntropy. Проблема решена сменой архитектуры в train_v4.py.

Запуск:
  python train_v3.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("  ОБУЧЕНИЕ v3 — EfficientNetB0 + Focal Loss")
print("  (Промежуточная версия — финальная: train_v4.py)")
print("=" * 60)

# ── Настройки ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4
DATA_DIR    = "data"
RESULTS_DIR = "results"
MODELS_DIR  = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

CLASS_NAMES  = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
CLASS_LABELS = list(CLASS_NAMES.keys())


# ── Focal Loss ────────────────────────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss для несбалансированной классификации.

    Формула: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Параметры:
        gamma — степень фокусировки на сложных примерах (обычно 2.0)
                Чем выше gamma, тем сильнее штраф за уверенные ошибки
        alpha — балансировочный коэффициент (обычно 0.25)
    """
    def loss_fn(y_true, y_pred):
        # Ограничиваем предсказания чтобы избежать log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        # Стандартный cross-entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Весовой коэффициент: (1-p)^gamma — большой для ошибочных предсказаний
        focal_weight  = alpha * y_true * tf.pow(1.0 - y_pred, gamma)

        return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=1))

    return loss_fn


# ── Загрузка метаданных ───────────────────────────────────────────────────────
print("\n[1/5] Загрузка данных...")

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

def find_image(image_id):
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']:
        path = os.path.join(DATA_DIR, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df['path']  = df['image_id'].apply(find_image)
df['label'] = df['dx'].map(CLASS_NAMES)
df = df.dropna(subset=['path'])

print(f"   Найдено: {len(df)} изображений")
print(f"   Классы: {df['dx'].value_counts().to_dict()}")

# ── Разбивка и лёгкий oversampling ───────────────────────────────────────────
print("\n[2/5] Разбивка и oversampling до среднего...")

tr, te = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
tr, va = train_test_split(tr, test_size=0.176, stratify=tr['label'], random_state=42)

# Дополняем редкие классы до среднего — мягче чем в v2
avg_count = int(tr['dx'].value_counts().mean())
balanced  = []
for cls in CLASS_LABELS:
    c = tr[tr['dx'] == cls].copy()
    if len(c) < avg_count:
        c = pd.concat([c] * (avg_count // len(c) + 1)).head(avg_count)
    balanced.append(c)

tr = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
print(f"   После oversampling: {tr['dx'].value_counts().to_dict()}")

# ── Загрузка изображений ──────────────────────────────────────────────────────
print("\n[3/5] Загрузка изображений...")

def load_images(dataframe):
    X, y = [], []
    for i, (_, row) in enumerate(dataframe.iterrows()):
        if i % 1000 == 0:
            print(f"   {i}/{len(dataframe)}...")
        try:
            img = load_img(row['path'], target_size=(IMG_SIZE, IMG_SIZE))
            # /255.0 — неправильная нормализация для EfficientNet
            # Правильная: preprocess_input из tensorflow.keras.applications.efficientnet
            X.append(img_to_array(img) / 255.0)
            y.append(row['label'])
        except Exception:
            pass
    return np.array(X), np.array(y)

X_tr, y_tr = load_images(tr)
X_va, y_va = load_images(va)
X_te, y_te = load_images(te)
print(f"   Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")

y_tr_c = to_categorical(y_tr, 7)
y_va_c = to_categorical(y_va, 7)

# ── Аугментация ───────────────────────────────────────────────────────────────
train_flow = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, vertical_flip=True, zoom_range=0.1, fill_mode='nearest'
).flow(X_tr, y_tr_c, batch_size=BATCH_SIZE, shuffle=True)

val_flow = ImageDataGenerator().flow(
    X_va, y_va_c, batch_size=BATCH_SIZE, shuffle=False
)

# ── Модель EfficientNetB0 ─────────────────────────────────────────────────────
print("\n[4/5] Построение модели EfficientNetB0 + Focal Loss...")

base = EfficientNetB0(
    weights='imagenet', include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

x   = GlobalAveragePooling2D()(base.output)
x   = BatchNormalization()(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(7, activation='softmax')(x)

model = Model(base.input, out)
model.compile(
    optimizer=Adam(LR),
    loss=focal_loss(gamma=2.0, alpha=0.25),  # Focal Loss вместо CrossEntropy
    metrics=['accuracy']
)
print(f"   Параметров: {model.count_params():,}")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# ── Фаза 1: обучение классификатора ──────────────────────────────────────────
print("\n[5/5] Фаза 1 — Обучение классификатора...")

h1 = model.fit(
    train_flow,
    steps_per_epoch=len(X_tr) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_flow,
    validation_steps=len(X_va) // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── Фаза 2: Fine-tuning ───────────────────────────────────────────────────────
print("\nФаза 2 — Fine-tuning (последние 20 слоёв)...")

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(LR / 10),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

h2 = model.fit(
    train_flow,
    steps_per_epoch=len(X_tr) // BATCH_SIZE,
    epochs=20,
    validation_data=val_flow,
    validation_steps=len(X_va) // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── Результаты ────────────────────────────────────────────────────────────────
print("\n== РЕЗУЛЬТАТЫ ==")
y_pred = model.predict(X_te, batch_size=BATCH_SIZE).argmax(axis=1)
print(classification_report(y_te, y_pred, target_names=CLASS_LABELS))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
ax.set_title('Confusion Matrix v3 (Focal Loss)')
ax.set_xlabel('Предсказано'); ax.set_ylabel('Реально')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_v3.png'), dpi=150)

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
a  = h1.history['accuracy']     + h2.history['accuracy']
v  = h1.history['val_accuracy'] + h2.history['val_accuracy']
ft = len(h1.history['accuracy'])
ep = range(1, len(a) + 1)

axes[0].plot(ep, a, label='Train', color='steelblue', lw=2)
axes[0].plot(ep, v, label='Val',   color='darkorange', lw=2)
axes[0].axvline(ft, color='red', ls='--', alpha=0.5, label='Fine-tuning')
axes[0].set_title('Accuracy (v3)'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, h1.history['loss'] + h2.history['loss'], label='Train', color='steelblue', lw=2)
axes[1].plot(ep, h1.history['val_loss'] + h2.history['val_loss'], label='Val', color='darkorange', lw=2)
axes[1].axvline(ft, color='red', ls='--', alpha=0.5)
axes[1].set_title('Loss (v3)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history_v3.png'), dpi=150)

model.save(os.path.join(MODELS_DIR, 'best_model.h5'))

acc   = accuracy_score(y_te, y_pred)
f1    = f1_score(y_te, y_pred, average='weighted')
mel_r = cm[4, 4] / cm[4].sum() if cm[4].sum() > 0 else 0

print(f"\n{'─' * 50}")
print(f"  Accuracy        : {acc:.3f} ({acc*100:.1f}%)")
print(f"  F1-Score        : {f1:.3f}  ({f1*100:.1f}%)")
print(f"  Recall меланомы : {mel_r:.3f} ({mel_r*100:.1f}%)")
print(f"{'─' * 50}")
print("\n  Финальная версия с лучшим результатом: python train_v4.py")
