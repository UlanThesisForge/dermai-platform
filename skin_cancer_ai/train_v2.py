"""
train_v2.py
-----------
Вторая версия обучения — попытка исправить коллапс модели в majority class.

Проблема v1: модель всегда предсказывала только невус (nv, 67% датасета),
игнорируя остальные классы. Accuracy была 66.9%, но recall меланомы = 0%.

Что изменилось в v2:
  - Oversampling до 50% от максимального класса (агрессивнее чем в v4)
  - Умеренные веса классов с ограничением [0.5, 5.0]
  - Более глубокий классификатор: Dense(512) + Dense(256)

Результат: модель ушла в другую крайность — стала предсказывать только BCC.
Проблема решена в финальной версии train_v4.py.

Архитектура: EfficientNetB0 (не рекомендуется — используйте train_v4.py)

Запуск:
  python train_v2.py
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
print("  ОБУЧЕНИЕ v2 — EfficientNetB0 + Oversampling до 50% макс")
print("  (Промежуточная версия — финальная: train_v4.py)")
print("=" * 60)

# ── Настройки ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 40
LR          = 1e-4
DATA_DIR    = "data"
RESULTS_DIR = "results"
MODELS_DIR  = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

CLASS_NAMES  = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
CLASS_LABELS = list(CLASS_NAMES.keys())

# ── Загрузка метаданных ───────────────────────────────────────────────────────
print("\n[1/6] Загрузка метаданных...")

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

def find_image(image_id):
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(DATA_DIR, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df['path']  = df['image_id'].apply(find_image)
df['label'] = df['dx'].map(CLASS_NAMES)
df = df.dropna(subset=['path'])
print(f"   Найдено {len(df)} изображений")

# ── Разбивка Train / Val / Test ───────────────────────────────────────────────
print("\n[2/6] Разбивка данных...")

X_temp, X_test_df = train_test_split(
    df, test_size=0.15, stratify=df['label'], random_state=42
)
X_train_df, X_val_df = train_test_split(
    X_temp, test_size=0.176, stratify=X_temp['label'], random_state=42
)
print(f"   Train: {len(X_train_df)} | Val: {len(X_val_df)} | Test: {len(X_test_df)}")

# ── Oversampling до 50% от максимального класса ───────────────────────────────
# В v2 дополняем до 50% от max_count — это агрессивнее чем в v4 (до среднего)
# Итог: модель переобучилась на дублях редких классов → коллапс в BCC
print("\n[3/6] Oversampling (до 50% от максимального класса)...")

max_count    = X_train_df['dx'].value_counts().max()
target_count = max_count // 2   # 50% от максимума

balanced = []
for cls in CLASS_LABELS:
    cls_df  = X_train_df[X_train_df['dx'] == cls].copy()
    current = len(cls_df)
    if current < target_count:
        mult   = target_count // current + 1
        cls_df = pd.concat([cls_df] * mult, ignore_index=True).head(target_count)
        print(f"   {cls:6s}: {current} → {len(cls_df)}")
    else:
        print(f"   {cls:6s}: {current} (без изменений)")
    balanced.append(cls_df)

train_df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   Итого train: {len(train_df)}")

# ── Загрузка изображений ──────────────────────────────────────────────────────
# Внимание: нормализация /255.0 — неправильна для EfficientNet
# Это одна из причин проблем v2 (исправлено в v4 через preprocess_input)
print("\n[4/6] Загрузка изображений...")

def load_images(dataframe):
    imgs, lbls = [], []
    total = len(dataframe)
    for i, (_, row) in enumerate(dataframe.iterrows()):
        if i % 1000 == 0:
            print(f"   {i}/{total}...")
        try:
            img = load_img(row['path'], target_size=(IMG_SIZE, IMG_SIZE))
            imgs.append(img_to_array(img) / 255.0)
            lbls.append(row['label'])
        except Exception:
            pass
    return np.array(imgs), np.array(lbls)

X_train, y_train = load_images(train_df)
X_val,   y_val   = load_images(X_val_df)
X_test,  y_test  = load_images(X_test_df)
print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

y_train_cat = to_categorical(y_train, 7)
y_val_cat   = to_categorical(y_val,   7)

# Умеренные веса: ограничиваем диапазон [0.5, 5.0] чтобы не перекосить обучение
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = np.clip(cw, 0.5, 5.0)
class_weights = dict(enumerate(cw))

# ── Аугментация ───────────────────────────────────────────────────────────────
print("\n[5/6] Настройка аугментации...")

train_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],  # работает только с uint8, может давать артефакты
    fill_mode='nearest'
).flow(X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True)

val_gen = ImageDataGenerator().flow(
    X_val, y_val_cat, batch_size=BATCH_SIZE, shuffle=False
)

# ── Построение модели EfficientNetB0 ─────────────────────────────────────────
print("\n[6/6] Построение модели EfficientNetB0...")

base = EfficientNetB0(
    weights='imagenet', include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

x   = GlobalAveragePooling2D()(base.output)
x   = BatchNormalization()(x)
x   = Dense(512, activation='relu')(x)
x   = Dropout(0.4)(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(7, activation='softmax')(x)

model = Model(base.input, out)
model.compile(
    optimizer=Adam(LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# ── Фаза 1: обучение классификатора ──────────────────────────────────────────
print("\n== ФАЗА 1: Обучение классификатора ==")

h1 = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=len(X_val) // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ── Фаза 2: Fine-tuning ───────────────────────────────────────────────────────
print("\n== ФАЗА 2: Fine-tuning (последние 30 слоёв) ==")

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(LR / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

h2 = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=15,
    validation_data=val_gen,
    validation_steps=len(X_val) // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ── Результаты ────────────────────────────────────────────────────────────────
print("\n== РЕЗУЛЬТАТЫ ==")
y_pred = model.predict(X_test, batch_size=BATCH_SIZE).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

# Графики
all_acc = h1.history['accuracy']     + h2.history['accuracy']
all_val = h1.history['val_accuracy'] + h2.history['val_accuracy']
ft_start = len(h1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, len(all_acc) + 1)

axes[0].plot(ep, all_acc, label='Train', color='steelblue', lw=2)
axes[0].plot(ep, all_val, label='Val',   color='darkorange', lw=2)
axes[0].axvline(ft_start, color='red', ls='--', alpha=0.5, label='Fine-tuning')
axes[0].set_title('Accuracy (v2)'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, h1.history['loss'] + h2.history['loss'], label='Train', color='steelblue', lw=2)
axes[1].plot(ep, h1.history['val_loss'] + h2.history['val_loss'], label='Val', color='darkorange', lw=2)
axes[1].axvline(ft_start, color='red', ls='--', alpha=0.5)
axes[1].set_title('Loss (v2)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history_v2.png'), dpi=150)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
ax.set_title('Confusion Matrix v2')
ax.set_xlabel('Предсказано'); ax.set_ylabel('Реально')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_v2.png'), dpi=150)

model.save(os.path.join(MODELS_DIR, 'best_model.h5'))

acc   = accuracy_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred, average='weighted')
mel_r = cm[4, 4] / cm[4].sum() if cm[4].sum() > 0 else 0

print(f"\n{'─' * 50}")
print(f"  Accuracy        : {acc:.3f} ({acc*100:.1f}%)")
print(f"  F1-Score        : {f1:.3f}  ({f1*100:.1f}%)")
print(f"  Recall меланомы : {mel_r:.3f} ({mel_r*100:.1f}%)")
print(f"{'─' * 50}")
print("\n  Финальная версия с лучшим результатом: python train_v4.py")
