"""
train_v4.py
-----------
Обучение модели классификации кожных заболеваний на датасете HAM10000.

Архитектура: MobileNetV2 (предобученная на ImageNet) + собственный классификатор
Подход: Transfer Learning в два этапа
  - Фаза 1: обучаем только классификатор, backbone заморожен
  - Фаза 2: fine-tuning — размораживаем последние 30 слоёв backbone

Ключевые технические решения:
  - preprocess_input от MobileNetV2 (нормализация в [-1, 1])
  - Oversampling редких классов до среднего значения по датасету
  - Обратные веса классов для дополнительной борьбы с дисбалансом
  - EarlyStopping — останавливает обучение при отсутствии прогресса

Запуск:
  python train_v4.py
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

import tensorflow as tf
print(f"TensorFlow версия: {tf.__version__}")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("  ОБУЧЕНИЕ МОДЕЛИ ДИАГНОСТИКИ РАКА КОЖИ")
print("  Архитектура: MobileNetV2 + Transfer Learning")
print("=" * 60)

# ── Глобальные настройки ──────────────────────────────────────────────────────
IMG_SIZE    = 224   # размер входного изображения (224x224 пикселей)
BATCH_SIZE  = 32    # количество изображений за одну итерацию
DATA_DIR    = "data"
RESULTS_DIR = "results"
MODELS_DIR  = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# Словарь: название класса → числовая метка
CLASS_NAMES  = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
CLASS_LABELS = list(CLASS_NAMES.keys())

# ── Шаг 1: Загрузка метаданных ────────────────────────────────────────────────
print("\n[1/5] Загрузка метаданных датасета...")

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

def find_image(image_id):
    """Ищем файл изображения в обеих частях датасета."""
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']:
        path = os.path.join(DATA_DIR, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df['path']  = df['image_id'].apply(find_image)
df['label'] = df['dx'].map(CLASS_NAMES)
df = df.dropna(subset=['path'])  # убираем записи без изображений

print(f"   Найдено изображений: {len(df)}")
print(f"   Распределение классов:")
for cls in CLASS_LABELS:
    n = len(df[df['dx'] == cls])
    print(f"   {cls:6s}: {n:5d}")

# ── Шаг 2: Разбивка на Train / Validation / Test ──────────────────────────────
# Важно: random_state=42 фиксирует разбивку — она должна быть одинаковой
# при обучении и при оценке (evaluate.py)
print("\n[2/5] Разбивка данных...")

tr_df, te_df = train_test_split(
    df, test_size=0.15,
    stratify=df['label'], random_state=42
)
tr_df, va_df = train_test_split(
    tr_df, test_size=0.176,
    stratify=tr_df['label'], random_state=42
)
# 0.176 от 85% ≈ 15% от всего датасета

print(f"   Train : {len(tr_df)} изображений (70%)")
print(f"   Val   : {len(va_df)} изображений (15%)")
print(f"   Test  : {len(te_df)} изображений (15%)")

# ── Шаг 3: Oversampling редких классов ───────────────────────────────────────
# Проблема: невус (nv) имеет 6705 фото, дерматофиброма (df) — только 115.
# Решение: дублируем редкие классы до среднего значения по датасету.
# Это мягче чем дублировать до максимума — не перегружает модель повторами.
print("\n[3/5] Балансировка классов (oversampling)...")

avg_count = int(tr_df['dx'].value_counts().mean())
print(f"   Среднее по классам: {avg_count} → до этого значения дополняем редкие классы")

balanced_parts = []
for cls in CLASS_LABELS:
    cls_data = tr_df[tr_df['dx'] == cls].copy()
    original_count = len(cls_data)

    if original_count < avg_count:
        # Дублируем записи нужное количество раз
        repeats  = avg_count // original_count + 1
        cls_data = pd.concat([cls_data] * repeats).head(avg_count)
        print(f"   {cls:6s}: {original_count} → {len(cls_data)} (дополнено)")
    else:
        print(f"   {cls:6s}: {original_count} (без изменений)")

    balanced_parts.append(cls_data)

# Перемешиваем итоговый датафрейм
tr_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   Итого в Train после oversampling: {len(tr_df)}")

# ── Шаг 4: Загрузка изображений ───────────────────────────────────────────────
# КРИТИЧНО: используем preprocess_input от MobileNetV2
# Эта функция принимает пиксели [0, 255] и нормализует их в [-1, 1]
# Именно такой диапазон ожидает предобученная MobileNetV2
print("\n[4/5] Загрузка и предобработка изображений...")

def load_dataset(dataframe, description=""):
    """
    Загружает изображения из датафрейма и применяет предобработку MobileNetV2.

    Возвращает:
        X — массив numpy формата [N, 224, 224, 3], значения в [-1, 1]
        y — массив меток классов [N]
    """
    X, y = [], []
    total = len(dataframe)

    for i, (_, row) in enumerate(dataframe.iterrows()):
        if i % 500 == 0:
            print(f"   {description}: {i}/{total}...")
        try:
            img = load_img(row['path'], target_size=(IMG_SIZE, IMG_SIZE))
            arr = img_to_array(img)       # загружаем в диапазоне [0, 255]
            arr = preprocess_input(arr)   # нормализуем в [-1, 1] для MobileNetV2
            X.append(arr)
            y.append(int(row['label']))
        except Exception:
            pass  # пропускаем повреждённые файлы

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

X_tr, y_tr = load_dataset(tr_df, "Train")
X_va, y_va = load_dataset(va_df, "Val")
X_te, y_te = load_dataset(te_df, "Test")

# Проверяем что нормализация прошла правильно
print(f"\n   Train : {X_tr.shape}, min={X_tr.min():.2f}, max={X_tr.max():.2f}")
print(f"   Val   : {X_va.shape}")
print(f"   Test  : {X_te.shape}")

# One-hot encoding меток для обучения (нужен для categorical_crossentropy)
y_tr_cat = to_categorical(y_tr, num_classes=7)
y_va_cat = to_categorical(y_va, num_classes=7)

# ── Веса классов ──────────────────────────────────────────────────────────────
# Дополнительно к oversampling — даём редким классам больший вес в функции потерь
# Формула: weight_i = N_total / (7 * N_i)
counts  = np.bincount(y_tr)
total   = len(y_tr)
weights = {i: total / (7 * c) for i, c in enumerate(counts)}

print("\n   Веса классов для функции потерь:")
for i, cls in enumerate(CLASS_LABELS):
    print(f"   {cls:6s}: {weights[i]:.3f}")

# ── Шаг 5: Построение модели ──────────────────────────────────────────────────
print("\n[5/5] Построение модели MobileNetV2...")

# Загружаем MobileNetV2 без финального классификатора (include_top=False)
# weights='imagenet' — используем веса предобученные на 1.2 млн изображений
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # замораживаем backbone на первом этапе

# Добавляем свой классификатор для 7 классов кожных заболеваний
x   = base_model.output
x   = GlobalAveragePooling2D()(x)   # усредняем карты признаков
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.35)(x)              # регуляризация — снижает переобучение
out = Dense(7, activation='softmax')(x)  # 7 классов, вероятности в сумме = 1

model = Model(inputs=base_model.input, outputs=out)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable_params = sum(p.numpy().size for p in model.trainable_weights)
print(f"   Обучаемых параметров: {trainable_params:,}")
print(f"   Замороженных (backbone): {model.count_params() - trainable_params:,}")

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    # Останавливаем обучение если val_accuracy не растёт 10 эпох подряд
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Уменьшаем learning rate если val_loss не улучшается 4 эпохи
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    # Сохраняем лучшую модель по val_accuracy
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ── Фаза 1: обучение только классификатора ────────────────────────────────────
print("\n" + "=" * 60)
print("  ФАЗА 1: Обучение классификатора (backbone заморожен)")
print("  Learning rate: 1e-3 | Максимум эпох: 25")
print("=" * 60)

history_1 = model.fit(
    X_tr, y_tr_cat,
    batch_size=BATCH_SIZE,
    epochs=25,
    validation_data=(X_va, y_va_cat),
    class_weight=weights,
    callbacks=callbacks,
    verbose=1
)

# ── Фаза 2: fine-tuning последних слоёв backbone ──────────────────────────────
# Размораживаем последние 30 слоёв — они обучаются под медицинские изображения
# Learning rate снижен в 10 раз чтобы не разрушить предобученные веса
print("\n" + "=" * 60)
print("  ФАЗА 2: Fine-tuning (последние 30 слоёв backbone)")
print("  Learning rate: 1e-4 | Максимум эпох: 20")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False  # первые слои остаются замороженными

model.compile(
    optimizer=Adam(learning_rate=1e-4),  # меньший LR для fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    X_tr, y_tr_cat,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(X_va, y_va_cat),
    class_weight=weights,
    callbacks=callbacks,
    verbose=1
)

# ── Оценка на тестовых данных ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  РЕЗУЛЬТАТЫ НА ТЕСТОВЫХ ДАННЫХ")
print("=" * 60)

y_pred  = model.predict(X_te, batch_size=BATCH_SIZE).argmax(axis=1)
report  = classification_report(y_te, y_pred, target_names=CLASS_LABELS)
print("\n" + report)

# Сохраняем отчёт
with open(os.path.join(RESULTS_DIR, 'report_v4.txt'), 'w', encoding='utf-8') as f:
    f.write("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ — MobileNetV2\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)

# ── Графики ───────────────────────────────────────────────────────────────────

# 1. Матрица ошибок
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_te, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax
)
ax.set_title('Матрица ошибок (Confusion Matrix)', fontsize=13, fontweight='bold')
ax.set_xlabel('Предсказанный класс')
ax.set_ylabel('Реальный класс')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_v4.png'), dpi=150)
print("   ✓ results/confusion_matrix_v4.png")

# 2. Графики обучения (Accuracy и Loss по эпохам)
# Объединяем историю двух фаз в один непрерывный график
acc_train = history_1.history['accuracy']     + history_2.history['accuracy']
acc_val   = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
loss_train= history_1.history['loss']         + history_2.history['loss']
loss_val  = history_1.history['val_loss']     + history_2.history['val_loss']
ft_start  = len(history_1.history['accuracy'])  # эпоха начала fine-tuning
epochs    = range(1, len(acc_train) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, acc_train, label='Train', color='steelblue', lw=2)
axes[0].plot(epochs, acc_val,   label='Val',   color='darkorange', lw=2)
axes[0].axvline(ft_start, color='red', ls='--', alpha=0.6, label='Начало fine-tuning')
axes[0].set_title('Точность (Accuracy) по эпохам', fontsize=12)
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, loss_train, label='Train', color='steelblue', lw=2)
axes[1].plot(epochs, loss_val,   label='Val',   color='darkorange', lw=2)
axes[1].axvline(ft_start, color='red', ls='--', alpha=0.6, label='Начало fine-tuning')
axes[1].set_title('Функция потерь (Loss) по эпохам', fontsize=12)
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history_v4.png'), dpi=150)
print("   ✓ results/training_history_v4.png")

# Сохраняем финальную модель
model.save(os.path.join(MODELS_DIR, 'best_model.h5'))
print(f"   ✓ models/best_model.h5")

# ── Итоговые метрики ──────────────────────────────────────────────────────────
acc    = accuracy_score(y_te, y_pred)
f1     = f1_score(y_te, y_pred, average='weighted')
mel_r  = cm[4, 4] / cm[4].sum() if cm[4].sum() > 0 else 0

print(f"\n{'═' * 55}")
print(f"  ИТОГ")
print(f"{'═' * 55}")
print(f"  Accuracy (общая точность) : {acc:.3f}  ({acc*100:.1f}%)")
print(f"  F1-Score (взвешенный)     : {f1:.3f}  ({f1*100:.1f}%)")
print(f"  Recall меланомы           : {mel_r:.3f} ({mel_r*100:.1f}%)")
print(f"{'═' * 55}")
print(f"\n  Следующий шаг: python app/app.py")
