"""
evaluate.py
-----------
Полная оценка качества обученной модели на тестовых данных.

Что делает этот скрипт:
  1. Загружает сохранённую модель из models/best_model.h5
  2. Формирует тестовую выборку (15% от датасета, та же разбивка что при обучении)
  3. Вычисляет метрики: Accuracy, F1, Precision, Recall по каждому классу
  4. Строит и сохраняет 4 графика:
     - Confusion Matrix
     - ROC-кривые
     - Распределение уверенности модели
  5. Выводит итоговые метрики в консоль

Запуск:
  python evaluate.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # без GUI — работает на любой машине
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Пути и настройки ──────────────────────────────────────────────────────────
MODEL_PATH  = "models/best_model.h5"
DATA_DIR    = "data"
RESULTS_DIR = "results"
IMG_SIZE    = 224
BATCH_SIZE  = 32

# Соответствие названий классов и числовых меток
CLASS_NAMES  = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
CLASS_LABELS = list(CLASS_NAMES.keys())

# Цвета для графиков — по одному на каждый класс
CLASS_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 55)
print("  ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
print("=" * 55)

# ── Загрузка модели ───────────────────────────────────────────────────────────
print("\nЗагрузка модели...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Модель не найдена: {MODEL_PATH}")
    print("   Сначала запустите: python train_v4.py")
    exit(1)

model = load_model(MODEL_PATH)
print(f"   ✓ Модель загружена ({model.count_params():,} параметров)")

# ── Загрузка метаданных и поиск изображений ───────────────────────────────────
print("\nПодготовка тестовых данных...")

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

def find_image(image_id):
    """Ищем изображение в обеих частях датасета."""
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(DATA_DIR, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df['path']  = df['image_id'].apply(find_image)
df['label'] = df['dx'].map(CLASS_NAMES)
df = df.dropna(subset=['path'])

# Воспроизводим ту же разбивку что использовалась при обучении (random_state=42)
# Это гарантирует что тестовые данные модель не видела во время обучения
_, df_test = train_test_split(
    df, test_size=0.15,
    stratify=df['label'],
    random_state=42
)

print(f"   Тестовых примеров: {len(df_test)}")
print(f"   Распределение классов в тесте:")
for cls, cnt in df_test['dx'].value_counts().items():
    print(f"   {cls:6s}: {cnt}")

# ── Загрузка изображений и предобработка ──────────────────────────────────────
print("\nЗагрузка изображений...")

X_test, y_test = [], []
for _, row in df_test.iterrows():
    try:
        img = load_img(row['path'], target_size=(IMG_SIZE, IMG_SIZE))
        # Используем preprocess_input от MobileNetV2 — приводит к [-1, 1]
        arr = preprocess_input(img_to_array(img))
        X_test.append(arr)
        y_test.append(row['label'])
    except Exception:
        pass  # пропускаем повреждённые файлы

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"   ✓ Загружено {len(X_test)} изображений")

# ── Получение предсказаний модели ─────────────────────────────────────────────
print("\nВычисление предсказаний...")
y_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
y_pred  = y_proba.argmax(axis=1)  # класс с максимальной вероятностью

# ── 1. Отчёт по классам (Precision, Recall, F1) ───────────────────────────────
print("\n" + "─" * 55)
print("  МЕТРИКИ ПО КЛАССАМ")
print("─" * 55)

report = classification_report(y_test, y_pred, target_names=CLASS_LABELS)
print(report)

# Сохраняем отчёт в текстовый файл
report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ ПО КЛАССИФИКАЦИИ — HAM10000\n")
    f.write("=" * 55 + "\n\n")
    f.write(report)

print(f"   ✓ {report_path}")

# ── 2. Матрица ошибок (Confusion Matrix) ──────────────────────────────────────
# Показывает в процентах сколько примеров каждого класса
# было правильно/неправильно классифицировано
fig, ax = plt.subplots(figsize=(10, 8))

cm = confusion_matrix(y_test, y_pred)
# Нормализуем по строкам — получаем recall каждого класса
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(
    cm_pct,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    xticklabels=CLASS_LABELS,
    yticklabels=CLASS_LABELS,
    ax=ax
)
ax.set_title('Матрица ошибок (% по строкам — Recall каждого класса)', fontsize=13)
ax.set_xlabel('Предсказанный класс')
ax.set_ylabel('Реальный класс')
plt.tight_layout()

out_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(out_path, dpi=150)
print(f"   ✓ {out_path}")

# ── 3. ROC-кривые (один класс против всех остальных) ──────────────────────────
# AUC (площадь под кривой) — чем ближе к 1.0, тем лучше модель
y_test_bin = label_binarize(y_test, classes=list(range(7)))

fig, ax = plt.subplots(figsize=(10, 8))

for i, (cls, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:.2f})')

# Диагональная линия — уровень случайного классификатора
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Случайный классификатор')
ax.set_xlabel('False Positive Rate (1 - Специфичность)')
ax.set_ylabel('True Positive Rate (Чувствительность)')
ax.set_title('ROC-кривые по каждому классу (один против всех)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()

out_path = os.path.join(RESULTS_DIR, 'roc_curves.png')
plt.savefig(out_path, dpi=150)
print(f"   ✓ {out_path}")

# ── 4. Распределение уверенности модели ───────────────────────────────────────
# Показывает насколько уверена модель когда она права и когда ошибается
fig, ax = plt.subplots(figsize=(10, 5))

correct_conf   = y_proba.max(axis=1)[y_pred == y_test]
incorrect_conf = y_proba.max(axis=1)[y_pred != y_test]

ax.hist(correct_conf,   bins=20, alpha=0.7, color='#2ecc71',
        label=f'Правильные предсказания ({len(correct_conf)})')
ax.hist(incorrect_conf, bins=20, alpha=0.7, color='#e74c3c',
        label=f'Ошибочные предсказания ({len(incorrect_conf)})')

ax.set_xlabel('Уверенность модели (максимальная вероятность softmax)')
ax.set_ylabel('Количество примеров')
ax.set_title('Распределение уверенности модели')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

out_path = os.path.join(RESULTS_DIR, 'confidence_distribution.png')
plt.savefig(out_path, dpi=150)
print(f"   ✓ {out_path}")

# ── Итоговые метрики ──────────────────────────────────────────────────────────
acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='weighted')
# Recall меланомы — особо важная метрика: сколько реальных меланом нашли
mel_recall = cm[4, 4] / cm[4].sum() if cm[4].sum() > 0 else 0

print("\n" + "─" * 55)
print("  ИТОГОВЫЕ МЕТРИКИ")
print("─" * 55)
print(f"  Accuracy (общая точность) : {acc:.3f}  ({acc*100:.1f}%)")
print(f"  F1-Score (взвешенный)     : {f1:.3f}  ({f1*100:.1f}%)")
print(f"  Recall меланомы           : {mel_recall:.3f}  ({mel_recall*100:.1f}%)")
print("─" * 55)
print(f"\n  Все результаты сохранены в: {RESULTS_DIR}/")
