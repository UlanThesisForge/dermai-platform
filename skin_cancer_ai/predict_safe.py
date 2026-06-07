"""
predict_safe.py
---------------
Предсказание диагноза с пониженным порогом срабатывания для меланомы.

Отличие от predict.py:
  Стандартный классификатор выбирает класс с максимальной вероятностью.
  Это значит меланома определяется только если её вероятность выше всех остальных.

  predict_safe.py дополнительно проверяет: если вероятность меланомы
  превышает MEL_THRESHOLD (20%) — выдаётся предупреждение, даже если
  топ-предсказание другое. Это снижает количество пропущенных случаев
  за счёт небольшого роста ложных тревог.

Запуск:
  python predict_safe.py <путь_к_изображению.jpg>

Пример:
  python predict_safe.py data/HAM10000_images_part_1/ISIC_0025964.jpg
"""

import sys
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Настройки ─────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/best_model.h5"
CLASS_LABELS  = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MEL_INDEX     = 4       # индекс меланомы в списке классов
MEL_THRESHOLD = 0.20    # порог срабатывания: если P(mel) > 20% — предупреждаем

# ── Проверка аргументов ───────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Использование: python predict_safe.py <путь_к_изображению.jpg>")
    print("Пример:        python predict_safe.py data/HAM10000_images_part_1/ISIC_0025964.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(MODEL_PATH):
    print(f"❌ Модель не найдена: {MODEL_PATH}")
    print("   Запустите обучение: python train_v4.py")
    sys.exit(1)

if not os.path.exists(image_path):
    print(f"❌ Изображение не найдено: {image_path}")
    sys.exit(1)

# ── Загрузка модели и предобработка изображения ───────────────────────────────
print(f"\nАнализ изображения: {os.path.basename(image_path)}")

model = load_model(MODEL_PATH)

img   = load_img(image_path, target_size=(224, 224))
x     = img_to_array(img)          # [0, 255]
x     = preprocess_input(x)        # [-1, 1] — правильная нормализация для MobileNetV2
x     = np.expand_dims(x, axis=0)  # добавляем batch-размерность: [1, 224, 224, 3]

# ── Предсказание ──────────────────────────────────────────────────────────────
probs     = model.predict(x, verbose=0)[0]
top_idx   = probs.argmax()
top_class = CLASS_LABELS[top_idx]
mel_prob  = probs[MEL_INDEX]

# ── Вывод результата ──────────────────────────────────────────────────────────
print(f"\n{'═' * 50}")
print(f"  Топ-предсказание: {top_class.upper()} ({probs[top_idx]:.1%})")

# Специальная проверка меланомы — независимо от топ-предсказания
if mel_prob >= MEL_THRESHOLD:
    print(f"\n  ⚠️  ВНИМАНИЕ: вероятность меланомы {mel_prob:.1%}")
    print(f"     Превышен порог {MEL_THRESHOLD:.0%} — рекомендуется")
    print(f"     срочная консультация дерматолога!")
else:
    print(f"\n  Вероятность меланомы: {mel_prob:.1%}")
    print(f"  Ниже порога {MEL_THRESHOLD:.0%} — признаков меланомы не выявлено")

# Все вероятности по убыванию
print(f"\n  Вероятности по всем классам:")
for cls, prob in sorted(zip(CLASS_LABELS, probs), key=lambda x: -x[1]):
    bar    = '█' * int(prob * 30)
    marker = ' ← топ' if cls == top_class else ''
    print(f"  {cls:6s}: {prob:5.1%}  {bar}{marker}")

print(f"\n  ⚕️  Система является вспомогательным инструментом.")
print(f"     Окончательный диагноз ставит только врач.")
print(f"{'═' * 50}\n")
