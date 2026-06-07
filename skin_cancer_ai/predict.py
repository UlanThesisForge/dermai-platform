"""
predict.py
----------
Предсказание диагноза для одного дерматоскопического изображения.
Дополнительно строит тепловую карту Grad-CAM — визуализацию того,
на какие области изображения модель обращала внимание при постановке диагноза.

Запуск:
  python predict.py <путь_к_изображению.jpg>

Пример:
  python predict.py data/HAM10000_images_part_1/ISIC_0025964.jpg
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # без GUI
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Пути и настройки ──────────────────────────────────────────────────────────
MODEL_PATH  = "models/best_model.h5"
IMG_SIZE    = 224
RESULTS_DIR = "results"

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Описания классов для вывода в консоль
CLASS_DESCRIPTIONS = {
    'akiec': 'Актинический кератоз / внутриэпителиальная карцинома',
    'bcc':   'Базальноклеточная карцинома',
    'bkl':   'Доброкачественный кератоз',
    'df':    'Дерматофиброма',
    'mel':   'Меланома ⚠️  (необходима срочная консультация врача)',
    'nv':    'Меланоцитарный невус (родинка)',
    'vasc':  'Сосудистое поражение'
}

# Классы с высоким риском — выводим предупреждение
HIGH_RISK_CLASSES = ['mel', 'bcc', 'akiec']


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Вычисляет тепловую карту Grad-CAM для заданного изображения.

    Grad-CAM (Gradient-weighted Class Activation Mapping) показывает,
    какие области изображения наиболее сильно повлияли на предсказание модели.
    Красные области — высокая важность, синие — низкая.

    Параметры:
        img_array          — предобработанное изображение [1, H, W, 3]
        model              — загруженная модель Keras
        last_conv_layer_name — название последнего свёрточного слоя
        pred_index         — индекс класса (по умолчанию — предсказанный класс)

    Возвращает:
        heatmap — тепловая карта в диапазоне [0, 1]
    """
    # Создаём вспомогательную модель с двумя выходами:
    # активации последнего свёрточного слоя и финальные предсказания
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        # Выбираем уверенность для нужного класса
        class_score = preds[:, pred_index]

    # Вычисляем градиент класса по карте признаков
    grads = tape.gradient(class_score, last_conv_output)

    # Усредняем градиенты по пространственным измерениям (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Взвешиваем карту признаков по важности каждого канала
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Нормализуем в диапазон [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def find_last_conv_layer(model):
    """Автоматически находит последний свёрточный слой в модели."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("Свёрточный слой не найден в модели")


def overlay_heatmap(heatmap, img_normalized, alpha=0.4):
    """
    Накладывает тепловую карту на оригинальное изображение.

    Параметры:
        heatmap        — карта Grad-CAM [H', W']
        img_normalized — нормализованное изображение [H, W, 3] в диапазоне [0, 1]
        alpha          — прозрачность тепловой карты (0.0 — невидима, 1.0 — полностью)
    """
    import cv2

    # Масштабируем карту до размера оригинального изображения
    h, w = img_normalized.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Применяем цветовую карту jet (синий → зелёный → красный)
    heatmap_colored = mpl_cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Накладываем тепловую карту на оригинал
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    overlay   = (img_uint8 * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return overlay


# ── Основная функция предсказания ────────────────────────────────────────────

def predict_image(image_path, show_gradcam=True):
    """
    Выполняет предсказание для одного изображения.

    Параметры:
        image_path  — путь к файлу изображения (.jpg, .png)
        show_gradcam — строить ли тепловую карту Grad-CAM (по умолчанию True)

    Возвращает словарь:
        {'class': str, 'confidence': float, 'all_probs': dict}
    """
    # Проверяем наличие модели и файла
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        print("   Сначала запустите обучение: python train_v4.py")
        return None

    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return None

    print(f"\n{'=' * 55}")
    print(f"  АНАЛИЗ: {os.path.basename(image_path)}")
    print(f"{'=' * 55}")

    # Загрузка модели
    print("\nЗагрузка модели...")
    model = load_model(MODEL_PATH)

    # Предобработка изображения
    # Важно: используем preprocess_input от MobileNetV2 (не просто /255.0)
    img       = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)                       # [0, 255]
    img_norm  = img_array / 255.0                       # [0, 1] — для отображения
    img_prep  = preprocess_input(img_array.copy())      # [-1, 1] — для модели
    img_batch = np.expand_dims(img_prep, axis=0)        # добавляем batch-размерность

    # Получаем предсказание
    print("Выполнение предсказания...")
    probs      = model.predict(img_batch, verbose=0)[0]
    top_idx    = probs.argmax()
    top_class  = CLASS_LABELS[top_idx]
    confidence = probs[top_idx]

    # ── Вывод результата ─────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  РЕЗУЛЬТАТ:")
    print(f"  Диагноз     : {top_class.upper()}")
    print(f"  Описание    : {CLASS_DESCRIPTIONS[top_class]}")
    print(f"  Уверенность : {confidence:.1%}")
    print(f"{'─' * 55}")

    print("\n  Вероятности по всем классам:")
    for i, (cls, prob) in enumerate(zip(CLASS_LABELS, probs)):
        bar    = '█' * int(prob * 30)
        marker = ' ← предсказание' if i == top_idx else ''
        print(f"  {cls:6s}: {prob:5.1%}  {bar}{marker}")

    # Предупреждение при высоком риске
    if top_class in HIGH_RISK_CLASSES:
        print(f"\n  ⚠️  ВНИМАНИЕ: обнаружено потенциально злокачественное образование!")
        print(f"     Рекомендуется консультация дерматолога.")

    # ── Grad-CAM визуализация ─────────────────────────────────────────────────
    if show_gradcam:
        print("\nПостроение тепловой карты Grad-CAM...")
        try:
            last_conv = find_last_conv_layer(model)
            heatmap   = make_gradcam_heatmap(img_batch, model, last_conv, top_idx)
            overlay   = overlay_heatmap(heatmap, img_norm)

            # Три панели: оригинал | тепловая карта | наложение
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                f"Диагноз: {top_class.upper()} (уверенность {confidence:.1%})\n"
                f"{CLASS_DESCRIPTIONS[top_class]}",
                fontsize=13, fontweight='bold'
            )

            axes[0].imshow(img_norm)
            axes[0].set_title('Исходное изображение')
            axes[0].axis('off')

            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM\n(красный = важная область)')
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('Наложение на оригинал')
            axes[2].axis('off')

            os.makedirs(RESULTS_DIR, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path  = os.path.join(RESULTS_DIR, f"gradcam_{base_name}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Тепловая карта сохранена: {out_path}")

        except Exception as e:
            print(f"   ⚠️  Не удалось построить Grad-CAM: {e}")

    print(f"\n  ⚕️  Система является вспомогательным инструментом скрининга.")
    print(f"     Окончательный диагноз устанавливает только врач.\n")

    return {
        'class':      top_class,
        'confidence': float(confidence),
        'all_probs':  {cls: float(p) for cls, p in zip(CLASS_LABELS, probs)}
    }


# ── Точка входа ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py <путь_к_изображению.jpg>")
        print("Пример:        python predict.py data/HAM10000_images_part_1/ISIC_0025964.jpg")
        sys.exit(1)

    predict_image(sys.argv[1])
