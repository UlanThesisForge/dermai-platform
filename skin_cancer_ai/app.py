"""
app.py
------
Веб-приложение для диагностики кожных заболеваний.
Написано на Flask — простом Python-фреймворке для веб-сервисов.

Две вкладки:
  1. Диагностика  — загружаешь фото, получаешь предсказание модели
  2. Тестирование — автоматически проверяет модель на случайных
                    примерах из датасета HAM10000

Как запустить:
  python app/app.py

Открыть в браузере:
  http://localhost:5000
"""

import os
import sys
import uuid
import numpy as np
import pandas as pd

# Добавляем корневую папку проекта в sys.path чтобы найти папку models/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # скрываем лишние сообщения TensorFlow

from flask import Flask, request, render_template_string, jsonify, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Инициализация приложения ──────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # максимум 16 МБ на загрузку
app.config['UPLOAD_FOLDER']      = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Классы и их характеристики ────────────────────────────────────────────────
CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Для каждого класса: полное название, уровень риска и цвет в интерфейсе
CLASS_INFO = {
    'akiec': {'name': 'Актинический кератоз',          'risk': 'medium', 'color': '#f39c12'},
    'bcc':   {'name': 'Базальноклеточная карцинома',   'risk': 'high',   'color': '#e74c3c'},
    'bkl':   {'name': 'Доброкачественный кератоз',     'risk': 'low',    'color': '#2ecc71'},
    'df':    {'name': 'Дерматофиброма',                 'risk': 'low',    'color': '#2ecc71'},
    'mel':   {'name': 'Меланома',                       'risk': 'high',   'color': '#c0392b'},
    'nv':    {'name': 'Меланоцитарный невус (родинка)', 'risk': 'low',    'color': '#27ae60'},
    'vasc':  {'name': 'Сосудистое поражение',           'risk': 'medium', 'color': '#e67e22'},
}

# ── Загрузка модели ───────────────────────────────────────────────────────────
# Модель загружается один раз при старте сервера и хранится в памяти
MODEL = None

def get_model():
    """Возвращает загруженную модель. Загружает при первом вызове."""
    global MODEL
    if MODEL is None:
        # Ищем файл модели в нескольких возможных местах
        for model_path in [
            os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.h5'),
            os.path.join('models', 'best_model.h5'),
        ]:
            if os.path.exists(model_path):
                MODEL = load_model(model_path)
                print(f"✓ Модель загружена: {model_path}")
                break

        if MODEL is None:
            print("⚠️  Модель не найдена. Запустите: python train_v4.py")

    return MODEL


def find_image(image_id, data_dir='data'):
    """Ищет файл изображения по ID в обеих частях датасета."""
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']:
        path = os.path.join(data_dir, folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None


def run_prediction(filepath):
    """
    Выполняет предсказание для изображения по указанному пути.
    Возвращает словарь с классом, уверенностью и вероятностями всех классов.
    """
    model = get_model()
    if model is None:
        return None

    # Загружаем и предобрабатываем изображение
    img   = load_img(filepath, target_size=(224, 224))
    x     = img_to_array(img)      # [0, 255]
    x     = preprocess_input(x)    # [-1, 1] — нужно для MobileNetV2
    x     = np.expand_dims(x, 0)   # добавляем batch-размерность

    probs = model.predict(x, verbose=0)[0]
    top_i = int(probs.argmax())

    return {
        'class':      CLASS_LABELS[top_i],
        'confidence': float(probs[top_i]),
        'all_probs':  {cls: float(p) for cls, p in zip(CLASS_LABELS, probs)}
    }


# ── HTML шаблон интерфейса ────────────────────────────────────────────────────
# Весь интерфейс написан в одной строке для простоты деплоя
# В реальном проекте лучше выносить в отдельные .html файлы в папке templates/
HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Диагностика рака кожи</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f0f4f8; color:#2d3748; }

/* Шапка */
.header { background:linear-gradient(135deg,#1a365d,#2b6cb0); color:white; padding:20px 40px; }
.header h1 { font-size:22px; font-weight:600; }
.header p  { font-size:13px; opacity:.8; margin-top:4px; }

/* Вкладки */
.tabs { display:flex; border-bottom:2px solid #e2e8f0; background:white;
        padding:0 30px; position:sticky; top:0; z-index:10;
        box-shadow:0 1px 3px rgba(0,0,0,.06); }
.tab { padding:14px 24px; cursor:pointer; font-size:14px; font-weight:500;
       color:#718096; border-bottom:3px solid transparent; margin-bottom:-2px; transition:all .2s; }
.tab.active { color:#3182ce; border-bottom-color:#3182ce; }
.tab:hover  { color:#2d3748; }
.tab-content { display:none; }
.tab-content.active { display:block; }

/* Карточки */
.container { max-width:960px; margin:30px auto; padding:0 20px; }
.card { background:white; border-radius:12px; padding:28px;
        box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:20px; }
.section-title { font-size:16px; font-weight:600; color:#4a5568; margin-bottom:14px; }

/* Зона загрузки */
.upload-zone { border:2px dashed #bee3f8; border-radius:10px; padding:40px;
               text-align:center; cursor:pointer; background:#ebf8ff; transition:all .2s; }
.upload-zone:hover { border-color:#3182ce; background:#bee3f8; }
.upload-icon { font-size:48px; margin-bottom:10px; }
#fileInput { display:none; }

/* Кнопки */
.btn { padding:11px 26px; background:#3182ce; color:white; border:none;
       border-radius:8px; font-size:14px; cursor:pointer; font-weight:500; transition:background .2s; }
.btn:hover    { background:#2c5282; }
.btn:disabled { background:#a0aec0; cursor:not-allowed; }
.btn-gray { background:#718096; }
.btn-gray:hover { background:#4a5568; }

/* Предпросмотр */
.preview-wrap { display:none; text-align:center; margin:16px 0; }
#preview { max-width:280px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,.15); }

/* Блок результата */
.result-box { display:none; }
.diag-main { display:flex; align-items:center; gap:16px; padding:20px; border-radius:10px; margin-bottom:18px; }
.diag-class { font-size:30px; font-weight:700; }
.diag-name  { font-size:15px; opacity:.9; margin-top:4px; }
.risk-badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; margin-top:8px; }
.risk-high   { background:#fed7d7; color:#9b2c2c; }
.risk-medium { background:#fefcbf; color:#744210; }
.risk-low    { background:#c6f6d5; color:#22543d; }
.conf-big { font-size:38px; font-weight:700; }

/* Полоски вероятностей */
.prob-row    { display:flex; align-items:center; gap:10px; margin:7px 0; font-size:13px; }
.prob-label  { width:48px; font-weight:500; }
.prob-bar-bg { flex:1; background:#edf2f7; border-radius:4px; height:9px; }
.prob-bar    { height:9px; border-radius:4px; transition:width .7s ease; }
.prob-pct    { width:44px; text-align:right; color:#718096; }

/* Информационные блоки */
.warning-box { background:#fff5f5; border:1px solid #feb2b2; border-radius:8px;
               padding:13px 16px; margin-top:14px; color:#c53030; font-size:13px; }
.info-box    { background:#ebf8ff; border:1px solid #bee3f8; border-radius:8px;
               padding:13px 16px; margin-top:10px; color:#2c5282; font-size:13px; }

/* Загрузчик */
.loading  { display:none; text-align:center; padding:28px; }
.spinner  { width:38px; height:38px; border:4px solid #bee3f8; border-top-color:#3182ce;
            border-radius:50%; animation:spin 1s linear infinite; margin:0 auto 10px; }
.spinner-inline { display:inline-block; width:16px; height:16px; border:2px solid #bee3f8;
                  border-top-color:#3182ce; border-radius:50%; animation:spin 1s linear infinite;
                  vertical-align:middle; margin-right:6px; }
@keyframes spin { to { transform:rotate(360deg); } }

/* Сетка тестирования */
.test-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; }
.test-card { border:1px solid #e2e8f0; border-radius:10px; overflow:hidden;
             background:white; transition:box-shadow .2s; }
.test-card:hover { box-shadow:0 4px 12px rgba(0,0,0,.1); }
.test-img  { width:100%; height:160px; object-fit:cover; display:block; }
.test-body { padding:12px 14px; }
.test-id   { font-size:11px; color:#a0aec0; margin-bottom:6px; }
.test-row  { display:flex; justify-content:space-between; align-items:center; margin:4px 0; font-size:13px; }
.test-label{ color:#718096; }
.test-real { font-weight:600; }
.test-pred { font-weight:700; padding:2px 10px; border-radius:12px; font-size:12px; }
.correct   { background:#c6f6d5; color:#22543d; }
.incorrect { background:#fed7d7; color:#9b2c2c; }

/* Мини-полоски в карточках тестирования */
.mini-bar-row { display:flex; align-items:center; gap:6px; margin:3px 0; font-size:11px; }
.mini-label   { width:36px; color:#4a5568; }
.mini-bg      { flex:1; background:#edf2f7; border-radius:3px; height:6px; }
.mini-fill    { height:6px; border-radius:3px; }
.mini-pct     { width:34px; text-align:right; color:#718096; }

/* Сводная статистика */
.summary-row  { display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap; }
.summary-card { flex:1; min-width:120px; background:white; border-radius:10px;
                padding:16px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,.08); }
.summary-val  { font-size:28px; font-weight:700; }
.summary-lbl  { font-size:12px; color:#718096; margin-top:4px; }
</style>
</head>
<body>

<div class="header">
  <h1>Система ранней диагностики рака кожи</h1>
  <p>MobileNetV2 · Датасет HAM10000 · Точность 80.8% · Дипломный проект</p>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('diagnose', this)">🔬 Диагностика</div>
  <div class="tab"        onclick="switchTab('test', this)">📊 Тестирование модели</div>
</div>

<!-- Вкладка 1: Диагностика -->
<div id="tab-diagnose" class="tab-content active">
<div class="container">

  <div class="card">
    <p class="section-title">Загрузить дерматоскопическое изображение</p>
    <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
      <div class="upload-icon">🔬</div>
      <p style="color:#4a5568;font-size:15px">Нажмите или перетащите изображение сюда</p>
      <p style="color:#718096;font-size:13px;margin-top:6px">Форматы: JPG, PNG · Максимум 16 МБ</p>
    </div>
    <input type="file" id="fileInput" accept="image/*">

    <div class="preview-wrap" id="previewWrap">
      <img id="preview" src="" alt="Предпросмотр изображения">
      <p style="color:#718096;font-size:13px;margin-top:8px" id="fileName"></p>
    </div>

    <div style="text-align:center;margin-top:18px">
      <button class="btn" id="analyzeBtn" onclick="analyze()" disabled>
        Анализировать
      </button>
    </div>
  </div>

  <!-- Индикатор загрузки -->
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p style="color:#718096">Нейронная сеть анализирует изображение...</p>
  </div>

  <!-- Результат -->
  <div class="result-box card" id="result">
    <p class="section-title">Результат диагностики</p>

    <div class="diag-main" id="diagMain">
      <div style="flex:1">
        <div class="diag-class" id="diagClass">—</div>
        <div class="diag-name"  id="diagName">—</div>
        <span class="risk-badge" id="riskBadge">—</span>
      </div>
      <div style="text-align:right">
        <div style="color:#718096;font-size:13px">Уверенность модели</div>
        <div class="conf-big" id="confidence">—</div>
      </div>
    </div>

    <p class="section-title">Вероятности по всем классам</p>
    <div id="probBars"></div>

    <div class="warning-box" id="warningBox" style="display:none">
      ⚠️ Обнаружено потенциально злокачественное образование.
      Необходима срочная консультация дерматолога!
    </div>
    <div class="info-box">
      ℹ️ Система является вспомогательным инструментом скрининга.
      Окончательный диагноз устанавливает только квалифицированный врач.
    </div>

    <div style="text-align:center;margin-top:16px">
      <button class="btn btn-gray" onclick="resetForm()">Новый анализ</button>
    </div>
  </div>

</div>
</div>

<!-- Вкладка 2: Тестирование -->
<div id="tab-test" class="tab-content">
<div class="container">

  <div class="card">
    <p class="section-title">Автоматическое тестирование на примерах из HAM10000</p>
    <p style="color:#718096;font-size:13px;margin-bottom:16px">
      Каждый запуск выбирает случайные фото по одному из каждого класса
      и сравнивает предсказание модели с реальным диагнозом.
    </p>
    <button class="btn" id="runTestBtn" onclick="runTests()">
      ▶ Запустить тестирование
    </button>
  </div>

  <div id="summarySection" style="display:none">
    <div class="summary-row" id="summaryRow"></div>
  </div>

  <div id="testGrid" class="test-grid"></div>

</div>
</div>

<script>
// Данные о классах переданы из Python через Jinja2
const classInfo = {{ class_info | tojson }};
let selectedFile = null;

// Переключение вкладок
function switchTab(name, el) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  el.classList.add('active');
}

// Обработка выбора файла
document.getElementById('fileInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = function(ev) {
    document.getElementById('preview').src = ev.target.result;
    document.getElementById('previewWrap').style.display = 'block';
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('result').style.display = 'none';
  };
  reader.readAsDataURL(file);
});

// Отправка фото на сервер и получение предсказания
async function analyze() {
  if (!selectedFile) return;

  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').style.display  = 'none';
  document.getElementById('analyzeBtn').disabled = true;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data     = await response.json();

    if (data.error) {
      alert('Ошибка: ' + data.error);
      return;
    }
    showResult(data);
  } catch(e) {
    alert('Ошибка соединения с сервером');
  } finally {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = false;
  }
}

// Отображение результата предсказания
function showResult(data) {
  const info   = classInfo[data.class];
  const riskRu = { high: 'Высокий риск', medium: 'Средний риск', low: 'Низкий риск' };

  document.getElementById('diagClass').textContent    = data.class.toUpperCase();
  document.getElementById('diagName').textContent     = info.name;
  document.getElementById('confidence').textContent   = (data.confidence * 100).toFixed(1) + '%';

  // Цветовое оформление блока диагноза
  const diagMain = document.getElementById('diagMain');
  diagMain.style.background = info.color + '18';
  diagMain.style.border     = '1px solid ' + info.color + '44';

  const badge = document.getElementById('riskBadge');
  badge.textContent = riskRu[info.risk];
  badge.className   = 'risk-badge risk-' + info.risk;

  // Полоски вероятностей — сортируем по убыванию
  const barsDiv = document.getElementById('probBars');
  barsDiv.innerHTML = '';
  Object.entries(data.all_probs)
    .sort((a, b) => b[1] - a[1])
    .forEach(([cls, prob]) => {
      const isTop = cls === data.class;
      barsDiv.innerHTML += `
        <div class="prob-row">
          <span class="prob-label" style="${isTop ? 'color:' + classInfo[cls].color + ';font-weight:700' : ''}">${cls}</span>
          <div class="prob-bar-bg">
            <div class="prob-bar" style="width:${(prob * 100).toFixed(1)}%;background:${isTop ? classInfo[cls].color : '#a0aec0'}"></div>
          </div>
          <span class="prob-pct">${(prob * 100).toFixed(1)}%</span>
        </div>`;
    });

  document.getElementById('warningBox').style.display = info.risk === 'high' ? 'block' : 'none';
  document.getElementById('result').style.display = 'block';
}

// Сброс формы
function resetForm() {
  selectedFile = null;
  document.getElementById('fileInput').value      = '';
  document.getElementById('previewWrap').style.display = 'none';
  document.getElementById('result').style.display = 'none';
  document.getElementById('analyzeBtn').disabled  = true;
}

// Запуск тестирования модели
async function runTests() {
  const btn  = document.getElementById('runTestBtn');
  const grid = document.getElementById('testGrid');

  btn.disabled    = true;
  btn.innerHTML   = '<span class="spinner-inline"></span>Тестирование...';
  grid.innerHTML  = '';
  document.getElementById('summarySection').style.display = 'none';

  // Запрашиваем случайные примеры с сервера
  const samplesResp = await fetch('/get_test_samples');
  const samples     = await samplesResp.json();

  const results = [];

  for (const sample of samples) {
    // Сразу показываем карточку с изображением
    const card = document.createElement('div');
    card.className = 'test-card';
    card.innerHTML = `
      <img src="/image/${sample.image_id}" class="test-img" alt="${sample.image_id}">
      <div class="test-body">
        <div class="test-id">${sample.image_id}</div>
        <div class="test-row">
          <span class="test-label">Реальный:</span>
          <span class="test-real" style="color:${classInfo[sample.real_class]?.color}">
            ${sample.real_class.toUpperCase()}
          </span>
        </div>
        <div class="test-row">
          <span class="test-label">Предсказание:</span>
          <span class="spinner-inline"></span>
        </div>
      </div>`;
    grid.appendChild(card);

    // Получаем предсказание для этого изображения
    try {
      const r    = await fetch('/predict_by_id', {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body:    JSON.stringify({ image_id: sample.image_id })
      });
      const data    = await r.json();
      const correct = data.class === sample.real_class;
      results.push({ ...sample, pred: data.class, correct });

      // Топ-4 вероятности в виде мини-полосок
      const miniProbs = Object.entries(data.all_probs)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
        .map(([cls, p]) => `
          <div class="mini-bar-row">
            <span class="mini-label">${cls}</span>
            <div class="mini-bg">
              <div class="mini-fill" style="width:${(p * 100).toFixed(0)}%;background:${cls === data.class ? classInfo[cls].color : '#a0aec0'}"></div>
            </div>
            <span class="mini-pct">${(p * 100).toFixed(0)}%</span>
          </div>`).join('');

      // Обновляем карточку с результатом
      card.querySelector('.test-body').innerHTML = `
        <div class="test-id">${sample.image_id} · ${sample.age}л · ${sample.sex}</div>
        <div class="test-row">
          <span class="test-label">Реальный:</span>
          <span class="test-real" style="color:${classInfo[sample.real_class]?.color}">${sample.real_class.toUpperCase()}</span>
        </div>
        <div class="test-row">
          <span class="test-label">Предсказание:</span>
          <span class="test-pred ${correct ? 'correct' : 'incorrect'}">${data.class.toUpperCase()} ${correct ? '✓' : '✗'}</span>
        </div>
        <div style="font-size:12px;color:#718096;margin-top:4px">
          Уверенность: ${(data.confidence * 100).toFixed(1)}%
        </div>
        <div style="margin-top:8px">${miniProbs}</div>`;

      // Зелёная рамка — правильно, красная — ошибка
      card.style.borderColor = correct ? '#68d391' : '#fc8181';

    } catch(e) {
      card.querySelector('.test-body').innerHTML +=
        '<p style="color:red;font-size:12px;margin-top:8px">Ошибка предсказания</p>';
    }
  }

  // Сводная статистика
  const total   = results.length;
  const correct = results.filter(r => r.correct).length;
  const acc     = (correct / total * 100).toFixed(0);
  const melRes  = results.find(r => r.real_class === 'mel');
  const melOk   = melRes?.correct ? '✓' : '✗';

  document.getElementById('summaryRow').innerHTML = `
    <div class="summary-card">
      <div class="summary-val" style="color:#3182ce">${total}</div>
      <div class="summary-lbl">Тестов всего</div>
    </div>
    <div class="summary-card">
      <div class="summary-val" style="color:#38a169">${correct}</div>
      <div class="summary-lbl">Правильных</div>
    </div>
    <div class="summary-card">
      <div class="summary-val" style="color:${acc >= 70 ? '#38a169' : '#e53e3e'}">${acc}%</div>
      <div class="summary-lbl">Точность</div>
    </div>
    <div class="summary-card">
      <div class="summary-val">${melOk}</div>
      <div class="summary-lbl">Меланома найдена</div>
    </div>`;

  document.getElementById('summarySection').style.display = 'block';
  btn.disabled  = false;
  btn.innerHTML = '▶ Запустить снова';
}
</script>
</body>
</html>"""


# ── Роуты Flask ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Главная страница — передаём данные о классах в шаблон."""
    # Начальные тестовые примеры (по одному на класс)
    # При нажатии кнопки тестирования они обновляются через /get_test_samples
    test_samples = []
    try:
        df = pd.read_csv(os.path.join('data', 'HAM10000_metadata.csv'))
        for cls in CLASS_LABELS:
            row = df[df['dx'] == cls].sample(1).iloc[0]
            test_samples.append({
                'image_id':   row['image_id'],
                'real_class': cls,
                'age':        int(row['age']) if pd.notna(row['age']) else '?',
                'sex':        row.get('sex', '?')
            })
    except Exception as e:
        print(f"Не удалось загрузить тестовые примеры: {e}")

    return render_template_string(HTML, class_info=CLASS_INFO, test_samples=test_samples)


@app.route('/image/<image_id>')
def serve_image(image_id):
    """Отдаёт изображение из датасета по его ID."""
    path = find_image(image_id)
    if path:
        return send_file(path, mimetype='image/jpeg')
    return '', 404


@app.route('/predict', methods=['POST'])
def predict():
    """Принимает загруженный файл и возвращает предсказание модели."""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не загружен'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Файл не выбран'}), 400

    model = get_model()
    if model is None:
        return jsonify({'error': 'Модель не найдена. Запустите train_v4.py'}), 500

    try:
        # Сохраняем файл во временную папку
        fname  = secure_filename(f"{uuid.uuid4()}.jpg")
        fpath  = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        result = run_prediction(fpath)
        os.remove(fpath)  # удаляем временный файл

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_by_id', methods=['POST'])
def predict_by_id():
    """Делает предсказание для изображения из датасета по его ID."""
    data     = request.get_json()
    image_id = data.get('image_id')
    path     = find_image(image_id)

    if not path:
        return jsonify({'error': f'Изображение {image_id} не найдено'}), 404

    model = get_model()
    if model is None:
        return jsonify({'error': 'Модель не найдена'}), 500

    try:
        return jsonify(run_prediction(path))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_test_samples')
def get_test_samples():
    """Возвращает случайные примеры по одному из каждого класса.
    Вызывается при каждом нажатии кнопки тестирования — даёт новые фото."""
    samples = []
    try:
        df = pd.read_csv(os.path.join('data', 'HAM10000_metadata.csv'))
        for cls in CLASS_LABELS:
            row = df[df['dx'] == cls].sample(1).iloc[0]
            samples.append({
                'image_id':   row['image_id'],
                'real_class': cls,
                'age':        int(row['age']) if pd.notna(row['age']) else '?',
                'sex':        row.get('sex', '?')
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify(samples)


# ── Точка входа ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  ВЕБ-ПРИЛОЖЕНИЕ ЗАПУЩЕНО")
    print("=" * 55)
    print("  Открыть в браузере: http://localhost:5000")
    print("  Остановить:         Ctrl+C")
    print("=" * 55 + "\n")
    get_model()
    app.run(debug=True, port=5000)
