import { useState, useCallback, useRef } from "react";
import { api } from "../api/client";
import "./AnalysePage.css";

const CLASS_COLORS = {
  mel: "#ef4444",
  bcc: "#f97316",
  akiec: "#f59e0b",
  bkl: "#22c55e",
  nv: "#3b82f6",
  df: "#8b5cf6",
  vasc: "#06b6d4",
};
const CLASS_INFO = {
  akiec: { name: "Актинический кератоз", risk: "medium" },
  bcc: { name: "Базальноклеточная карцинома", risk: "high" },
  bkl: { name: "Доброкачественный кератоз", risk: "low" },
  df: { name: "Дерматофиброма", risk: "low" },
  mel: { name: "Меланома", risk: "high" },
  nv: { name: "Меланоцитарный невус (родинка)", risk: "low" },
  vasc: { name: "Сосудистое поражение", risk: "medium" },
};
const RISK_LABEL = {
  high: "Высокий риск",
  medium: "Средний риск",
  low: "Низкий риск",
};

// ── Вкладка Диагностика ───────────────────────────────────────────────────────
function DiagnoseTab() {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showGrad, setShowGrad] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError("");
    setShowGrad(false);
    const r = new FileReader();
    r.onload = (e) => setPreview(e.target.result);
    r.readAsDataURL(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const analyse = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const data = await api.upload("/diagnosis/analyse", fd);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setShowGrad(false);
  };

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="analyse-grid">
      {/* Левая колонка */}
      <div className="analyse-left">
        <div className="card upload-card">
          {!preview ? (
            <div
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current.click()}
            >
              <div className="drop-icon">
                <svg
                  width="40"
                  height="40"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <rect x="3" y="3" width="18" height="18" rx="3" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="m21 15-5-5L5 21" />
                </svg>
              </div>
              <p className="drop-title">Перетащите изображение сюда</p>
              <p className="drop-sub">или нажмите для выбора файла</p>
              <p className="drop-hint">JPG, PNG · Максимум 16 МБ</p>
            </div>
          ) : (
            <div className="preview-area">
              <img
                src={
                  showGrad && result
                    ? `http://localhost:8000/diagnosis/gradcam/${result.image_id}`
                    : preview
                }
                alt="Предпросмотр"
                className="preview-img"
              />
              {result && (
                <div className="gradcam-toggle">
                  <button
                    className={`btn ${showGrad ? "btn-primary" : "btn-ghost"}`}
                    onClick={() => setShowGrad((g) => !g)}
                  >
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                    {showGrad ? "Оригинал" : "Grad-CAM"}
                  </button>
                  <span className="gradcam-hint">
                    {showGrad
                      ? "Тепловая карта активаций"
                      : "Показать зоны внимания модели"}
                  </span>
                </div>
              )}
              <p className="preview-filename">{file?.name}</p>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        {error && (
          <div className="error-box">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
            </svg>
            {error}
          </div>
        )}

        <div className="analyse-actions">
          {preview && !result && (
            <button
              className="btn btn-primary"
              onClick={analyse}
              disabled={loading}
              style={{ flex: 1 }}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 16, height: 16 }} />{" "}
                  Анализирую...
                </>
              ) : (
                <>
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.35-4.35" />
                  </svg>
                  Анализировать
                </>
              )}
            </button>
          )}
          {preview && (
            <button className="btn btn-ghost" onClick={reset}>
              Сбросить
            </button>
          )}
        </div>
      </div>

      {/* Правая колонка */}
      <div className="analyse-right">
        {!result && !loading && (
          <div className="no-result card">
            <div className="no-result-icon">🔬</div>
            <p>Загрузите изображение и нажмите «Анализировать»</p>
          </div>
        )}
        {loading && (
          <div className="card loading-card">
            <div
              className="spinner"
              style={{ width: 32, height: 32, margin: "0 auto 16px" }}
            />
            <p style={{ textAlign: "center", color: "var(--text-secondary)" }}>
              Нейросеть анализирует...
            </p>
          </div>
        )}
        {result && (
          <div className="result-panel fade-in">
            <div
              className="card diagnosis-card"
              style={{ borderColor: CLASS_COLORS[result.prediction] + "44" }}
            >
              <div className="diagnosis-header">
                <div>
                  <div
                    className="diagnosis-class"
                    style={{ color: CLASS_COLORS[result.prediction] }}
                  >
                    {result.prediction.toUpperCase()}
                  </div>
                  <div className="diagnosis-name">{result.class_name}</div>
                </div>
                <div className="diagnosis-conf">
                  <div className="conf-val">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="conf-lbl">уверенность</div>
                </div>
              </div>
              <span
                className={`badge badge-${result.risk}`}
                style={{ marginTop: 8 }}
              >
                <span className="risk-dot" />
                {RISK_LABEL[result.risk]}
              </span>
              {result.risk === "high" && (
                <div className="warning-block">
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    style={{ flexShrink: 0 }}
                  >
                    <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
                  </svg>
                  Обнаружено потенциально злокачественное образование.
                  Необходима консультация дерматолога.
                </div>
              )}
            </div>

            <div className="card">
              <h3 style={{ marginBottom: 16 }}>Вероятности классов</h3>
              <div className="prob-list">
                {sortedProbs.map(([cls, prob]) => (
                  <div key={cls} className="prob-row">
                    <span className="prob-cls">{cls}</span>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar-fill"
                        style={{
                          width: `${(prob * 100).toFixed(1)}%`,
                          background: CLASS_COLORS[cls],
                          opacity: cls === result.prediction ? 1 : 0.45,
                        }}
                      />
                    </div>
                    <span className="prob-val">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card meta-card">
              <div className="meta-row">
                <span className="meta-lbl">Время обработки</span>
                <span className="meta-val mono">{result.processing_ms} мс</span>
              </div>
              <div className="meta-row">
                <span className="meta-lbl">Модель</span>
                <span className="meta-val mono">MobileNetV2</span>
              </div>
              <div className="meta-row">
                <span className="meta-lbl">Датасет</span>
                <span className="meta-val">HAM10000</span>
              </div>
            </div>

            <p className="disclaimer">
              ⚕️ Система является вспомогательным инструментом скрининга.
              Финальный диагноз устанавливает только квалифицированный врач.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Вкладка Тестирование ──────────────────────────────────────────────────────
function TestingTab() {
  const [samples, setSamples] = useState([]);
  const [results, setResults] = useState({}); // image_id -> prediction result
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [summary, setSummary] = useState(null);

  const runTests = async () => {
    setLoading(true);
    setError("");
    setResults({});
    setSummary(null);
    try {
      // 1. Получаем случайные примеры
      const newSamples = await api.get("/diagnosis/test_samples");
      setSamples(newSamples);

      // 2. Прогоняем каждый через модель
      const newResults = {};
      for (const sample of newSamples) {
        try {
          const res = await api.post("/diagnosis/test_predict", {
            image_id: sample.image_id,
          });
          newResults[sample.image_id] = res;
          setResults((prev) => ({ ...prev, [sample.image_id]: res }));
        } catch {
          newResults[sample.image_id] = null;
        }
      }

      // 3. Считаем итог
      const total = newSamples.length;
      const correct = newSamples.filter(
        (s) => newResults[s.image_id]?.prediction === s.real_class,
      ).length;
      const melSample = newSamples.find((s) => s.real_class === "mel");
      setSummary({
        total,
        correct,
        accuracy: total > 0 ? ((correct / total) * 100).toFixed(0) : 0,
        mel_correct: melSample
          ? newResults[melSample.image_id]?.prediction === "mel"
          : null,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Заголовок и кнопка */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <div>
            <h3 style={{ marginBottom: 6 }}>
              Автоматическое тестирование на HAM10000
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Случайные примеры по одному из каждого класса — предсказание
              сравнивается с реальным диагнозом.
            </p>
          </div>
          <button
            className="btn btn-primary"
            onClick={runTests}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16 }} />{" "}
                Тестирую...
              </>
            ) : (
              <>
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Запустить тестирование
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="error-box" style={{ marginTop: 14 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
            </svg>
            {error.includes("не найден")
              ? "Датасет HAM10000 не смонтирован. Проверьте docker-compose.yml"
              : error}
          </div>
        )}
      </div>

      {/* Сводка */}
      {summary && (
        <div className="test-summary">
          {[
            { val: summary.total, lbl: "Тестов", color: "var(--accent-light)" },
            {
              val: summary.correct,
              lbl: "Правильно",
              color: "var(--color-success, #22c55e)",
            },
            {
              val: summary.accuracy + "%",
              lbl: "Точность",
              color: parseInt(summary.accuracy) >= 70 ? "#22c55e" : "#ef4444",
            },
            {
              val:
                summary.mel_correct === null
                  ? "—"
                  : summary.mel_correct
                    ? "✓"
                    : "✗",
              lbl: "Меланома",
              color: summary.mel_correct ? "#22c55e" : "#ef4444",
            },
          ].map((m, i) => (
            <div key={i} className="summary-card2">
              <div className="summary-val2" style={{ color: m.color }}>
                {m.val}
              </div>
              <div className="summary-lbl2">{m.lbl}</div>
            </div>
          ))}
        </div>
      )}

      {/* Карточки результатов */}
      {samples.length > 0 && (
        <div className="test-grid">
          {samples.map((sample) => {
            const res = results[sample.image_id];
            const correct = res ? res.prediction === sample.real_class : null;
            const probs = res
              ? Object.entries(res.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 4)
              : [];

            return (
              <div
                key={sample.image_id}
                className="test-card2"
                style={{
                  borderColor:
                    correct === true
                      ? "#22c55e44"
                      : correct === false
                        ? "#ef444444"
                        : "var(--border)",
                }}
              >
                {/* Изображение */}
                <div className="test-img-wrap">
                  <img
                    src={`http://localhost:8000/diagnosis/ham_image/${sample.image_id}`}
                    alt={sample.image_id}
                    className="test-img2"
                    onError={(e) => {
                      e.target.style.display = "none";
                    }}
                  />
                  {correct !== null && (
                    <div
                      className={`test-badge ${correct ? "correct" : "incorrect"}`}
                    >
                      {correct ? "✓" : "✗"}
                    </div>
                  )}
                </div>

                {/* Тело */}
                <div className="test-body2">
                  <div className="test-id2">
                    {sample.image_id.slice(0, 18)}...
                  </div>
                  {sample.age && (
                    <div className="test-meta2">
                      {sample.age}л · {sample.sex}
                    </div>
                  )}

                  {/* Реальный */}
                  <div className="test-row2">
                    <span className="test-lbl2">Реальный</span>
                    <span
                      style={{
                        fontWeight: 600,
                        color: CLASS_COLORS[sample.real_class],
                      }}
                    >
                      {sample.real_class.toUpperCase()}
                    </span>
                  </div>

                  {/* Предсказание */}
                  <div className="test-row2">
                    <span className="test-lbl2">Модель</span>
                    {res ? (
                      <span
                        style={{
                          fontWeight: 700,
                          color: CLASS_COLORS[res.prediction],
                        }}
                      >
                        {res.prediction.toUpperCase()}
                        <span
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.75rem",
                            marginLeft: 6,
                            color: "var(--text-muted)",
                          }}
                        >
                          {(res.confidence * 100).toFixed(0)}%
                        </span>
                      </span>
                    ) : loading ? (
                      <span
                        className="spinner"
                        style={{ width: 14, height: 14 }}
                      />
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>—</span>
                    )}
                  </div>

                  {/* Мини-полоски */}
                  {probs.length > 0 && (
                    <div className="mini-probs">
                      {probs.map(([cls, p]) => (
                        <div key={cls} className="mini-prob-row">
                          <span className="mini-cls">{cls}</span>
                          <div className="mini-bg2">
                            <div
                              className="mini-fill2"
                              style={{
                                width: `${(p * 100).toFixed(0)}%`,
                                background: CLASS_COLORS[cls],
                                opacity: cls === res?.prediction ? 1 : 0.4,
                              }}
                            />
                          </div>
                          <span className="mini-pct2">
                            {(p * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {samples.length === 0 && !loading && (
        <div
          className="card"
          style={{
            textAlign: "center",
            padding: 60,
            color: "var(--text-muted)",
          }}
        >
          <p style={{ fontSize: "2rem", marginBottom: 12 }}>🧪</p>
          <p>
            Нажмите «Запустить тестирование» чтобы проверить модель на реальных
            примерах из HAM10000
          </p>
        </div>
      )}
    </div>
  );
}

// ── Главная страница ──────────────────────────────────────────────────────────
export default function AnalysePage() {
  const [tab, setTab] = useState("diagnose");

  return (
    <div className="analyse-page fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">
            {tab === "diagnose" ? "Анализ снимка" : "Тестирование модели"}
          </h1>
          <p className="page-subtitle">
            {tab === "diagnose"
              ? "Загрузите дерматоскопическое изображение для классификации"
              : "Автоматическая проверка на случайных примерах из датасета HAM10000"}
          </p>
        </div>
      </div>

      {/* Вкладки */}
      <div className="analyse-tabs">
        <button
          className={`analyse-tab ${tab === "diagnose" ? "active" : ""}`}
          onClick={() => setTab("diagnose")}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
          </svg>
          Диагностика
        </button>
        <button
          className={`analyse-tab ${tab === "test" ? "active" : ""}`}
          onClick={() => setTab("test")}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18" />
          </svg>
          Тестирование
        </button>
      </div>

      {tab === "diagnose" && <DiagnoseTab />}
      {tab === "test" && <TestingTab />}
    </div>
  );
}
