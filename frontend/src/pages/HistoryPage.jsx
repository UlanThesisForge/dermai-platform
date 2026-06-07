import { useState, useEffect } from "react";
import { api } from "../api/client";

const CLASS_COLORS = {
  mel: "#ef4444",
  bcc: "#f97316",
  akiec: "#f59e0b",
  bkl: "#22c55e",
  nv: "#3b82f6",
  df: "#8b5cf6",
  vasc: "#06b6d4",
};
const RISK_LABEL = {
  high: "Высокий риск",
  medium: "Средний риск",
  low: "Низкий риск",
};

function AuthImage({ imageId, style, alt = "" }) {
  return (
    <img
      src={imageId ? `http://localhost:8000/diagnosis/image/${imageId}` : ""}
      alt={alt}
      style={{ ...style, objectFit: "cover", flexShrink: 0 }}
      onError={(e) => {
        e.target.style.display = "none";
      }}
    />
  );
}

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    api
      .get("/diagnosis/history?limit=50")
      .then((d) => setHistory(d))
      .finally(() => setLoading(false));
  }, []);

  // Grad-CAM URL — теперь публичный эндпоинт
  const gradcamUrl = (imageId) =>
    `http://localhost:8000/diagnosis/gradcam/${imageId}`;

  const toggleSelect = (item) => {
    if (selected?.report_id === item.report_id) {
      setSelected(null);
    } else {
      setSelected(item);
      // gradcam загружается по URL
    }
  };

  return (
    <div className="fade-in" style={{ maxWidth: 900 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">История диагностик</h1>
          <p className="page-subtitle">Все выполненные анализы</p>
        </div>
        <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
          {history.length} записей
        </span>
      </div>

      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: 60 }}>
          <div className="spinner" style={{ width: 32, height: 32 }} />
        </div>
      ) : history.length === 0 ? (
        <div
          className="card"
          style={{
            textAlign: "center",
            padding: 60,
            color: "var(--text-muted)",
          }}
        >
          <p style={{ fontSize: "2rem", marginBottom: 12 }}>📋</p>
          <p>История пуста. Выполните первый анализ!</p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {history.map((item) => (
            <div
              key={item.report_id}
              className="card"
              style={{
                cursor: "pointer",
                transition: "border-color .15s",
                borderColor:
                  selected?.report_id === item.report_id
                    ? CLASS_COLORS[item.prediction]
                    : undefined,
                padding: "16px 20px",
              }}
              onClick={() => toggleSelect(item)}
            >
              {/* Основная строка */}
              <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                <AuthImage
                  imageId={item.image_id}
                  style={{ width: 52, height: 52, borderRadius: 8 }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontWeight: 500,
                      color: CLASS_COLORS[item.prediction],
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {item.prediction.toUpperCase()} — {item.class_name}
                  </div>
                  <div
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--text-muted)",
                      marginTop: 3,
                    }}
                  >
                    {new Date(item.created_at).toLocaleString("ru-RU")}
                  </div>
                </div>
                <span className={`badge badge-${item.risk}`}>
                  {RISK_LABEL[item.risk]}
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "1rem",
                    fontWeight: 600,
                    color: CLASS_COLORS[item.prediction],
                    minWidth: 52,
                    textAlign: "right",
                  }}
                >
                  {(item.confidence * 100).toFixed(0)}%
                </span>
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="var(--text-muted)"
                  strokeWidth="2"
                  style={{
                    flexShrink: 0,
                    transform:
                      selected?.report_id === item.report_id
                        ? "rotate(180deg)"
                        : "rotate(0deg)",
                    transition: "transform .2s",
                  }}
                >
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </div>

              {/* Раскрытая деталь */}
              {selected?.report_id === item.report_id && (
                <div
                  style={{
                    marginTop: 16,
                    paddingTop: 16,
                    borderTop: "1px solid var(--border-light)",
                    display: "flex",
                    gap: 20,
                    flexWrap: "wrap",
                  }}
                >
                  {/* Grad-CAM */}
                  <div style={{ display: "flex", gap: 12 }}>
                    <div>
                      <p
                        style={{
                          fontSize: "0.75rem",
                          color: "var(--text-muted)",
                          marginBottom: 6,
                        }}
                      >
                        Grad-CAM — зоны внимания модели
                      </p>
                      {true ? (
                        <img
                          src={true}
                          alt="Grad-CAM"
                          style={{
                            width: 160,
                            height: 160,
                            borderRadius: 8,
                            objectFit: "cover",
                          }}
                        />
                      ) : (
                        <div
                          style={{
                            width: 160,
                            height: 160,
                            borderRadius: 8,
                            background: "var(--bg-elevated)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            color: "var(--text-muted)",
                            fontSize: "0.8rem",
                          }}
                        >
                          <div
                            className="spinner"
                            style={{ width: 20, height: 20 }}
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Вероятности */}
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <p
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--text-muted)",
                        marginBottom: 8,
                      }}
                    >
                      Вероятности классов
                    </p>
                    {Object.entries(item.probabilities || {})
                      .sort((a, b) => b[1] - a[1])
                      .map(([cls, prob]) => (
                        <div
                          key={cls}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            marginBottom: 6,
                            fontSize: "0.8rem",
                          }}
                        >
                          <span
                            style={{
                              width: 36,
                              fontFamily: "var(--font-mono)",
                              color: CLASS_COLORS[cls],
                              fontSize: "0.75rem",
                            }}
                          >
                            {cls}
                          </span>
                          <div
                            style={{
                              flex: 1,
                              background: "var(--bg-elevated)",
                              borderRadius: 4,
                              height: 7,
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${(prob * 100).toFixed(1)}%`,
                                background: CLASS_COLORS[cls],
                                opacity: cls === item.prediction ? 1 : 0.4,
                                height: 7,
                                borderRadius: 4,
                              }}
                            />
                          </div>
                          <span
                            style={{
                              width: 40,
                              textAlign: "right",
                              fontFamily: "var(--font-mono)",
                              color: "var(--text-muted)",
                            }}
                          >
                            {(prob * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                  </div>

                  {/* Мета */}
                  <div
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <div style={{ marginBottom: 4 }}>
                      <span style={{ color: "var(--text-muted)" }}>ID: </span>
                      <span className="mono" style={{ fontSize: "0.75rem" }}>
                        {item.report_id.slice(0, 16)}...
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
