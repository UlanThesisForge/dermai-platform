import { useState, useCallback, useRef } from "react"
import { api } from "../api/client"
import "./AnalysePage.css"

const CLASS_COLORS = {
  mel:   "#ef4444", bcc: "#f97316", akiec: "#f59e0b",
  bkl:   "#22c55e", nv:  "#3b82f6", df:    "#8b5cf6", vasc: "#06b6d4"
}
const RISK_LABEL = { high: "Высокий риск", medium: "Средний риск", low: "Низкий риск" }

export default function AnalysePage() {
  const [dragging, setDragging]  = useState(false)
  const [preview,  setPreview]   = useState(null)   // base64
  const [file,     setFile]      = useState(null)
  const [result,   setResult]    = useState(null)
  const [loading,  setLoading]   = useState(false)
  const [error,    setError]     = useState("")
  const [showGrad, setShowGrad]  = useState(false)
  const inputRef = useRef()

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return
    setFile(f)
    setResult(null)
    setError("")
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const analyse = async () => {
    if (!file) return
    setLoading(true)
    setError("")
    try {
      const fd = new FormData()
      fd.append("file", file)
      const data = await api.upload("/diagnosis/analyse", fd)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null); setPreview(null); setResult(null); setError(""); setShowGrad(false)
  }

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="analyse-page fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Анализ снимка</h1>
          <p className="page-subtitle">Загрузите дерматоскопическое изображение для классификации</p>
        </div>
      </div>

      <div className="analyse-grid">
        {/* Левая колонка: загрузка */}
        <div className="analyse-left">
          <div className="card upload-card">
            {!preview ? (
              <div
                className={`drop-zone ${dragging ? "dragging" : ""}`}
                onDragOver={e => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current.click()}
              >
                <div className="drop-icon">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="3"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <path d="m21 15-5-5L5 21"/>
                  </svg>
                </div>
                <p className="drop-title">Перетащите изображение сюда</p>
                <p className="drop-sub">или нажмите для выбора файла</p>
                <p className="drop-hint">JPG, PNG · Максимум 16 МБ</p>
              </div>
            ) : (
              <div className="preview-area">
                <img
                  src={showGrad && result
                    ? `http://localhost:8000/diagnosis/gradcam/${result.image_id}`
                    : preview}
                  alt="Предпросмотр"
                  className="preview-img"
                />
                {result && (
                  <div className="gradcam-toggle">
                    <button
                      className={`btn ${showGrad ? "btn-primary" : "btn-ghost"}`}
                      onClick={() => setShowGrad(g => !g)}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/>
                        <circle cx="12" cy="12" r="3"/>
                      </svg>
                      {showGrad ? "Оригинал" : "Grad-CAM"}
                    </button>
                    <span className="gradcam-hint">
                      {showGrad ? "Тепловая карта активаций" : "Показать зоны внимания модели"}
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
              onChange={e => handleFile(e.target.files[0])}
            />
          </div>

          {error && (
            <div className="error-box">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
              </svg>
              {error}
            </div>
          )}

          <div className="analyse-actions">
            {preview && !result && (
              <button className="btn btn-primary" onClick={analyse} disabled={loading} style={{flex:1}}>
                {loading
                  ? <><span className="spinner" style={{width:16,height:16}} /> Анализирую...</>
                  : <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                      </svg>
                      Анализировать
                    </>
                }
              </button>
            )}
            {preview && (
              <button className="btn btn-ghost" onClick={reset}>Сбросить</button>
            )}
          </div>
        </div>

        {/* Правая колонка: результат */}
        <div className="analyse-right">
          {!result && !loading && (
            <div className="no-result card">
              <div className="no-result-icon">🔬</div>
              <p>Загрузите изображение и нажмите «Анализировать»</p>
            </div>
          )}

          {loading && (
            <div className="card loading-card">
              <div className="spinner" style={{width:32,height:32,margin:"0 auto 16px"}} />
              <p style={{textAlign:"center",color:"var(--text-secondary)"}}>Нейросеть анализирует...</p>
            </div>
          )}

          {result && (
            <div className="result-panel fade-in">
              {/* Главный диагноз */}
              <div className="card diagnosis-card" style={{borderColor: CLASS_COLORS[result.prediction] + "44"}}>
                <div className="diagnosis-header">
                  <div>
                    <div
                      className="diagnosis-class"
                      style={{color: CLASS_COLORS[result.prediction]}}
                    >
                      {result.prediction.toUpperCase()}
                    </div>
                    <div className="diagnosis-name">{result.class_name}</div>
                  </div>
                  <div className="diagnosis-conf">
                    <div className="conf-val">{(result.confidence * 100).toFixed(1)}%</div>
                    <div className="conf-lbl">уверенность</div>
                  </div>
                </div>

                <span className={`badge badge-${result.risk}`} style={{marginTop:8}}>
                  <span className="risk-dot" />
                  {RISK_LABEL[result.risk]}
                </span>

                {result.risk === "high" && (
                  <div className="warning-block">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{flexShrink:0}}>
                      <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
                    </svg>
                    Обнаружено потенциально злокачественное образование. Необходима консультация дерматолога.
                  </div>
                )}
              </div>

              {/* Вероятности */}
              <div className="card">
                <h3 style={{marginBottom:16}}>Вероятности классов</h3>
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

              {/* Мета */}
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
    </div>
  )
}
