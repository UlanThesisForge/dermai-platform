// StatisticsPage.jsx
import { useState, useEffect } from "react"
import { api } from "../api/client"

const CLASS_COLORS = {
  mel:"#ef4444",bcc:"#f97316",akiec:"#f59e0b",
  bkl:"#22c55e",nv:"#3b82f6",df:"#8b5cf6",vasc:"#06b6d4"
}
const CLASS_NAMES = {
  mel:"Меланома",bcc:"Базальноклеточная карцинома",akiec:"Актинический кератоз",
  bkl:"Доброкачественный кератоз",nv:"Невус",df:"Дерматофиброма",vasc:"Сосудистое поражение"
}

export function StatisticsPage() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get("/diagnosis/stats")
      .then(d => setStats(d))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{display:"flex",justifyContent:"center",padding:60}}>
      <div className="spinner" style={{width:32,height:32}} />
    </div>
  )

  const total = stats?.total_diagnoses || 0
  const byClass = stats?.by_class || {}

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Статистика</h1>
          <p className="page-subtitle">Сводка по выполненным диагностикам</p>
        </div>
      </div>

      {/* Метрики */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(180px,1fr))",gap:14,marginBottom:20}}>
        {[
          { val: total,   lbl: "Всего диагностик" },
          { val: `${stats?.avg_processing_ms || 0} мс`, lbl: "Среднее время" },
          { val: stats?.model?.version || "—", lbl: "Версия модели" },
          { val: stats?.model ? `${(stats.model.accuracy * 100).toFixed(1)}%` : "—", lbl: "Точность модели" },
        ].map((m, i) => (
          <div key={i} className="card" style={{textAlign:"center"}}>
            <div style={{fontSize:"1.8rem",fontWeight:700,fontFamily:"var(--font-mono)",color:"var(--accent-light)"}}>{m.val}</div>
            <div style={{fontSize:"0.8rem",color:"var(--text-secondary)",marginTop:4}}>{m.lbl}</div>
          </div>
        ))}
      </div>

      {/* По классам */}
      {total > 0 && (
        <div className="card">
          <h3 style={{marginBottom:16}}>Распределение по классам</h3>
          <div style={{display:"flex",flexDirection:"column",gap:12}}>
            {Object.entries(byClass).sort((a,b) => b[1]-a[1]).map(([cls, count]) => {
              const pct = total > 0 ? (count / total * 100) : 0
              return (
                <div key={cls} style={{display:"flex",alignItems:"center",gap:12,fontSize:"0.875rem"}}>
                  <span style={{width:44,fontFamily:"var(--font-mono)",fontSize:"0.8rem",color:CLASS_COLORS[cls]}}>{cls}</span>
                  <span style={{width:180,color:"var(--text-secondary)",fontSize:"0.8rem"}}>{CLASS_NAMES[cls]}</span>
                  <div style={{flex:1,background:"var(--bg-elevated)",borderRadius:4,height:10}}>
                    <div style={{width:`${pct.toFixed(1)}%`,background:CLASS_COLORS[cls],height:10,borderRadius:4,transition:"width .8s ease"}} />
                  </div>
                  <span style={{width:48,textAlign:"right",fontFamily:"var(--font-mono)",fontSize:"0.8rem",color:"var(--text-secondary)"}}>{count}</span>
                  <span style={{width:44,textAlign:"right",fontFamily:"var(--font-mono)",fontSize:"0.8rem",color:"var(--text-muted)"}}>{pct.toFixed(0)}%</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {total === 0 && (
        <div className="card" style={{textAlign:"center",padding:60,color:"var(--text-muted)"}}>
          <p style={{fontSize:"2rem",marginBottom:12}}>📊</p>
          <p>Статистика появится после первых анализов</p>
        </div>
      )}
    </div>
  )
}

export default StatisticsPage

// ── AboutPage ─────────────────────────────────────────────────────────────────
export function AboutPage() {
  const rows = [
    ["Архитектура",       "MobileNetV2 (Transfer Learning)"],
    ["Датасет",           "HAM10000 — 10 015 изображений"],
    ["Классов",           "7 (mel, nv, bcc, bkl, akiec, df, vasc)"],
    ["Точность (Test)",   "80.8%"],
    ["F1-Score",          "80.6% (взвешенный)"],
    ["Recall меланомы",   "51.5%"],
    ["Нормализация",      "preprocess_input MobileNetV2 → [-1, 1]"],
    ["Балансировка",      "Oversampling до среднего + обратные веса"],
    ["Фазы обучения",     "2: голова (LR=1e-3) + fine-tuning (LR=1e-4)"],
    ["Fine-tuning слоёв", "Последние 30 из 153 слоёв backbone"],
    ["Интерпретируемость","Grad-CAM (тепловая карта активаций)"],
    ["Безопасный порог",  "P(mel) > 20% → предупреждение"],
  ]

  const classes = [
    {cls:"mel",  name:"Меланома",                    risk:"Высокий"},
    {cls:"bcc",  name:"Базальноклеточная карцинома", risk:"Высокий"},
    {cls:"akiec",name:"Актинический кератоз",         risk:"Средний"},
    {cls:"bkl",  name:"Доброкачественный кератоз",   risk:"Низкий"},
    {cls:"nv",   name:"Меланоцитарный невус",         risk:"Низкий"},
    {cls:"df",   name:"Дерматофиброма",               risk:"Низкий"},
    {cls:"vasc", name:"Сосудистое поражение",         risk:"Низкий"},
  ]
  const riskBadge = r => r === "Высокий" ? "high" : r === "Средний" ? "medium" : "low"

  return (
    <div className="fade-in" style={{maxWidth:800}}>
      <div className="page-header">
        <div>
          <h1 className="page-title">О модели</h1>
          <p className="page-subtitle">Технические характеристики системы диагностики</p>
        </div>
      </div>

      <div className="card" style={{marginBottom:20}}>
        <h3 style={{marginBottom:16}}>Технические характеристики</h3>
        <div style={{border:"1px solid var(--border)",borderRadius:8,overflow:"hidden"}}>
          {rows.map(([lbl, val], i) => (
            <div key={i} style={{
              display:"flex",padding:"9px 14px",
              background: i % 2 ? "var(--bg-elevated)" : "transparent",
              fontSize:"0.875rem"
            }}>
              <span style={{width:220,color:"var(--text-secondary)"}}>{lbl}</span>
              <span style={{fontWeight:500}}>{val}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 style={{marginBottom:16}}>Классы диагностики</h3>
        <div style={{display:"flex",flexDirection:"column",gap:8}}>
          {classes.map(({cls, name, risk}) => (
            <div key={cls} style={{
              display:"flex",alignItems:"center",gap:12,
              padding:"10px 14px",background:"var(--bg-elevated)",
              borderRadius:8,fontSize:"0.875rem"
            }}>
              <div style={{width:10,height:10,borderRadius:"50%",background:CLASS_COLORS[cls],flexShrink:0}} />
              <span className="mono" style={{width:50,color:CLASS_COLORS[cls]}}>{cls}</span>
              <span style={{flex:1}}>{name}</span>
              <span className={`badge badge-${riskBadge(risk)}`}>{risk} риск</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{marginTop:20,padding:"14px 20px"}}>
        <p style={{fontSize:"0.85rem",color:"var(--text-secondary)",lineHeight:1.6}}>
          ⚕️ Система является вспомогательным инструментом скрининга кожных заболеваний.
          Не предназначена для замены квалифицированного дерматолога.
          Recall меланомы 51.5% означает обнаружение примерно каждого второго случая.
          Финальный диагноз устанавливает только врач на основе клинического осмотра.
        </p>
      </div>
    </div>
  )
}
