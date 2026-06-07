import { useState, useEffect } from "react"
import { api } from "../api/client"
import { useAuth } from "../App"
import "./SettingsPage.css"

const ROLES = { admin: "Администратор", doctor: "Врач", viewer: "Наблюдатель" }

export default function SettingsPage() {
  const { user } = useAuth()
  const isAdmin = user?.role === "admin"

  const [tab, setTab] = useState("profile")

  return (
    <div className="settings-page fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Настройки</h1>
          <p className="page-subtitle">Управление профилем и пользователями</p>
        </div>
      </div>

      <div className="settings-layout">
        <div className="settings-sidebar">
          <button className={`stab ${tab==="profile"?"active":""}`} onClick={() => setTab("profile")}>
            Мой профиль
          </button>
          {isAdmin && (
            <button className={`stab ${tab==="users"?"active":""}`} onClick={() => setTab("users")}>
              Пользователи
            </button>
          )}
        </div>

        <div className="settings-body">
          {tab === "profile" && <ProfileTab />}
          {tab === "users" && isAdmin && <UsersTab />}
        </div>
      </div>
    </div>
  )
}

// ── Профиль ───────────────────────────────────────────────────────────────────
function ProfileTab() {
  const { user } = useAuth()
  const [name, setName]   = useState(user?.full_name || "")
  const [pass, setPass]   = useState("")
  const [pass2, setPass2] = useState("")
  const [msg,  setMsg]    = useState("")
  const [loading, setLoading] = useState(false)

  const save = async () => {
    if (pass && pass !== pass2) { setMsg("Пароли не совпадают"); return }
    setLoading(true)
    setMsg("")
    try {
      const body = {}
      if (name !== user.full_name) body.full_name = name
      if (pass)                    body.password  = pass
      await api.patch(`/users/${user.doctor_id}`, body)
      setMsg("Сохранено успешно")
      setPass(""); setPass2("")
    } catch (e) {
      setMsg(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="settings-section">
      <h2 style={{marginBottom:20}}>Мой профиль</h2>

      <div className="profile-avatar-row">
        <div className="profile-avatar">
          {user?.full_name?.charAt(0).toUpperCase()}
        </div>
        <div>
          <div className="profile-name">{user?.full_name}</div>
          <span className={`badge badge-${user?.role}`}>{ROLES[user?.role]}</span>
        </div>
      </div>

      <div className="divider" />

      <div className="settings-form">
        <div className="form-group">
          <label className="label">Полное имя</label>
          <input className="input" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div className="form-group">
          <label className="label">Email</label>
          <input className="input" value={user?.email} disabled style={{opacity:.6}} />
        </div>
        <div className="form-group">
          <label className="label">Новый пароль</label>
          <input className="input" type="password" placeholder="Оставьте пустым если не меняете"
            value={pass} onChange={e => setPass(e.target.value)} />
        </div>
        <div className="form-group">
          <label className="label">Повторите пароль</label>
          <input className="input" type="password" placeholder="••••••••"
            value={pass2} onChange={e => setPass2(e.target.value)} />
        </div>

        {msg && (
          <div className={`settings-msg ${msg.includes("успешно") ? "success" : "error"}`}>
            {msg}
          </div>
        )}

        <button className="btn btn-primary" onClick={save} disabled={loading}>
          {loading ? "Сохраняю..." : "Сохранить изменения"}
        </button>
      </div>
    </div>
  )
}

// ── Управление пользователями ─────────────────────────────────────────────────
function UsersTab() {
  const [users,   setUsers]   = useState([])
  const [loading, setLoading] = useState(true)
  const [showAdd, setShowAdd] = useState(false)
  const [editUser, setEditUser] = useState(null)
  const [error,   setError]   = useState("")

  const load = async () => {
    setLoading(true)
    try {
      const data = await api.get("/users/")
      setUsers(data)
    } catch(e) { setError(e.message) }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const toggleActive = async (u) => {
    try {
      await api.patch(`/users/${u.doctor_id}`, { is_active: !u.is_active })
      load()
    } catch(e) { setError(e.message) }
  }

  const deleteUser = async (u) => {
    if (!confirm(`Удалить пользователя ${u.full_name}?`)) return
    try {
      await api.delete(`/users/${u.doctor_id}`)
      load()
    } catch(e) { setError(e.message) }
  }

  return (
    <div className="settings-section">
      <div className="users-header">
        <h2>Пользователи</h2>
        <button className="btn btn-primary" onClick={() => { setShowAdd(true); setEditUser(null) }}>
          + Добавить
        </button>
      </div>

      {error && <div className="settings-msg error" style={{marginBottom:12}}>{error}</div>}

      {loading ? (
        <div style={{display:"flex",justifyContent:"center",padding:40}}>
          <div className="spinner" />
        </div>
      ) : (
        <div className="users-table">
          <div className="users-thead">
            <span>Имя</span>
            <span>Email</span>
            <span>Роль</span>
            <span>Статус</span>
            <span>Последний вход</span>
            <span></span>
          </div>
          {users.map(u => (
            <div key={u.doctor_id} className="users-row">
              <span className="user-row-name">{u.full_name}</span>
              <span className="user-row-email mono">{u.email}</span>
              <span><span className={`badge badge-${u.role}`}>{ROLES[u.role]}</span></span>
              <span>
                <button
                  className={`status-toggle ${u.is_active ? "active" : "inactive"}`}
                  onClick={() => toggleActive(u)}
                  title={u.is_active ? "Деактивировать" : "Активировать"}
                >
                  {u.is_active ? "Активен" : "Отключён"}
                </button>
              </span>
              <span className="user-row-date">
                {u.last_login ? new Date(u.last_login).toLocaleDateString("ru-RU") : "—"}
              </span>
              <span className="user-row-actions">
                <button className="icon-btn" onClick={() => { setEditUser(u); setShowAdd(true) }} title="Редактировать">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                  </svg>
                </button>
                <button className="icon-btn danger" onClick={() => deleteUser(u)} title="Удалить">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6"/>
                    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                    <path d="M10 11v6m4-6v6"/>
                  </svg>
                </button>
              </span>
            </div>
          ))}
        </div>
      )}

      {showAdd && (
        <UserModal
          user={editUser}
          onClose={() => { setShowAdd(false); setEditUser(null) }}
          onSaved={load}
        />
      )}
    </div>
  )
}

// ── Модальное окно добавления/редактирования ──────────────────────────────────
function UserModal({ user, onClose, onSaved }) {
  const [name,  setName]  = useState(user?.full_name || "")
  const [email, setEmail] = useState(user?.email || "")
  const [pass,  setPass]  = useState("")
  const [role,  setRole]  = useState(user?.role || "doctor")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const save = async () => {
    setLoading(true); setError("")
    try {
      if (user) {
        const body = { full_name: name, email, role }
        if (pass) body.password = pass
        await api.patch(`/users/${user.doctor_id}`, body)
      } else {
        if (!pass) { setError("Пароль обязателен"); setLoading(false); return }
        await api.post("/users/", { full_name: name, email, password: pass, role })
      }
      onSaved()
      onClose()
    } catch(e) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-box card">
        <div className="modal-header">
          <h3>{user ? "Редактировать пользователя" : "Новый пользователь"}</h3>
          <button className="icon-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        <div className="modal-body">
          <div className="form-group">
            <label className="label">Полное имя</label>
            <input className="input" value={name} onChange={e => setName(e.target.value)} required />
          </div>
          <div className="form-group">
            <label className="label">Email</label>
            <input className="input" type="email" value={email} onChange={e => setEmail(e.target.value)} required />
          </div>
          <div className="form-group">
            <label className="label">{user ? "Новый пароль (оставьте пустым)" : "Пароль"}</label>
            <input className="input" type="password" value={pass} onChange={e => setPass(e.target.value)} />
          </div>
          <div className="form-group">
            <label className="label">Роль</label>
            <select className="input" value={role} onChange={e => setRole(e.target.value)}>
              <option value="doctor">Врач</option>
              <option value="viewer">Наблюдатель</option>
              <option value="admin">Администратор</option>
            </select>
          </div>

          {error && <div className="settings-msg error">{error}</div>}
        </div>

        <div className="modal-footer">
          <button className="btn btn-ghost" onClick={onClose}>Отмена</button>
          <button className="btn btn-primary" onClick={save} disabled={loading}>
            {loading ? "Сохраняю..." : "Сохранить"}
          </button>
        </div>
      </div>
    </div>
  )
}
