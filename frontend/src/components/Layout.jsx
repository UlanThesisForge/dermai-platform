import { useState } from "react"
import { NavLink, Outlet, useNavigate } from "react-router-dom"
import { useAuth } from "../App"
import "./Layout.css"

const NAV_ITEMS = [
  { to: "/analyse",    icon: "analyse",    label: "Анализ снимка" },
  { to: "/history",    icon: "history",    label: "История" },
  { to: "/statistics", icon: "statistics", label: "Статистика" },
  { to: "/about",      icon: "about",      label: "О модели" },
  { to: "/settings",   icon: "settings",   label: "Настройки" },
]

function Icon({ name }) {
  const icons = {
    analyse:    <path d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" strokeWidth="2" strokeLinecap="round"/>,
    history:    <><circle cx="12" cy="12" r="10" strokeWidth="2"/><path d="M12 6v6l4 2" strokeWidth="2" strokeLinecap="round"/></>,
    statistics: <><path d="M3 3v18h18" strokeWidth="2" strokeLinecap="round"/><path d="M7 16l4-4 4 4 4-4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></>,
    about:      <><circle cx="12" cy="12" r="10" strokeWidth="2"/><path d="M12 16v-4m0-4h.01" strokeWidth="2" strokeLinecap="round"/></>,
    settings:   <><circle cx="12" cy="12" r="3" strokeWidth="2"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.4 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" strokeWidth="2"/></>,
    logout:     <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>,
  }
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
      {icons[name]}
    </svg>
  )
}

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [collapsed, setCollapsed] = useState(false)

  const handleLogout = async () => {
    await logout()
    navigate("/login")
  }

  return (
    <div className={`layout ${collapsed ? "collapsed" : ""}`}>
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="8" fill="#2563eb"/>
              <circle cx="16" cy="13" r="5" stroke="#fff" strokeWidth="2"/>
              <path d="M8 26c0-4.4 3.6-8 8-8s8 3.6 8 8" stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            {!collapsed && <span className="sidebar-brand">DermAI</span>}
          </div>
          <button
            className="collapse-btn"
            onClick={() => setCollapsed(c => !c)}
            title={collapsed ? "Развернуть" : "Свернуть"}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {collapsed
                ? <path d="M9 18l6-6-6-6"/>
                : <path d="M15 18l-6-6 6-6"/>
              }
            </svg>
          </button>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
              title={collapsed ? item.label : undefined}
            >
              <Icon name={item.icon} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-user">
            <div className="user-avatar">
              {user?.full_name?.charAt(0).toUpperCase() || "U"}
            </div>
            {!collapsed && (
              <div className="user-info">
                <span className="user-name">{user?.full_name}</span>
                <span className={`badge badge-${user?.role}`}>{user?.role}</span>
              </div>
            )}
          </div>
          <button className="nav-item logout-btn" onClick={handleLogout} title="Выход">
            <Icon name="logout" />
            {!collapsed && <span>Выход</span>}
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}
