import { createContext, useContext, useState, useEffect } from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import Login from "./pages/Login"
import Layout from "./components/Layout"
import AnalysePage from "./pages/AnalysePage"
import HistoryPage from "./pages/HistoryPage"
import StatisticsPage from "./pages/StatisticsPage"
import AboutPage from "./pages/AboutPage"
import SettingsPage from "./pages/SettingsPage"
import "./styles/global.css"

const AuthContext = createContext(null)

export function useAuth() {
  return useContext(AuthContext)
}

function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // Читаем из localStorage один раз при монтировании
  useEffect(() => {
    try {
      const stored = localStorage.getItem("dermai_user")
      if (stored) setUser(JSON.parse(stored))
    } catch {}
    setLoading(false)
  }, [])

  const login = (userData) => {
    localStorage.setItem("dermai_user",    JSON.stringify(userData))
    localStorage.setItem("dermai_access",  userData.access_token)
    localStorage.setItem("dermai_refresh", userData.refresh_token)
    setUser(userData)   // state обновляется — PrivateRoute перерендерится
  }

  const logout = async () => {
    const refresh = localStorage.getItem("dermai_refresh")
    if (refresh) {
      try {
        await fetch("http://localhost:8000/auth/logout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refresh }),
        })
      } catch {}
    }
    setUser(null)
    localStorage.removeItem("dermai_user")
    localStorage.removeItem("dermai_access")
    localStorage.removeItem("dermai_refresh")
  }

  // Пока читаем localStorage — не рендерим ничего (избегаем flash)
  if (loading) return null

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

function PrivateRoute({ children }) {
  const { user } = useAuth()
  return user ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<PrivateRoute><Layout /></PrivateRoute>}>
            <Route index element={<Navigate to="/analyse" replace />} />
            <Route path="analyse"    element={<AnalysePage />} />
            <Route path="history"    element={<HistoryPage />} />
            <Route path="statistics" element={<StatisticsPage />} />
            <Route path="about"      element={<AboutPage />} />
            <Route path="settings"   element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
