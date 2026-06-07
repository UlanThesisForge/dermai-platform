import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../App";
import { api } from "../api/client";
import "./Login.css";

export default function Login() {
  const { login, user } = useAuth();
  const navigate = useNavigate();

  const [tab, setTab] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // ВСЕ хуки должны быть до любого return
  // Редирект через useEffect — не через early return
  useEffect(() => {
    if (user) navigate("/analyse", { replace: true });
  }, [user, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      let data;
      if (tab === "login") {
        data = await api.post("/auth/login", { email, password });
      } else {
        data = await api.post("/auth/register", {
          full_name: name,
          email,
          password,
        });
      }
      login(data);
      // navigate вызовет useEffect выше когда user обновится
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Если уже залогинен — показываем ничего пока useEffect не сработает
  if (user) return null;

  return (
    <div className="login-page">
      <div className="login-bg">
        <div className="login-bg-grid" />
      </div>

      <div className="login-left">
        <div className="login-brand">
          <div className="login-logo">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="10" fill="#2563eb" />
              <circle cx="16" cy="13" r="5" stroke="#fff" strokeWidth="2" />
              <path
                d="M8 26c0-4.4 3.6-8 8-8s8 3.6 8 8"
                stroke="#fff"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <span className="login-brand-name">DermAI</span>
        </div>

        <div className="login-tagline">
          <h1>
            Диагностика
            <br />
            кожных заболеваний
          </h1>
          <p>
            Система на основе нейронной сети MobileNetV2, обученной на датасете
            HAM10000. Точность 80.8%.
          </p>
        </div>

        <div className="login-stats">
          <div className="login-stat">
            <span className="login-stat-val">80.8%</span>
            <span className="login-stat-lbl">Точность</span>
          </div>
          <div className="login-stat">
            <span className="login-stat-val">7</span>
            <span className="login-stat-lbl">Классов</span>
          </div>
          <div className="login-stat">
            <span className="login-stat-val">10K+</span>
            <span className="login-stat-lbl">Изображений</span>
          </div>
        </div>
      </div>

      <div className="login-right">
        <div className="login-card fade-in">
          <div className="login-tabs">
            <button
              className={`login-tab ${tab === "login" ? "active" : ""}`}
              onClick={() => {
                setTab("login");
                setError("");
              }}
            >
              Вход
            </button>
            <button
              className={`login-tab ${tab === "register" ? "active" : ""}`}
              onClick={() => {
                setTab("register");
                setError("");
              }}
            >
              Регистрация
            </button>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            {tab === "register" && (
              <div className="form-group">
                <label className="label">Полное имя</label>
                <input
                  className="input"
                  placeholder="Иванов Иван Иванович"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
            )}

            <div className="form-group">
              <label className="label">Email</label>
              <input
                className="input"
                type="email"
                placeholder="doctor@clinic.kz"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="label">Пароль</label>
              <input
                className="input"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
              />
            </div>

            {error && (
              <div className="login-error">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                </svg>
                {error}
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary login-submit"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 16, height: 16 }} />{" "}
                  Подождите...
                </>
              ) : tab === "login" ? (
                "Войти"
              ) : (
                "Создать аккаунт"
              )}
            </button>
          </form>

          {tab === "login" && (
            <p className="login-hint">
              Демо: <span className="mono">admin@dermai.kz</span> /{" "}
              <span className="mono">admin123</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
