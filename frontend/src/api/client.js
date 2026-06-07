// api/client.js — централизованный HTTP клиент с автообновлением токена
const BASE = "http://localhost:8000"

async function request(path, options = {}) {
  const access = localStorage.getItem("dermai_access")

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...(access ? { Authorization: `Bearer ${access}` } : {}),
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
    },
  })

  // Автообновление токена
  if (res.status === 401) {
    const refresh = localStorage.getItem("dermai_refresh")
    if (refresh) {
      const refreshRes = await fetch(`${BASE}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      })
      if (refreshRes.ok) {
        const data = await refreshRes.json()
        localStorage.setItem("dermai_access",  data.access_token)
        localStorage.setItem("dermai_refresh", data.refresh_token)
        const user = JSON.parse(localStorage.getItem("dermai_user") || "{}")
        localStorage.setItem("dermai_user", JSON.stringify({ ...user, ...data }))

        // Повторяем оригинальный запрос
        return request(path, options)
      }
    }
    // Токен не обновился — выходим
    localStorage.clear()
    window.location.href = "/login"
    return
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Ошибка сервера" }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }

  return res.json()
}

export const api = {
  get:    (path)              => request(path),
  post:   (path, body)        => request(path, { method: "POST", body: JSON.stringify(body) }),
  patch:  (path, body)        => request(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: (path)              => request(path, { method: "DELETE" }),
  upload: (path, formData)    => request(path, { method: "POST", body: formData }),
  imgUrl: (path)              => `${BASE}${path}?token=${localStorage.getItem("dermai_access")}`,
}
