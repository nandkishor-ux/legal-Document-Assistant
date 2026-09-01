// Thin client for the RTI Legal Document Assistant FastAPI backend.
//
// The Vite dev server (vite.config.js) proxies /ask and /health to
// http://127.0.0.1:8000, which keeps the app single-origin (no CORS in dev).
// As a fallback we also try the backend directly.

const BACKEND_BASE = 'http://127.0.0.1:8000'

const DEFAULT_TIMEOUT_MS = 120_000 // the RAG pipeline can take a while

async function request(path, { method = 'GET', body } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS)

  const fetchAttempt = async (url) => {
    const res = await fetch(url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    if (!res.ok) {
      let detail = `Request failed with status ${res.status}`
      try {
        const data = await res.json()
        if (data && data.detail) detail = data.detail
      } catch {
        /* keep default message */
      }
      const err = new Error(detail)
      err.status = res.status
      throw err
    }
    return res.json()
  }

  try {
    // Try the same-origin Vite proxy first, then the direct backend.
    const proxyUrl = path
    try {
      return await fetchAttempt(proxyUrl)
    } catch (e) {
      if (e.name === 'AbortError') throw e
      // Fall back to the direct backend (handles CORS on the FastAPI side).
      return await fetchAttempt(BACKEND_BASE + path)
    }
  } finally {
    clearTimeout(timer)
  }
}

export function ask(question) {
  return request('/ask', { method: 'POST', body: { question } })
}

export function health() {
  return request('/health')
}
