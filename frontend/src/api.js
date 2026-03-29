const BASE = import.meta.env.VITE_API_URL || '/api'

async function request(path, options = {}) {
  const url = BASE.replace(/\/$/, '') + path
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health:   ()      => request('/health'),
  predict:  (body)  => request('/predict',  { method: 'POST', body: JSON.stringify(body) }),
  simulate: (body)  => request('/simulate', { method: 'POST', body: JSON.stringify(body) }),
  markovInfo: ()    => request('/markov/info'),
  zonePrior:  ()    => request('/zones/prior'),
}
