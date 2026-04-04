import axios from "axios"

/*
Backend base URL
Uses VITE_API_URL in production (Vercel)
Falls back to Render backend locally
*/
const BASE_URL =
  import.meta.env.VITE_API_URL || "https://volleyball-ai.onrender.com"

/*
Axios instance
*/
const client = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
})

/*
Dropdown string → numeric mapping
Matches backend encoding
*/

const SET_TYPE_MAP = {
  quick: 0,
  pipe: 1,
  outside: 2,
  opposite: 3,
  back_row: 4,
}

const HIT_TYPE_MAP = {
  joust: 0,
  line: 1,
  cross: 2,
  tip: 3,
  roll: 4,
  power: 5,
}

const SERVE_TYPE_MAP = {
  jump_serve: 0,
  float_serve: 1,
  jump_float: 2,
  topspin: 3,
}

/*
Remove empty values
Prevents FastAPI validation errors
*/

function cleanPayload(obj) {
  const cleaned = {}

  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue
    if (typeof value === "string" && value.trim() === "") continue
    if (typeof value === "number" && Number.isNaN(value)) continue

    cleaned[key] = value
  }

  return cleaned
}

//
// ─────────────────────────────────────────
// Hybrid ML prediction endpoint
// ─────────────────────────────────────────
//

export async function predictLandingZones(params) {
  const payload = cleanPayload({
    hitter_location: Number(params.hitterZone),

    set_location: Number(params.setLocation),

    pass_rating: params.passQuality === "good" ? 1 : 0,

    block_touch: params.blockTouch ? 1 : 0,

    num_blockers:
      typeof params.numBlockers === "string"
        ? parseInt(params.numBlockers, 10)
        : params.numBlockers,

    set_type:
      params.setType && SET_TYPE_MAP[params.setType] !== undefined
        ? SET_TYPE_MAP[params.setType]
        : undefined,

    hit_type:
      params.hitType && HIT_TYPE_MAP[params.hitType] !== undefined
        ? HIT_TYPE_MAP[params.hitType]
        : undefined,

    serve_type:
      params.serveType && SERVE_TYPE_MAP[params.serveType] !== undefined
        ? SERVE_TYPE_MAP[params.serveType]
        : undefined,
  })

  console.log("Sending payload:", payload) // debug log

  const { data } = await client.post("/predict", payload)

  return data
}

//
// ─────────────────────────────────────────
// Markov prediction endpoint
// ─────────────────────────────────────────
//

export async function predictMarkov(params) {
  const payload = cleanPayload({
    hitter_location: Number(params.hitterZone),

    set_location: Number(params.setLocation),

    pass_rating: params.passQuality === "good" ? 1 : 0,

    block_touch: params.blockTouch ? 1 : 0,

    num_blockers:
      typeof params.numBlockers === "string"
        ? parseInt(params.numBlockers, 10)
        : params.numBlockers,

    set_type:
      params.setType && SET_TYPE_MAP[params.setType] !== undefined
        ? SET_TYPE_MAP[params.setType]
        : undefined,

    hit_type:
      params.hitType && HIT_TYPE_MAP[params.hitType] !== undefined
        ? HIT_TYPE_MAP[params.hitType]
        : undefined,

    serve_type:
      params.serveType && SERVE_TYPE_MAP[params.serveType] !== undefined
        ? SERVE_TYPE_MAP[params.serveType]
        : undefined,
  })

  const { data } = await client.post("/predict/markov", payload)

  return data
}

//
// ─────────────────────────────────────────
// Simulation endpoint
// ─────────────────────────────────────────
//

export async function runSimulation(params) {
  const payload = cleanPayload({
    num_rallies: params.numRallies || 100,

    hitter_zone:
      params.hitterZone !== undefined
        ? Number(params.hitterZone)
        : undefined,

    pass_rating:
      params.passRating !== undefined
        ? Number(params.passRating)
        : 1,

    set_loc:
      params.setLoc !== undefined
        ? Number(params.setLoc)
        : 1,

    strategy: params.strategy || undefined,
  })

  const { data } = await client.post("/simulate", payload)

  return data
}

//
// ─────────────────────────────────────────
// Dashboard stats endpoint
// ─────────────────────────────────────────
//

export async function getDashboardStats() {
  const { data } = await client.get("/stats")

  return data
}

export default client
