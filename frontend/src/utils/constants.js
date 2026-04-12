// Volleyball landing zones shown in the predictor UI.
// Zones 1-15 are the playable landing court.
// Zones 16+ are treated as Outside.
export const LANDING_ZONES = [
  [1,  2,  3,  4,  5],
  [6,  7,  8,  9,  10],
  [11, 12, 13, 14, 15],  // net row
]

export const OUTSIDE_ZONE_LABEL = 'Outside'
export const OUTSIDE_ZONE_KEY = 'outside'

// Hitter zones 8-15 (attacker's side of the court for this UI flow)
export const HITTER_ZONE_OPTIONS = Array.from({ length: 8 }, (_, i) => ({
  value: i + 8,
  label: `Zone ${i + 8}`,
}))

export const SET_LOCATION_OPTIONS = [
  { value: 1, label: 'Outside' },
  { value: 2, label: 'Oppo' },
  { value: 3, label: 'Quick' },
  { value: 4, label: 'Bic' },
  { value: 5, label: 'Dump' },
  { value: 6, label: 'D-Ball' },
  { value: 7, label: 'In' },
  { value: 8, label: 'Blocked' },
]

export const HIT_TYPE_OPTIONS = [
  { value: '', label: '— any —' },
  { value: 'joust',   label: 'Joust' },
  { value: 'line',    label: 'Line' },
  { value: 'cross',   label: 'Cross' },
  { value: 'tip',     label: 'Tip' },
  { value: 'roll',    label: 'Roll' },
  { value: 'power',   label: 'Power' },
]

export const SERVE_TYPE_OPTIONS = [
  { value: '', label: '— any —' },
  { value: 'jump_serve',   label: 'Jump Serve' },
  { value: 'float_serve',  label: 'Float Serve' },
  { value: 'jump_float',   label: 'Jump Float' },
  { value: 'topspin',      label: 'Topspin' },
]

// Heatmap color interpolation: 0→low, 1→high
export function heatColor(value, max) {
  if (!max || max === 0) return 'rgba(26,35,50,0.6)'
  const t = Math.min(value / max, 1)
  if (t === 0) return 'rgba(26,35,50,0.6)'

  // 4-stop gradient: dark-teal → teal → lime-green → orange-red
  const stops = [
    { t: 0.0,  r: 13,  g: 79,  b: 60  },
    { t: 0.3,  r: 0,   g: 160, b: 100 },
    { t: 0.65, r: 0,   g: 229, b: 160 },
    { t: 0.85, r: 255, g: 180, b: 0   },
    { t: 1.0,  r: 255, g: 70,  b: 30  },
  ]

  let lo = stops[0], hi = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].t && t <= stops[i + 1].t) {
      lo = stops[i]; hi = stops[i + 1]; break
    }
  }

  const range = hi.t - lo.t || 1
  const f = (t - lo.t) / range
  const r = Math.round(lo.r + f * (hi.r - lo.r))
  const g = Math.round(lo.g + f * (hi.g - lo.g))
  const b = Math.round(lo.b + f * (hi.b - lo.b))
  const alpha = 0.25 + t * 0.75
  return `rgba(${r},${g},${b},${alpha})`
}

export function heatTextColor(value, max) {
  if (!max || max === 0) return 'var(--text-muted)'
  const t = value / max
  if (t < 0.1) return 'var(--text-muted)'
  if (t < 0.4) return 'var(--accent)'
  return '#fff'
}
