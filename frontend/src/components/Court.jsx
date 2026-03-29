import './Court.css'

/* 
  25-zone court: 5 cols × 5 rows
  Zones 1-5: row 1 (top/back row opponent)
  Zones 6-10: row 2
  Zones 11-15: row 3 (net area)
  Zones 16-20: row 4
  Zones 21-25: row 5 (back row own side)
*/

const ROWS = 5
const COLS = 5

function getZoneCoords(zone) {
  const idx  = zone - 1
  const row  = Math.floor(idx / COLS)
  const col  = idx % COLS
  return { row, col }
}

function lerp(a, b, t) { return a + (b - a) * t }

function probToColor(p, maxP) {
  if (p === 0 || maxP === 0) return 'transparent'
  const t = p / maxP
  if (t < 0.33) {
    const s = t / 0.33
    return `rgba(0,134,255,${lerp(0.05, 0.25, s)})`
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33
    return `rgba(245,166,35,${lerp(0.25, 0.6, s)})`
  } else {
    const s = (t - 0.66) / 0.34
    return `rgba(0,201,167,${lerp(0.6, 0.95, s)})`
  }
}

export default function Court({ probs = [], topZones = [], highlighted = [] }) {
  const maxP = probs.length ? Math.max(...probs) : 0

  const CELL_W = 72
  const CELL_H = 52
  const PAD    = 4
  const W      = COLS * CELL_W + PAD * 2
  const H      = ROWS * CELL_H + PAD * 2

  const top3Set = new Set(topZones.slice(0, 3))
  const top1    = topZones[0]

  return (
    <div className="court-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="court-svg">
        {/* Net line */}
        <line
          x1={PAD}
          y1={H / 2}
          x2={W - PAD}
          y2={H / 2}
          stroke="rgba(255,255,255,0.25)"
          strokeWidth="2"
          strokeDasharray="6 4"
        />
        <text x={W - PAD - 2} y={H / 2 - 4} textAnchor="end" fontSize="9" fill="rgba(255,255,255,0.3)">NET</text>

        {Array.from({ length: 25 }, (_, i) => {
          const zone = i + 1
          const { row, col } = getZoneCoords(zone)
          const x = PAD + col * CELL_W
          const y = PAD + row * CELL_H
          const p = probs[i] || 0
          const bg = probToColor(p, maxP)
          const isTop1  = zone === top1
          const isTop3  = top3Set.has(zone)
          const isHigh  = highlighted.includes(zone)

          return (
            <g key={zone}>
              <rect
                x={x + 1}
                y={y + 1}
                width={CELL_W - 2}
                height={CELL_H - 2}
                rx={4}
                fill={bg}
                stroke={isTop1 ? 'var(--accent)' : isTop3 ? 'rgba(0,201,167,0.4)' : isHigh ? 'var(--accent3)' : 'rgba(255,255,255,0.06)'}
                strokeWidth={isTop1 ? 2 : isTop3 ? 1.5 : 1}
              />
              <text
                x={x + CELL_W / 2}
                y={y + CELL_H / 2 - (p > 0 ? 7 : 0)}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize="11"
                fontWeight={isTop1 ? '700' : '400'}
                fill={p > 0.5 * maxP ? '#fff' : 'rgba(255,255,255,0.45)'}
              >
                {zone}
              </text>
              {p > 0 && (
                <text
                  x={x + CELL_W / 2}
                  y={y + CELL_H / 2 + 9}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize="9"
                  fill={p > 0.5 * maxP ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)'}
                  fontFamily="'DM Mono', monospace"
                >
                  {(p * 100).toFixed(1)}%
                </text>
              )}
              {isTop1 && (
                <rect
                  x={x + CELL_W - 16}
                  y={y + 4}
                  width={12}
                  height={12}
                  rx={6}
                  fill="var(--accent)"
                />
              )}
            </g>
          )
        })}

        {/* Grid lines */}
        {Array.from({ length: COLS + 1 }, (_, i) => (
          <line
            key={`v${i}`}
            x1={PAD + i * CELL_W}
            y1={PAD}
            x2={PAD + i * CELL_W}
            y2={H - PAD}
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="1"
          />
        ))}
        {Array.from({ length: ROWS + 1 }, (_, i) => (
          <line
            key={`h${i}`}
            x1={PAD}
            y1={PAD + i * CELL_H}
            x2={W - PAD}
            y2={PAD + i * CELL_H}
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="1"
          />
        ))}
      </svg>

      <div className="court-legend">
        <span>Low</span>
        <div className="legend-bar" />
        <span>High</span>
      </div>
    </div>
  )
}
