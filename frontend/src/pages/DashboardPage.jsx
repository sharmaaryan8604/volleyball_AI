import { useState, useEffect } from 'react'
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine, Legend
} from 'recharts'
import { BarChart2, TrendingUp, Target, Cpu } from 'lucide-react'
import './DashboardPage.css'

/* ── Static evaluation data pulled from the dashboard screenshot ── */

const TOP_K_DATA = [
  { k: 1,  coverage: 58.9 },
  { k: 2,  coverage: 71.2 },
  { k: 3,  coverage: 78.6 },
  { k: 4,  coverage: 83.1 },
  { k: 5,  coverage: 85.7 },
  { k: 6,  coverage: 87.4 },
  { k: 7,  coverage: 88.9 },
  { k: 8,  coverage: 90.1 },
  { k: 9,  coverage: 91.0 },
  { k: 10, coverage: 91.8 },
]

const PER_ZONE_DATA = [
  { zone: 1,  acc: 73, status: 'good' },
  { zone: 2,  acc: 50, status: 'mid'  },
  { zone: 3,  acc: 63, status: 'good' },
  { zone: 4,  acc: 50, status: 'mid'  },
  { zone: 5,  acc: 52, status: 'mid'  },
  { zone: 6,  acc: 61, status: 'good' },
  { zone: 7,  acc: 55, status: 'mid'  },
  { zone: 8,  acc: 64, status: 'good' },
  { zone: 9,  acc: 40, status: 'mid'  },
  { zone: 10, acc: 60, status: 'good' },
  { zone: 11, acc: 65, status: 'good' },
  { zone: 12, acc: 64, status: 'good' },
  { zone: 13, acc: 65, status: 'good' },
  { zone: 14, acc: 46, status: 'mid'  },
  { zone: 15, acc: 45, status: 'mid'  },
  { zone: 16, acc: 41, status: 'mid'  },
  { zone: 17, acc: 54, status: 'mid'  },
  { zone: 18, acc: 29, status: 'bad'  },
  { zone: 19, acc: 55, status: 'mid'  },
  { zone: 20, acc: 38, status: 'bad'  },
  { zone: 21, acc: 57, status: 'mid'  },
  { zone: 22, acc: 58, status: 'mid'  },
  { zone: 23, acc: 63, status: 'good' },
  { zone: 24, acc: 64, status: 'good' },
  { zone: 25, acc: 67, status: 'good' },
]

const MRR_HITTER_DATA = [
  { hitter: 4,  mrr: 0.91 },
  { hitter: 8,  mrr: 0.80 },
  { hitter: 3,  mrr: 0.74 },
  { hitter: 6,  mrr: 0.73 },
  { hitter: 15, mrr: 0.72 },
  { hitter: 11, mrr: 0.71 },
  { hitter: 9,  mrr: 0.70 },
  { hitter: 7,  mrr: 0.70 },
  { hitter: 14, mrr: 0.69 },
  { hitter: 13, mrr: 0.68 },
  { hitter: 12, mrr: 0.67 },
  { hitter: 10, mrr: 0.66 },
  { hitter: 5,  mrr: 0.64 },
]

const CALIBRATION_DATA = [
  { predicted: 0.05, actual: 0.02 },
  { predicted: 0.15, actual: 0.08 },
  { predicted: 0.25, actual: 0.43 },
  { predicted: 0.35, actual: 0.61 },
  { predicted: 0.55, actual: 0.64 },
  { predicted: 0.65, actual: 0.67 },
  { predicted: 0.75, actual: 0.76 },
  { predicted: 0.85, actual: 0.80 },
  { predicted: 0.92, actual: 0.82 },
]

const STAT_CARDS = [
  { label: 'Top-1 Accuracy', value: '58.9%', sub: '+26.3pp over majority',   icon: Target,   color: 'green' },
  { label: 'Top-3 Coverage', value: '78.6%', sub: '3 zones capture 4 in 5',  icon: TrendingUp, color: 'blue'  },
  { label: 'MRR',            value: '0.707', sub: 'Mean Reciprocal Rank',     icon: BarChart2, color: 'orange' },
  { label: 'ECE',            value: '0.0183', sub: 'Very well calibrated',    icon: Cpu,       color: 'green' },
]

const ZONE_COLOR = {
  good: '#00c9a7',
  mid:  '#f5a623',
  bad:  '#ff4d6a',
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      {label !== undefined && <div className="tt-label">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="tt-row">
          <span style={{ color: p.color || '#00c9a7' }}>{p.name || p.dataKey}:</span>
          <strong>{typeof p.value === 'number' ? p.value.toFixed(typeof p.value === 'number' && p.value < 2 ? 3 : 1) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

export default function DashboardPage() {
  const [animated, setAnimated] = useState(false)
  useEffect(() => { const t = setTimeout(() => setAnimated(true), 100); return () => clearTimeout(t) }, [])

  return (
    <div className="dashboard-page">
      <header className="page-header">
        <h1>Evaluation Dashboard</h1>
        <p className="subtext">XGB + LGB + Markov Hybrid — full model evaluation</p>
      </header>

      {/* Stat cards */}
      <div className="stat-grid">
        {STAT_CARDS.map(({ label, value, sub, icon: Icon, color }) => (
          <div key={label} className={`stat-card color-${color}`}>
            <div className="stat-icon"><Icon size={18} /></div>
            <div className="stat-body">
              <div className="stat-value">{value}</div>
              <div className="stat-label">{label}</div>
              <div className="stat-sub">{sub}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts row 1 */}
      <div className="charts-row">
        <div className="chart-card">
          <div className="chart-title">Top-k Zone Coverage</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={TOP_K_DATA} margin={{ top: 8, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="k" tick={{ fill: '#8ba3bf', fontSize: 11 }} />
              <YAxis tick={{ fill: '#8ba3bf', fontSize: 11 }} domain={[50, 95]} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="coverage"
                stroke="#00c9a7"
                strokeWidth={2.5}
                dot={{ fill: '#00c9a7', r: 3 }}
                animationDuration={animated ? 1200 : 0}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Calibration Diagram <span className="ece-badge">ECE = 0.0183</span></div>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 8, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="predicted" name="Mean Predicted" type="number" domain={[0,1]} tick={{ fill: '#8ba3bf', fontSize: 11 }} />
              <YAxis dataKey="actual"    name="Actual Freq"    type="number" domain={[0,1]} tick={{ fill: '#8ba3bf', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
              <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke="rgba(255,255,255,0.15)" strokeDasharray="6 4" label={{ value: 'Perfect', fill: '#4d6880', fontSize: 10 }} />
              <Scatter data={CALIBRATION_DATA} fill="#0086ff" opacity={0.85} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-zone accuracy */}
      <div className="chart-card wide">
        <div className="chart-title">
          Per-Zone Top-1 Accuracy
          <div className="legend-inline">
            {Object.entries({ Good: 'good', Mid: 'mid', Low: 'bad' }).map(([label, status]) => (
              <span key={label} className="legend-dot-item">
                <span className="legend-dot" style={{ background: ZONE_COLOR[status] }} />
                {label}
              </span>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={PER_ZONE_DATA} margin={{ top: 8, right: 20, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="zone" tick={{ fill: '#8ba3bf', fontSize: 10 }} />
            <YAxis tick={{ fill: '#8ba3bf', fontSize: 11 }} domain={[0, 80]} unit="%" />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={58.9} stroke="rgba(255,255,255,0.2)" strokeDasharray="5 5" label={{ value: 'Overall', fill: '#4d6880', fontSize: 10 }} />
            <Bar dataKey="acc" name="Top-1 %" radius={[3, 3, 0, 0]}>
              {PER_ZONE_DATA.map((d, i) => (
                <Cell key={i} fill={ZONE_COLOR[d.status]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* MRR by hitter */}
      <div className="chart-card wide">
        <div className="chart-title">MRR by Hitter Zone <span className="ece-badge">Mean 0.707</span></div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart
            data={[...MRR_HITTER_DATA].sort((a, b) => a.mrr - b.mrr)}
            layout="vertical"
            margin={{ top: 8, right: 20, bottom: 0, left: 30 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis type="number" domain={[0.5, 1.0]} tick={{ fill: '#8ba3bf', fontSize: 11 }} />
            <YAxis type="category" dataKey="hitter" tick={{ fill: '#8ba3bf', fontSize: 11 }} width={40} tickFormatter={v => `Z${v}`} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x={0.707} stroke="rgba(255,77,106,0.5)" strokeDasharray="5 5" />
            <Bar dataKey="mrr" name="MRR" fill="#9b6dff" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
