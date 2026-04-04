import React, { useState } from 'react'
import { Shuffle, AlertCircle, TrendingUp, Target, Activity } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'
import { Select, NumberInput } from '../components/FormControls'
import { HITTER_ZONE_OPTIONS } from '../utils/constants'
import { runSimulation } from '../api/client'
import './Simulate.css'

const STRATEGY_OPTIONS = [
  { value: '',           label: '— any —'     },
  { value: 'aggressive', label: 'Aggressive'  },
  { value: 'safe',       label: 'Safe'        },
  { value: 'balanced',   label: 'Balanced'    },
  { value: 'cross',      label: 'Cross Court' },
  { value: 'line',       label: 'Line Shot'   },
]

const ZONE_OPTIONS = [
  { value: '', label: '— all zones —' },
  ...HITTER_ZONE_OPTIONS,
]

export default function Simulate() {
  const [numRallies, setNumRallies]   = useState(100)
  const [hitterZone, setHitterZone]   = useState('')
  const [strategy, setStrategy]       = useState('')
  const [result, setResult]           = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)

  async function handleSimulate() {
    setLoading(true)
    setError(null)
    try {
      const data = await runSimulation({
        numRallies: parseInt(numRallies, 10),
        hitterZone: hitterZone ? parseInt(hitterZone, 10) : undefined,
        strategy:   strategy || undefined,
      })
      setResult(data)
    } catch (err) {
      const detail = err.response?.data?.detail
      if (Array.isArray(detail)) {
        setError(detail.map(d => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join(' · '))
      } else {
        setError(typeof detail === 'string' ? detail : `Simulation failed: ${err.response?.status ?? err.message}`)
      }
    } finally {
      setLoading(false)
    }
  }

  // Build chart data from zone_distribution/distribution if present.
  // Zones 16+ are collapsed into a single Outside bucket.
  const distribution = result?.zone_distribution ?? result?.distribution
  const chartData = distribution
    ? Object.entries(distribution)
        .reduce((acc, [zone, count]) => {
          const zoneNum = Number(zone)

          if (!Number.isNaN(zoneNum) && zoneNum >= 16) {
            acc.outside = (acc.outside ?? 0) + count
          } else {
            acc[zoneNum] = count
          }

          return acc
        }, {})
    : {}

  const chartRows = Object.entries(chartData)
    .map(([zone, count]) => ({
      zone: zone === 'outside' ? 'Outside' : `Z${zone}`,
      count,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 15)

  const ACCENT = '#00e5a0'
  const DIM    = '#1a8a5a'

  return (
    <div className="simulate-page">
      <div className="page-header">
        <h1 className="page-title">Simulate</h1>
        <p className="page-sub">Run rally simulations across attack scenarios</p>
      </div>

      <div className="simulate-layout">
        {/* Controls */}
        <div className="sim-card">
          <h2 className="param-card__title">SIMULATION PARAMETERS</h2>

          <div className="sim-grid">
            <NumberInput
              label="Num Rallies"
              value={numRallies}
              onChange={setNumRallies}
              min={10}
              max={10000}
            />
            <Select
              label="Hitter Zone"
              options={ZONE_OPTIONS}
              value={hitterZone}
              onChange={setHitterZone}
            />
            <Select
              label="Strategy"
              options={STRATEGY_OPTIONS}
              value={strategy}
              onChange={setStrategy}
            />
          </div>

          <AnimatePresence>
            {error && (
              <motion.div className="error-box"
                initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <AlertCircle size={14} /><span>{error}</span>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            className={`predict-btn${loading ? ' predict-btn--loading' : ''}`}
            onClick={handleSimulate}
            disabled={loading}
          >
            {loading ? <span className="spinner" /> : <Shuffle size={16} strokeWidth={2.5} />}
            {loading ? 'Simulating…' : 'Run Simulation'}
          </button>
        </div>

        {/* Results */}
        <AnimatePresence>
          {result && (
            <motion.div className="sim-results"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>

              {/* Stat pills */}
              <div className="sim-stats">
                {result.total_rallies != null && (
                  <StatPill icon={<Activity size={14}/>} label="Total Rallies" value={result.total_rallies} />
                )}
                {result.win_rate != null && (
                  <StatPill icon={<TrendingUp size={14}/>} label="Win Rate"
                    value={`${(result.win_rate * 100).toFixed(1)}%`} accent />
                )}
                {result.avg_zone != null && (
                  <StatPill icon={<Target size={14}/>} label="Avg Landing Zone" value={`Z${result.avg_zone}`} />
                )}
              </div>

              {/* Zone distribution chart */}
              {chartRows.length > 0 && (
                <div className="sim-chart-card">
                  <p className="sim-chart-title">Zone Distribution</p>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chartRows} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
                      <XAxis dataKey="zone" tick={{ fill: '#7a9cc0', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                      <YAxis tick={{ fill: '#7a9cc0', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                      <Tooltip
                        contentStyle={{ background: '#1a2332', border: '1px solid #243447', borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 12 }}
                        itemStyle={{ color: ACCENT }}
                        cursor={{ fill: 'rgba(0,229,160,0.06)' }}
                      />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {chartRows.map((row, i) => (
                          <Cell
                            key={i}
                            fill={row.zone === 'Outside' ? '#ffb347' : i === 0 ? ACCENT : i < 3 ? '#00b87d' : DIM}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Raw JSON fallback */}
              {chartRows.length === 0 && (
                <pre className="sim-raw">{JSON.stringify(result, null, 2)}</pre>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function StatPill({ icon, label, value, accent }) {
  return (
    <div className={`stat-pill${accent ? ' stat-pill--accent' : ''}`}>
      <span className="stat-pill__icon">{icon}</span>
      <div>
        <p className="stat-pill__label">{label}</p>
        <p className="stat-pill__value">{value}</p>
      </div>
    </div>
  )
}
