import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'
import { Shuffle, Loader2, AlertCircle, Play } from 'lucide-react'
import { api } from '../api.js'
import Court from '../components/Court.jsx'
import './SimulatePage.css'

const SCENARIOS = [
  { label: 'Good pass + Outside', pass_rating: 1, set_loc: 1 },
  { label: 'Bad pass + Outside',  pass_rating: 0, set_loc: 1 },
  { label: 'Good pass + Quick',   pass_rating: 1, set_loc: 3 },
  { label: 'Good pass + Opposite',pass_rating: 1, set_loc: 2 },
]

function distToProbs(dist, n) {
  const arr = new Array(25).fill(0)
  for (const [zone, count] of Object.entries(dist)) {
    const idx = parseInt(zone) - 1
    if (idx >= 0 && idx < 25) arr[idx] = count / n
  }
  return arr
}

function distToChartData(dist) {
  return Object.entries(dist)
    .map(([zone, count]) => ({ zone: `Z${zone}`, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 12)
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="tt-row">
        <span>Zone:</span><strong>{payload[0].payload.zone}</strong>
      </div>
      <div className="tt-row">
        <span>Simulated hits:</span><strong>{payload[0].value}</strong>
      </div>
    </div>
  )
}

export default function SimulatePage() {
  const [hitterZone, setHitterZone] = useState(11)
  const [n, setN]                   = useState(5000)
  const [results, setResults]       = useState([])
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [activeIdx, setActiveIdx]   = useState(0)

  async function runAll() {
    setLoading(true)
    setError(null)
    try {
      const runs = await Promise.all(
        SCENARIOS.map(sc =>
          api.simulate({ hitter_zone: hitterZone, n, ...sc })
            .then(res => ({ ...sc, ...res }))
        )
      )
      setResults(runs)
      setActiveIdx(0)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const active = results[activeIdx]
  const probs  = active ? distToProbs(active.distribution, active.n) : []
  const chartData = active ? distToChartData(active.distribution) : []
  const topZone  = active
    ? parseInt(Object.entries(active.distribution).sort((a,b) => b[1]-a[1])[0]?.[0] || '0')
    : null

  return (
    <div className="simulate-page">
      <header className="page-header">
        <h1>Scenario Simulator</h1>
        <p className="subtext">Run Monte-Carlo simulations via the Markov transition matrix</p>
      </header>

      {/* Controls */}
      <div className="sim-controls">
        <div className="ctrl-group">
          <label>Hitter Zone</label>
          <select value={hitterZone} onChange={e => setHitterZone(Number(e.target.value))}>
            {Array.from({ length: 15 }, (_, i) => (
              <option key={i+1} value={i+1}>Zone {i+1}</option>
            ))}
          </select>
        </div>

        <div className="ctrl-group">
          <label>Simulations (n)</label>
          <select value={n} onChange={e => setN(Number(e.target.value))}>
            {[1000, 2500, 5000, 10000].map(v => (
              <option key={v} value={v}>{v.toLocaleString()}</option>
            ))}
          </select>
        </div>

        <button className="run-btn" onClick={runAll} disabled={loading}>
          {loading
            ? <><Loader2 size={15} className="spin" /> Simulating…</>
            : <><Play size={15} /> Run 4 Scenarios</>
          }
        </button>
      </div>

      {error && (
        <div className="error-bar"><AlertCircle size={15} />{error}</div>
      )}

      {results.length > 0 && (
        <div className="sim-results fade-up">
          {/* Scenario tabs */}
          <div className="scenario-tabs">
            {results.map((r, i) => (
              <button
                key={i}
                className={`scenario-tab ${i === activeIdx ? 'active' : ''}`}
                onClick={() => setActiveIdx(i)}
              >
                <span className={`pass-dot ${r.pass_rating === 1 ? 'good' : 'bad'}`} />
                {r.label}
              </button>
            ))}
          </div>

          {active && (
            <div className="sim-detail">
              {/* Stats strip */}
              <div className="sim-stats">
                <div className="sim-stat">
                  <span className="s-val">Zone {topZone}</span>
                  <span className="s-lbl">Top landing zone</span>
                </div>
                <div className="sim-stat">
                  <span className="s-val">{((active.distribution[String(topZone)] || 0) / active.n * 100).toFixed(1)}%</span>
                  <span className="s-lbl">Hit rate to top zone</span>
                </div>
                <div className="sim-stat">
                  <span className="s-val">{active.n.toLocaleString()}</span>
                  <span className="s-lbl">Simulated rallies</span>
                </div>
                <div className="sim-stat">
                  <span className={`s-val ${active.pass_rating ? 'good-text' : 'bad-text'}`}>
                    {active.pass_rating ? 'Good' : 'Bad'} pass
                  </span>
                  <span className="s-lbl">Set loc {active.set_loc}</span>
                </div>
              </div>

              <div className="sim-viz">
                {/* Court heatmap */}
                <div className="court-col">
                  <div className="viz-label">Landing distribution</div>
                  <Court probs={probs} topZones={[topZone]} />
                </div>

                {/* Bar chart */}
                <div className="bar-col">
                  <div className="viz-label">Top 12 zones</div>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={chartData} margin={{ top: 8, right: 10, bottom: 10, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="zone" tick={{ fill: '#8ba3bf', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#8ba3bf', fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
                        {chartData.map((d, i) => (
                          <Cell
                            key={i}
                            fill={i === 0 ? '#00c9a7' : i < 3 ? '#0086ff' : '#253348'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Comparison table across scenarios */}
              {results.length === 4 && (
                <div className="comparison-table">
                  <div className="table-title">Scenario comparison — Zone {hitterZone}</div>
                  <table>
                    <thead>
                      <tr>
                        <th>Scenario</th>
                        <th>Top zone</th>
                        <th>Hit rate</th>
                        <th>Top 3 zones</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((r, i) => {
                        const sorted = Object.entries(r.distribution).sort((a,b) => b[1]-a[1])
                        const t1 = sorted[0]
                        const top3 = sorted.slice(0, 3).map(([z]) => `Z${z}`).join(', ')
                        return (
                          <tr key={i} className={i === activeIdx ? 'active-row' : ''}>
                            <td>
                              <span className={`pass-dot ${r.pass_rating ? 'good' : 'bad'}`} />
                              {r.label}
                            </td>
                            <td><strong>Zone {t1?.[0]}</strong></td>
                            <td className="mono">{((t1?.[1] || 0) / r.n * 100).toFixed(1)}%</td>
                            <td className="mono dimmed">{top3}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {results.length === 0 && !loading && (
        <div className="sim-empty">
          <Shuffle size={32} strokeWidth={1.2} />
          <p>Select a hitter zone and run scenarios to see<br />the Markov-based landing distribution.</p>
        </div>
      )}
    </div>
  )
}
