import { useState } from 'react'
import { Zap, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import { api } from '../api.js'
import Court from '../components/Court.jsx'
import './PredictPage.css'

const DEFAULTS = {
  hitter_location: 4,
  set_location:    1,
  pass_rating:     1,
  num_blockers:    2,
  block_touch:     0,
  set_type:        '',
  hit_type:        '',
  serve_type:      '',
}

const SET_TYPES  = [1, 2, 3, 4, 5]
const HIT_TYPES  = [1, 2, 3, 4]
const SERVE_TYPES = [1, 2, 3]

export default function PredictPage() {
  const [form, setForm]       = useState(DEFAULTS)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  function set(k, v) { setForm(f => ({ ...f, [k]: v })) }

  async function handleSubmit() {
    setLoading(true)
    setError(null)
    try {
      const body = {
        hitter_location: Number(form.hitter_location),
        set_location:    Number(form.set_location),
        pass_rating:     Number(form.pass_rating),
        num_blockers:    form.num_blockers !== '' ? Number(form.num_blockers) : null,
        block_touch:     form.block_touch !== '' ? Number(form.block_touch) : null,
        set_type:        form.set_type !== '' ? Number(form.set_type) : null,
        hit_type:        form.hit_type !== '' ? Number(form.hit_type) : null,
        serve_type:      form.serve_type !== '' ? Number(form.serve_type) : null,
      }
      const res = await api.predict(body)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="predict-page">
      <header className="page-header">
        <h1>Zone Predictor</h1>
        <p className="subtext">XGBoost + LightGBM + Markov hybrid — predicts hit landing zones 1-25</p>
      </header>

      <div className="predict-layout">
        {/* ── Form ── */}
        <section className="form-card">
          <h2 className="section-title">Attack Parameters</h2>

          <div className="form-grid">
            <FormField label="Hitter Zone" hint="1-15">
              <select value={form.hitter_location} onChange={e => set('hitter_location', e.target.value)}>
                {Array.from({ length: 15 }, (_, i) => (
                  <option key={i+1} value={i+1}>Zone {i+1}</option>
                ))}
              </select>
            </FormField>

            <FormField label="Set Location" hint="1-8">
              <select value={form.set_location} onChange={e => set('set_location', e.target.value)}>
                {Array.from({ length: 8 }, (_, i) => (
                  <option key={i+1} value={i+1}>Location {i+1}</option>
                ))}
              </select>
            </FormField>

            <FormField label="Pass Quality">
              <div className="toggle-row">
                {[{v:0,l:'Bad'},{v:1,l:'Good'}].map(({v,l}) => (
                  <button
                    key={v}
                    className={`toggle-btn ${form.pass_rating === v ? 'active' : ''}`}
                    onClick={() => set('pass_rating', v)}
                  >
                    {l}
                  </button>
                ))}
              </div>
            </FormField>

            <FormField label="Block Touch">
              <div className="toggle-row">
                {[{v:0,l:'No'},{v:1,l:'Yes'}].map(({v,l}) => (
                  <button
                    key={v}
                    className={`toggle-btn ${form.block_touch === v ? 'active' : ''}`}
                    onClick={() => set('block_touch', v)}
                  >
                    {l}
                  </button>
                ))}
              </div>
            </FormField>

            <FormField label="Num Blockers">
              <input
                type="number"
                min={0} max={3}
                value={form.num_blockers}
                onChange={e => set('num_blockers', e.target.value)}
              />
            </FormField>

            <FormField label="Set Type" hint="optional">
              <select value={form.set_type} onChange={e => set('set_type', e.target.value)}>
                <option value="">— any —</option>
                {SET_TYPES.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </FormField>

            <FormField label="Hit Type" hint="optional">
              <select value={form.hit_type} onChange={e => set('hit_type', e.target.value)}>
                <option value="">— any —</option>
                {HIT_TYPES.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </FormField>

            <FormField label="Serve Type" hint="optional">
              <select value={form.serve_type} onChange={e => set('serve_type', e.target.value)}>
                <option value="">— any —</option>
                {SERVE_TYPES.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </FormField>
          </div>

          <button className="predict-btn" onClick={handleSubmit} disabled={loading}>
            {loading
              ? <><Loader2 size={16} className="spin" /> Running model…</>
              : <><Zap size={16} /> Predict Landing Zones</>
            }
          </button>

          {error && (
            <div className="error-bar">
              <AlertCircle size={15} />
              {error}
            </div>
          )}
        </section>

        {/* ── Results ── */}
        <section className="results-card">
          <h2 className="section-title">
            Heatmap
            {result && (
              <span className={`tag ${result.markov_hit ? 'tag-green' : 'tag-blue'}`}>
                {result.markov_hit ? 'Markov hit' : 'ML only'}
              </span>
            )}
          </h2>

          <Court
            probs={result?.all_probs || []}
            topZones={result?.top5_zones || []}
          />

          {result && (
            <div className="top-zones fade-up">
              <div className="top-zones-label">Top predicted zones</div>
              <div className="top-zones-list">
                {result.top_zones.slice(0, 5).map((z, i) => (
                  <div key={z.zone} className={`zone-chip rank-${i}`}>
                    <span className="zone-rank">#{i+1}</span>
                    <span className="zone-num">Zone {z.zone}</span>
                    <span className="zone-prob">{(z.probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!result && (
            <div className="court-placeholder">
              <p>Fill in attack parameters and click <strong>Predict</strong></p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

function FormField({ label, hint, children }) {
  return (
    <div className="form-field">
      <label>
        {label}
        {hint && <span className="hint">{hint}</span>}
      </label>
      {children}
    </div>
  )
}
