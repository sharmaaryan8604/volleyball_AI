import React, { useState } from 'react'
import { Zap, AlertCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

import Heatmap from '../components/Heatmap'
import { Select, Toggle, NumberInput } from '../components/FormControls'
import {
  HITTER_ZONE_OPTIONS,
  SET_LOCATION_OPTIONS,
  HIT_TYPE_OPTIONS,
  SERVE_TYPE_OPTIONS,
} from '../utils/constants'
import { predictLandingZones, predictMarkov } from '../api/client'
import './Predict.css'

const PASS_OPTS  = [{ value: 'bad', label: 'Bad' }, { value: 'good', label: 'Good' }]
const BLOCK_OPTS = [{ value: false, label: 'No'  }, { value: true,  label: 'Yes'  }]

const defaultForm = {
  hitterZone:  8,
  setLocation: 1,
  passQuality: 'good',
  blockTouch:  false,
  numBlockers: 2,
  hitType:     '',
  serveType:   '',
}

export default function Predict() {
  const [form, setForm]             = useState(defaultForm)
  const [xgbResult, setXgbResult]   = useState(null)
  const [markovResult, setMarkov]   = useState(null)
  const [activeTab, setActiveTab]   = useState('xgb')
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)

  const set = (key, val) => setForm(f => ({ ...f, [key]: val }))

  async function handlePredict() {
    setLoading(true)
    setError(null)
    try {
      const [xgb, markov] = await Promise.all([
        predictLandingZones(form),
        predictMarkov(form).catch(() => null),  // ignore markov errors if the request fails
      ])
      setXgbResult(xgb)
      setMarkov(markov)
      setActiveTab('xgb')
    } catch (err) {
      const detail = err.response?.data?.detail
      if (Array.isArray(detail)) {
        const msgs = detail.map(d => `${d.loc?.slice(-1)[0] ?? 'field'}: ${d.msg}`).join(' · ')
        setError(`Validation error — ${msgs}`)
      } else if (typeof detail === 'string') {
        setError(detail)
      } else {
        setError(`Predict failed: ${err.response?.status ?? err.message}`)
      }
    } finally {
      setLoading(false)
    }
  }

  const activeResult = activeTab === 'xgb' ? xgbResult : markovResult

  return (
    <div className="predict-page">
      {/* ── Left panel: attack params ── */}
      <div className="predict-form-panel">
        <div className="page-header">
          <h1 className="page-title">Zone Predictor</h1>
        </div>

        <div className="param-card">
          <h2 className="param-card__title">ATTACK PARAMETERS</h2>

          <div className="param-grid">
            <Select
              label="Hitter Zone"
              tag="8–15"
              options={HITTER_ZONE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
              value={form.hitterZone}
              onChange={v => set('hitterZone', parseInt(v, 10))}
            />
            <Select
              label="Set Type"
              options={SET_LOCATION_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
              value={form.setLocation}
              onChange={v => set('setLocation', parseInt(v, 10))}
            />

            <Toggle
              label="Pass Quality"
              options={PASS_OPTS}
              value={form.passQuality}
              onChange={v => set('passQuality', v)}
            />
            <Toggle
              label="Block Touch"
              options={BLOCK_OPTS}
              value={form.blockTouch}
              onChange={v => set('blockTouch', v === 'true' || v === true)}
            />

            <NumberInput
              label="Num Blockers"
              value={form.numBlockers}
              onChange={v => set('numBlockers', v)}
              min={0}
              max={3}
            />

            <Select
              label="Hit Type"
              options={HIT_TYPE_OPTIONS}
              value={form.hitType}
              onChange={v => set('hitType', v)}
            />
            <Select
              label="Serve Type"
              options={SERVE_TYPE_OPTIONS}
              value={form.serveType}
              onChange={v => set('serveType', v)}
            />
          </div>

          <AnimatePresence>
            {error && (
              <motion.div
                className="error-box"
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
              >
                <AlertCircle size={14} />
                <span>{error}</span>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            className={`predict-btn${loading ? ' predict-btn--loading' : ''}`}
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? (
              <span className="spinner" />
            ) : (
              <Zap size={16} strokeWidth={2.5} />
            )}
            {loading ? 'Predicting…' : 'Predict Landing Zones'}
          </button>
        </div>
      </div>

      {/* ── Right panel: heatmap ── */}
      <div className="predict-heatmap-panel">
        {/* Model tabs */}
        {(xgbResult || markovResult) && (
          <div className="model-tabs">
            <button
              className={`model-tab${activeTab === 'xgb' ? ' model-tab--active' : ''}`}
              onClick={() => setActiveTab('xgb')}
            >
              XGB + LGB
            </button>
            {markovResult && (
              <button
                className={`model-tab${activeTab === 'markov' ? ' model-tab--active' : ''}`}
                onClick={() => setActiveTab('markov')}
              >
                Markov
              </button>
            )}
          </div>
        )}

        <Heatmap predictions={activeResult} loading={loading} />
      </div>
    </div>
  )
}
