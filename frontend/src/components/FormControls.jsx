import React from 'react'
import { ChevronDown } from 'lucide-react'
import './FormControls.css'

/* ── Select ──────────────────────────────────────────────────────────── */
export function Select({ label, tag, options, value, onChange }) {
  return (
    <div className="fc-field">
      {label && (
        <label className="fc-label">
          {label}
          {tag && <span className="fc-tag">{tag}</span>}
        </label>
      )}
      <div className="fc-select-wrap">
        <select
          className="fc-select"
          value={value}
          onChange={e => onChange(e.target.value)}
        >
          {options.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown size={14} className="fc-select-icon" />
      </div>
    </div>
  )
}

/* ── Toggle button pair ───────────────────────────────────────────────── */
export function Toggle({ label, options, value, onChange }) {
  return (
    <div className="fc-field">
      {label && <label className="fc-label">{label}</label>}
      <div className="fc-toggle">
        {options.map(opt => (
          <button
            key={opt.value}
            type="button"
            className={`fc-toggle-btn${value === opt.value ? ' fc-toggle-btn--active' : ''}`}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ── Number input ─────────────────────────────────────────────────────── */
export function NumberInput({ label, tag, value, onChange, min = 0, max = 10 }) {
  return (
    <div className="fc-field">
      {label && (
        <label className="fc-label">
          {label}
          {tag && <span className="fc-tag">{tag}</span>}
        </label>
      )}
      <input
        type="number"
        className="fc-input"
        value={value}
        min={min}
        max={max}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  )
}
