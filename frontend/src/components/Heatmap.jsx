import React, { useMemo } from 'react'
import {
  LANDING_ZONES,
  OUTSIDE_ZONE_KEY,
  OUTSIDE_ZONE_LABEL,
  heatColor,
  heatTextColor,
} from '../utils/constants'
import './Heatmap.css'

/**
 * props:
 *  predictions  – structured API response or a flat probability map
 *  loading      – bool
 */
export default function Heatmap({ predictions, loading }) {
  const { probMap, maxProb, sorted } = useMemo(() => {
    if (!predictions) return { probMap: {}, maxProb: 0, sorted: [] }

    let raw = predictions.zone_probabilities

    if (!raw && Array.isArray(predictions.all_probs)) {
      raw = Object.fromEntries(
        predictions.all_probs.map((prob, index) => [String(index + 1), prob])
      )
    }

    if (!raw && Array.isArray(predictions.top_zones)) {
      raw = Object.fromEntries(
        predictions.top_zones.map(zone => [
          String(zone.zone ?? zone),
          zone.probability ?? zone.prob ?? 0,
        ])
      )
    }

    if (!raw) raw = predictions

    const probMap = {}
    let outsideProb = 0
    let maxProb = 0

    for (const [k, vRaw] of Object.entries(raw)) {
      const v = Number(vRaw) || 0
      const zone = parseInt(k, 10)

      if (!Number.isNaN(zone)) {
        if (zone >= 16) {
          outsideProb += v
        } else {
          probMap[zone] = v
          if (v > maxProb) maxProb = v
        }
      } else if (k === OUTSIDE_ZONE_KEY || k === OUTSIDE_ZONE_LABEL) {
        outsideProb += v
      }
    }

    if (outsideProb > 0) {
      probMap[OUTSIDE_ZONE_KEY] = outsideProb
      if (outsideProb > maxProb) maxProb = outsideProb
    }

    const sorted = Object.entries(probMap)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)

    return { probMap, maxProb, sorted }
  }, [predictions])

  return (
    <div className="heatmap-wrapper">
      <div className="heatmap-header">
        <h3 className="heatmap-title">HEATMAP</h3>
        {predictions && (
          <span className="heatmap-subtitle">Landing Zone Probability</span>
        )}
      </div>

      <div className="heatmap-court">
        <div className="heatmap-inside-shell">
          {/* Inside court: zones 1-15 */}
          {LANDING_ZONES.map((row, ri) => (
            <div key={ri} className="heatmap-row">
              {row.map(zone => {
                const prob = probMap[zone] ?? 0
                const pct = maxProb ? prob / maxProb : 0
                return (
                  <HeatCell
                    key={zone}
                    zone={zone}
                    prob={prob}
                    pct={pct}
                    maxProb={maxProb}
                    loading={loading}
                  />
                )
              })}
            </div>
          ))}
        </div>

        {/* Divider between inside and outside */}
        <div className="heatmap-divider" />
        <div className="heatmap-outside-row">
          <HeatCell
          zone="Outside"
            label="Outside"
            prob={probMap[OUTSIDE_ZONE_KEY] ?? 0}
            pct={maxProb ? (probMap[OUTSIDE_ZONE_KEY] ?? 0) / maxProb : 0}
            maxProb={maxProb}
            loading={loading}
            isOutsideZone
          />
        </div>
      </div>

      {/* Legend */}
      <div className="heatmap-legend">
        <span className="legend-label">Low</span>
        <div className="legend-gradient" />
        <span className="legend-label">High</span>
      </div>

      {/* Top zones list */}
      {sorted.length > 0 && (
        <div className="heatmap-topzones">
          <p className="topzones-title">Top predicted zones</p>
          <div className="topzones-list">
            {sorted.map(([zone, prob]) => (
              <div key={zone} className="topzone-item">
                <span className={`topzone-num${zone === OUTSIDE_ZONE_KEY ? ' topzone-num--outside' : ''}`}>
                  {zone === OUTSIDE_ZONE_KEY ? 'Outside' : `Z${zone}`}
                </span>
                <div className="topzone-bar-wrap">
                  <div
                    className="topzone-bar"
                    style={{ width: `${(prob / maxProb) * 100}%` }}
                  />
                </div>
                <span className="topzone-pct">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {!predictions && !loading && (
        <p className="heatmap-placeholder">
          Run a prediction to see landing zone probabilities
        </p>
      )}
    </div>
  )
}

function HeatCell({ zone, label, prob, pct, maxProb, loading, isOutsideZone }) {
  const bg = loading ? undefined : heatColor(prob, maxProb)
  const color = loading ? undefined : heatTextColor(prob, maxProb)
  const isHot = pct > 0.6

  return (
    <div
      className={`heat-cell${loading ? ' heat-cell--loading' : ''}${isHot ? ' heat-cell--hot' : ''}${isOutsideZone ? ' heat-cell--outside' : ''}`}
      style={loading ? {} : { background: bg, color }}
      title={prob > 0 ? `${zone}: ${(prob * 100).toFixed(1)}%` : `${zone}`}
    >
      <span className="heat-cell__zone">{label ?? zone}</span>
      {prob > 0 && !loading && (
        <span className="heat-cell__pct">{(prob * 100).toFixed(0)}%</span>
      )}
    </div>
  )
}
