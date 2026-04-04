import React from 'react'
import { BarChart3, ShieldCheck, Trophy, Users } from 'lucide-react'
import './Dashboard.css'

const METRICS = [
  {
    icon: <Trophy size={16} />,
    label: 'Top-1 accuracy',
    value: '58.08%',
    note: 'Primary landing-zone prediction',
  },
  {
    icon: <BarChart3 size={16} />,
    label: 'Top-3 accuracy',
    value: '77.88%',
    note: 'Shortlist coverage',
  },
  {
    icon: <Users size={16} />,
    label: 'Top-5 accuracy',
    value: '84.52%',
    note: 'Expanded shortlist coverage',
  },
  {
    icon: <ShieldCheck size={16} />,
    label: 'Top-10 accuracy',
    value: '92.87%',
    note: 'Broad landing-zone coverage',
  },
  {
    icon: <BarChart3 size={16} />,
    label: 'ECE score',
    value: '0.0183',
    note: 'Calibration is nearly perfect',
  },
]

export default function Dashboard() {
  return (
    <div className="dashboard-page summary-dashboard">
      <section className="summary-hero">
        <div>
          <p className="summary-eyebrow">Volleyball attack prediction</p>
          <h1 className="page-title summary-title">Key model performance</h1>
        </div>

        <div className="summary-badge">
          <Trophy size={14} />
          <span>Hybrid ML + Markov Chain</span>
        </div>
      </section>

      <section className="summary-grid" aria-label="Key performance metrics">
        {METRICS.map(metric => (
          <MetricCard key={metric.label} {...metric} />
        ))}
      </section>
    </div>
  )
}

function MetricCard({ icon, label, value, note }) {
  return (
    <article className="summary-card">
      <div className="summary-card__icon">{icon}</div>
      <div className="summary-card__body">
        <p className="summary-card__label">{label}</p>
        <p className="summary-card__value">{value}</p>
        <p className="summary-card__note">{note}</p>
      </div>
    </article>
  )
}
