import React from 'react'
import { NavLink } from 'react-router-dom'
import { Activity, Shuffle, BarChart2 } from 'lucide-react'
import './Sidebar.css'

const nav = [
  { to: '/',          icon: Activity,  label: 'Predict'   },
  { to: '/simulate',  icon: Shuffle,   label: 'Simulate'  },
  { to: '/dashboard', icon: BarChart2, label: 'Dashboard' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar__logo">
        <div className="sidebar__logo-icon">
          <Activity size={18} strokeWidth={2.5} />
        </div>
        <div className="sidebar__logo-text">
          <span className="sidebar__logo-vol">Volleyball</span>
          <span className="sidebar__logo-title">Attack Zone</span>
          <span className="sidebar__logo-subtitle">Predictor</span>
        </div>
      </div>

      <nav className="sidebar__nav">
        {nav.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `sidebar__link${isActive ? ' sidebar__link--active' : ''}`
            }
          >
            <Icon size={16} strokeWidth={2} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__badges">
        <span className="badge badge--xgb">XGB + LGB</span>
        <span className="badge badge--markov">MARKOV</span>
      </div>
    </aside>
  )
}
