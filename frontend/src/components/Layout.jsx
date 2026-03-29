import { Outlet, NavLink } from 'react-router-dom'
import { Activity, BarChart2, Shuffle, Zap } from 'lucide-react'
import './Layout.css'

const NAV = [
  { to: '/',          icon: Zap,       label: 'Predict'   },
  { to: '/simulate',  icon: Shuffle,   label: 'Simulate'  },
  { to: '/dashboard', icon: BarChart2, label: 'Dashboard' },
]

export default function Layout() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <Activity size={20} strokeWidth={2.5} />
          <span>Volleyball<br /><strong>AI</strong></span>
        </div>

        <nav className="sidebar-nav">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => 'nav-item' + (isActive ? ' active' : '')}
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="tag tag-green">XGB + LGB</span>
          <span className="tag tag-blue">Markov</span>
        </div>
      </aside>

      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}
