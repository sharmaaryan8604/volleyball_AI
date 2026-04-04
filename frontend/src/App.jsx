import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar   from './components/Sidebar'
import Predict   from './pages/Predict'
import Simulate  from './pages/Simulate'
import Dashboard from './pages/Dashboard'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="app-main">
          <Routes>
            <Route path="/"          element={<Predict />}   />
            <Route path="/simulate"  element={<Simulate />}  />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="*"          element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
