import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import PredictPage from './pages/PredictPage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import SimulatePage from './pages/SimulatePage.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<PredictPage />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="simulate" element={<SimulatePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
