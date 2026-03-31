import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import StudentView from './components/StudentView.jsx'

const isStudent = window.location.hash === '#student'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {isStudent ? <StudentView onBack={() => { window.location.hash = ''; window.location.reload(); }} /> : <App />}
  </StrictMode>
)