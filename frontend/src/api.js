// import axios from 'axios'

// const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

// export async function startSession(sessionId) {
//   // ensures session exists on backend (optional)
//   try {
//     const res = await axios.post(`${API_BASE}/sessions`, { user_id: sessionId })
//     return res.data
//   } catch (err) {
//     return null
//   }
// }

// export async function sendMessage(sessionId, text) {
//   const res = await axios.post(`${API_BASE}/sessions/${sessionId}/message`, { text })
//   return res.data
// }
import axios from 'axios'

// Base URL of your backend API
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

// Get session ID from localStorage or create a new one
function getSessionId() {
  let sessionId = localStorage.getItem('session_id')
  if (!sessionId) {
    // Generate a random session ID if not present
    sessionId = 'sess_' + Math.random().toString(36).substring(2, 10)
    localStorage.setItem('session_id', sessionId)
  }
  return sessionId
}

// Start a session with the backend
export async function startSession() {
  const sessionId = getSessionId()
  try {
    const res = await axios.post(`${API_BASE}/sessions`, { user_id: sessionId })
    return res.data
  } catch (err) {
    console.error('Failed to start session', err)
    return null
  }
}

// Send a message to the backend
export async function sendMessage(message) {
  const sessionId = getSessionId()
  try {
    const res = await axios.post(`${API_BASE}/sessions/${sessionId}/message`, { message })
    return res.data
  } catch (err) {
    console.error('Failed to send message', err)
    return null
  }
}
