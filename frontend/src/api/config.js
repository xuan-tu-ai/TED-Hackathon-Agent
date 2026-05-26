// ── Single Source of Truth for all API endpoints ────────────────
// In dev (Vite proxy) & Docker (Nginx): VITE_API_URL = '' → relative URLs
// In Azure (direct):                  VITE_API_URL = 'https://...' → full URLs
// Set the value in frontend/.env or CI env vars.
//
// Usage:  fetch(`${API_BASE}/sim/start`, ...)
//         fetch(`${API_BASE}/robots/${id}/error-history`, ...)

export const API_BASE = `${import.meta.env.VITE_API_URL || ''}/api`;
