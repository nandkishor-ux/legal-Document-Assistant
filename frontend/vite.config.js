import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // The API runs on 127.0.0.1:8000. Routing /ask and /health through this
    // proxy keeps the browser on a single origin (avoids CORS entirely),
    // while allowing direct calls to http://127.0.0.1:8000 too.
    proxy: {
      '/ask': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
