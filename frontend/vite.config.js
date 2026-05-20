import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy all /api requests to the backend
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        ws: true,
        // Needed for Gradio SSE (Server-Sent Events) streaming
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            // Ensure proper headers for SSE
            proxyReq.setHeader('Connection', 'keep-alive');
          });
        },
      },
    },
  },
})
