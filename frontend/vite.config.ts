import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // Оптимизации для production
    minify: 'terser',
    rollupOptions: {
      output: {
        // Разделение кода для лучшего кэширования
        manualChunks: {
          vendor: ['react', 'react-dom'],
          plotly: ['react-plotly.js', 'plotly.js'],
          ui: ['framer-motion', 'lucide-react']
        }
      }
    },
    // Сжатие и оптимизация
    cssCodeSplit: true,
    sourcemap: false, // Отключаем sourcemap для production
    chunkSizeWarningLimit: 1000
  },
  server: {
    // Настройки для dev сервера
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/load-tic': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/analyze': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
