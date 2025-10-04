import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // Оптимизация сборки
    target: 'es2015',
    minify: 'esbuild', // Используем esbuild для лучшей производительности
    rollupOptions: {
      output: {
        // Разделение кода для лучшего кэширования
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['lucide-react', 'framer-motion'],
          charts: ['plotly.js-dist-min', 'react-plotly.js'],
          i18n: ['i18next', 'react-i18next'],
          particles: ['react-tsparticles', 'tsparticles-slim'],
        },
      },
    },
    // Увеличиваем лимит для больших чанков
    chunkSizeWarningLimit: 1000,
  },
  server: {
    // Настройки dev сервера
    port: 5176,
    host: true,
    open: true,
  },
  preview: {
    port: 4173,
    host: true,
  },
  optimizeDeps: {
    // Предварительная сборка зависимостей
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'lucide-react',
      'plotly.js-dist-min',
    ],
  },
})
