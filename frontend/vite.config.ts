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
        manualChunks: (id) => {
          if (id.includes('node_modules')) {
            if (id.includes('plotly')) {
              return 'charts';
            }
            if (id.includes('react') && !id.includes('plotly')) {
              return 'vendor';
            }
            if (id.includes('i18next')) {
              return 'i18n';
            }
            if (id.includes('particles')) {
              return 'particles';
            }
            return 'vendor';
          }
        },
      },
    },
    // Увеличиваем лимит для больших чанков
    chunkSizeWarningLimit: 1000,
  },
  server: {
    // Настройки dev сервера
    port: 5177,
    host: true,
    open: true,
    proxy: {
      // Проксируем API запросы на бэкенд
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      }
    }
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
