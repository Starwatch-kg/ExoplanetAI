<<<<<<< HEAD
<<<<<<< HEAD
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
=======
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
>>>>>>> ef5c656 (Версия 1.5.1)
=======
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
>>>>>>> 975c3a7 (Версия 1.5.1)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
<<<<<<< HEAD
<<<<<<< HEAD
  build: {
    // Оптимизации для production
    minify: 'terser',
    cssMinify: true,
    rollupOptions: {
      output: {
        // Разделение кода для лучшего кэширования
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'ml-vendor': ['tensorflow/tfjs'],
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
=======
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  server: {
<<<<<<< HEAD
>>>>>>> ef5c656 (Версия 1.5.1)
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
    port: 5173,
    host: true,
    proxy: {
      '/api': {
<<<<<<< HEAD
<<<<<<< HEAD
        target: process.env.VITE_API_URL || 'http://localhost:8001',
        changeOrigin: true,
      },
      '/health': {
        target: process.env.VITE_API_URL || 'http://localhost:8001',
        changeOrigin: true,
      },
      '/load-tic': {
        target: process.env.VITE_API_URL || 'http://localhost:8001',
        changeOrigin: true,
      },
      '/analyze': {
        target: process.env.VITE_API_URL || 'http://localhost:8001',
        changeOrigin: true,
      },
      '/amateur/analyze': {
        target: process.env.VITE_API_URL || 'http://localhost:8001',
        changeOrigin: true,
      },
      '/pro/analyze': {
        target: process.env.VITE_API_URL || 'http://localhost:8001',
        changeOrigin: true,
=======
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          plotly: ['plotly.js', 'react-plotly.js'],
        },
<<<<<<< HEAD
>>>>>>> ef5c656 (Версия 1.5.1)
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
      },
    },
  },
})
