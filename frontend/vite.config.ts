import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react({
      // Исключение node_modules из трансформации
      exclude: /node_modules/,
    })
  ],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      }
    },
    // Оптимизация dev сервера
    hmr: {
      overlay: false
    },
    host: true
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info'],
        passes: 2
      },
      mangle: {
        safari10: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks для лучшего кэширования
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'charts-vendor': ['plotly.js-dist-min', 'react-plotly.js'],
          'ui-vendor': ['framer-motion', 'lucide-react', 'clsx'],
          'particles-vendor': ['react-tsparticles', 'tsparticles-slim'],
          'i18n-vendor': ['i18next', 'react-i18next'],
          'query-vendor': ['@tanstack/react-query', 'axios']
        },
        // Оптимизация имен файлов
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      },
      // Внешние зависимости для CDN (опционально)
      external: []
    },
    chunkSizeWarningLimit: 1000,
    // Оптимизация CSS
    cssCodeSplit: true,
    // Генерация source maps только для production
    sourcemap: false,
    // Оптимизация ассетов
    assetsInlineLimit: 4096,
    // Очистка dist директории
    emptyOutDir: true
  },
  optimizeDeps: {
    include: [
      'react', 
      'react-dom', 
      'react-router-dom',
      'plotly.js-dist-min',
      'framer-motion',
      'lucide-react'
    ],
    exclude: ['@vite/client', '@vite/env']
  },
  // Разрешение путей
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@pages': '/src/pages',
      '@hooks': '/src/hooks',
      '@utils': '/src/utils',
      '@types': '/src/types'
    }
  },
  // Переменные окружения
  define: {
    __DEV__: JSON.stringify(process.env.NODE_ENV === 'development'),
    __PROD__: JSON.stringify(process.env.NODE_ENV === 'production')
  },
  // Оптимизация CSS
  css: {
    devSourcemap: false,
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`
      }
    }
  }
})
