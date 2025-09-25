<<<<<<< HEAD
<<<<<<< HEAD
import React, { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import * as Sentry from '@sentry/react'
import './index.css'
import App from './App.tsx'

const Landing = React.lazy(() => import('./pages/Landing'))
const HowItWorks = React.lazy(() => import('./pages/HowItWorks'))

if (import.meta.env.VITE_SENTRY_DSN) {
  Sentry.init({ dsn: import.meta.env.VITE_SENTRY_DSN });
}

const router = createBrowserRouter([
  { path: '/', element: <React.Suspense fallback={<div />}> <Landing /> </React.Suspense> },
  { path: '/how-it-works', element: <React.Suspense fallback={<div />}> <HowItWorks /> </React.Suspense> },
  { path: '/demo', element: <App /> },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
=======
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import App from './App.tsx'
import './index.css'

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>,
<<<<<<< HEAD
>>>>>>> ef5c656 (Версия 1.5.1)
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
)
