import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import LoadingSpinner from './components/ui/LoadingSpinner'

// Lazy loading компонентов
const HomePage = lazy(() => import('./pages/HomePage'))
const SearchPage = lazy(() => import('./pages/SearchPage'))
const AboutPage = lazy(() => import('./pages/AboutPage'))
const DatabasePage = lazy(() => import('./pages/DatabasePage'))
const CatalogPage = lazy(() => import('./pages/CatalogPage'))
const AITrainingPage = lazy(() => import('./pages/AITrainingPage'))
const GPIAnalysisPage = lazy(() => import('./pages/GPIAnalysisPage'))

// Wrapper для Suspense
const SuspenseWrapper = ({ children }: { children: React.ReactNode }) => (
  <Suspense fallback={<LoadingSpinner />}>
    {children}
  </Suspense>
)

export const router = createBrowserRouter([
  {
    path: '/',
    element: <SuspenseWrapper><HomePage /></SuspenseWrapper>
  },
  {
    path: '/search',
    element: <SuspenseWrapper><SearchPage /></SuspenseWrapper>
  },
  {
    path: '/about',
    element: <SuspenseWrapper><AboutPage /></SuspenseWrapper>
  },
  {
    path: '/database',
    element: <SuspenseWrapper><DatabasePage /></SuspenseWrapper>
  },
  {
    path: '/catalog',
    element: <SuspenseWrapper><CatalogPage /></SuspenseWrapper>
  },
  {
    path: '/ai-training',
    element: <SuspenseWrapper><AITrainingPage /></SuspenseWrapper>
  },
  {
    path: '/gpi-analysis',
    element: <SuspenseWrapper><GPIAnalysisPage /></SuspenseWrapper>
  }
])
