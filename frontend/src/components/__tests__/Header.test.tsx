/**
 * Тесты компонента Header
 */
import { render, screen, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, it, expect, vi, beforeEach } from 'vitest'

import Header from '../layout/Header'
import type { HealthStatus } from '../../types/api'

// Mock для React Router
const MockRouter = ({ children }: { children: React.ReactNode }) => (
  <BrowserRouter>{children}</BrowserRouter>
)

// Mock для React Query
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
})

const MockQueryProvider = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={createTestQueryClient()}>
    {children}
  </QueryClientProvider>
)

const MockProviders = ({ children }: { children: React.ReactNode }) => (
  <MockRouter>
    <MockQueryProvider>
      {children}
    </MockQueryProvider>
  </MockRouter>
)

describe('Header Component', () => {
  const mockHealthyStatus: HealthStatus = {
    status: 'healthy',
    uptime: 3600,
    components: {
      data_sources: {
        status: 'healthy',
        initialized: 4,
        total: 4
      },
      cache: {
        status: 'healthy',
        redis_connected: true
      },
      authentication: {
        status: 'healthy'
      }
    },
    timestamp: new Date().toISOString()
  }

  const mockDegradedStatus: HealthStatus = {
    status: 'degraded',
    uptime: 1800,
    timestamp: new Date().toISOString(),
    components: {
      data_sources: {
        status: 'healthy',
        initialized: 2,
        total: 4
      },
      cache: {
        status: 'unhealthy',
        redis_connected: false
      }
    }
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders header with healthy status', () => {
    render(
      <MockProviders>
        <Header healthStatus={mockHealthyStatus} />
      </MockProviders>
    )

    // Проверяем наличие основных элементов
    expect(screen.getByText(/ExoplanetAI/i)).toBeInTheDocument()
    expect(screen.getByText(/healthy/i)).toBeInTheDocument()
  })

  it('renders header with degraded status', () => {
    render(
      <MockProviders>
        <Header healthStatus={mockDegradedStatus} />
      </MockProviders>
    )

    expect(screen.getByText(/degraded/i)).toBeInTheDocument()
  })

  it('renders header without health status', () => {
    render(
      <MockProviders>
        <Header healthStatus={null} />
      </MockProviders>
    )

    // Компонент должен рендериться даже без статуса
    expect(screen.getByText(/ExoplanetAI/i)).toBeInTheDocument()
  })

  it('displays correct uptime format', () => {
    render(
      <MockProviders>
        <Header healthStatus={mockHealthyStatus} />
      </MockProviders>
    )

    // Проверяем форматирование времени работы (3600 сек = 1 час)
    expect(screen.getByText(/1h/i)).toBeInTheDocument()
  })

  it('shows component status details', async () => {
    render(
      <MockProviders>
        <Header healthStatus={mockHealthyStatus} />
      </MockProviders>
    )

    // Ищем индикаторы статуса компонентов
    await waitFor(() => {
      expect(screen.getByText(/4\/4/)).toBeInTheDocument() // data sources
    })
  })

  it('handles navigation links correctly', () => {
    render(
      <MockProviders>
        <Header healthStatus={mockHealthyStatus} />
      </MockProviders>
    )

    // Проверяем наличие навигационных ссылок
    expect(screen.getByRole('link', { name: /home/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /search/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /about/i })).toBeInTheDocument()
  })

  it('applies correct CSS classes for different health states', () => {
    const { rerender } = render(
      <MockProviders>
        <Header healthStatus={mockHealthyStatus} />
      </MockProviders>
    )

    // Проверяем CSS классы для здорового состояния
    const healthIndicator = screen.getByText(/healthy/i)
    expect(healthIndicator).toHaveClass('text-green-500')

    // Перерендерим с деградированным статусом
    rerender(
      <MockProviders>
        <Header healthStatus={mockDegradedStatus} />
      </MockProviders>
    )

    const degradedIndicator = screen.getByText(/degraded/i)
    expect(degradedIndicator).toHaveClass('text-yellow-500')
  })
})
