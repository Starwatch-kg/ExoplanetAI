/**
 * Настройка тестовой среды для Vitest
 */
import '@testing-library/jest-dom'
import { expect, afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

// Очистка после каждого теста
afterEach(() => {
  cleanup()
})

// Mock для matchMedia (для компонентов с responsive дизайном)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock для ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock для IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock для window.scrollTo
Object.defineProperty(window, 'scrollTo', {
  value: vi.fn(),
  writable: true,
})

// Расширение expect с custom matchers
expect.extend({
  toBeInTheDocument: (received: any) => {
    const pass = received && received.ownerDocument && received.ownerDocument.body.contains(received)
    return {
      pass,
      message: () => `Expected element ${pass ? 'not ' : ''}to be in the document`,
    }
  },
})
