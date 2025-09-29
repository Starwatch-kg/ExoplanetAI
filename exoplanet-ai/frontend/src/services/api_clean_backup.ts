/**
 * Clean API Service for Production
 * Очищенный API сервис для продакшена
 */

import axios, { AxiosResponse } from 'axios'

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

// Types
export interface SearchRequest {
  target_name: string
  catalog: 'TIC' | 'KIC' | 'EPIC'
  mission: 'TESS' | 'Kepler' | 'K2'
  period_min: number
  period_max: number
  snr_threshold: number
}

export interface SearchResult {
  target_name: string
  catalog: string
  mission: string
  bls_result: {
    best_period: number
    best_t0: number
    best_duration: number
    best_power: number
    snr: number
    depth: number
    depth_err: number
    significance: number
    is_significant: boolean
  } | null
  lightcurve_info: {
    points_count: number
    time_span_days: number
    cadence_minutes: number
    noise_level_ppm: number
    data_source: string
  }
  star_info: {
    target_id: string
    catalog: string
    ra: number
    dec: number
    magnitude: number
    temperature?: number
    radius?: number
    mass?: number
    stellar_type?: string
  }
  candidates_found: number
  processing_time_ms: number
  status: string
}

export interface HealthStatus {
  status: string
  timestamp: string
  version: string
  services: Record<string, string>
}

export interface CatalogInfo {
  catalogs: string[]
  missions: string[]
  description: Record<string, string>
}

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for analysis operations
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message)
    
    if (error.response?.status === 500) {
      throw new Error('Внутренняя ошибка сервера. Попробуйте позже.')
    } else if (error.response?.status === 404) {
      throw new Error('Данные не найдены для указанной цели.')
    } else if (error.response?.status === 400) {
      throw new Error(error.response.data?.detail || 'Неверный запрос.')
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('Превышено время ожидания запроса.')
    } else if (!error.response) {
      throw new Error('Нет соединения с сервером.')
    }
    
    return Promise.reject(error)
  }
)

export class ApiService {
  // Health check
  static async getHealth(): Promise<HealthStatus> {
    const response: AxiosResponse<HealthStatus> = await apiClient.get('/api/v1/health')
    return response.data
  }

  // Search for exoplanets
  static async searchExoplanets(request: SearchRequest): Promise<SearchResult> {
    const response: AxiosResponse<SearchResult> = await apiClient.post('/api/v1/search', request)
    return response.data
  }

  // Get available catalogs
  static async getCatalogs(): Promise<CatalogInfo> {
    const response: AxiosResponse<CatalogInfo> = await apiClient.get('/api/v1/catalogs')
    return response.data
  }

  // Test connection
  static async testConnection(): Promise<boolean> {
    try {
      await this.getHealth()
      return true
    } catch {
      return false
    }
  }
}

export default ApiService
