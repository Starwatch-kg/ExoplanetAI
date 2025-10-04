import axios, { AxiosResponse } from 'axios'
import type { 
  SearchRequest, 
  SearchResult, 
  HealthStatus, 
  CatalogsResponse
} from '../types/api'

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds timeout for most requests
  headers: {
    'Content-Type': 'application/json',
  },
})

// Create a special client for long-running operations
const longRunningClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 0, // No timeout for analysis operations
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
const requestInterceptor = (config: any) => {
  const fullUrl = `${config.baseURL}${config.url}`
  console.log(`API Request: ${config.method?.toUpperCase()} ${fullUrl}`)
  return config
}

const requestErrorInterceptor = (error: any) => {
  console.error('API Request Error:', error)
  return Promise.reject(error)
}

// Response interceptor for error handling
const responseInterceptor = (response: any) => {
  const fullUrl = `${response.config.baseURL}${response.config.url}`
  console.log(`API Response: ${response.status} ${fullUrl}`)
  return response
}

const responseErrorInterceptor = (error: any) => {
  console.error('API Response Error:', error.response?.data || error.message)
  
  if (error.response?.status === 500) {
    throw new Error('Внутренняя ошибка сервера. Попробуйте позже.')
  } else if (error.response?.status === 404) {
    throw new Error('Ресурс не найден.')
  } else if (error.response?.status === 400) {
    throw new Error(error.response.data?.detail || 'Неверный запрос.')
  } else if (error.code === 'ECONNABORTED') {
    throw new Error('Превышено время ожидания запроса.')
  } else if (!error.response) {
    throw new Error('Нет соединения с сервером.')
  }
  
  return Promise.reject(error)
}

// Apply interceptors to both clients
apiClient.interceptors.request.use(requestInterceptor, requestErrorInterceptor)
apiClient.interceptors.response.use(responseInterceptor, responseErrorInterceptor)
longRunningClient.interceptors.request.use(requestInterceptor, requestErrorInterceptor)
longRunningClient.interceptors.response.use(responseInterceptor, responseErrorInterceptor)

export class ApiService {
  // Health check
  static async getHealth(): Promise<HealthStatus> {
    try {
      const response: AxiosResponse<HealthStatus> = await apiClient.get('/api/v1/health')
      return response.data
    } catch (error) {
      throw new Error(`Health check failed: ${error}`)
    }
  }

  // Search for exoplanets (main endpoint)
  static async searchExoplanets(request: SearchRequest): Promise<SearchResult> {
    try {
      const response: AxiosResponse<SearchResult> = await longRunningClient.post('/api/v1/search', request)
      return response.data
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Search failed'
      throw new Error(message)
    }
  }

  // Get available catalogs
  static async getCatalogs(): Promise<CatalogsResponse> {
    try {
      const response: AxiosResponse<CatalogsResponse> = await apiClient.get('/api/v1/catalogs')
      return response.data
    } catch (error) {
      throw new Error(`Failed to get catalogs: ${error}`)
    }
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

// Export individual clients for special cases
export { apiClient, longRunningClient }

// Export default service
export default ApiService
