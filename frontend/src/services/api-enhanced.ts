import axios, { AxiosResponse, AxiosProgressEvent } from 'axios'
import type { 
  SearchRequest, 
  SearchResult, 
  HealthStatus, 
  CatalogsResponse,
  UploadResponse,
  TrainingRequest,
  TrainingResponse,
  ModelMetrics,
  PredictionRequest,
  PredictionResponse
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

// Create a client for file uploads
const uploadClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for uploads
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})

// JWT token management
class TokenManager {
  private static TOKEN_KEY = 'exoplanet_ai_token'
  
  static getToken(): string | null {
    return localStorage.getItem(this.TOKEN_KEY)
  }
  
  static setToken(token: string): void {
    localStorage.setItem(this.TOKEN_KEY, token)
  }
  
  static removeToken(): void {
    localStorage.removeItem(this.TOKEN_KEY)
  }
  
  static isAuthenticated(): boolean {
    const token = this.getToken()
    if (!token) return false
    
    try {
      const payload = JSON.parse(atob(token.split('.')[1]))
      return payload.exp > Date.now() / 1000
    } catch {
      return false
    }
  }
}

// Request interceptor for authentication
const authInterceptor = (config: any) => {
  const token = TokenManager.getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
}

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
  
  // Handle authentication errors
  if (error.response?.status === 401) {
    TokenManager.removeToken()
    window.location.href = '/login'
    return Promise.reject(new Error('Authentication required'))
  }
  
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

// Apply interceptors to all clients
[apiClient, longRunningClient, uploadClient].forEach(client => {
  client.interceptors.request.use(authInterceptor)
  client.interceptors.request.use(requestInterceptor, requestErrorInterceptor)
  client.interceptors.response.use(responseInterceptor, responseErrorInterceptor)
})

export class ApiService {
  // Authentication
  static async login(email: string, password: string): Promise<{ token: string; user: any }> {
    try {
      const response = await apiClient.post('/api/v1/auth/login', { email, password })
      const { token, user } = response.data
      TokenManager.setToken(token)
      return { token, user }
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Login failed')
    }
  }
  
  static async register(email: string, password: string, name: string): Promise<{ token: string; user: any }> {
    try {
      const response = await apiClient.post('/api/v1/auth/register', { email, password, name })
      const { token, user } = response.data
      TokenManager.setToken(token)
      return { token, user }
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Registration failed')
    }
  }
  
  static async logout(): Promise<void> {
    TokenManager.removeToken()
  }
  
  static async getProfile(): Promise<any> {
    try {
      const response = await apiClient.get('/api/v1/auth/profile')
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get profile')
    }
  }
  
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
  
  // File upload with progress tracking
  static async uploadFile(
    file: File, 
    onProgress?: (progress: number) => void
  ): Promise<UploadResponse> {
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await uploadClient.post('/api/v1/upload', formData, {
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (progressEvent.total && onProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress(progress)
          }
        }
      })
      
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Upload failed')
    }
  }
  
  // Model training
  static async trainModel(request: TrainingRequest): Promise<TrainingResponse> {
    try {
      const response = await longRunningClient.post('/api/v1/train', request)
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Training failed')
    }
  }
  
  // Get model statistics
  static async getModelStats(): Promise<ModelMetrics> {
    try {
      const response = await apiClient.get('/api/v1/stats')
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get model stats')
    }
  }
  
  // Make predictions
  static async predict(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      const response = await longRunningClient.post('/api/v1/predict', request)
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Prediction failed')
    }
  }
  
  // Light curve operations
  static async getLightCurve(ticId: string): Promise<any> {
    try {
      const response = await apiClient.get(`/api/v1/lightcurves/${ticId}`)
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get light curve')
    }
  }
  
  static async analyzeLightCurve(data: any): Promise<any> {
    try {
      const response = await longRunningClient.post('/api/v1/lightcurves/analyze', data)
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Light curve analysis failed')
    }
  }
  
  // Data management
  static async ingestData(data: any): Promise<any> {
    try {
      const response = await longRunningClient.post('/api/v1/ingest/table', data)
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Data ingestion failed')
    }
  }
  
  static async validateData(data: any): Promise<any> {
    try {
      const response = await apiClient.post('/api/v1/validate', data)
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Data validation failed')
    }
  }
  
  // Admin operations (requires admin role)
  static async getAdminStats(): Promise<any> {
    try {
      const response = await apiClient.get('/api/v1/admin/stats')
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get admin stats')
    }
  }
  
  static async cleanupData(): Promise<any> {
    try {
      const response = await apiClient.post('/api/v1/admin/cleanup')
      return response.data
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Cleanup failed')
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
  
  // WebSocket connection for real-time updates
  static createWebSocket(endpoint: string): WebSocket {
    const wsUrl = API_BASE_URL.replace('http', 'ws')
    const token = TokenManager.getToken()
    const url = token ? `${wsUrl}${endpoint}?token=${token}` : `${wsUrl}${endpoint}`
    return new WebSocket(url)
  }
}

// Export individual clients for special cases
export { apiClient, longRunningClient, uploadClient, TokenManager }

// Export default service
export default ApiService
