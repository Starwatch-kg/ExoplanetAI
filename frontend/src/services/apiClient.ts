/**
 * ExoplanetAI Production API Client
 * Типобезопасный клиент для взаимодействия с FastAPI backend
 * 
 * Особенности:
 * - Автоматическая обработка JWT токенов
 * - TypeScript типы для всех запросов/ответов
 * - Interceptors для логирования и error handling
 * - Поддержка file uploads с прогрессом
 * - Retry механизм для failed requests
 */

import axios, { AxiosInstance, AxiosProgressEvent } from 'axios'

// ===== TYPE DEFINITIONS =====

export interface ApiConfig {
  baseURL: string
  timeout: number
  retryAttempts: number
  retryDelay: number
}

export interface AuthTokens {
  access_token: string
  refresh_token?: string
  token_type: string
  expires_in: number
}

export interface User {
  id: string
  email: string
  name: string
  role: 'guest' | 'user' | 'researcher' | 'admin'
  created_at: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
  user: User
}

export interface PredictionRequest {
  target_name: string
  data_source?: string
  analysis_type?: string
  model_type?: string
}

export interface BLSResults {
  period: number
  epoch: number
  depth: number
  duration: number
  snr: number
  significance: number
}

export interface PredictionResponse {
  target_name: string
  prediction: string
  confidence: number
  probability_planet: number
  bls_results: BLSResults
  processing_time: number
  model_version: string
}

export interface TrainingRequest {
  model_type: string
  dataset_path: string
  epochs?: number
  batch_size?: number
}

export interface TrainingResponse {
  job_id: string
  status: string
  message: string
  websocket_url: string
}

export interface HealthStatus {
  status: string
  timestamp: string
  version: string
  components: {
    cache: string
    database: string
    ml_models: string
  }
}

export interface ApiError {
  error: string
  status_code: number
  timestamp: string
  details?: any
}

// ===== TOKEN MANAGER =====

class TokenManager {
  private static readonly TOKEN_KEY = 'exoplanet_ai_tokens'
  private static readonly USER_KEY = 'exoplanet_ai_user'

  static getTokens(): AuthTokens | null {
    try {
      const tokens = localStorage.getItem(this.TOKEN_KEY)
      return tokens ? JSON.parse(tokens) : null
    } catch {
      return null
    }
  }

  static setTokens(tokens: AuthTokens): void {
    localStorage.setItem(this.TOKEN_KEY, JSON.stringify(tokens))
  }

  static removeTokens(): void {
    localStorage.removeItem(this.TOKEN_KEY)
    localStorage.removeItem(this.USER_KEY)
  }

  static getUser(): User | null {
    try {
      const user = localStorage.getItem(this.USER_KEY)
      return user ? JSON.parse(user) : null
    } catch {
      return null
    }
  }

  static setUser(user: User): void {
    localStorage.setItem(this.USER_KEY, JSON.stringify(user))
  }

  static isTokenExpired(tokens: AuthTokens): boolean {
    try {
      // Простая проверка на основе expires_in
      // В production лучше использовать JWT decode для проверки exp claim
      const tokenData = JSON.parse(atob(tokens.access_token.split('.')[1]))
      const currentTime = Math.floor(Date.now() / 1000)
      return tokenData.exp < currentTime
    } catch {
      return true
    }
  }

  static isAuthenticated(): boolean {
    const tokens = this.getTokens()
    return tokens !== null && !this.isTokenExpired(tokens)
  }
}

// ===== API CLIENT CLASS =====

export class ExoplanetApiClient {
  private client: AxiosInstance
  private config: ApiConfig
  private refreshPromise: Promise<void> | null = null

  constructor(config: Partial<ApiConfig> = {}) {
    this.config = {
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8001',
      timeout: 10000, // Уменьшили таймаут до 10 секунд
      retryAttempts: 3,
      retryDelay: 1000,
      ...config
    }

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json'
      }
    })

    this.setupInterceptors()
  }

  private setupInterceptors(): void {
    // Request interceptor - добавляем JWT токен
    this.client.interceptors.request.use(
      (config) => {
        const tokens = TokenManager.getTokens()
        if (tokens && !TokenManager.isTokenExpired(tokens)) {
          config.headers.Authorization = `Bearer ${tokens.access_token}`
        }

        // Логирование запросов
        console.log(`🔄 API Request: ${config.method?.toUpperCase()} ${config.url}`)
        return config
      },
      (error) => {
        console.error('❌ Request interceptor error:', error)
        return Promise.reject(error)
      }
    )

    // Response interceptor - обработка ошибок и refresh токенов
    this.client.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`)
        return response
      },
      async (error) => {
        const originalRequest = error.config

        // Если 401 и есть refresh токен, пытаемся обновить
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true

          try {
            await this.refreshToken()
            // Повторяем оригинальный запрос с новым токеном
            const tokens = TokenManager.getTokens()
            if (tokens) {
              originalRequest.headers.Authorization = `Bearer ${tokens.access_token}`
              return this.client(originalRequest)
            }
          } catch (refreshError) {
            // Refresh не удался, перенаправляем на логин
            TokenManager.removeTokens()
            window.location.href = '/login'
            return Promise.reject(refreshError)
          }
        }

        // Обработка других ошибок
        const apiError: ApiError = {
          error: error.response?.data?.error || error.message || 'Unknown error',
          status_code: error.response?.status || 500,
          timestamp: new Date().toISOString(),
          details: error.response?.data
        }

        console.error('❌ API Error:', apiError)
        return Promise.reject(apiError)
      }
    )
  }

  private async refreshToken(): Promise<void> {
    // Предотвращаем множественные одновременные refresh запросы
    if (this.refreshPromise) {
      return this.refreshPromise
    }

    this.refreshPromise = this.performTokenRefresh()
    try {
      await this.refreshPromise
    } finally {
      this.refreshPromise = null
    }
  }

  private async performTokenRefresh(): Promise<void> {
    const tokens = TokenManager.getTokens()
    if (!tokens?.refresh_token) {
      throw new Error('No refresh token available')
    }

    try {
      const response = await axios.post(`${this.config.baseURL}/api/auth/refresh`, {
        refresh_token: tokens.refresh_token
      })

      const newTokens: AuthTokens = response.data
      TokenManager.setTokens(newTokens)
    } catch (error) {
      TokenManager.removeTokens()
      throw error
    }
  }

  // ===== AUTHENTICATION METHODS =====

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    try {
      const response = await this.client.post<LoginResponse>('/api/auth/login', credentials)
      
      // Сохраняем токены и пользователя
      const { access_token, refresh_token, token_type, expires_in, user } = response.data
      TokenManager.setTokens({ access_token, refresh_token, token_type, expires_in })
      TokenManager.setUser(user)
      
      return response.data
    } catch (error) {
      throw this.handleError(error, 'Login failed')
    }
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/api/auth/logout')
    } catch (error) {
      console.warn('Logout request failed:', error)
    } finally {
      TokenManager.removeTokens()
    }
  }

  async getCurrentUser(): Promise<User> {
    try {
      const response = await this.client.get<User>('/api/auth/me')
      TokenManager.setUser(response.data)
      return response.data
    } catch (error) {
      throw this.handleError(error, 'Failed to get current user')
    }
  }

  // ===== HEALTH CHECK =====

  async getHealth(): Promise<HealthStatus> {
    try {
      const response = await this.client.get<HealthStatus>('/api/v1/health')
      return response.data
    } catch (error) {
      throw this.handleError(error, 'Health check failed')
    }
  }

  // ===== PREDICTION METHODS =====

  async predictExoplanet(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      console.log(`🔮 Starting prediction for ${request.target_name}`)
      
      const response = await this.client.post<PredictionResponse>('/api/predict', request)
      
      console.log(`✨ Prediction completed: ${response.data.prediction} (${(response.data.confidence * 100).toFixed(1)}%)`)
      return response.data
    } catch (error) {
      throw this.handleError(error, `Prediction failed for ${request.target_name}`)
    }
  }

  // ===== TRAINING METHODS =====

  async startTraining(request: TrainingRequest): Promise<TrainingResponse> {
    try {
      console.log(`🚀 Starting training: ${request.model_type}`)
      
      const response = await this.client.post<TrainingResponse>('/api/train', request)
      
      console.log(`📊 Training started with job ID: ${response.data.job_id}`)
      return response.data
    } catch (error) {
      throw this.handleError(error, 'Failed to start training')
    }
  }

  // ===== FILE UPLOAD =====

  async uploadFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ file_id: string; filename: string; size: number }> {
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await this.client.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (progressEvent.total && onProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress(progress)
          }
        }
      })

      return response.data
    } catch (error) {
      throw this.handleError(error, 'File upload failed')
    }
  }

  // ===== UTILITY METHODS =====

  private handleError(error: any, message: string): ApiError {
    if (error.error && error.status_code) {
      // Уже обработанная ошибка из interceptor
      return error
    }

    return {
      error: error.response?.data?.error || error.message || message,
      status_code: error.response?.status || 500,
      timestamp: new Date().toISOString(),
      details: error.response?.data
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.getHealth()
      return true
    } catch {
      return false
    }
  }

  // ===== STATIC METHODS =====

  static isAuthenticated(): boolean {
    return TokenManager.isAuthenticated()
  }

  static getCurrentUser(): User | null {
    return TokenManager.getUser()
  }

  static logout(): void {
    TokenManager.removeTokens()
  }
}

// ===== DEFAULT INSTANCE =====

// Создаем единственный экземпляр клиента для использования в приложении
export const apiClient = new ExoplanetApiClient()

// ===== CONVENIENCE FUNCTIONS =====

export const api = {
  // Auth
  login: (credentials: LoginRequest) => apiClient.login(credentials),
  logout: () => apiClient.logout(),
  getCurrentUser: () => apiClient.getCurrentUser(),
  
  // Health
  getHealth: () => apiClient.getHealth(),
  testConnection: () => apiClient.testConnection(),
  
  // Predictions
  predict: (request: PredictionRequest) => apiClient.predictExoplanet(request),
  
  // Training
  startTraining: (request: TrainingRequest) => apiClient.startTraining(request),
  
  // Files
  uploadFile: (file: File, onProgress?: (progress: number) => void) => 
    apiClient.uploadFile(file, onProgress),
  
  // Utils
  isAuthenticated: () => ExoplanetApiClient.isAuthenticated(),
  getUser: () => ExoplanetApiClient.getCurrentUser()
}

export default api
