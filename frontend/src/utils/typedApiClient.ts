/**
 * Типизированный API клиент для ExoplanetAI
 * Заменяет небезопасные any типы на строгую типизацию
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError, InternalAxiosRequestConfig } from 'axios'
import type { SearchRequest, SearchResult, HealthStatus, CatalogsResponse, ApiError, ApiErrorResponse, SystemStatistics } from '../types/api'

export interface ApiSuccessResponse<T = any> {
  data: T
  status: 'success'
  request_id?: string
}

class TypedApiClient {
  private client: AxiosInstance
  private readonly isDevelopment: boolean

  constructor(baseURL?: string) {
    this.isDevelopment = import.meta.env.DEV
    // Автоматически определяем URL в зависимости от окружения
    const apiUrl = baseURL || (this.isDevelopment 
      ? 'http://localhost:8001/api/v1' 
      : '/api/v1')
    this.client = this.createClient(apiUrl)
    this.setupInterceptors()
  }

  private createClient(baseURL: string): AxiosInstance {
    return axios.create({
      baseURL,
      timeout: 300000, // 5 минут вместо бесконечности
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }

  private setupInterceptors(): void {
    // Request interceptor с типизацией
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Генерируем request ID для трассировки
        const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        if (config.headers) {
          config.headers['X-Request-ID'] = requestId
        }

        // Безопасное логирование только в development
        if (this.isDevelopment && config.url) {
          const sanitizedUrl = this.sanitizeUrl(`${config.baseURL}${config.url}`)
          console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${sanitizedUrl}`)
        }

        return config
      },
      (error: AxiosError) => {
        if (this.isDevelopment) {
          console.error('❌ Request Error:', error.message)
        }
        return Promise.reject(this.transformError(error as AxiosError<ApiErrorResponse>))
      }
    )

    // Response interceptor с обработкой ошибок
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        if (this.isDevelopment && response.config.url) {
          const sanitizedUrl = this.sanitizeUrl(`${response.config.baseURL}${response.config.url}`)
          console.log(`✅ API Response: ${response.status} ${sanitizedUrl}`)
        }
        return response
      },
      (error: AxiosError) => {
        if (this.isDevelopment) {
          console.error('❌ Response Error:', error.response?.data || error.message)
        }
        return Promise.reject(this.transformError(error as AxiosError<ApiErrorResponse>))
      }
    )
  }

  private sanitizeUrl(url: string): string {
    // Удаляем чувствительные данные из URL для логирования
    return url
      .replace(/token=[^&]+/gi, 'token=***')
      .replace(/password=[^&]+/gi, 'password=***')
  }

  private transformError(error: AxiosError<ApiErrorResponse>): ApiError {
    const response = error.response
    const data = response?.data

    return {
      message: data?.message || error.message || 'Unknown API error',
      status: response?.status || 500,
      code: data?.code || response?.statusText,
      details: data?.details,
      requestId: data?.requestId
    }
  }

  // Типизированные методы API
  async get<T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> {
    try {
      const response = await this.client.get<ApiSuccessResponse<T>>(url, config)
      return response.data.data || response.data as T
    } catch (error) {
      throw error // Уже трансформирована в interceptor
    }
  }

  async post<T>(url: string, data?: any, config?: InternalAxiosRequestConfig): Promise<T> {
    try {
      const response = await this.client.post<ApiSuccessResponse<T>>(url, data, config)
      return response.data.data || response.data as T
    } catch (error) {
      throw error
    }
  }

  async put<T>(url: string, data?: any, config?: InternalAxiosRequestConfig): Promise<T> {
    try {
      const response = await this.client.put<ApiSuccessResponse<T>>(url, data, config)
      return response.data.data || response.data as T
    } catch (error) {
      throw error
    }
  }

  async delete<T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> {
    try {
      const response = await this.client.delete<ApiSuccessResponse<T>>(url, config)
      return response.data.data || response.data as T
    } catch (error) {
      throw error
    }
  }

  // Специализированные методы для ExoplanetAI API
  async getHealth(): Promise<HealthStatus> {
    return this.get<HealthStatus>('/health')
  }

  async searchExoplanets(request: SearchRequest): Promise<SearchResult> {
    return this.post<SearchResult>('/planets/search', request)
  }

  async getCatalogs(): Promise<CatalogsResponse> {
    return this.get<CatalogsResponse>('/catalogs')
  }

  async validateTarget(targetName: string): Promise<{ valid: boolean; suggestions: string[] }> {
    return this.get<{ valid: boolean; suggestions: string[] }>(`/validate-target?target_name=${encodeURIComponent(targetName)}`)
  }

  async getStatistics(): Promise<SystemStatistics> {
    return this.get<SystemStatistics>('/statistics')
  }
}

// Создаем и экспортируем экземпляр клиента
export const typedApiClient = new TypedApiClient()

// Функция для создания клиента с длительным таймаутом (для долгих операций)
export const createLongRunningClient = (timeout: number = 600000) => {
  const client = new TypedApiClient('/api/v1')
  // Apply custom timeout to the client
  client['client'].defaults.timeout = timeout
  return client
}

// Функция для создания cancel token
export const createCancelToken = () => {
  return axios.CancelToken.source()
}

export default typedApiClient
