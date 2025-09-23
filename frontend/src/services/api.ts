import axios, { AxiosResponse } from 'axios'
import { 
  SearchRequest, 
  SearchResult, 
  LightCurveData, 
  HealthStatus, 
  Target,
  ExportFormat
} from '../types/api'

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 0, // Убираем таймаут полностью
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
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

// Response interceptor for error handling
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
)

export class ApiService {
  // Health check
  static async getHealth(): Promise<HealthStatus> {
    const response: AxiosResponse<HealthStatus> = await apiClient.get('/api/health')
    return response.data
  }

  // Basic search
  static async searchExoplanets(request: Partial<SearchRequest>): Promise<SearchResult> {
    const response: AxiosResponse<SearchResult> = await apiClient.post('/api/search', request)
    return response.data
  }

  // AI-enhanced search
  static async aiEnhancedSearch(request: SearchRequest): Promise<SearchResult> {
    const response: AxiosResponse<SearchResult> = await apiClient.post('/api/ai-search', request)
    return response.data
  }

  // Get light curve data
  static async getLightCurve(
    targetName: string, 
    catalog: string = 'TIC', 
    mission: string = 'TESS'
  ): Promise<LightCurveData> {
    const response: AxiosResponse<LightCurveData> = await apiClient.get(
      `/api/lightcurve/${encodeURIComponent(targetName)}`,
      { params: { catalog, mission } }
    )
    return response.data
  }

  // Get available catalogs and missions
  static async getCatalogs(): Promise<{ catalogs: string[], missions: string[], description: Record<string, string> }> {
    const response = await apiClient.get('/api/catalogs')
    return response.data
  }

  // Search targets
  static async searchTargets(
    query: string, 
    catalog: string = 'TIC', 
    limit: number = 10
  ): Promise<{ targets: Target[] }> {
    const response = await apiClient.get('/api/targets/search', {
      params: { query, catalog, limit }
    })
    return response.data
  }

  // Export results
  static async exportResults(results: any, format: 'csv' | 'json' = 'csv'): Promise<ExportFormat> {
    const response = await apiClient.post('/api/export', results, {
      params: { format }
    })
    return response.data
  }

  // AI-specific endpoints
  static async getAIExplanation(
    targetName: string, 
    mode: string = 'detailed'
  ): Promise<{ explanation: string, confidence: number }> {
    const response = await apiClient.get(`/api/ai/explanation/${encodeURIComponent(targetName)}`, {
      params: { mode }
    })
    return response.data
  }

  static async submitFeedback(
    targetName: string,
    userId: string,
    feedbackType: string,
    confidenceRating: number,
    comments?: string
  ): Promise<{ success: boolean }> {
    const response = await apiClient.post('/api/ai/feedback', {
      target_name: targetName,
      user_id: userId,
      feedback_type: feedbackType,
      confidence_rating: confidenceRating,
      comments
    })
    return response.data
  }

  static async getSimilarTargets(
    targetName: string, 
    topK: number = 5
  ): Promise<{ similar_targets: Array<{ name: string, similarity: number, prediction: any }> }> {
    const response = await apiClient.get(`/api/ai/similar/${encodeURIComponent(targetName)}`, {
      params: { top_k: topK }
    })
    return response.data
  }

  static async retrainModel(
    datasetName: string = 'user_feedback', 
    epochs: number = 10
  ): Promise<{ success: boolean, message: string }> {
    const response = await apiClient.post('/api/ai/retrain', {
      dataset_name: datasetName,
      epochs
    })
    return response.data
  }
}

export default ApiService
