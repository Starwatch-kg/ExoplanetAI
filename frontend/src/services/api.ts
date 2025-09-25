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
<<<<<<< HEAD
  baseURL: API_BASE_URL,
=======
  baseURL: `${API_BASE_URL}/api/v1`, // Обновляем базовый URL для v1 API
>>>>>>> 975c3a7 (Версия 1.5.1)
  timeout: 0, // Убираем таймаут полностью
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
<<<<<<< HEAD
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
=======
    const fullUrl = `${config.baseURL}${config.url}`
    console.log(`API Request: ${config.method?.toUpperCase()} ${fullUrl}`)
>>>>>>> 975c3a7 (Версия 1.5.1)
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
<<<<<<< HEAD
    console.log(`API Response: ${response.status} ${response.config.url}`)
=======
    const fullUrl = `${response.config.baseURL}${response.config.url}`
    console.log(`API Response: ${response.status} ${fullUrl}`)
>>>>>>> 975c3a7 (Версия 1.5.1)
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
<<<<<<< HEAD
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
=======
    const response: AxiosResponse<HealthStatus> = await apiClient.get('/health')
    return response.data
  }

  // Detailed health check
  static async getDetailedHealth(): Promise<any> {
    const response = await apiClient.get('/health/detailed')
    return response.data
  }

  // Comprehensive exoplanet search with BLS and AI
  static async searchExoplanets(request: {
    target_name: string
    catalog?: string
    mission?: string
    use_bls?: boolean
    use_ai?: boolean
    use_ensemble?: boolean
    search_mode?: 'single' | 'ensemble' | 'comprehensive'
    period_min?: number
    period_max?: number
    snr_threshold?: number
  }): Promise<any> {
    try {
      console.log('🚀 API Request:', request)
      
      const response = await apiClient.post('/search', {
        target_name: request.target_name,
        catalog: request.catalog || 'TIC',
        mission: request.mission || 'TESS',
        use_bls: request.use_bls !== false,
        use_ai: request.use_ai !== false,
        use_ensemble: request.use_ensemble !== false,
        search_mode: request.search_mode || 'ensemble',
        period_min: request.period_min || 0.5,
        period_max: request.period_max || 20.0,
        snr_threshold: request.snr_threshold || 7.0
      })
      
      console.log('✅ API Response:', response.data)
      return response.data
    } catch (error) {
      console.error('❌ API Error:', error)
      throw error
    }
  }

  // BLS analysis
  static async analyzeBLS(request: {
    target_name: string
    catalog?: string
    mission?: string
    period_min?: number
    period_max?: number
    duration_min?: number
    duration_max?: number
    snr_threshold?: number
    use_enhanced?: boolean
  }): Promise<{
    target_name: string
    best_period: number
    best_t0: number
    best_duration: number
    best_power: number
    snr: number
    depth: number
    depth_err: number
    significance: number
    is_significant: boolean
    enhanced_analysis: boolean
    ml_confidence: number
    physical_validation: boolean
    processing_time_ms: number
    request_id?: string
    trace_id?: string
  }> {
    const response = await apiClient.post('/bls', {
      target_name: request.target_name,
      catalog: request.catalog || 'TIC',
      mission: request.mission || 'TESS',
      period_min: request.period_min || 0.5,
      period_max: request.period_max || 20.0,
      duration_min: request.duration_min || 0.05,
      duration_max: request.duration_max || 0.3,
      snr_threshold: request.snr_threshold || 7.0,
      use_enhanced: request.use_enhanced !== false
    })
    return response.data
  }

  // AI-enhanced search (legacy compatibility)
  static async aiEnhancedSearch(request: SearchRequest): Promise<SearchResult> {
    // Map to new search endpoint
    return this.searchExoplanets({
      target_name: request.target_name,
      catalog: request.catalog,
      mission: request.mission,
      use_ai: true,
      use_bls: false
    }) as any
  }

  // Batch search
  static async batchSearch(targets: SearchRequest[]): Promise<{ batch_id: string, status: string }> {
    const response = await apiClient.post('/search/batch', targets)
    return response.data
  }

  // Get batch status
  static async getBatchStatus(batchId: string): Promise<any> {
    const response = await apiClient.get(`/search/batch/${batchId}`)
>>>>>>> 975c3a7 (Версия 1.5.1)
    return response.data
  }

  // Get light curve data
  static async getLightCurve(
    targetName: string, 
    catalog: string = 'TIC', 
    mission: string = 'TESS'
<<<<<<< HEAD
  ): Promise<LightCurveData> {
    const response: AxiosResponse<LightCurveData> = await apiClient.get(
      `/api/lightcurve/${encodeURIComponent(targetName)}`,
=======
  ): Promise<{
    target_name: string
    catalog: string
    mission: string
    time: number[]
    flux: number[]
    flux_err: number[]
    cadence_minutes: number
    noise_level_ppm: number
    data_source: string
    points_count: number
    time_span_days: number
    request_id?: string
    trace_id?: string
  }> {
    const response = await apiClient.get(
      `/lightcurve/${encodeURIComponent(targetName)}`,
>>>>>>> 975c3a7 (Версия 1.5.1)
      { params: { catalog, mission } }
    )
    return response.data
  }

<<<<<<< HEAD
  // Get available catalogs and missions
  static async getCatalogs(): Promise<{ catalogs: string[], missions: string[], description: Record<string, string> }> {
    const response = await apiClient.get('/api/catalogs')
    return response.data
  }

  // Search targets
=======
  // Get lightcurve preview
  static async getLightCurvePreview(
    targetName: string,
    catalog: string = 'TIC',
    mission: string = 'TESS',
    maxPoints: number = 1000
  ): Promise<any> {
    const response = await apiClient.get(
      `/lightcurve/${encodeURIComponent(targetName)}/preview`,
      { params: { catalog, mission, max_points: maxPoints } }
    )
    return response.data
  }

  // Get available catalogs and missions
  static async getCatalogs(): Promise<{ 
    catalogs: string[]
    missions: string[]
    description: Record<string, string> 
  }> {
    const response = await apiClient.get('/catalogs')
    return response.data
  }

  // Get catalog info
  static async getCatalogInfo(catalogName: string): Promise<any> {
    const response = await apiClient.get(`/catalogs/${catalogName}`)
    return response.data
  }

  // Search targets in catalog
>>>>>>> 975c3a7 (Версия 1.5.1)
  static async searchTargets(
    query: string, 
    catalog: string = 'TIC', 
    limit: number = 10
<<<<<<< HEAD
  ): Promise<{ targets: Target[] }> {
    const response = await apiClient.get('/api/targets/search', {
      params: { query, catalog, limit }
=======
  ): Promise<{ 
    targets: Array<{
      target_id: string
      catalog: string
      ra: number
      dec: number
      magnitude: number
      temperature?: number
      radius?: number
      mass?: number
      distance?: number
      stellar_type?: string
    }>
    total_found: number
    query: string
    catalog: string
  }> {
    const response = await apiClient.get(`/catalogs/${catalog}/search`, {
      params: { query, limit }
>>>>>>> 975c3a7 (Версия 1.5.1)
    })
    return response.data
  }

<<<<<<< HEAD
  // Export results
  static async exportResults(results: any, format: 'csv' | 'json' = 'csv'): Promise<ExportFormat> {
    const response = await apiClient.post('/api/export', results, {
      params: { format }
=======
  // Get random targets
  static async getRandomTargets(
    catalog: string = 'TIC',
    count: number = 5,
    magnitudeMax?: number
  ): Promise<{
    targets: Array<{
      target_id: string
      catalog: string
      magnitude: number
      ra: number
      dec: number
      temperature: number
      stellar_type: string
    }>
    catalog: string
    count: number
  }> {
    const params: any = { count }
    if (magnitudeMax) params.magnitude_max = magnitudeMax
    
    const response = await apiClient.get(`/catalogs/${catalog}/random`, { params })
    return response.data
  }

  // Export results
  static async exportResults(results: any, format: 'csv' | 'json' = 'csv'): Promise<Blob> {
    const response = await apiClient.post('/export/results', results, {
      params: { format },
      responseType: 'blob'
    })
    return response.data
  }

  // Export batch results
  static async exportBatchResults(
    batchResults: any[], 
    format: 'csv' | 'json' = 'csv',
    combine: boolean = true
  ): Promise<Blob> {
    const response = await apiClient.post('/export/batch', batchResults, {
      params: { format, combine },
      responseType: 'blob'
>>>>>>> 975c3a7 (Версия 1.5.1)
    })
    return response.data
  }

  // AI-specific endpoints
  static async getAIExplanation(
    targetName: string, 
    mode: string = 'detailed'
  ): Promise<{ explanation: string, confidence: number }> {
<<<<<<< HEAD
    const response = await apiClient.get(`/api/ai/explanation/${encodeURIComponent(targetName)}`, {
=======
    const response = await apiClient.get(`/ai/explanation/${encodeURIComponent(targetName)}`, {
>>>>>>> 975c3a7 (Версия 1.5.1)
      params: { mode }
    })
    return response.data
  }

  static async submitFeedback(
    targetName: string,
<<<<<<< HEAD
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
=======
    feedbackType: string,
    isCorrect: boolean,
    userClassification?: string,
    comments?: string
  ): Promise<{ success: boolean }> {
    const response = await apiClient.post('/ai/feedback', {
      target_name: targetName,
      feedback_type: feedbackType,
      is_correct: isCorrect,
      user_classification: userClassification,
>>>>>>> 975c3a7 (Версия 1.5.1)
      comments
    })
    return response.data
  }

  static async getSimilarTargets(
    targetName: string, 
<<<<<<< HEAD
    topK: number = 5
  ): Promise<{ similar_targets: Array<{ name: string, similarity: number, prediction: any }> }> {
    const response = await apiClient.get(`/api/ai/similar/${encodeURIComponent(targetName)}`, {
      params: { top_k: topK }
=======
    topK: number = 5,
    similarityThreshold: number = 0.7
  ): Promise<{ similar_targets: Array<{ name: string, similarity: number, prediction: any }> }> {
    const response = await apiClient.get(`/ai/similar/${encodeURIComponent(targetName)}`, {
      params: { top_k: topK, similarity_threshold: similarityThreshold }
    })
    return response.data
  }

  static async getAvailableModels(): Promise<any> {
    const response = await apiClient.get('/ai/models')
    return response.data
  }

  static async getModelPredictions(
    targetName: string,
    includeUncertainty: boolean = true
  ): Promise<any> {
    const response = await apiClient.get(`/ai/predictions/${encodeURIComponent(targetName)}`, {
      params: { include_uncertainty: includeUncertainty }
>>>>>>> 975c3a7 (Версия 1.5.1)
    })
    return response.data
  }

  static async retrainModel(
    datasetName: string = 'user_feedback', 
<<<<<<< HEAD
    epochs: number = 10
  ): Promise<{ success: boolean, message: string }> {
    const response = await apiClient.post('/api/ai/retrain', {
      dataset_name: datasetName,
      epochs
    })
    return response.data
  }
=======
    epochs: number = 10,
    modelType: string = 'ensemble'
  ): Promise<{ success: boolean, message: string }> {
    const response = await apiClient.post('/ai/retrain', null, {
      params: { dataset_name: datasetName, epochs, model_type: modelType }
    })
    return response.data
  }

  // Statistics and monitoring
  static async getSearchStats(): Promise<any> {
    const response = await apiClient.get('/search/stats')
    return response.data
  }

  static async getExportFormats(): Promise<any> {
    const response = await apiClient.get('/export/formats')
    return response.data
  }
>>>>>>> 975c3a7 (Версия 1.5.1)
}

export default ApiService
