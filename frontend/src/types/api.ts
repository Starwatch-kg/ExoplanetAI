/**
 * Clean API types matching ExoplanetAI backend
 * Полностью соответствует реальному API
 */

// === REQUEST TYPES ===
export interface SearchRequest {
  target_name: string
  catalog: 'TIC' | 'KIC' | 'EPIC'
  mission: 'TESS' | 'Kepler' | 'K2'
  period_min: number
  period_max: number
  snr_threshold: number
}

// === RESPONSE TYPES ===
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
  lightcurve_data?: {
    time: number[]
    flux: number[]
    flux_err?: number[]
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
  status: 'healthy' | 'unhealthy' | 'degraded'
  timestamp: string
  version?: string
  uptime?: number
  database?: {
    status: 'connected' | 'disconnected'
    response_time_ms?: number
  }
  ai_models?: {
    loaded: boolean
    model_count?: number
    last_updated?: string
  }
  services?: Record<string, 'healthy' | 'unhealthy' | string>
  components?: {
    data_sources?: {
      status: string
      initialized?: number
      total?: number
    }
    cache?: {
      status: string
      redis_connected?: boolean
    }
    authentication?: {
      status: string
    }
  }
}

export interface SystemStatistics {
  stars_analyzed: number
  candidates_found: number
  confirmed_planets: number
  system_accuracy: number
  average_processing_time_seconds: number
  total_searches: number
  active_users: number
  database_size_gb: number
  last_updated: string
  uptime_hours: number
}

export interface ApiError {
  message: string
  status: number
  code?: string
  details?: Record<string, any>
  requestId?: string
}

export interface ApiErrorResponse {
  message: string
  code?: string
  details?: Record<string, any>
  requestId?: string
}

export interface CatalogsResponse {
  catalogs: string[]
  missions: string[]
  descriptions: Record<string, string>
}

export interface LoadingState {
  isLoading: boolean
  error: string | null
}

// === FORM TYPES ===
export interface SearchFormData {
  target_name: string
  catalog: 'TIC' | 'KIC' | 'EPIC'
  mission: 'TESS' | 'Kepler' | 'K2'
  period_min: number
  period_max: number
  snr_threshold: number
}

// === ENHANCED API TYPES ===
export interface UploadResponse {
  file_id: string
  filename: string
  size: number
  status: 'uploaded' | 'processing' | 'completed' | 'failed'
  message?: string
}

export interface TrainingRequest {
  model_type: 'detector' | 'classifier' | 'ensemble'
  dataset_id?: string
  parameters?: Record<string, any>
  epochs?: number
  batch_size?: number
}

export interface TrainingResponse {
  training_id: string
  status: 'started' | 'running' | 'completed' | 'failed'
  progress?: number
  metrics?: ModelMetrics
  message?: string
}

export interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  loss: number
  val_accuracy?: number
  val_loss?: number
  training_time_seconds?: number
  model_size_mb?: number
}

export interface PredictionRequest {
  model_id?: string
  data: number[] | number[][]
  target_name?: string
  preprocessing?: {
    normalize?: boolean
    detrend?: boolean
    remove_outliers?: boolean
  }
}

export interface PredictionResponse {
  prediction: 'planet' | 'star' | 'noise'
  confidence: number
  probabilities: {
    planet: number
    star: number
    noise: number
  }
  processing_time_ms: number
  model_used: string
}
