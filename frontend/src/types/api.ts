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
  status: 'healthy' | 'degraded' | 'unhealthy'
  uptime?: number
  version?: string
  services?: Record<string, string>
  components?: {
    data_sources?: {
      status: 'healthy' | 'unhealthy'
      initialized: number
      total: number
    }
    cache?: {
      status: 'healthy' | 'unhealthy'
      redis_connected: boolean
    }
    authentication?: {
      status: 'healthy' | 'unhealthy'
    }
  }
  timestamp?: number
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
