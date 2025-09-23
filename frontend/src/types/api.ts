export interface LightCurveData {
  time: number[]
  flux: number[]
  flux_err?: number[]
  target_name: string
  mission: string
  sector?: number
  quarter?: number
}

export interface SearchRequest {
  target_name: string
  catalog: 'TIC' | 'KIC' | 'EPIC'
  mission: 'TESS' | 'Kepler' | 'K2'
  period_min: number
  period_max: number
  duration_min: number
  duration_max: number
  snr_threshold: number
}

export interface BLSResult {
  best_period: number
  best_power: number
  best_duration: number
  best_t0: number
  snr: number
  depth: number
  depth_err: number
  significance: number
}

export interface TransitCandidate {
  period: number
  epoch: number
  duration: number
  depth: number
  snr: number
  significance: number
  is_planet_candidate: boolean
  confidence: number
}

export interface AIAnalysis {
  is_transit: boolean
  confidence: number
  confidence_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'VERY_HIGH'
  explanation: string
  model_predictions: {
    cnn: number
    lstm: number
    transformer: number
    ensemble: number
  }
  uncertainty: number
  similar_targets?: string[]
}

export interface SearchResult {
  target_name: string
  analysis_timestamp: string
  lightcurve_data: LightCurveData
  bls_results: BLSResult
  candidates: TransitCandidate[]
  ai_analysis?: AIAnalysis
  physical_parameters?: {
    planet_radius?: number
    planet_mass?: number
    orbital_period?: number
    equilibrium_temperature?: number
    habitability_score?: number
  }
  status: 'success' | 'error' | 'processing'
  message?: string
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'down'
  timestamp: string
  services_available: boolean
  scientific_libs: boolean
  services: {
    bls: 'active' | 'unavailable'
    data: 'active' | 'unavailable'
    ai: 'active' | 'unavailable'
  }
}

export interface Target {
  name: string
  catalog_id: string
  ra: number
  dec: number
  magnitude: number
  stellar_type?: string
  temperature?: number
  radius?: number
  mass?: number
}

export interface Catalog {
  name: string
  description: string
  missions: string[]
}

export interface ExportFormat {
  format: 'csv' | 'json'
  data: any
  filename: string
}

export type LoadingState = 'idle' | 'loading' | 'success' | 'error'
