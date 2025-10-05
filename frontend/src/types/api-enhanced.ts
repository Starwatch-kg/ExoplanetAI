// Enhanced API Type Definitions for ExoplanetAI

// Base types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  message?: string
  error?: string
  timestamp: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  per_page: number
  pages: number
}

// Authentication types
export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  email: string
  password: string
  name: string
  role?: 'guest' | 'user' | 'researcher' | 'admin'
}

export interface AuthResponse {
  token: string
  refresh_token: string
  user: User
  expires_in: number
}

export interface User {
  id: string
  email: string
  name: string
  role: 'guest' | 'user' | 'researcher' | 'admin'
  created_at: string
  last_login?: string
  preferences?: UserPreferences
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto'
  language: 'en' | 'ru'
  notifications: boolean
  default_catalog: string
}

// Health and status types
export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp: string
  version: string
  uptime: number
  components: {
    database: ComponentStatus
    cache: ComponentStatus
    data_sources: ComponentStatus
    ml_models: ComponentStatus
  }
}

export interface ComponentStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  response_time?: number
  last_check: string
  details?: string
}

// Search and catalog types
export interface SearchRequest {
  target?: string
  catalog?: string
  ra?: number
  dec?: number
  radius?: number
  magnitude_min?: number
  magnitude_max?: number
  period_min?: number
  period_max?: number
  filters?: SearchFilters
  limit?: number
  offset?: number
}

export interface SearchFilters {
  mission?: string[]
  object_type?: string[]
  disposition?: string[]
  has_lightcurve?: boolean
  quality_flags?: string[]
}

export interface SearchResult {
  targets: ExoplanetTarget[]
  total: number
  query_time: number
  cache_hit: boolean
  filters_applied: SearchFilters
  pagination: {
    page: number
    per_page: number
    total_pages: number
  }
}

export interface ExoplanetTarget {
  id: string
  tic_id?: string
  toi_id?: string
  koi_id?: string
  name: string
  ra: number
  dec: number
  magnitude: number
  stellar_properties: StellarProperties
  planetary_candidates?: PlanetaryCandidate[]
  lightcurve_available: boolean
  analysis_status: 'pending' | 'processing' | 'completed' | 'failed'
  last_updated: string
}

export interface StellarProperties {
  temperature?: number
  radius?: number
  mass?: number
  metallicity?: number
  distance?: number
  spectral_type?: string
}

export interface PlanetaryCandidate {
  id: string
  period: number
  period_error?: number
  epoch: number
  epoch_error?: number
  depth: number
  depth_error?: number
  duration: number
  duration_error?: number
  snr: number
  disposition: 'candidate' | 'confirmed' | 'false_positive'
  discovery_method: string
  confidence_score: number
}

export interface CatalogsResponse {
  catalogs: CatalogInfo[]
  total_targets: number
  last_updated: string
}

export interface CatalogInfo {
  name: string
  description: string
  source: string
  total_targets: number
  last_updated: string
  available_fields: string[]
}

// File upload types
export interface UploadResponse {
  file_id: string
  filename: string
  size: number
  mime_type: string
  upload_time: string
  validation_status: 'pending' | 'valid' | 'invalid'
  validation_errors?: string[]
  preview?: DataPreview
}

export interface DataPreview {
  columns: string[]
  sample_rows: any[]
  total_rows: number
  data_types: Record<string, string>
}

// Machine learning types
export interface TrainingRequest {
  dataset_id: string
  model_type: 'detector' | 'classifier' | 'ensemble'
  hyperparameters?: Record<string, any>
  validation_split?: number
  epochs?: number
  batch_size?: number
  early_stopping?: boolean
}

export interface TrainingResponse {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  estimated_time_remaining?: number
  current_epoch?: number
  metrics?: TrainingMetrics
}

export interface TrainingMetrics {
  loss: number
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  val_loss?: number
  val_accuracy?: number
  val_precision?: number
  val_recall?: number
  val_f1_score?: number
}

export interface ModelMetrics {
  model_id: string
  model_type: string
  version: string
  training_date: string
  performance: {
    accuracy: number
    precision: number
    recall: number
    f1_score: number
    auc_roc: number
    confusion_matrix: number[][]
  }
  validation_metrics: TrainingMetrics
  feature_importance?: FeatureImportance[]
}

export interface FeatureImportance {
  feature: string
  importance: number
  description?: string
}

// Prediction types
export interface PredictionRequest {
  model_id?: string
  data: LightCurveData | any[]
  return_probabilities?: boolean
  return_features?: boolean
}

export interface PredictionResponse {
  prediction: string
  confidence: number
  probabilities?: Record<string, number>
  features?: Record<string, number>
  processing_time: number
  model_info: {
    id: string
    version: string
    type: string
  }
}

// Light curve types
export interface LightCurveData {
  time: number[]
  flux: number[]
  flux_error?: number[]
  quality_flags?: number[]
  metadata: LightCurveMetadata
}

export interface LightCurveMetadata {
  tic_id: string
  sector?: number
  camera?: number
  ccd?: number
  mission: 'TESS' | 'Kepler' | 'K2'
  cadence: 'short' | 'long'
  start_time: number
  end_time: number
  total_points: number
  quality_summary: QualitySummary
}

export interface QualitySummary {
  good_points: number
  flagged_points: number
  gap_count: number
  outlier_count: number
  quality_score: number
}

export interface LightCurveAnalysis {
  bls_results: BLSResults
  periodogram: PeriodogramData
  statistics: LightCurveStatistics
  quality_assessment: QualityAssessment
  transit_candidates: TransitCandidate[]
}

export interface BLSResults {
  period: number
  period_error: number
  epoch: number
  epoch_error: number
  depth: number
  depth_error: number
  duration: number
  duration_error: number
  snr: number
  significance: number
  false_alarm_probability: number
}

export interface PeriodogramData {
  frequencies: number[]
  power: number[]
  peak_frequency: number
  peak_power: number
}

export interface LightCurveStatistics {
  mean_flux: number
  std_flux: number
  median_flux: number
  mad_flux: number
  skewness: number
  kurtosis: number
  autocorr_lag1: number
  trend_slope: number
}

export interface QualityAssessment {
  overall_score: number
  completeness: number
  noise_level: number
  systematic_trends: number
  outlier_fraction: number
  gap_analysis: GapAnalysis
}

export interface GapAnalysis {
  total_gaps: number
  max_gap_duration: number
  gap_fraction: number
  significant_gaps: number
}

export interface TransitCandidate {
  period: number
  epoch: number
  depth: number
  duration: number
  snr: number
  confidence: number
  shape_score: number
  periodicity_score: number
  uniqueness_score: number
}

// Data management types
export interface DataIngestionRequest {
  source: 'file' | 'api' | 'catalog'
  data_type: 'lightcurve' | 'catalog' | 'parameters'
  format: 'csv' | 'fits' | 'json'
  file_id?: string
  api_params?: Record<string, any>
  validation_rules?: ValidationRule[]
}

export interface ValidationRule {
  field: string
  rule_type: 'required' | 'range' | 'format' | 'custom'
  parameters: any
  error_message: string
}

export interface DataIngestionResponse {
  job_id: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  records_processed: number
  records_total: number
  validation_errors: ValidationError[]
  warnings: string[]
}

export interface ValidationError {
  row: number
  field: string
  value: any
  error: string
  severity: 'error' | 'warning'
}

// Version control types
export interface DataVersion {
  name: string
  description: string
  created_at: string
  created_by: string
  checksum: string
  size: number
  record_count: number
  changes: VersionChange[]
}

export interface VersionChange {
  type: 'added' | 'modified' | 'deleted'
  table: string
  record_id: string
  field?: string
  old_value?: any
  new_value?: any
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'training_progress' | 'analysis_update' | 'system_notification'
  data: any
  timestamp: string
  session_id?: string
}

export interface TrainingProgressMessage {
  job_id: string
  progress: number
  current_epoch: number
  total_epochs: number
  metrics: TrainingMetrics
  estimated_time_remaining: number
  status: 'running' | 'completed' | 'failed'
}

// Admin types
export interface AdminStats {
  system: SystemStats
  users: UserStats
  data: DataStats
  models: ModelStats
  performance: PerformanceStats
}

export interface SystemStats {
  uptime: number
  cpu_usage: number
  memory_usage: number
  disk_usage: number
  active_connections: number
  request_rate: number
  error_rate: number
}

export interface UserStats {
  total_users: number
  active_users: number
  new_users_today: number
  user_roles: Record<string, number>
  top_users: UserActivity[]
}

export interface UserActivity {
  user_id: string
  name: string
  requests: number
  last_active: string
}

export interface DataStats {
  total_targets: number
  total_lightcurves: number
  data_size: number
  ingestion_rate: number
  quality_distribution: Record<string, number>
}

export interface ModelStats {
  total_models: number
  active_models: number
  training_jobs: number
  prediction_rate: number
  model_performance: Record<string, ModelMetrics>
}

export interface PerformanceStats {
  avg_response_time: number
  cache_hit_rate: number
  error_rate: number
  throughput: number
  slow_queries: SlowQuery[]
}

export interface SlowQuery {
  endpoint: string
  avg_time: number
  count: number
  last_occurrence: string
}

// Error types
export interface ApiError {
  code: string
  message: string
  details?: any
  timestamp: string
  request_id?: string
}

// Export all types
export type {
  // Re-export existing types for compatibility
  SearchRequest as LegacySearchRequest,
  SearchResult as LegacySearchResult,
  HealthStatus as LegacyHealthStatus,
  CatalogsResponse as LegacyCatalogsResponse
}
