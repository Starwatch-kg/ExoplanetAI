export interface ToastMessage {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
}

export interface LoadingState {
  isLoading: boolean
  message?: string
  progress?: number
}

export interface PlotConfig {
  displayModeBar: boolean
  responsive: boolean
  toImageButtonOptions: {
    format: 'png' | 'svg' | 'jpeg' | 'webp'
    filename: string
    height: number
    width: number
    scale: number
  }
}

export interface ChartData {
  x: number[]
  y: number[]
  type: 'scatter' | 'line'
  mode?: 'lines' | 'markers' | 'lines+markers'
  name?: string
  line?: {
    color?: string
    width?: number
    dash?: string
  }
  marker?: {
    color?: string
    size?: number
    opacity?: number
  }
}

export interface AnalysisStep {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'error'
  progress?: number
  message?: string
  duration?: number
}

export interface FilterOptions {
  catalog: string[]
  mission: string[]
  dateRange: {
    start: Date | null
    end: Date | null
  }
  confidenceRange: {
    min: number
    max: number
  }
  sortBy: 'date' | 'confidence' | 'target_name' | 'snr'
  sortOrder: 'asc' | 'desc'
}

export interface PaginationState {
  page: number
  pageSize: number
  total: number
  hasNext: boolean
  hasPrevious: boolean
}

export interface FormField {
  name: string
  label: string
  type: 'text' | 'number' | 'select' | 'range' | 'checkbox'
  value: any
  options?: { label: string; value: any }[]
  min?: number
  max?: number
  step?: number
  required?: boolean
  placeholder?: string
  helpText?: string
}
