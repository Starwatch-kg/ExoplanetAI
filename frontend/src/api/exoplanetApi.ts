// API клиент для взаимодействия с backend
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export interface LightcurveData {
  tic_id: string;
  times: number[];
  fluxes: number[];
  sector?: string;
  camera?: string;
  ccd?: string;
}

export interface Candidate {
  id: string;
  period: number;
  depth: number;
  duration: number;
  confidence: number;
  start_time: number;
  end_time: number;
  method: string;
}

export interface AnalysisRequest {
  lightcurve_data: LightcurveData;
  model_type: string;
  parameters?: Record<string, any>;
}

export interface AnalysisResponse {
  success: boolean;
  candidates: Candidate[];
  processing_time: number;
  model_used: string;
  statistics: Record<string, any>;
  error?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface SystemMetrics {
  timestamp: string;
  system: {
    platform: string;
    architecture: string;
    python_version: string;
  };
  cpu: {
    usage_percent: number;
    cores: number;
    status: string;
  };
  memory: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
    available_gb: number;
    status: string;
  };
  disk: {
    usage_percent: number;
    free_gb: number;
    status: string;
  };
  network: {
    latency_ms: number;
    status: string;
  };
  application: {
    uptime: string;
    requests_total: number;
    active_analyses: number;
    ml_modules_loaded: boolean;
    cnn_available: boolean;
    python_processes: number;
  };
}

class ExoplanetAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    const token = localStorage.getItem('accessToken');
    if (token) {
      (defaultOptions.headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, { ...defaultOptions, ...options });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Проверка состояния API
  async healthCheck(): Promise<{ status: string; uptimeSec: number }> {
    return this.request('/health');
  }

  // Получение системных метрик
  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.request('/api/system/metrics');
  }

  // Загрузка данных TESS по TIC ID
  async loadTICData(ticId: string, sectors?: number[]): Promise<{
    success: boolean;
    data: LightcurveData;
  }> {
    return this.request('/load-tic', {
      method: 'POST',
      body: JSON.stringify({
        tic_id: ticId,
        sectors: sectors,
      }),
    });
  }

  // Анализ кривой блеска
  async analyzeLightcurve(
    lightcurveData: LightcurveData,
    modelType: string,
    parameters?: Record<string, any>
  ): Promise<AnalysisResponse> {
    return this.request('/analyze', {
      method: 'POST',
      body: JSON.stringify({
        lightcurve_data: lightcurveData,
        model_type: modelType,
        parameters: parameters,
      }),
    });
  }

  // Получение статистики NASA
  async getNASAStats(): Promise<{
    totalPlanets: number;
    totalHosts: number;
    lastUpdated: string;
    source: string;
  }> {
    return this.request('/api/nasa/stats');
  }

  // CNN методы
  async getCNNModels(): Promise<{
    available_architectures: any[];
    saved_models: any[];
    cnn_available: boolean;
  }> {
    return this.request('/api/cnn/models');
  }

  async startCNNTraining(request: {
    model_type: string;
    model_params?: Record<string, any>;
    training_params?: Record<string, any>;
    data_params?: Record<string, any>;
  }): Promise<{
    success: boolean;
    training_id: string;
    message: string;
    model_info?: Record<string, any>;
    error?: string;
  }> {
    return this.request('/api/cnn/train', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getCNNTrainingStatus(trainingId: string): Promise<{
    training_id: string;
    status: string;
    current_epoch: number;
    total_epochs: number;
    current_metrics?: Record<string, number>;
    best_metrics?: Record<string, number>;
    progress_percentage: number;
    estimated_time_remaining?: number;
  }> {
    return this.request(`/api/cnn/training/${trainingId}/status`);
  }

  async performCNNInference(request: {
    model_id: string;
    lightcurve_data: LightcurveData;
    preprocessing?: Record<string, any>;
  }): Promise<{
    success: boolean;
    prediction: Record<string, any>;
    confidence: number;
    processing_time: number;
    model_used: string;
    error?: string;
  }> {
    return this.request('/api/cnn/inference', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Получение списка доступных моделей
  async getModels(): Promise<ModelInfo[]> {
    return this.request('/models');
  }
}

// Создаем экземпляр API клиента
export const exoplanetApi = new ExoplanetAPI();

// Экспортируем типы для использования в компонентах
