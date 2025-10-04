// API клиент для взаимодействия с backend
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export interface LightcurveData {
  tic_id: string;
  times: number[];
  fluxes: number[];
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

  // Анализ в любительском режиме
  async amateurAnalyze(
    request: AmateurAnalysisRequest
  ): Promise<AmateurAnalysisResponse> {
    return this.request('/amateur/analyze', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Анализ в профессиональном режиме
  async proAnalyze(
    request: ProAnalysisRequest
  ): Promise<ProAnalysisResponse> {
    return this.request('/pro/analyze', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Получение статистики NASA
  async getNASAStats(): Promise<Record<string, any>> {
    return this.request('/api/nasa/stats');
  }
}

// Создаем экземпляр API клиента
export const exoplanetApi = new ExoplanetAPI();

// Экспортируем типы для использования в компонентах

export interface AmateurAnalysisRequest {
  tic_id: string;
}

export interface AmateurAnalysisResponse {
  success: boolean;
  candidate?: Candidate;
  summary: Record<string, any>;
  processing_time: number;
  error?: string;
}

export interface ProAnalysisRequest {
  lightcurve_data: LightcurveData;
  model_type: string;
  parameters?: Record<string, any>;
  advanced_settings?: Record<string, any>;
}

export interface ProAnalysisResponse {
  success: boolean;
  candidates: Candidate[];
  detailed_analysis: Record<string, any>;
  plots_data: Record<string, any>;
  processing_time: number;
  model_metrics: Record<string, any>;
  error?: string;
}
