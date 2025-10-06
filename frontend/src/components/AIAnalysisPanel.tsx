import React, { useState, useCallback } from 'react';
import { Brain, Upload, Download, BarChart3, AlertTriangle, CheckCircle, Info, Zap } from 'lucide-react';
import Plot from 'react-plotly.js';

interface AnalysisRequest {
  target_name?: string;
  time_data?: number[];
  flux_data?: number[];
  flux_err_data?: number[];
  mission: string;
  auto_detect_cadence: boolean;
  adaptive_detrending: boolean;
  include_uncertainty: boolean;
  explain_prediction: boolean;
}

interface AnalysisResult {
  target_name: string;
  predicted_class: string;
  confidence_score: number;
  uncertainty_bounds: [number, number];
  transit_probability: number;
  signal_characteristics: {
    snr_estimate: number;
    data_points: number;
    time_span_days: number;
  };
  feature_importance: number[];
  decision_reasoning: string[];
  recommendations: string[];
  data_quality_metrics: {
    photometric_precision: number;
    systematic_noise_level: number;
    data_completeness: number;
    instrumental_effects_score: number;
  };
  processing_time_ms: number;
  model_version: string;
}

const AIAnalysisPanel: React.FC = () => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string>('');
  const [targetName, setTargetName] = useState('');
  const [uploadedData, setUploadedData] = useState<{time: number[], flux: number[]} | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'target' | 'upload'>('target');

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split('\n').filter(line => line.trim());
        
        const data = lines.slice(1).map(line => {
          const values = line.split(/[,\s]+/).map(v => parseFloat(v.trim()));
          return { time: values[0], flux: values[1] };
        }).filter(row => !isNaN(row.time) && !isNaN(row.flux));

        setUploadedData({
          time: data.map(d => d.time),
          flux: data.map(d => d.flux)
        });
        
        setError('');
      } catch (err) {
        setError('Ошибка при чтении файла. Убедитесь, что файл содержит данные в формате CSV.');
      }
    };
    
    reader.readAsText(file);
  }, []);

  const performAnalysis = useCallback(async () => {
    if (analysisMode === 'target' && !targetName.trim()) {
      setError('Введите название цели');
      return;
    }
    
    if (analysisMode === 'upload' && !uploadedData) {
      setError('Загрузите файл с данными');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    try {
      const requestData: AnalysisRequest = {
        mission: 'TESS',
        auto_detect_cadence: true,
        adaptive_detrending: true,
        include_uncertainty: true,
        explain_prediction: true
      };

      if (analysisMode === 'target') {
        requestData.target_name = targetName;
      } else {
        requestData.time_data = uploadedData!.time;
        requestData.flux_data = uploadedData!.flux;
      }

      const response = await fetch('/api/v1/ai/analyze_lightcurve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`Анализ не удался: ${response.status}`);
      }

      const result: AnalysisResult = await response.json();
      setAnalysisResult(result);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Произошла ошибка при анализе');
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  }, [analysisMode, targetName, uploadedData]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getClassColor = (className: string) => {
    switch (className) {
      case 'Confirmed':
      case 'CANDIDATE':
        return 'bg-green-600';
      case 'PC':
        return 'bg-yellow-600';
      default:
        return 'bg-red-600';
    }
  };

  const renderFeatureImportance = () => {
    if (!analysisResult?.feature_importance) return null;

    const featureNames = [
      'Глубина транзита', 'SNR', 'Длительность', 'Асимметрия', 'Форма V',
      'Форма U', 'Форма Box', 'Частотная мощность', 'Качество данных', 'Шум'
    ];

    const plotData = [{
      x: analysisResult.feature_importance.slice(0, 10),
      y: featureNames.slice(0, 10),
      type: 'bar' as const,
      orientation: 'h' as const,
      marker: {
        color: 'rgba(59, 130, 246, 0.8)',
        line: {
          color: 'rgba(59, 130, 246, 1)',
          width: 1
        }
      }
    }];

    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Важность признаков
        </h3>
        <Plot
          data={plotData}
          layout={{
            height: 400,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { l: 120, r: 20, t: 20, b: 40 },
            xaxis: {
              gridcolor: 'rgba(255,255,255,0.1)',
              title: { text: 'Важность' }
            },
            yaxis: {
              gridcolor: 'rgba(255,255,255,0.1)'
            }
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>
    );
  };

  const renderDataQualityMetrics = () => {
    if (!analysisResult?.data_quality_metrics) return null;

    const metrics = analysisResult.data_quality_metrics;
    
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Метрики качества данных</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-700 rounded p-3">
            <div className="text-sm text-gray-400">Фотометрическая точность</div>
            <div className="text-lg font-semibold text-white">
              {(metrics.photometric_precision * 1000).toFixed(1)} ppm
            </div>
          </div>
          
          <div className="bg-gray-700 rounded p-3">
            <div className="text-sm text-gray-400">Систематический шум</div>
            <div className="text-lg font-semibold text-white">
              {(metrics.systematic_noise_level * 100).toFixed(2)}%
            </div>
          </div>
          
          <div className="bg-gray-700 rounded p-3">
            <div className="text-sm text-gray-400">Полнота данных</div>
            <div className="text-lg font-semibold text-white">
              {(metrics.data_completeness * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="bg-gray-700 rounded p-3">
            <div className="text-sm text-gray-400">Инструментальные эффекты</div>
            <div className="text-lg font-semibold text-white">
              {(metrics.instrumental_effects_score * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold text-white flex items-center justify-center gap-2">
          <Brain className="w-8 h-8 text-purple-400" />
          ИИ Анализ экзопланет
        </h1>
        <p className="text-gray-300">
          Продвинутый анализ кривых блеска с объяснимым ИИ и оценкой неопределенности
        </p>
      </div>

      {/* Analysis Mode Selection */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setAnalysisMode('target')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              analysisMode === 'target' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Анализ по названию
          </button>
          <button
            onClick={() => setAnalysisMode('upload')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              analysisMode === 'upload' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Загрузка файла
          </button>
        </div>

        {analysisMode === 'target' ? (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Название цели (TOI-715, Kepler-452b, TIC-441420236, etc.)
              </label>
              <input
                type="text"
                value={targetName}
                onChange={(e) => setTargetName(e.target.value)}
                placeholder="Введите название цели..."
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Загрузить файл с данными кривой блеска (CSV)
              </label>
              <div className="flex items-center justify-center w-full">
                <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-gray-700 hover:bg-gray-600">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Upload className="w-8 h-8 mb-4 text-gray-400" />
                    <p className="mb-2 text-sm text-gray-400">
                      <span className="font-semibold">Нажмите для загрузки</span> или перетащите файл
                    </p>
                    <p className="text-xs text-gray-500">CSV файлы (время, поток)</p>
                  </div>
                  <input type="file" accept=".csv,.txt" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>
              {uploadedData && (
                <div className="mt-2 text-sm text-green-400">
                  ✓ Загружено {uploadedData.time.length} точек данных
                </div>
              )}
            </div>
          </div>
        )}

        <button
          onClick={performAnalysis}
          disabled={isAnalyzing || (analysisMode === 'target' && !targetName.trim()) || (analysisMode === 'upload' && !uploadedData)}
          className="w-full mt-6 px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white transition-colors flex items-center justify-center gap-2"
        >
          {isAnalyzing ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Анализ...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              Запустить ИИ анализ
            </>
          )}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border border-red-600 rounded-lg p-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Analysis Results */}
      {analysisResult && (
        <div className="space-y-6">
          {/* Main Results */}
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">Результаты анализа</h2>
              <div className="text-sm text-gray-400">
                Время обработки: {analysisResult.processing_time_ms.toFixed(0)} мс
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Classification Result */}
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <h3 className="font-semibold text-white">Классификация</h3>
                </div>
                <div className={`inline-block px-3 py-1 rounded-full text-white text-sm font-medium ${getClassColor(analysisResult.predicted_class)}`}>
                  {analysisResult.predicted_class}
                </div>
              </div>

              {/* Confidence */}
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-5 h-5 text-blue-400" />
                  <h3 className="font-semibold text-white">Уверенность</h3>
                </div>
                <div className={`text-2xl font-bold ${getConfidenceColor(analysisResult.confidence_score)}`}>
                  {(analysisResult.confidence_score * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Диапазон: {(analysisResult.uncertainty_bounds[0] * 100).toFixed(1)}% - {(analysisResult.uncertainty_bounds[1] * 100).toFixed(1)}%
                </div>
              </div>

              {/* Transit Probability */}
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Info className="w-5 h-5 text-yellow-400" />
                  <h3 className="font-semibold text-white">Вероятность транзита</h3>
                </div>
                <div className="text-2xl font-bold text-yellow-400">
                  {(analysisResult.transit_probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Signal Characteristics */}
            <div className="mt-6 bg-gray-700 rounded-lg p-4">
              <h3 className="font-semibold text-white mb-3">Характеристики сигнала</h3>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">SNR:</span>
                  <span className="text-white ml-2 font-medium">{analysisResult.signal_characteristics.snr_estimate.toFixed(1)}</span>
                </div>
                <div>
                  <span className="text-gray-400">Точки данных:</span>
                  <span className="text-white ml-2 font-medium">{analysisResult.signal_characteristics.data_points.toLocaleString()}</span>
                </div>
                <div>
                  <span className="text-gray-400">Временной охват:</span>
                  <span className="text-white ml-2 font-medium">{analysisResult.signal_characteristics.time_span_days.toFixed(1)} дней</span>
                </div>
              </div>
            </div>
          </div>

          {/* Feature Importance */}
          {renderFeatureImportance()}

          {/* Data Quality Metrics */}
          {renderDataQualityMetrics()}

          {/* AI Reasoning */}
          {analysisResult.decision_reasoning.length > 0 && (
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Объяснение решения ИИ
              </h3>
              <ul className="space-y-2">
                {analysisResult.decision_reasoning.map((reason, index) => (
                  <li key={index} className="text-gray-300 flex items-start gap-2">
                    <span className="text-blue-400 mt-1">•</span>
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Recommendations */}
          {analysisResult.recommendations.length > 0 && (
            <div className="bg-blue-900 border border-blue-600 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-blue-200 mb-4">Рекомендации</h3>
              <ul className="space-y-2">
                {analysisResult.recommendations.map((rec, index) => (
                  <li key={index} className="text-blue-200 flex items-start gap-2">
                    <span className="text-blue-400 mt-1">→</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Export Results */}
          <div className="bg-gray-800 rounded-lg p-4">
            <button
              onClick={() => {
                const dataStr = JSON.stringify(analysisResult, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `exoplanet_analysis_${analysisResult.target_name}_${Date.now()}.json`;
                link.click();
              }}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            >
              <Download className="w-4 h-4" />
              Экспортировать результаты
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIAnalysisPanel;
