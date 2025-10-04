import React, { useState } from 'react';
import { Settings, Play, Download, BarChart3, Activity, Zap, Target } from 'lucide-react';
import { exoplanetApi } from '../api/exoplanetApi';
import type { ProAnalysisRequest, ProAnalysisResponse, Candidate } from '../api/exoplanetApi';

interface AnalysisResult {
  candidates: Candidate[];
  detailedAnalysis: Record<string, any>;
  plotsData: Record<string, any>;
  processingTime: number;
  modelMetrics: Record<string, any>;
  error?: string;
}

const ProView: React.FC = () => {
  const [ticId, setTicId] = useState('');
  const [modelType, setModelType] = useState('detector');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [advancedParams, setAdvancedParams] = useState({
    threshold: 4.5,
    smoothing: 'wavelet',
    period_range: [0.5, 100],
    depth_range: [0.001, 0.1]
  });

  const handleAnalyze = async () => {
    if (!ticId.trim()) {
      alert('Пожалуйста, введите TIC ID');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      // Сначала загружаем данные
      const ticData = await exoplanetApi.loadTICData(ticId.trim());

      if (!ticData.success) {
        throw new Error('Не удалось загрузить данные TESS');
      }

      const request: ProAnalysisRequest = {
        lightcurve_data: ticData.data,
        model_type: modelType,
        parameters: {
          threshold: advancedParams.threshold,
          smoothing: advancedParams.smoothing
        },
        advanced_settings: {
          period_range: advancedParams.period_range,
          depth_range: advancedParams.depth_range
        }
      };

      const response: ProAnalysisResponse = await exoplanetApi.proAnalyze(request);

      if (response.success) {
        setResult({
          candidates: response.candidates,
          detailedAnalysis: response.detailed_analysis,
          plotsData: response.plots_data,
          processingTime: response.processing_time,
          modelMetrics: response.model_metrics
        });
      } else {
        throw new Error(response.error || 'Ошибка анализа');
      }
    } catch (error) {
      console.error('Ошибка анализа:', error);
      setResult({
        candidates: [],
        detailedAnalysis: {},
        plotsData: {},
        processingTime: 0,
        modelMetrics: {},
        error: error instanceof Error ? error.message : 'Неизвестная ошибка'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400 bg-green-500/20';
    if (confidence >= 0.6) return 'text-yellow-400 bg-yellow-500/20';
    return 'text-red-400 bg-red-500/20';
  };

  const getModelDescription = (model: string) => {
    switch (model) {
      case 'detector': return 'Базовая CNN модель для детекции транзитов';
      case 'verifier': return 'CNN + LSTM для верификации кандидатов';
      case 'pro': return 'CNN + Attention для продвинутого анализа';
      default: return '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Заголовок */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center gap-3">
            <BarChart3 className="w-10 h-10 text-blue-400" />
            Профессиональный режим
            <Activity className="w-10 h-10 text-purple-400" />
          </h1>
          <p className="text-xl text-gray-300">
            Детальный анализ с расширенными возможностями
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Левая панель - управление */}
          <div className="lg:col-span-1 space-y-6">
            {/* Основные настройки */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4">Основные настройки</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-white text-sm font-medium mb-2">
                    TIC ID звезды
                  </label>
                  <input
                    type="text"
                    value={ticId}
                    onChange={(e) => setTicId(e.target.value)}
                    placeholder="TIC 123456789"
                    className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={isAnalyzing}
                  />
                </div>

                <div>
                  <label className="block text-white text-sm font-medium mb-2">
                    Модель анализа
                  </label>
                  <select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                    className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={isAnalyzing}
                  >
                    <option value="detector">Detector (CNN)</option>
                    <option value="verifier">Verifier (CNN+LSTM)</option>
                    <option value="pro">Pro (CNN+Attention)</option>
                  </select>
                  <p className="text-xs text-gray-400 mt-1">{getModelDescription(modelType)}</p>
                </div>

                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="w-full px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/30 rounded-lg text-white transition-all duration-200 flex items-center justify-center gap-2"
                >
                  <Settings className="w-4 h-4" />
                  {showSettings ? 'Скрыть' : 'Показать'} расширенные настройки
                </button>

                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !ticId.trim()}
                  className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-600 text-white rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2 disabled:cursor-not-allowed"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      Анализ...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Начать детальный анализ
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Расширенные настройки */}
            {showSettings && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4">Расширенные параметры</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-white text-sm font-medium mb-2">
                      Порог обнаружения: {advancedParams.threshold}
                    </label>
                    <input
                      type="range"
                      min="2"
                      max="10"
                      step="0.1"
                      value={advancedParams.threshold}
                      onChange={(e) => setAdvancedParams(prev => ({...prev, threshold: parseFloat(e.target.value)}))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-white text-sm font-medium mb-2">
                      Метод сглаживания
                    </label>
                    <select
                      value={advancedParams.smoothing}
                      onChange={(e) => setAdvancedParams(prev => ({...prev, smoothing: e.target.value}))}
                      className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
                    >
                      <option value="wavelet">Wavelet</option>
                      <option value="median">Median</option>
                      <option value="gaussian">Gaussian</option>
                      <option value="savitzky_golay">Savitzky-Golay</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-white text-xs font-medium mb-1">
                        Мин. период (дни)
                      </label>
                      <input
                        type="number"
                        value={advancedParams.period_range[0]}
                        onChange={(e) => setAdvancedParams(prev => ({
                          ...prev,
                          period_range: [parseFloat(e.target.value), prev.period_range[1]]
                        }))}
                        className="w-full px-2 py-1 bg-white/20 border border-white/30 rounded text-white text-sm"
                        step="0.1"
                      />
                    </div>
                    <div>
                      <label className="block text-white text-xs font-medium mb-1">
                        Макс. период (дни)
                      </label>
                      <input
                        type="number"
                        value={advancedParams.period_range[1]}
                        onChange={(e) => setAdvancedParams(prev => ({
                          ...prev,
                          period_range: [prev.period_range[0], parseFloat(e.target.value)]
                        }))}
                        className="w-full px-2 py-1 bg-white/20 border border-white/30 rounded text-white text-sm"
                        step="0.1"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Метрики модели */}
            {result && !result.error && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Метрики модели
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Точность:</span>
                    <span className="text-white font-semibold">{(result.modelMetrics.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Precision:</span>
                    <span className="text-white font-semibold">{(result.modelMetrics.precision * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Recall:</span>
                    <span className="text-white font-semibold">{(result.modelMetrics.recall * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">ROC-AUC:</span>
                    <span className="text-white font-semibold">{(result.modelMetrics.roc_auc * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Правая панель - результаты */}
          <div className="lg:col-span-2">
            {result && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h2 className="text-2xl font-bold text-white mb-6">Результаты анализа</h2>

                {result.error ? (
                  <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4">
                    <p className="text-red-300">Ошибка: {result.error}</p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Кандидаты */}
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-4">
                        Обнаружено кандидатов: {result.candidates.length}
                      </h3>

                      <div className="grid gap-4">
                        {result.candidates.map((candidate, index) => (
                          <div key={index} className={`p-4 rounded-lg border ${getConfidenceColor(candidate.confidence)}`}>
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="text-white font-semibold">Кандидат {index + 1}</h4>
                              <span className="px-2 py-1 bg-white/20 rounded text-sm text-white">
                                {(candidate.confidence * 100).toFixed(1)}% уверенность
                              </span>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                              <div>
                                <p className="text-gray-300">Период</p>
                                <p className="text-white font-semibold">{candidate.period.toFixed(3)} дней</p>
                              </div>
                              <div>
                                <p className="text-gray-300">Глубина</p>
                                <p className="text-white font-semibold">{(candidate.depth * 100).toFixed(3)}%</p>
                              </div>
                              <div>
                                <p className="text-gray-300">Длительность</p>
                                <p className="text-white font-semibold">{candidate.duration.toFixed(2)} часов</p>
                              </div>
                              <div>
                                <p className="text-gray-300">Метод</p>
                                <p className="text-white font-semibold">{candidate.method}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Детальный анализ */}
                    {Object.keys(result.detailedAnalysis).length > 0 && (
                      <div>
                        <h3 className="text-lg font-semibold text-white mb-4">Детальный анализ</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="bg-white/5 rounded-lg p-4">
                            <h4 className="text-white font-medium mb-3">SNR и качество сигнала</h4>
                            <p className="text-gray-300">SNR: {result.detailedAnalysis.snr?.toFixed(2) || 'N/A'}</p>
                            <p className="text-gray-300">False positive rate: {((result.detailedAnalysis.false_positive_rate || 0) * 100).toFixed(2)}%</p>
                          </div>

                          <div className="bg-white/5 rounded-lg p-4">
                            <h4 className="text-white font-medium mb-3">Анализ периода</h4>
                            <p className="text-gray-300">Обнаруженный период: {result.detailedAnalysis.period_analysis?.detected_period?.toFixed(3) || 'N/A'} дней</p>
                            <p className="text-gray-300">Уверенность: {((result.detailedAnalysis.period_analysis?.period_confidence || 0) * 100).toFixed(1)}%</p>
                          </div>

                          <div className="bg-white/5 rounded-lg p-4 md:col-span-2">
                            <h4 className="text-white font-medium mb-3">Метрики транзита</h4>
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <p className="text-gray-300">Диапазон глубины:</p>
                                <p className="text-white font-semibold">
                                  {(result.detailedAnalysis.transit_metrics?.depth_range?.[0] * 100).toFixed(3)}% - {(result.detailedAnalysis.transit_metrics?.depth_range?.[1] * 100).toFixed(3)}%
                                </p>
                              </div>
                              <div>
                                <p className="text-gray-300">Диапазон длительности:</p>
                                <p className="text-white font-semibold">
                                  {result.detailedAnalysis.transit_metrics?.duration_range?.[0]} - {result.detailedAnalysis.transit_metrics?.duration_range?.[1]} часов
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Время обработки */}
                    <div className="flex items-center justify-between pt-4 border-t border-white/20">
                      <span className="text-gray-300">Время анализа:</span>
                      <span className="text-white font-semibold">{result.processingTime.toFixed(2)} секунд</span>
                    </div>

                    {/* Действия */}
                    <div className="flex gap-4 pt-4">
                      <button className="px-6 py-3 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white rounded-lg font-medium transition-all duration-200 flex items-center gap-2">
                        <Download className="w-5 h-5" />
                        Экспорт данных
                      </button>
                      <button className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg font-medium transition-all duration-200 flex items-center gap-2">
                        <Zap className="w-5 h-5" />
                        Сравнить модели
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {!result && !isAnalyzing && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
                <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Готов к анализу</h3>
                <p className="text-gray-400">
                  Введите TIC ID и выберите параметры для детального анализа кривых блеска
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProView;
