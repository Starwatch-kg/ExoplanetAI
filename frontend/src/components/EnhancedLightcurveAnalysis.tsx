import React, { useState, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Target,
  Database,
  Play,
  Download,
  Loader,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  Zap
} from 'lucide-react';
import { exoplanetApi } from '../api/exoplanetApi';
import NASADataBrowser from './NASADataBrowser';
import { useBackgroundTask } from '../hooks/useBackgroundTask';

// Lazy load Plotly для оптимизации
const Plot = React.lazy(() => import('react-plotly.js'));

interface LightcurveData {
  tic_id: string;
  times: number[];
  fluxes: number[];
  flux_errors?: number[];
  star_parameters?: any;
  transit_parameters?: any;
}

interface AnalysisCandidate {
  id: string;
  period: number;
  depth: number;
  duration: number;
  confidence: number;
  start_time: number;
  end_time: number;
  method: string;
}

const EnhancedLightcurveAnalysis: React.FC = () => {
  const [ticId, setTicId] = useState('261136679');
  const [lightcurveData, setLightcurveData] = useState<LightcurveData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisMethod, setAnalysisMethod] = useState('bls');
  const [candidates, setCandidates] = useState<AnalysisCandidate[]>([]);
  const [showNASABrowser, setShowNASABrowser] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  
  // Используем хук для фоновых задач
  const { startLightcurveAnalysis } = useBackgroundTask();

  // Методы анализа с улучшенным дизайном
  const analysisMethods = [
    { id: 'bls', name: 'Box Least Squares', description: 'Классический метод поиска транзитов', color: 'blue' },
    { id: 'tls', name: 'Transit Least Squares', description: 'Улучшенный алгоритм TLS', color: 'purple' },
    { id: 'cnn', name: 'CNN Detection', description: 'Нейросетевая детекция', color: 'green' },
    { id: 'hybrid', name: 'Hybrid Method', description: 'Комбинированный подход', color: 'orange' },
    { id: 'ensemble', name: 'Ensemble', description: 'Ансамбль методов', color: 'red' }
  ];

  // Загрузка данных TESS
  const loadTICData = async () => {
    if (!ticId.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await exoplanetApi.loadTICData(ticId);
      if (response.success) {
        setLightcurveData(response.data);
      } else {
        setError('Не удалось загрузить данные');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка загрузки данных');
    } finally {
      setIsLoading(false);
    }
  };

  // Запуск анализа с анимацией прогресса
  const startAnalysis = async () => {
    if (!lightcurveData) return;
    
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setCandidates([]);
    setError(null);
    
    // Анимация прогресса
    const progressInterval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + Math.random() * 10;
      });
    }, 500);
    
    try {
      // Запускаем фоновую задачу анализа
      startLightcurveAnalysis(ticId);
      
      // Симуляция анализа для демонстрации
      const mockCandidates: AnalysisCandidate[] = [
        {
          id: 'candidate_1',
          period: 3.14159,
          depth: 0.0023,
          duration: 4.2,
          confidence: 0.87,
          start_time: 1234.5,
          end_time: 1238.7,
          method: analysisMethod
        }
      ];
      
      // Устанавливаем mock результаты
      setCandidates(mockCandidates);
      setAnalysisProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка анализа');
    } finally {
      clearInterval(progressInterval);
      setTimeout(() => {
        setIsAnalyzing(false);
        setAnalysisProgress(0);
      }, 1000);
    }
  };

  // Обработчик выбора метода с анимацией
  const handleMethodSelect = (methodId: string) => {
    setSelectedMethod(methodId);
    setAnalysisMethod(methodId);
    
    // Убираем подсветку через 2 секунды
    setTimeout(() => {
      setSelectedMethod(null);
    }, 2000);
  };

  // Экспорт данных
  const exportData = () => {
    if (!lightcurveData) return;
    
    const dataStr = JSON.stringify({
      tic_id: lightcurveData.tic_id,
      times: lightcurveData.times,
      fluxes: lightcurveData.fluxes,
      candidates: candidates,
      analysis_method: analysisMethod,
      export_date: new Date().toISOString()
    }, null, 2);
    
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = `TIC_${lightcurveData.tic_id}_analysis.json`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="space-y-lg">
      {/* Заголовок */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="card-title flex items-center gap-sm">
            <BarChart3 className="w-6 h-6 text-primary" />
            Анализ кривых блеска TESS
          </h2>
          <div className="text-sm text-secondary">
            Поиск экзопланет в данных космического телескопа TESS
          </div>
        </div>
      </motion.div>

      {/* Верхняя панель - Управление и NASA браузер */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-lg">
        {/* Левая панель - Управление */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="card-title flex items-center gap-sm">
              <Target className="w-5 h-5 text-blue-400" />
              Управление анализом
            </h3>
          </div>
          <div className="card-body space-y-md">
            {/* Ввод TIC ID */}
            <div>
              <label className="form-label">TIC ID звезды</label>
              <div className="flex gap-sm">
                <input
                  type="text"
                  value={ticId}
                  onChange={(e) => setTicId(e.target.value)}
                  placeholder="Например: 261136679"
                  className="form-input flex-1"
                />
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowNASABrowser(!showNASABrowser)}
                  className="btn btn-secondary"
                  title="NASA браузер"
                >
                  <Database className="w-4 h-4" />
                </motion.button>
              </div>
            </div>

            {/* Кнопка загрузки */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={loadTICData}
              disabled={isLoading || !ticId.trim()}
              className="btn btn-primary w-full"
            >
              {isLoading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin mr-2" />
                  Загрузка данных TESS...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Загрузить данные TESS
                </>
              )}
            </motion.button>

            {/* Выбор метода анализа */}
            {lightcurveData && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-sm"
              >
                <label className="form-label">Метод анализа</label>
                <div className="grid grid-cols-1 gap-2">
                  {analysisMethods.map((method) => (
                    <motion.button
                      key={method.id}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleMethodSelect(method.id)}
                      className={`p-3 rounded-lg border text-left transition-all duration-300 ${
                        analysisMethod === method.id
                          ? `border-${method.color}-500 bg-${method.color}-500/10`
                          : 'border-gray-600 hover:border-gray-500'
                      } ${
                        selectedMethod === method.id
                          ? `ring-2 ring-${method.color}-400 shadow-lg shadow-${method.color}-400/20`
                          : ''
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">{method.name}</div>
                          <div className="text-xs text-gray-400">{method.description}</div>
                        </div>
                        {analysisMethod === method.id && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className={`w-2 h-2 rounded-full bg-${method.color}-400`}
                          />
                        )}
                      </div>
                    </motion.button>
                  ))}
                </div>

                {/* Кнопка запуска анализа */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={startAnalysis}
                  disabled={isAnalyzing}
                  className="btn btn-success w-full"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader className="w-4 h-4 animate-spin mr-2" />
                      Анализ в процессе...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Запустить анализ
                    </>
                  )}
                </motion.button>
              </motion.div>
            )}
          </div>
        </motion.div>

        {/* Правая панель - NASA браузер и статистика */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-md"
        >
          {/* NASA браузер */}
          <AnimatePresence>
            {showNASABrowser && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="card"
              >
                <div className="card-header">
                  <h3 className="card-title flex items-center gap-sm">
                    <Database className="w-5 h-5 text-yellow-400" />
                    NASA Data Browser
                  </h3>
                </div>
                <div className="card-body">
                  <NASADataBrowser />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Прогресс анализа */}
          <AnimatePresence>
            {isAnalyzing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="card"
              >
                <div className="card-header">
                  <h3 className="card-title flex items-center gap-sm">
                    <Zap className="w-5 h-5 animate-pulse text-yellow-400" />
                    Анализ данных
                  </h3>
                </div>
                <div className="card-body">
                  <div className="space-y-sm">
                    <div className="flex justify-between text-sm">
                      <span>Прогресс:</span>
                      <span>{Math.round(analysisProgress)}%</span>
                    </div>
                    <div className="progress">
                      <motion.div
                        className="progress-bar bg-gradient-to-r from-yellow-400 to-orange-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${analysisProgress}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                    <div className="text-xs text-tertiary">
                      Применяется метод {analysisMethod.toUpperCase()}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Статистика данных */}
          {lightcurveData && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card"
            >
              <div className="card-header">
                <h3 className="card-title flex items-center gap-sm">
                  <TrendingUp className="w-5 h-5 text-green-400" />
                  Статистика данных
                </h3>
              </div>
              <div className="card-body">
                <div className="grid grid-cols-2 gap-sm text-sm">
                  <div>
                    <span className="text-gray-400">TIC ID:</span>
                    <span className="text-white ml-2">{lightcurveData.tic_id}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Точек данных:</span>
                    <span className="text-white ml-2">{lightcurveData.times.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Период наблюдений:</span>
                    <span className="text-white ml-2">
                      {((lightcurveData.times[lightcurveData.times.length - 1] - lightcurveData.times[0])).toFixed(1)} дней
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Кандидатов:</span>
                    <span className="text-white ml-2">{candidates.length}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>

      {/* Большой график внизу */}
      {lightcurveData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="card-title flex items-center gap-sm">
              <BarChart3 className="w-5 h-5 text-blue-400" />
              Кривая блеска TIC {lightcurveData.tic_id}
            </h3>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={exportData}
              className="btn btn-secondary btn-sm"
              title="Экспорт данных"
            >
              <Download className="w-4 h-4" />
            </motion.button>
          </div>
          <div className="card-body">
            <div className="bg-glass rounded-lg p-sm border border-primary/30">
              <Suspense fallback={
                <div className="h-96 flex items-center justify-center">
                  <Loader className="w-8 h-8 animate-spin text-blue-400" />
                  <span className="ml-2 text-secondary">Загрузка графика...</span>
                </div>
              }>
                <Plot
                  data={[
                    {
                      x: lightcurveData.times,
                      y: lightcurveData.fluxes,
                      type: 'scatter',
                      mode: 'lines+markers',
                      marker: { 
                        size: 2, 
                        color: '#60A5FA',
                        line: { width: 0 }
                      },
                      line: { 
                        color: '#60A5FA', 
                        width: 1 
                      },
                      name: 'Поток',
                      hovertemplate: 'Время: %{x:.3f} дней<br>Поток: %{y:.6f}<extra></extra>'
                    }
                  ]}
                  layout={{
                    title: {
                      text: `Кривая блеска TIC ${lightcurveData.tic_id}`,
                      font: { color: '#FFFFFF', size: 18 }
                    },
                    xaxis: {
                      title: { text: 'Время (дни)' },
                      color: '#9CA3AF',
                      gridcolor: '#374151',
                      tickfont: { size: 12 }
                    },
                    yaxis: {
                      title: { text: 'Нормализованный поток' },
                      color: '#9CA3AF',
                      gridcolor: '#374151',
                      tickfont: { size: 12 }
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#FFFFFF' },
                    margin: { t: 60, r: 20, b: 60, l: 80 },
                    showlegend: false,
                    hovermode: 'closest'
                  }}
                  style={{ width: '100%', height: '600px' }}
                  config={{
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                    displaylogo: false,
                    toImageButtonOptions: {
                      format: 'png',
                      filename: `TIC_${lightcurveData.tic_id}_lightcurve`,
                      height: 600,
                      width: 1200,
                      scale: 2
                    }
                  }}
                  useResizeHandler={true}
                />
              </Suspense>
            </div>
          </div>
        </motion.div>
      )}

      {/* Результаты анализа */}
      {candidates.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="card-title flex items-center gap-sm">
              <CheckCircle className="w-5 h-5 text-green-400" />
              Найденные кандидаты ({candidates.length})
            </h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-md">
              {candidates.map((candidate, index) => (
                <motion.div
                  key={candidate.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-md bg-gray-800/50 rounded-lg border border-green-500/30"
                >
                  <div className="space-y-sm">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-green-400">Кандидат #{index + 1}</span>
                      <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                        {(candidate.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="text-sm space-y-1">
                      <div>
                        <span className="text-gray-400">Период:</span>
                        <span className="text-white ml-2">{candidate.period.toFixed(3)} дней</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Глубина:</span>
                        <span className="text-white ml-2">{(candidate.depth * 100).toFixed(3)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Длительность:</span>
                        <span className="text-white ml-2">{(candidate.duration * 24).toFixed(1)} ч</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Ошибки */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="card border-red-500/30"
          >
            <div className="card-body">
              <div className="flex items-center gap-sm text-red-400">
                <AlertTriangle className="w-5 h-5" />
                <span>{error}</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default EnhancedLightcurveAnalysis;
