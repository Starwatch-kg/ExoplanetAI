import React, { useState, useEffect } from 'react';
import { Brain, Upload, Play, BarChart3, Settings, AlertCircle, CheckCircle, Clock } from 'lucide-react';

interface ClassificationResult {
  target_name: string;
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  model_predictions: Record<string, any>;
  processing_time: number;
  data_quality_score: number;
}

interface ModelStatus {
  is_trained: boolean;
  training_in_progress: boolean;
  last_training_time: string | null;
  model_metrics: any;
  available_features: string[];
}

const MLClassificationPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'classify' | 'train' | 'status'>('classify');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Форма для классификации
  const [targetName, setTargetName] = useState('');
  const [lightcurveData, setLightcurveData] = useState({
    time: '',
    flux: '',
    flux_err: ''
  });
  const [transitParams, setTransitParams] = useState({
    period: '',
    epoch: '',
    duration: ''
  });

  // Форма для обучения
  const [trainingConfig, setTrainingConfig] = useState({
    use_synthetic_data: true,
    n_synthetic_samples: 1000,
    test_size: 0.2,
    cv_folds: 5
  });

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const response = await fetch('/api/v1/ml/model-status');
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Model status API Error:', response.status, errorText);
        return;
      }
      
      const responseText = await response.text();
      let status;
      try {
        status = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Model status JSON Parse Error:', parseError);
        console.error('Response text:', responseText);
        return;
      }
      
      setModelStatus(status);
    } catch (err) {
      console.error('Failed to fetch model status:', err);
    }
  };

  const handleClassify = async () => {
    if (!targetName || !lightcurveData.time || !lightcurveData.flux) {
      setError('Пожалуйста, заполните все обязательные поля');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const timeArray = lightcurveData.time.split(',').map(x => parseFloat(x.trim()));
      const fluxArray = lightcurveData.flux.split(',').map(x => parseFloat(x.trim()));
      const fluxErrArray = lightcurveData.flux_err ? 
        lightcurveData.flux_err.split(',').map(x => parseFloat(x.trim())) :
        new Array(fluxArray.length).fill(0.001);

      const request = {
        target_name: targetName,
        lightcurve: {
          time: timeArray,
          flux: fluxArray,
          flux_err: fluxErrArray
        },
        transit_params: {
          period: parseFloat(transitParams.period),
          epoch: parseFloat(transitParams.epoch),
          duration: parseFloat(transitParams.duration)
        }
      };

      const response = await fetch('/api/v1/ml/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Classification API Error:', response.status, errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseText = await response.text();
      let result;
      try {
        result = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Classification JSON Parse Error:', parseError);
        console.error('Response text:', responseText);
        throw new Error('Invalid JSON response from classification API');
      }
      
      setClassificationResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка классификации');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrainModel = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/ml/train-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(trainingConfig)
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Training API Error:', response.status, errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseText = await response.text();
      let result;
      try {
        result = JSON.parse(responseText);
        console.log('Training started:', result);
      } catch (parseError) {
        console.error('Training JSON Parse Error:', parseError);
        console.error('Response text:', responseText);
        throw new Error('Invalid JSON response from training API');
      }
      
      alert('Обучение модели запущено в фоне. Проверьте статус через несколько минут.');
      fetchModelStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка запуска обучения');
    } finally {
      setIsLoading(false);
    }
  };

  const getClassColor = (className: string) => {
    switch (className) {
      case 'CANDIDATE': return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'PC': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'FP': return 'text-red-400 bg-red-400/10 border-red-400/20';
      default: return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const getClassDescription = (className: string) => {
    switch (className) {
      case 'CANDIDATE': return 'Подтвержденная экзопланета';
      case 'PC': return 'Планетарный кандидат';
      case 'FP': return 'Ложный позитив';
      default: return 'Неизвестный класс';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Brain className="w-12 h-12 text-purple-400 mr-4" />
            <h1 className="text-4xl font-bold text-white">
              ML Классификация Экзопланет
            </h1>
          </div>
          <p className="text-gray-300 text-lg">
            Искусственный интеллект для анализа и классификации экзопланет
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex justify-center mb-8">
          <div className="bg-white/5 backdrop-blur-sm rounded-lg p-1 border border-white/10">
            {[
              { id: 'classify', label: 'Классификация', icon: Brain },
              { id: 'train', label: 'Обучение', icon: Settings },
              { id: 'status', label: 'Статус', icon: BarChart3 }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`px-6 py-3 rounded-md flex items-center space-x-2 transition-all ${
                  activeTab === id
                    ? 'bg-purple-500 text-white shadow-lg'
                    : 'text-gray-300 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center">
            <AlertCircle className="w-5 h-5 text-red-400 mr-3" />
            <span className="text-red-300">{error}</span>
          </div>
        )}

        {/* Classification Tab */}
        {activeTab === 'classify' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Input Form */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <Upload className="w-6 h-6 mr-3 text-purple-400" />
                Данные для классификации
              </h2>

              <div className="space-y-6">
                {/* Target Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Название объекта
                  </label>
                  <input
                    type="text"
                    value={targetName}
                    onChange={(e) => setTargetName(e.target.value)}
                    placeholder="TOI-715 b"
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                {/* Light Curve Data */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Время (BJD, через запятую)
                  </label>
                  <textarea
                    value={lightcurveData.time}
                    onChange={(e) => setLightcurveData({...lightcurveData, time: e.target.value})}
                    placeholder="2459000.5, 2459000.52, 2459000.54, ..."
                    rows={3}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Поток (нормализованный, через запятую)
                  </label>
                  <textarea
                    value={lightcurveData.flux}
                    onChange={(e) => setLightcurveData({...lightcurveData, flux: e.target.value})}
                    placeholder="1.0, 0.999, 0.998, 0.985, 0.998, 0.999, 1.0, ..."
                    rows={3}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Ошибки потока (опционально, через запятую)
                  </label>
                  <textarea
                    value={lightcurveData.flux_err}
                    onChange={(e) => setLightcurveData({...lightcurveData, flux_err: e.target.value})}
                    placeholder="0.001, 0.001, 0.001, ..."
                    rows={2}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                {/* Transit Parameters */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Период (дни)
                    </label>
                    <input
                      type="number"
                      step="0.000001"
                      value={transitParams.period}
                      onChange={(e) => setTransitParams({...transitParams, period: e.target.value})}
                      placeholder="19.3"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Эпоха (BJD)
                    </label>
                    <input
                      type="number"
                      step="0.000001"
                      value={transitParams.epoch}
                      onChange={(e) => setTransitParams({...transitParams, epoch: e.target.value})}
                      placeholder="2459000.5"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Длительность (часы)
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={transitParams.duration}
                      onChange={(e) => setTransitParams({...transitParams, duration: e.target.value})}
                      placeholder="4.2"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>

                {/* Classify Button */}
                <button
                  onClick={handleClassify}
                  disabled={isLoading || !modelStatus?.is_trained}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-4 px-6 rounded-lg font-semibold hover:from-purple-600 hover:to-pink-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <Clock className="w-5 h-5 mr-2 animate-spin" />
                      Классификация...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 mr-2" />
                      Классифицировать
                    </>
                  )}
                </button>

                {!modelStatus?.is_trained && (
                  <p className="text-yellow-400 text-sm text-center">
                    Модель не обучена. Перейдите на вкладку "Обучение" для обучения модели.
                  </p>
                )}
              </div>
            </div>

            {/* Results */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <BarChart3 className="w-6 h-6 mr-3 text-purple-400" />
                Результаты классификации
              </h2>

              {classificationResult ? (
                <div className="space-y-6">
                  {/* Main Result */}
                  <div className="text-center">
                    <h3 className="text-xl font-semibold text-white mb-2">
                      {classificationResult.target_name}
                    </h3>
                    <div className={`inline-flex items-center px-6 py-3 rounded-full border text-lg font-semibold ${getClassColor(classificationResult.predicted_class)}`}>
                      <CheckCircle className="w-5 h-5 mr-2" />
                      {classificationResult.predicted_class}
                    </div>
                    <p className="text-gray-300 mt-2">
                      {getClassDescription(classificationResult.predicted_class)}
                    </p>
                    <p className="text-2xl font-bold text-white mt-2">
                      Уверенность: {(classificationResult.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Class Probabilities */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-3">
                      Вероятности классов
                    </h4>
                    <div className="space-y-3">
                      {Object.entries(classificationResult.class_probabilities).map(([className, probability]) => (
                        <div key={className} className="flex items-center justify-between">
                          <span className="text-gray-300">{className}</span>
                          <div className="flex items-center space-x-3">
                            <div className="w-32 bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all"
                                style={{ width: `${probability * 100}%` }}
                              />
                            </div>
                            <span className="text-white font-semibold w-12 text-right">
                              {(probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Processing Info */}
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t border-white/10">
                    <div>
                      <p className="text-gray-400 text-sm">Время обработки</p>
                      <p className="text-white font-semibold">
                        {classificationResult.processing_time.toFixed(2)}s
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Качество данных</p>
                      <p className="text-white font-semibold">
                        {(classificationResult.data_quality_score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-400 py-12">
                  <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Результаты классификации появятся здесь</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Training Tab */}
        {activeTab === 'train' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <Settings className="w-6 h-6 mr-3 text-purple-400" />
                Обучение модели
              </h2>

              <div className="space-y-6">
                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    id="synthetic"
                    checked={trainingConfig.use_synthetic_data}
                    onChange={(e) => setTrainingConfig({...trainingConfig, use_synthetic_data: e.target.checked})}
                    className="w-4 h-4 text-purple-600 bg-gray-100 border-gray-300 rounded focus:ring-purple-500"
                  />
                  <label htmlFor="synthetic" className="text-white">
                    Использовать синтетические данные (для демонстрации)
                  </label>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Количество образцов
                  </label>
                  <input
                    type="number"
                    value={trainingConfig.n_synthetic_samples}
                    onChange={(e) => setTrainingConfig({...trainingConfig, n_synthetic_samples: parseInt(e.target.value)})}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Размер тестовой выборки
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.1"
                      max="0.5"
                      value={trainingConfig.test_size}
                      onChange={(e) => setTrainingConfig({...trainingConfig, test_size: parseFloat(e.target.value)})}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Фолды кросс-валидации
                    </label>
                    <input
                      type="number"
                      min="3"
                      max="10"
                      value={trainingConfig.cv_folds}
                      onChange={(e) => setTrainingConfig({...trainingConfig, cv_folds: parseInt(e.target.value)})}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>

                <button
                  onClick={handleTrainModel}
                  disabled={isLoading || modelStatus?.training_in_progress}
                  className="w-full bg-gradient-to-r from-green-500 to-blue-500 text-white py-4 px-6 rounded-lg font-semibold hover:from-green-600 hover:to-blue-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isLoading || modelStatus?.training_in_progress ? (
                    <>
                      <Clock className="w-5 h-5 mr-2 animate-spin" />
                      Обучение в процессе...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 mr-2" />
                      Начать обучение
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Status Tab */}
        {activeTab === 'status' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <BarChart3 className="w-6 h-6 mr-3 text-purple-400" />
                Статус модели
              </h2>

              {modelStatus ? (
                <div className="space-y-6">
                  {/* Status Overview */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center">
                      <div className={`inline-flex items-center px-4 py-2 rounded-full ${
                        modelStatus.is_trained ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {modelStatus.is_trained ? <CheckCircle className="w-4 h-4 mr-2" /> : <AlertCircle className="w-4 h-4 mr-2" />}
                        {modelStatus.is_trained ? 'Обучена' : 'Не обучена'}
                      </div>
                      <p className="text-gray-400 text-sm mt-2">Статус модели</p>
                    </div>
                    <div className="text-center">
                      <div className={`inline-flex items-center px-4 py-2 rounded-full ${
                        modelStatus.training_in_progress ? 'bg-yellow-500/20 text-yellow-400' : 'bg-gray-500/20 text-gray-400'
                      }`}>
                        {modelStatus.training_in_progress ? <Clock className="w-4 h-4 mr-2 animate-spin" /> : <CheckCircle className="w-4 h-4 mr-2" />}
                        {modelStatus.training_in_progress ? 'В процессе' : 'Готова'}
                      </div>
                      <p className="text-gray-400 text-sm mt-2">Обучение</p>
                    </div>
                    <div className="text-center">
                      <div className="text-white font-semibold">
                        {modelStatus.available_features.length}
                      </div>
                      <p className="text-gray-400 text-sm mt-2">Признаков</p>
                    </div>
                  </div>

                  {/* Last Training */}
                  {modelStatus.last_training_time && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">
                        Последнее обучение
                      </h3>
                      <p className="text-gray-300">
                        {new Date(modelStatus.last_training_time).toLocaleString('ru-RU')}
                      </p>
                    </div>
                  )}

                  {/* Model Metrics */}
                  {modelStatus.model_metrics && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">
                        Метрики модели
                      </h3>
                      <div className="bg-black/20 rounded-lg p-4">
                        <pre className="text-gray-300 text-sm overflow-auto">
                          {JSON.stringify(modelStatus.model_metrics, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-12">
                  <Clock className="w-16 h-16 mx-auto mb-4 opacity-50 animate-spin" />
                  <p>Загрузка статуса модели...</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MLClassificationPage;
