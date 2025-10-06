import React, { useState, useEffect, useCallback } from 'react';
import { 
  Brain, 
  Play, 
  Settings, 
  Activity, 
  Database, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle, 
  Clock,
  Zap,
  BarChart3
} from 'lucide-react';

interface TrainingStatus {
  is_training: boolean;
  model_version: number;
  last_training_time: string | null;
  training_history: Array<{
    timestamp: string;
    version: number;
    quality?: number;
    status: string;
    training_samples?: number;
  }>;
  real_data_cache_size: number;
  next_check_in_hours: number;
  quality_threshold: number;
  min_real_samples: number;
  trainer_type: string;
}

interface TrainingMetrics {
  time_since_last_training: number;
  new_real_data_count: number;
  model_performance_score: number;
  data_quality_score: number;
  training_recommendation: string;
}

interface TrainingConfig {
  training_interval_hours: number;
  min_real_samples: number;
  quality_threshold: number;
  synthetic_ratio: number;
  use_enhanced_trainer: boolean;
}

const AutoTrainingPanel: React.FC = () => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [config, setConfig] = useState<TrainingConfig>({
    training_interval_hours: 12,
    min_real_samples: 20,
    quality_threshold: 0.80,
    synthetic_ratio: 0.7,
    use_enhanced_trainer: true
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [showConfig, setShowConfig] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/auto-training/status');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error('Error fetching status:', err);
      setError('Ошибка загрузки статуса');
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/auto-training/metrics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchMetrics();
    
    // Обновляем каждые 30 секунд
    const interval = setInterval(() => {
      fetchStatus();
      fetchMetrics();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [fetchStatus, fetchMetrics]);

  const handleStartTraining = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/v1/auto-training/start', {
        method: 'POST'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ошибка запуска');
      }
      
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка запуска обучения');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTriggerImmediate = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/v1/auto-training/trigger', {
        method: 'POST'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ошибка запуска');
      }
      
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка немедленного обучения');
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfigUpdate = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/v1/auto-training/configure', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ошибка обновления конфигурации');
      }
      
      setShowConfig(false);
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка обновления конфигурации');
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (isTraining: boolean) => {
    return isTraining ? 'text-green-400' : 'text-blue-400';
  };

  const getStatusIcon = (isTraining: boolean) => {
    return isTraining ? (
      <Activity className="w-5 h-5 text-green-400 animate-pulse" />
    ) : (
      <Clock className="w-5 h-5 text-blue-400" />
    );
  };

  const formatTimeAgo = (timestamp: string | null) => {
    if (!timestamp) return 'Никогда';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);
    
    if (diffHours < 1) return 'Менее часа назад';
    if (diffHours < 24) return `${Math.floor(diffHours)} ч. назад`;
    return `${Math.floor(diffHours / 24)} дн. назад`;
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold text-white flex items-center justify-center gap-2">
          <Brain className="w-8 h-8 text-purple-400" />
          Автоматическое обучение ИИ
        </h1>
        <p className="text-gray-300">
          Система автоматического обучения на реальных данных NASA с fallback на синтетику
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border border-red-600 rounded-lg p-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Main Status */}
      {status && (
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">Статус системы</h2>
            <div className="flex items-center gap-2">
              {getStatusIcon(status.is_training)}
              <span className={`font-semibold ${getStatusColor(status.is_training)}`}>
                {status.is_training ? 'Обучение активно' : 'Ожидание'}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-5 h-5 text-blue-400" />
                <h3 className="font-semibold text-white">Версия модели</h3>
              </div>
              <div className="text-2xl font-bold text-blue-400">v{status.model_version}</div>
            </div>

            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Database className="w-5 h-5 text-green-400" />
                <h3 className="font-semibold text-white">Реальные данные</h3>
              </div>
              <div className="text-2xl font-bold text-green-400">{status.real_data_cache_size}</div>
            </div>

            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-yellow-400" />
                <h3 className="font-semibold text-white">Порог качества</h3>
              </div>
              <div className="text-2xl font-bold text-yellow-400">
                {(status.quality_threshold * 100).toFixed(0)}%
              </div>
            </div>

            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-white">Последнее обучение</h3>
              </div>
              <div className="text-sm font-medium text-purple-400">
                {formatTimeAgo(status.last_training_time)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Metrics */}
      {metrics && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-4">Метрики обучения</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Производительность модели:</span>
                <span className={`font-semibold ${
                  metrics.model_performance_score >= 0.8 ? 'text-green-400' : 
                  metrics.model_performance_score >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {(metrics.model_performance_score * 100).toFixed(1)}%
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Качество данных:</span>
                <span className="font-semibold text-blue-400">
                  {(metrics.data_quality_score * 100).toFixed(1)}%
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Новые данные:</span>
                <span className="font-semibold text-green-400">
                  {metrics.new_real_data_count} образцов
                </span>
              </div>
            </div>
            
            <div className="bg-blue-900 border border-blue-600 rounded-lg p-4">
              <h3 className="font-semibold text-blue-200 mb-2">Рекомендация ИИ</h3>
              <p className="text-blue-200 text-sm">{metrics.training_recommendation}</p>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">Управление</h2>
        
        <div className="flex flex-wrap gap-4">
          <button
            onClick={handleStartTraining}
            disabled={isLoading || (status?.is_training ?? false)}
            className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white transition-colors flex items-center gap-2"
          >
            <Play className="w-5 h-5" />
            Запустить автообучение
          </button>
          
          <button
            onClick={handleTriggerImmediate}
            disabled={isLoading || (status?.is_training ?? false)}
            className="px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white transition-colors flex items-center gap-2"
          >
            <Zap className="w-5 h-5" />
            Немедленное обучение
          </button>
          
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-white transition-colors flex items-center gap-2"
          >
            <Settings className="w-5 h-5" />
            Настройки
          </button>
        </div>
      </div>

      {/* Configuration Panel */}
      {showConfig && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-4">Конфигурация автообучения</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Интервал проверки (часы): {config.training_interval_hours}
              </label>
              <input
                type="range"
                min="1"
                max="168"
                value={config.training_interval_hours}
                onChange={(e) => setConfig(prev => ({ ...prev, training_interval_hours: parseInt(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Минимум реальных образцов: {config.min_real_samples}
              </label>
              <input
                type="range"
                min="5"
                max="100"
                value={config.min_real_samples}
                onChange={(e) => setConfig(prev => ({ ...prev, min_real_samples: parseInt(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Порог качества: {(config.quality_threshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.01"
                value={config.quality_threshold}
                onChange={(e) => setConfig(prev => ({ ...prev, quality_threshold: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Доля синтетики: {(config.synthetic_ratio * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={config.synthetic_ratio}
                onChange={(e) => setConfig(prev => ({ ...prev, synthetic_ratio: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
          
          <div className="mt-6 flex gap-4">
            <button
              onClick={handleConfigUpdate}
              disabled={isLoading}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg text-white transition-colors"
            >
              Применить настройки
            </button>
            
            <button
              onClick={() => setShowConfig(false)}
              className="px-6 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-white transition-colors"
            >
              Отмена
            </button>
          </div>
        </div>
      )}

      {/* Training History */}
      {status?.training_history && status.training_history.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-4">История обучения</h2>
          
          <div className="space-y-3">
            {status.training_history.slice(-5).reverse().map((entry, index) => (
              <div key={index} className="bg-gray-700 rounded-lg p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {entry.status === 'success' ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-red-400" />
                  )}
                  
                  <div>
                    <div className="text-white font-medium">
                      Версия {entry.version} - {entry.status === 'success' ? 'Успешно' : 'Ошибка'}
                    </div>
                    <div className="text-gray-400 text-sm">
                      {new Date(entry.timestamp).toLocaleString('ru-RU')}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  {entry.quality && (
                    <div className="text-white font-medium">
                      {(entry.quality * 100).toFixed(1)}%
                    </div>
                  )}
                  {entry.training_samples && (
                    <div className="text-gray-400 text-sm">
                      {entry.training_samples} образцов
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AutoTrainingPanel;
