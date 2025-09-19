import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Play,
  Pause,
  Square,
  Settings,
  TrendingUp,
  Zap,
  CheckCircle,
  AlertTriangle,
  Clock,
  Cpu,
  MemoryStick,
  Activity
} from 'lucide-react';
import { useBackgroundTask } from '../hooks/useBackgroundTask';

interface TrainingSession {
  id: string;
  model_type: string;
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error';
  progress: number;
  current_epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  estimated_time_remaining: string;
  start_time: string;
}

interface ModelConfig {
  id: string;
  name: string;
  description: string;
  complexity: 'low' | 'medium' | 'high';
  color: string;
  icon: React.ReactNode;
  parameters: {
    layers: number;
    parameters: string;
    memory: string;
  };
}

const EnhancedCNNTraining: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('exoplanet_cnn');
  const [trainingSession, setTrainingSession] = useState<TrainingSession | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [backgroundTasks, setBackgroundTasks] = useState<TrainingSession[]>([]);
  
  // Используем хук для фоновых задач
  const { startCNNTraining } = useBackgroundTask();
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [selectedModelHighlight, setSelectedModelHighlight] = useState<string | null>(null);

  // Конфигурации моделей
  const modelConfigs: ModelConfig[] = [
    {
      id: 'exoplanet_cnn',
      name: 'ExoplanetCNN',
      description: 'Базовая CNN архитектура для детекции транзитов',
      complexity: 'low',
      color: 'blue',
      icon: <Brain className="w-6 h-6" />,
      parameters: {
        layers: 4,
        parameters: '2.1M',
        memory: '150MB'
      }
    },
    {
      id: 'exoplanet_resnet',
      name: 'ExoplanetResNet',
      description: 'ResNet с остаточными связями и SE-блоками',
      complexity: 'medium',
      color: 'purple',
      icon: <Zap className="w-6 h-6" />,
      parameters: {
        layers: 8,
        parameters: '5.3M',
        memory: '320MB'
      }
    },
    {
      id: 'exoplanet_densenet',
      name: 'ExoplanetDenseNet',
      description: 'DenseNet с плотными связями между слоями',
      complexity: 'medium',
      color: 'green',
      icon: <Activity className="w-6 h-6" />,
      parameters: {
        layers: 12,
        parameters: '4.8M',
        memory: '280MB'
      }
    },
    {
      id: 'exoplanet_attention',
      name: 'ExoplanetAttention',
      description: 'CNN с механизмом self-attention',
      complexity: 'high',
      color: 'orange',
      icon: <TrendingUp className="w-6 h-6" />,
      parameters: {
        layers: 16,
        parameters: '8.7M',
        memory: '450MB'
      }
    }
  ];

  // Параметры обучения
  const [trainingParams, setTrainingParams] = useState({
    epochs: 50,
    batch_size: 32,
    learning_rate: 0.001,
    optimizer: 'adamw',
    scheduler: 'cosine',
    early_stopping: true,
    patience: 10
  });

  // Обработчик выбора модели с анимацией подсветки
  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
    setSelectedModelHighlight(modelId);
    
    // Убираем подсветку через 2 секунды
    setTimeout(() => {
      setSelectedModelHighlight(null);
    }, 2000);
  };

  // Запуск обучения с фоновой задачей
  const startTraining = async () => {
    try {
      // Сначала запускаем обучение на backend
      const response = await fetch('/api/cnn/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: selectedModel,
          ...trainingParams
        })
      });

      if (response.ok) {
        const data = await response.json();
        
        // Создаем фоновую задачу для мониторинга
        startCNNTraining({
          training_id: data.training_id,
          model_type: selectedModel
        });
        
        const newSession: TrainingSession = {
          id: data.training_id,
          model_type: selectedModel,
          status: 'training',
          progress: 0,
          current_epoch: 0,
          total_epochs: trainingParams.epochs,
          loss: 0,
          accuracy: 0,
          estimated_time_remaining: 'Вычисляется...',
          start_time: new Date().toISOString()
        };
        
        setTrainingSession(newSession);
        setIsTraining(true);
        
        // Добавляем в фоновые задачи
        setBackgroundTasks(prev => [...prev, newSession]);
      }
    } catch (error) {
      console.error('Ошибка запуска обучения:', error);
    }
  };

  // Пауза обучения
  const pauseTraining = async () => {
    if (!trainingSession) return;
    
    try {
      await fetch(`/api/cnn/training/${trainingSession.id}/pause`, {
        method: 'POST'
      });
      
      setTrainingSession(prev => prev ? { ...prev, status: 'paused' } : null);
    } catch (error) {
      console.error('Ошибка паузы обучения:', error);
    }
  };

  // Остановка обучения
  const stopTraining = async () => {
    if (!trainingSession) return;
    
    try {
      await fetch(`/api/cnn/training/${trainingSession.id}/stop`, {
        method: 'POST'
      });
      
      setTrainingSession(null);
      setIsTraining(false);
      
      // Удаляем из фоновых задач
      setBackgroundTasks(prev => prev.filter(task => task.id !== trainingSession.id));
    } catch (error) {
      console.error('Ошибка остановки обучения:', error);
    }
  };

  // Обновление статуса обучения
  const updateTrainingStatus = useCallback(async () => {
    if (!trainingSession || trainingSession.status !== 'training') return;
    
    try {
      const response = await fetch(`/api/cnn/training/${trainingSession.id}/status`);
      if (response.ok) {
        const data = await response.json();
        setTrainingSession(prev => prev ? { ...prev, ...data } : null);
        
        // Обновляем фоновые задачи
        setBackgroundTasks(prev => 
          prev.map(task => 
            task.id === trainingSession.id ? { ...task, ...data } : task
          )
        );
        
        if (data.status === 'completed') {
          setIsTraining(false);
        }
      }
    } catch (error) {
      console.error('Ошибка получения статуса:', error);
    }
  }, [trainingSession]);

  // Периодическое обновление статуса
  useEffect(() => {
    if (isTraining) {
      const interval = setInterval(updateTrainingStatus, 2000);
      return () => clearInterval(interval);
    }
  }, [isTraining, updateTrainingStatus]);

  // Получение цвета сложности
  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  // Получение статуса иконки
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'training': return <Play className="w-4 h-4 text-green-400 animate-pulse" />;
      case 'paused': return <Pause className="w-4 h-4 text-yellow-400" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
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
            <Brain className="w-6 h-6 text-primary" />
            Обучение CNN моделей
          </h2>
          <div className="text-sm text-secondary">
            Обучение нейронных сетей для детекции экзопланет
          </div>
        </div>
      </motion.div>

      {/* Фоновые задачи */}
      <AnimatePresence>
        {backgroundTasks.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="card border-blue-500/30"
          >
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <Activity className="w-5 h-5 text-blue-400" />
                Фоновые процессы ({backgroundTasks.length})
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-sm">
                {backgroundTasks.map((task) => (
                  <motion.div
                    key={task.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center justify-between p-sm bg-gray-800/50 rounded-lg"
                  >
                    <div className="flex items-center gap-sm">
                      {getStatusIcon(task.status)}
                      <div>
                        <div className="text-sm font-medium text-white">
                          {modelConfigs.find(m => m.id === task.model_type)?.name}
                        </div>
                        <div className="text-xs text-gray-400">
                          Эпоха {task.current_epoch}/{task.total_epochs} • {task.progress.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-white">Loss: {task.loss.toFixed(4)}</div>
                      <div className="text-xs text-gray-400">Acc: {(task.accuracy * 100).toFixed(1)}%</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-lg">
        {/* Левая панель - Выбор модели */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="space-y-md"
        >
          {/* Выбор архитектуры */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <Brain className="w-5 h-5 text-purple-400" />
                Архитектура модели
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-sm">
                {modelConfigs.map((model) => (
                  <motion.button
                    key={model.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleModelSelect(model.id)}
                    className={`w-full p-md rounded-lg border text-left transition-all duration-300 ${
                      selectedModel === model.id
                        ? `border-${model.color}-500 bg-${model.color}-500/10`
                        : 'border-gray-600 hover:border-gray-500'
                    } ${
                      selectedModelHighlight === model.id
                        ? `ring-2 ring-${model.color}-400 shadow-lg shadow-${model.color}-400/30 animate-pulse`
                        : ''
                    }`}
                  >
                    <div className="flex items-start gap-sm">
                      <div className={`text-${model.color}-400 mt-1`}>
                        {model.icon}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="font-medium text-white">{model.name}</h4>
                          <span className={`text-xs px-2 py-1 rounded ${getComplexityColor(model.complexity)}`}>
                            {model.complexity}
                          </span>
                        </div>
                        <p className="text-sm text-gray-400 mb-2">{model.description}</p>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">Слои:</span>
                            <span className="text-white ml-1">{model.parameters.layers}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Параметры:</span>
                            <span className="text-white ml-1">{model.parameters.parameters}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Память:</span>
                            <span className="text-white ml-1">{model.parameters.memory}</span>
                          </div>
                        </div>
                      </div>
                      {selectedModel === model.id && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className={`w-3 h-3 rounded-full bg-${model.color}-400 mt-1`}
                        />
                      )}
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>
          </div>

          {/* Продвинутые настройки */}
          <div className="card">
            <div className="card-header">
              <button
                onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                className="card-title flex items-center gap-sm hover:text-blue-400 transition-colors"
              >
                <Settings className="w-5 h-5" />
                Параметры обучения
                <motion.div
                  animate={{ rotate: showAdvancedSettings ? 180 : 0 }}
                  className="ml-auto"
                >
                  ▼
                </motion.div>
              </button>
            </div>
            <AnimatePresence>
              {showAdvancedSettings && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="card-body"
                >
                  <div className="grid grid-cols-2 gap-md">
                    <div>
                      <label className="form-label">Эпохи</label>
                      <input
                        type="number"
                        value={trainingParams.epochs}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          epochs: parseInt(e.target.value)
                        }))}
                        className="form-input"
                        min="1"
                        max="200"
                      />
                    </div>
                    <div>
                      <label className="form-label">Batch Size</label>
                      <select
                        value={trainingParams.batch_size}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          batch_size: parseInt(e.target.value)
                        }))}
                        className="form-select"
                      >
                        <option value={16}>16</option>
                        <option value={32}>32</option>
                        <option value={64}>64</option>
                        <option value={128}>128</option>
                      </select>
                    </div>
                    <div>
                      <label className="form-label">Learning Rate</label>
                      <select
                        value={trainingParams.learning_rate}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          learning_rate: parseFloat(e.target.value)
                        }))}
                        className="form-select"
                      >
                        <option value={0.01}>0.01</option>
                        <option value={0.001}>0.001</option>
                        <option value={0.0001}>0.0001</option>
                      </select>
                    </div>
                    <div>
                      <label className="form-label">Оптимизатор</label>
                      <select
                        value={trainingParams.optimizer}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          optimizer: e.target.value
                        }))}
                        className="form-select"
                      >
                        <option value="adamw">AdamW</option>
                        <option value="adam">Adam</option>
                        <option value="sgd">SGD</option>
                      </select>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Правая панель - Управление и мониторинг */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-md"
        >
          {/* Управление обучением */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <Play className="w-5 h-5 text-green-400" />
                Управление обучением
              </h3>
            </div>
            <div className="card-body">
              <div className="flex gap-sm">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={startTraining}
                  disabled={isTraining}
                  className="btn btn-success flex-1"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Запустить
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={pauseTraining}
                  disabled={!isTraining}
                  className="btn btn-warning"
                >
                  <Pause className="w-4 h-4" />
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={stopTraining}
                  disabled={!isTraining}
                  className="btn btn-danger"
                >
                  <Square className="w-4 h-4" />
                </motion.button>
              </div>
            </div>
          </div>

          {/* Прогресс обучения */}
          {trainingSession && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card"
            >
              <div className="card-header">
                <h3 className="card-title flex items-center gap-sm">
                  <TrendingUp className="w-5 h-5 text-blue-400" />
                  Прогресс обучения
                </h3>
              </div>
              <div className="card-body space-y-md">
                {/* Прогресс бар */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Эпоха {trainingSession.current_epoch}/{trainingSession.total_epochs}</span>
                    <span>{trainingSession.progress.toFixed(1)}%</span>
                  </div>
                  <div className="progress">
                    <motion.div
                      className="progress-bar bg-gradient-to-r from-blue-500 to-purple-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${trainingSession.progress}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>

                {/* Метрики */}
                <div className="grid grid-cols-2 gap-md">
                  <div className="bg-gray-800/50 p-sm rounded-lg">
                    <div className="text-xs text-gray-400">Loss</div>
                    <div className="text-lg font-bold text-red-400">
                      {trainingSession.loss.toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-gray-800/50 p-sm rounded-lg">
                    <div className="text-xs text-gray-400">Accuracy</div>
                    <div className="text-lg font-bold text-green-400">
                      {(trainingSession.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Время */}
                <div className="text-sm text-gray-400">
                  <div>Осталось: {trainingSession.estimated_time_remaining}</div>
                  <div>Статус: {trainingSession.status}</div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Системные ресурсы */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <Cpu className="w-5 h-5 text-orange-400" />
                Системные ресурсы
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-blue-400" />
                    <span className="text-sm">CPU</span>
                  </div>
                  <span className="text-sm">25%</span>
                </div>
                <div className="progress">
                  <div className="progress-bar bg-blue-500" style={{ width: '25%' }} />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MemoryStick className="w-4 h-4 text-purple-400" />
                    <span className="text-sm">Memory</span>
                  </div>
                  <span className="text-sm">45%</span>
                </div>
                <div className="progress">
                  <div className="progress-bar bg-purple-500" style={{ width: '45%' }} />
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default EnhancedCNNTraining;
