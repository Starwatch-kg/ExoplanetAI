import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Play,
  Pause,
  CheckCircle,
  AlertTriangle,
  X,
  Clock,
  Brain,
  BarChart3,
  Camera,
  ChevronDown
} from 'lucide-react';
import { useBackgroundTasks, type BackgroundTask } from '../contexts/BackgroundTasksContext';

const BackgroundTasksIndicator: React.FC = () => {
  const { tasks, pauseTask, resumeTask, removeTask, getRunningTasks } = useBackgroundTasks();
  const [isExpanded, setIsExpanded] = useState(false);
  
  const runningTasks = getRunningTasks();
  const hasActiveTasks = tasks.length > 0;

  // Получение иконки по типу задачи
  const getTaskIcon = (type: BackgroundTask['type']) => {
    switch (type) {
      case 'cnn_training':
        return <Brain className="w-4 h-4" />;
      case 'lightcurve_analysis':
        return <BarChart3 className="w-4 h-4" />;
      case 'image_classification':
        return <Camera className="w-4 h-4" />;
      case 'data_loading':
        return <Activity className="w-4 h-4" />;
      case 'model_inference':
        return <Brain className="w-4 h-4" />;
      case 'file_processing':
        return <Activity className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  // Получение иконки статуса
  const getStatusIcon = (status: BackgroundTask['status']) => {
    switch (status) {
      case 'running':
        return <Play className="w-3 h-3 text-green-400 animate-pulse" />;
      case 'paused':
        return <Pause className="w-3 h-3 text-yellow-400" />;
      case 'completed':
        return <CheckCircle className="w-3 h-3 text-green-400" />;
      case 'error':
        return <AlertTriangle className="w-3 h-3 text-red-400" />;
      default:
        return <Clock className="w-3 h-3 text-gray-400" />;
    }
  };

  // Получение цвета прогресс-бара
  const getProgressColor = (status: BackgroundTask['status']) => {
    switch (status) {
      case 'running':
        return 'bg-blue-500';
      case 'paused':
        return 'bg-yellow-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Форматирование времени
  const formatDuration = (startTime: Date) => {
    const now = new Date();
    const diff = now.getTime() - startTime.getTime();
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    
    if (minutes > 0) {
      return `${minutes}м ${seconds}с`;
    }
    return `${seconds}с`;
  };

  if (!hasActiveTasks) {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 z-50">
      <motion.div
        initial={{ opacity: 0, x: 100 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-gray-800/95 backdrop-blur-sm border border-blue-500/30 rounded-lg shadow-lg"
      >
        {/* Заголовок с индикатором */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full p-3 flex items-center justify-between text-left"
        >
          <div className="flex items-center gap-2">
            <div className="relative">
              <Activity className="w-5 h-5 text-blue-400" />
              {runningTasks.length > 0 && (
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full"
                />
              )}
            </div>
            <span className="text-white font-medium">
              Фоновые задачи ({tasks.length})
            </span>
          </div>
          <motion.div
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronDown className="w-4 h-4 text-gray-400" />
          </motion.div>
        </motion.button>

        {/* Список задач */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="border-t border-gray-700"
            >
              <div className="max-h-96 overflow-y-auto">
                {tasks.map((task, index) => (
                  <motion.div
                    key={task.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-3 border-b border-gray-700/50 last:border-b-0"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-start gap-2 flex-1 min-w-0">
                        <div className="text-blue-400 mt-0.5">
                          {getTaskIcon(task.type)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1 mb-1">
                            {getStatusIcon(task.status)}
                            <span className="text-sm font-medium text-white truncate">
                              {task.name}
                            </span>
                          </div>
                          
                          {/* Прогресс бар */}
                          <div className="mb-2">
                            <div className="flex justify-between text-xs text-gray-400 mb-1">
                              <span>{task.progress.toFixed(1)}%</span>
                              <span>{formatDuration(task.startTime)}</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-1.5">
                              <motion.div
                                className={`h-1.5 rounded-full ${getProgressColor(task.status)}`}
                                initial={{ width: 0 }}
                                animate={{ width: `${task.progress}%` }}
                                transition={{ duration: 0.5 }}
                              />
                            </div>
                          </div>

                          {/* Дополнительная информация */}
                          {task.data && (
                            <div className="text-xs text-gray-400">
                              {task.type === 'cnn_training' && task.data.epoch && (
                                <span>Эпоха {task.data.epoch}/{task.data.total_epochs}</span>
                              )}
                              {task.data.loss && (
                                <span className="ml-2">Loss: {task.data.loss.toFixed(4)}</span>
                              )}
                              {task.data.accuracy && (
                                <span className="ml-2">Acc: {(task.data.accuracy * 100).toFixed(1)}%</span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Кнопки управления */}
                      <div className="flex gap-1">
                        {task.status === 'running' && (
                          <motion.button
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => pauseTask(task.id)}
                            className="p-1 text-yellow-400 hover:bg-yellow-400/20 rounded"
                            title="Пауза"
                          >
                            <Pause className="w-3 h-3" />
                          </motion.button>
                        )}
                        
                        {task.status === 'paused' && (
                          <motion.button
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => resumeTask(task.id)}
                            className="p-1 text-green-400 hover:bg-green-400/20 rounded"
                            title="Продолжить"
                          >
                            <Play className="w-3 h-3" />
                          </motion.button>
                        )}
                        
                        {(task.status === 'completed' || task.status === 'error') && (
                          <motion.button
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => removeTask(task.id)}
                            className="p-1 text-red-400 hover:bg-red-400/20 rounded"
                            title="Удалить"
                          >
                            <X className="w-3 h-3" />
                          </motion.button>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Кнопки управления всеми задачами */}
              {tasks.length > 1 && (
                <div className="p-2 border-t border-gray-700/50 bg-gray-800/50">
                  <div className="flex gap-2 text-xs">
                    <button
                      onClick={() => {
                        tasks.forEach(task => {
                          if (task.status === 'running') pauseTask(task.id);
                        });
                      }}
                      className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded hover:bg-yellow-500/30 transition-colors"
                    >
                      Пауза всех
                    </button>
                    <button
                      onClick={() => {
                        tasks.forEach(task => {
                          if (task.status === 'completed' || task.status === 'error') {
                            removeTask(task.id);
                          }
                        });
                      }}
                      className="px-2 py-1 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors"
                    >
                      Очистить завершенные
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default BackgroundTasksIndicator;
