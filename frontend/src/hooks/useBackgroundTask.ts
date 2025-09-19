import { useCallback } from 'react';
import { useBackgroundTasks, type BackgroundTask } from '../contexts/BackgroundTasksContext';

interface UseBackgroundTaskOptions {
  type: BackgroundTask['type'];
  name: string;
  priority?: BackgroundTask['priority'];
  canPause?: boolean;
  apiEndpoint?: string;
  pollInterval?: number;
}

export const useBackgroundTask = () => {
  const {
    tasks,
    addTask,
    updateTask,
    removeTask,
    startPollingTask,
    stopPollingTask,
    createWorkerTask,
    getTasksByType
  } = useBackgroundTasks();

  // Создание и запуск API polling задачи
  const startApiTask = useCallback((options: UseBackgroundTaskOptions) => {
    const taskId = addTask({
      type: options.type,
      name: options.name,
      status: 'running',
      progress: 0,
      priority: options.priority || 'medium',
      canPause: options.canPause || false,
      apiEndpoint: options.apiEndpoint,
      pollInterval: options.pollInterval
    });

    if (options.apiEndpoint) {
      startPollingTask(taskId, options.apiEndpoint, options.pollInterval);
    }

    return taskId;
  }, [addTask, startPollingTask]);

  // Создание и запуск Worker задачи
  const startWorkerTask = useCallback((
    options: UseBackgroundTaskOptions,
    workerScript: string,
    data: any
  ) => {
    const taskId = addTask({
      type: options.type,
      name: options.name,
      status: 'running',
      progress: 0,
      priority: options.priority || 'medium',
      canPause: options.canPause || false
    });

    createWorkerTask(taskId, workerScript, data);
    return taskId;
  }, [addTask, createWorkerTask]);

  // CNN Training задача
  const startCNNTraining = useCallback((config: any) => {
    return startApiTask({
      type: 'cnn_training',
      name: `CNN Training - ${config.model_type}`,
      priority: 'high',
      canPause: true,
      apiEndpoint: `/api/cnn/training/${config.training_id}/status`,
      pollInterval: 2000
    });
  }, [startApiTask]);

  // Lightcurve Analysis задача
  const startLightcurveAnalysis = useCallback((targetId: string) => {
    return startApiTask({
      type: 'lightcurve_analysis',
      name: `Lightcurve Analysis - ${targetId}`,
      priority: 'medium',
      canPause: false,
      apiEndpoint: `/api/lightcurve/analysis/${targetId}/status`,
      pollInterval: 1000
    });
  }, [startApiTask]);

  // Image Classification задача
  const startImageClassification = useCallback((imageId: string, model: string) => {
    return startApiTask({
      type: 'image_classification',
      name: `Image Classification - ${model}`,
      priority: 'medium',
      canPause: false,
      apiEndpoint: `/api/cnn/classify/${imageId}/status`,
      pollInterval: 500
    });
  }, [startApiTask]);

  // Data Loading задача
  const startDataLoading = useCallback((source: string) => {
    return startApiTask({
      type: 'data_loading',
      name: `Loading ${source} Data`,
      priority: 'low',
      canPause: true,
      apiEndpoint: `/api/data/load/${source}/status`,
      pollInterval: 3000
    });
  }, [startApiTask]);

  // Model Inference задача
  const startModelInference = useCallback((modelId: string, data: any) => {
    return startWorkerTask(
      {
        type: 'model_inference',
        name: `Model Inference - ${modelId}`,
        priority: 'high',
        canPause: false
      },
      '/workers/inference-worker.js',
      { modelId, data }
    );
  }, [startWorkerTask]);

  // File Processing задача
  const startFileProcessing = useCallback((fileName: string, processingType: string) => {
    return startWorkerTask(
      {
        type: 'file_processing',
        name: `Processing ${fileName}`,
        priority: 'medium',
        canPause: true
      },
      '/workers/file-processor.js',
      { fileName, processingType }
    );
  }, [startWorkerTask]);

  // Остановка задачи
  const stopTask = useCallback((taskId: string) => {
    const task = tasks.find(t => t.id === taskId);
    if (task) {
      if (task.apiEndpoint) {
        stopPollingTask(taskId);
      }
      if (task.backgroundWorker) {
        task.backgroundWorker.terminate();
      }
      updateTask(taskId, { status: 'paused' });
    }
  }, [tasks, stopPollingTask, updateTask]);

  // Получение активных задач по типу
  const getActiveTasksByType = useCallback((type: BackgroundTask['type']) => {
    return getTasksByType(type).filter(task => 
      task.status === 'running' || task.status === 'queued'
    );
  }, [getTasksByType]);

  return {
    tasks,
    startApiTask,
    startWorkerTask,
    startCNNTraining,
    startLightcurveAnalysis,
    startImageClassification,
    startDataLoading,
    startModelInference,
    startFileProcessing,
    stopTask,
    removeTask,
    updateTask,
    getActiveTasksByType,
    getTasksByType
  };
};

export default useBackgroundTask;
