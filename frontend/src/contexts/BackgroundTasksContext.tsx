import React, { createContext, useContext, useState, useEffect, type ReactNode } from 'react';

export interface BackgroundTask {
  id: string;
  type: 'cnn_training' | 'lightcurve_analysis' | 'image_classification' | 'data_loading' | 'model_inference' | 'file_processing';
  name: string;
  status: 'running' | 'paused' | 'completed' | 'error' | 'queued';
  progress: number;
  startTime: Date;
  estimatedEndTime?: Date;
  data?: any;
  priority: 'low' | 'medium' | 'high';
  canPause: boolean;
  backgroundWorker?: Worker;
  apiEndpoint?: string;
  pollInterval?: number;
}

interface BackgroundTasksContextType {
  tasks: BackgroundTask[];
  addTask: (task: Omit<BackgroundTask, 'id' | 'startTime'>) => string;
  updateTask: (id: string, updates: Partial<BackgroundTask>) => void;
  removeTask: (id: string) => void;
  pauseTask: (id: string) => void;
  resumeTask: (id: string) => void;
  getTasksByType: (type: BackgroundTask['type']) => BackgroundTask[];
  getRunningTasks: () => BackgroundTask[];
  startPollingTask: (id: string, endpoint: string, interval?: number) => void;
  stopPollingTask: (id: string) => void;
  createWorkerTask: (id: string, workerScript: string, data: any) => void;
}

const BackgroundTasksContext = createContext<BackgroundTasksContextType | undefined>(undefined);

export const useBackgroundTasks = () => {
  const context = useContext(BackgroundTasksContext);
  if (!context) {
    throw new Error('useBackgroundTasks must be used within a BackgroundTasksProvider');
  }
  return context;
};

interface BackgroundTasksProviderProps {
  children: ReactNode;
}

export const BackgroundTasksProvider: React.FC<BackgroundTasksProviderProps> = ({ children }) => {
  const [tasks, setTasks] = useState<BackgroundTask[]>([]);
  const [pollingIntervals, setPollingIntervals] = useState<Map<string, NodeJS.Timeout>>(new Map());

  // Генерация уникального ID
  const generateId = () => {
    return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  // Добавление новой задачи
  const addTask = (taskData: Omit<BackgroundTask, 'id' | 'startTime'>): string => {
    const id = generateId();
    const newTask: BackgroundTask = {
      ...taskData,
      id,
      startTime: new Date()
    };
    
    setTasks(prev => [...prev, newTask]);
    
    // Уведомление о новой задаче
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(`Запущена задача: ${newTask.name}`, {
        icon: '/favicon.ico',
        body: `Тип: ${newTask.type}`,
        tag: newTask.id
      });
    }
    
    return id;
  };

  // Обновление задачи
  const updateTask = (id: string, updates: Partial<BackgroundTask>) => {
    setTasks(prev => prev.map(task => 
      task.id === id ? { ...task, ...updates } : task
    ));
    
    // Уведомление о завершении
    if (updates.status === 'completed') {
      const task = tasks.find(t => t.id === id);
      if (task && 'Notification' in window && Notification.permission === 'granted') {
        new Notification(`Задача завершена: ${task.name}`, {
          icon: '/favicon.ico',
          body: 'Нажмите для просмотра результатов',
          tag: task.id
        });
      }
    }
  };

  // Удаление задачи
  const removeTask = (id: string) => {
    setTasks(prev => prev.filter(task => task.id !== id));
  };

  // Пауза задачи
  const pauseTask = (id: string) => {
    updateTask(id, { status: 'paused' });
  };

  // Возобновление задачи
  const resumeTask = (id: string) => {
    updateTask(id, { status: 'running' });
  };

  // Получение задач по типу
  const getTasksByType = (type: BackgroundTask['type']) => {
    return tasks.filter(task => task.type === type);
  };

  // Получение активных задач
  const getRunningTasks = () => {
    return tasks.filter(task => task.status === 'running');
  };

  // Автоматическая очистка завершенных задач (через 1 час)
  useEffect(() => {
    const cleanup = setInterval(() => {
      const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
      setTasks(prev => prev.filter(task => 
        task.status !== 'completed' || task.startTime > oneHourAgo
      ));
    }, 5 * 60 * 1000); // Проверяем каждые 5 минут

    return () => clearInterval(cleanup);
  }, []);

  // Запуск polling задачи
  const startPollingTask = (id: string, endpoint: string, interval: number = 2000) => {
    // Остановить существующий polling если есть
    stopPollingTask(id);
    
    const pollFunction = async () => {
      try {
        const response = await fetch(endpoint);
        if (response.ok) {
          const data = await response.json();
          updateTask(id, {
            progress: data.progress || 0,
            status: data.status || 'running',
            data: data
          });
          
          // Если задача завершена, остановить polling
          if (data.status === 'completed' || data.status === 'error') {
            stopPollingTask(id);
          }
        }
      } catch (error) {
        console.error(`Polling error for task ${id}:`, error);
        updateTask(id, { status: 'error' });
        stopPollingTask(id);
      }
    };
    
    // Запустить немедленно и затем по интервалу
    pollFunction();
    const intervalId = setInterval(pollFunction, interval);
    
    setPollingIntervals(prev => new Map(prev.set(id, intervalId)));
  };

  // Остановка polling задачи
  const stopPollingTask = (id: string) => {
    const intervalId = pollingIntervals.get(id);
    if (intervalId) {
      clearInterval(intervalId);
      setPollingIntervals(prev => {
        const newMap = new Map(prev);
        newMap.delete(id);
        return newMap;
      });
    }
  };

  // Создание Web Worker задачи
  const createWorkerTask = (id: string, workerScript: string, data: any) => {
    try {
      const worker = new Worker(workerScript);
      
      worker.postMessage(data);
      
      worker.onmessage = (event) => {
        const { type, progress, result, error } = event.data;
        
        switch (type) {
          case 'progress':
            updateTask(id, { progress });
            break;
          case 'completed':
            updateTask(id, { status: 'completed', progress: 100, data: result });
            worker.terminate();
            break;
          case 'error':
            updateTask(id, { status: 'error', data: { error } });
            worker.terminate();
            break;
        }
      };
      
      worker.onerror = (error) => {
        console.error(`Worker error for task ${id}:`, error);
        updateTask(id, { status: 'error', data: { error: error.message } });
        worker.terminate();
      };
      
      // Сохранить worker в задаче
      updateTask(id, { backgroundWorker: worker });
      
    } catch (error) {
      console.error(`Failed to create worker for task ${id}:`, error);
      updateTask(id, { status: 'error', data: { error: 'Failed to create worker' } });
    }
  };

  // Очистка при размонтировании
  useEffect(() => {
    return () => {
      // Остановить все polling интервалы
      pollingIntervals.forEach((intervalId) => {
        clearInterval(intervalId);
      });
      
      // Завершить все workers
      tasks.forEach(task => {
        if (task.backgroundWorker) {
          task.backgroundWorker.terminate();
        }
      });
    };
  }, []);

  // Запрос разрешения на уведомления
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const value: BackgroundTasksContextType = {
    tasks,
    addTask,
    updateTask,
    removeTask,
    pauseTask,
    resumeTask,
    getTasksByType,
    getRunningTasks,
    startPollingTask,
    stopPollingTask,
    createWorkerTask
  };

  return (
    <BackgroundTasksContext.Provider value={value}>
      {children}
    </BackgroundTasksContext.Provider>
  );
};
