"""
Модуль GPU оптимизации и параллельных вычислений
Обеспечивает эффективное использование GPU и многопоточность
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from pathlib import Path
import psutil
import gc

logger = logging.getLogger(__name__)

class GPUManager:
    """
    Менеджер для управления GPU ресурсами
    """
    
    def __init__(self, device_id: Optional[int] = None, 
                 memory_fraction: float = 0.8):
        """
        Инициализация GPU менеджера
        
        Args:
            device_id: ID GPU устройства (None для автоматического выбора)
            memory_fraction: Доля GPU памяти для использования
        """
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.device = self._setup_device()
        self.memory_info = self._get_memory_info()
        
        logger.info(f"GPU Manager инициализирован: {self.device}")
        logger.info(f"GPU память: {self.memory_info['total']:.2f} GB")
    
    def _setup_device(self) -> torch.device:
        """Настройка GPU устройства"""
        if torch.cuda.is_available():
            if self.device_id is not None:
                device = torch.device(f"cuda:{self.device_id}")
            else:
                device = torch.device("cuda")
            
            # Настройка памяти GPU
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Очистка кэша
            torch.cuda.empty_cache()
            
            return device
        else:
            logger.warning("CUDA недоступна, используется CPU")
            return torch.device("cpu")
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Получение информации о памяти GPU"""
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            cached_memory = torch.cuda.memory_reserved(self.device)
            
            return {
                'total': total_memory / 1024**3,  # GB
                'allocated': allocated_memory / 1024**3,  # GB
                'cached': cached_memory / 1024**3,  # GB
                'free': (total_memory - allocated_memory) / 1024**3  # GB
            }
        else:
            # CPU память
            memory = psutil.virtual_memory()
            return {
                'total': memory.total / 1024**3,  # GB
                'allocated': memory.used / 1024**3,  # GB
                'cached': 0,  # GB
                'free': memory.available / 1024**3  # GB
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Получение текущего использования памяти"""
        return self._get_memory_info()
    
    def clear_cache(self):
        """Очистка кэша GPU"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def optimize_memory(self):
        """Оптимизация использования памяти"""
        if self.device.type == 'cuda':
            # Очистка кэша
            self.clear_cache()
            
            # Настройка оптимизации памяти
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            logger.info("Оптимизация памяти GPU выполнена")
    
    def move_to_device(self, data: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
        """Перемещение данных/модели на устройство"""
        return data.to(self.device)
    
    def is_available(self) -> bool:
        """Проверка доступности GPU"""
        return torch.cuda.is_available()


class ParallelDataProcessor:
    """
    Класс для параллельной обработки данных
    """
    
    def __init__(self, num_workers: Optional[int] = None, 
                 use_multiprocessing: bool = True):
        """
        Инициализация параллельного процессора
        
        Args:
            num_workers: Количество рабочих процессов (None для автоматического)
            use_multiprocessing: Использовать ли многопроцессность
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.use_multiprocessing = use_multiprocessing and self.num_workers > 1
        
        logger.info(f"ParallelDataProcessor инициализирован: {self.num_workers} workers")
    
    def process_batch_parallel(self, data_list: List, 
                              process_func: callable,
                              batch_size: int = 32) -> List:
        """
        Параллельная обработка батчей данных
        
        Args:
            data_list: Список данных для обработки
            process_func: Функция обработки
            batch_size: Размер батча
            
        Returns:
            List: Обработанные данные
        """
        logger.info(f"Параллельная обработка {len(data_list)} элементов")
        
        # Разделение на батчи
        batches = [data_list[i:i + batch_size] 
                  for i in range(0, len(data_list), batch_size)]
        
        if self.use_multiprocessing:
            # Многопроцессная обработка
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(process_func, batches))
        else:
            # Многопоточная обработка
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(process_func, batches))
        
        # Объединение результатов
        processed_data = []
        for batch_result in results:
            processed_data.extend(batch_result)
        
        logger.info(f"Обработано {len(processed_data)} элементов")
        return processed_data
    
    def process_with_queue(self, data_queue: queue.Queue,
                          result_queue: queue.Queue,
                          process_func: callable):
        """
        Обработка данных с использованием очередей
        
        Args:
            data_queue: Очередь входных данных
            result_queue: Очередь результатов
            process_func: Функция обработки
        """
        while True:
            try:
                data = data_queue.get(timeout=1)
                if data is None:  # Сигнал завершения
                    break
                
                result = process_func(data)
                result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка обработки: {e}")
                result_queue.put(None)


class OptimizedDataLoader:
    """
    Оптимизированный DataLoader с поддержкой GPU
    """
    
    def __init__(self, dataset, batch_size: int = 32,
                 num_workers: int = 4, pin_memory: bool = True,
                 prefetch_factor: int = 2):
        """
        Инициализация оптимизированного DataLoader
        
        Args:
            dataset: Датасет для загрузки
            batch_size: Размер батча
            num_workers: Количество рабочих процессов
            pin_memory: Использовать ли pin_memory для GPU
            prefetch_factor: Коэффициент предзагрузки
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.prefetch_factor = prefetch_factor
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False
        )
        
        logger.info(f"OptimizedDataLoader создан: batch_size={batch_size}, "
                   f"num_workers={num_workers}, pin_memory={pin_memory}")
    
    def __iter__(self):
        """Итератор по DataLoader"""
        return iter(self.dataloader)
    
    def __len__(self):
        """Длина DataLoader"""
        return len(self.dataloader)


class MemoryEfficientModel:
    """
    Базовый класс для моделей с эффективным использованием памяти
    """
    
    def __init__(self, device: torch.device):
        """
        Инициализация модели
        
        Args:
            device: Устройство для вычислений
        """
        self.device = device
        self.gradient_accumulation_steps = 1
        self.mixed_precision = torch.cuda.is_available()
        
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def forward_pass(self, model: nn.Module, data: torch.Tensor,
                    target: Optional[torch.Tensor] = None,
                    criterion: Optional[nn.Module] = None) -> Dict:
        """
        Эффективный прямой проход с поддержкой mixed precision
        
        Args:
            model: Модель для прямого прохода
            data: Входные данные
            target: Целевые значения (опционально)
            criterion: Функция потерь (опционально)
            
        Returns:
            Dict: Результаты прямого прохода
        """
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(data)
                if target is not None and criterion is not None:
                    loss = criterion(output, target)
                else:
                    loss = None
        else:
            output = model(data)
            if target is not None and criterion is not None:
                loss = criterion(output, target)
            else:
                loss = None
        
        return {
            'output': output,
            'loss': loss
        }
    
    def backward_pass(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                     model: nn.Module):
        """
        Эффективный обратный проход с накоплением градиентов
        
        Args:
            loss: Значение функции потерь
            optimizer: Оптимизатор
            model: Модель
        """
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            
            if (self.gradient_accumulation_steps == 1 or 
                hasattr(self, '_step_count') and 
                self._step_count % self.gradient_accumulation_steps == 0):
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            if (self.gradient_accumulation_steps == 1 or 
                hasattr(self, '_step_count') and 
                self._step_count % self.gradient_accumulation_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
    
    def set_gradient_accumulation_steps(self, steps: int):
        """Установка количества шагов накопления градиентов"""
        self.gradient_accumulation_steps = steps


class BatchProcessor:
    """
    Класс для эффективной обработки батчей
    """
    
    def __init__(self, device: torch.device, 
                 max_batch_size: int = 64):
        """
        Инициализация процессора батчей
        
        Args:
            device: Устройство для вычислений
            max_batch_size: Максимальный размер батча
        """
        self.device = device
        self.max_batch_size = max_batch_size
        self.memory_efficient_model = MemoryEfficientModel(device)
    
    def process_large_dataset(self, model: nn.Module, 
                             dataloader: DataLoader,
                             process_func: callable) -> List:
        """
        Обработка большого датасета с контролем памяти
        
        Args:
            model: Модель для обработки
            dataloader: DataLoader с данными
            process_func: Функция обработки батча
            
        Returns:
            List: Результаты обработки
        """
        logger.info("Начинаем обработку большого датасета")
        
        model.eval()
        results = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                try:
                    # Перемещение данных на устройство
                    if isinstance(batch_data, (list, tuple)):
                        batch_data = [data.to(self.device) for data in batch_data]
                    else:
                        batch_data = batch_data.to(self.device)
                    
                    # Обработка батча
                    batch_result = process_func(model, batch_data)
                    results.append(batch_result)
                    
                    # Очистка памяти каждые 10 батчей
                    if batch_idx % 10 == 0:
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                    if batch_idx % 100 == 0:
                        logger.info(f"Обработано {batch_idx} батчей")
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"Недостаточно памяти для батча {batch_idx}, "
                                     "уменьшаем размер батча")
                        # Уменьшение размера батча
                        self._reduce_batch_size()
                        continue
                    else:
                        raise e
        
        logger.info(f"Обработка завершена: {len(results)} батчей")
        return results
    
    def _reduce_batch_size(self):
        """Уменьшение размера батча при нехватке памяти"""
        self.max_batch_size = max(1, self.max_batch_size // 2)
        logger.info(f"Размер батча уменьшен до {self.max_batch_size}")


class PerformanceMonitor:
    """
    Монитор производительности для отслеживания GPU/CPU использования
    """
    
    def __init__(self, device: torch.device):
        """
        Инициализация монитора производительности
        
        Args:
            device: Устройство для мониторинга
        """
        self.device = device
        self.start_time = None
        self.memory_usage_history = []
        self.performance_metrics = {}
    
    def start_monitoring(self):
        """Начало мониторинга"""
        self.start_time = time.time()
        logger.info("Мониторинг производительности запущен")
    
    def log_memory_usage(self):
        """Логирование использования памяти"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            cached = torch.cuda.memory_reserved(self.device) / 1024**3
            
            self.memory_usage_history.append({
                'timestamp': time.time(),
                'allocated_gb': allocated,
                'cached_gb': cached
            })
        else:
            memory = psutil.virtual_memory()
            self.memory_usage_history.append({
                'timestamp': time.time(),
                'allocated_gb': memory.used / 1024**3,
                'cached_gb': 0
            })
    
    def log_performance_metric(self, metric_name: str, value: float):
        """Логирование метрики производительности"""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append({
            'timestamp': time.time(),
            'value': value
        })
    
    def get_summary(self) -> Dict:
        """Получение сводки производительности"""
        if self.start_time is None:
            return {}
        
        duration = time.time() - self.start_time
        
        summary = {
            'duration_seconds': duration,
            'memory_usage_history': self.memory_usage_history,
            'performance_metrics': self.performance_metrics
        }
        
        # Статистика по памяти
        if self.memory_usage_history:
            allocated_values = [m['allocated_gb'] for m in self.memory_usage_history]
            summary['memory_stats'] = {
                'max_allocated_gb': max(allocated_values),
                'avg_allocated_gb': np.mean(allocated_values),
                'min_allocated_gb': min(allocated_values)
            }
        
        return summary
    
    def stop_monitoring(self) -> Dict:
        """Остановка мониторинга и получение итоговой сводки"""
        summary = self.get_summary()
        logger.info("Мониторинг производительности остановлен")
        return summary


def optimize_model_for_inference(model: nn.Module, 
                                device: torch.device) -> nn.Module:
    """
    Оптимизация модели для инференса
    
    Args:
        model: Модель для оптимизации
        device: Устройство для вычислений
        
    Returns:
        nn.Module: Оптимизированная модель
    """
    logger.info("Оптимизируем модель для инференса")
    
    # Перемещение на устройство
    model = model.to(device)
    
    # Переключение в режим оценки
    model.eval()
    
    # Оптимизация для GPU
    if device.type == 'cuda':
        # Компиляция модели (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Модель скомпилирована с torch.compile")
            except Exception as e:
                logger.warning(f"Не удалось скомпилировать модель: {e}")
        
        # Оптимизация cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Заморозка параметров
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info("Оптимизация модели завершена")
    return model


def create_optimized_dataloader(dataset, batch_size: int = 32,
                              num_workers: Optional[int] = None,
                              device: torch.device = torch.device('cpu')) -> DataLoader:
    """
    Создание оптимизированного DataLoader
    
    Args:
        dataset: Датасет
        batch_size: Размер батча
        num_workers: Количество рабочих процессов
        device: Устройство для вычислений
        
    Returns:
        DataLoader: Оптимизированный DataLoader
    """
    if num_workers is None:
        num_workers = min(4, mp.cpu_count())
    
    pin_memory = device.type == 'cuda'
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )


if __name__ == "__main__":
    # Пример использования
    logger.info("Тестирование модуля GPU оптимизации")
    
    # Инициализация GPU менеджера
    gpu_manager = GPUManager()
    
    # Проверка доступности GPU
    if gpu_manager.is_available():
        logger.info("GPU доступна")
        memory_info = gpu_manager.get_memory_usage()
        logger.info(f"Использование памяти: {memory_info}")
    else:
        logger.info("GPU недоступна, используется CPU")
    
    # Создание тестовой модели
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3)
            self.conv2 = nn.Conv1d(32, 64, 3)
            self.fc = nn.Linear(64, 1)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.adaptive_avg_pool1d(x, 1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = TestModel()
    model = optimize_model_for_inference(model, gpu_manager.device)
    
    # Тестирование производительности
    monitor = PerformanceMonitor(gpu_manager.device)
    monitor.start_monitoring()
    
    # Создание тестовых данных
    test_data = torch.randn(100, 1, 2000)
    test_data = gpu_manager.move_to_device(test_data)
    
    # Тестирование модели
    with torch.no_grad():
        for i in range(10):
            output = model(test_data)
            monitor.log_memory_usage()
            monitor.log_performance_metric('inference_time', time.time())
    
    # Получение сводки
    summary = monitor.stop_monitoring()
    logger.info(f"Сводка производительности: {summary}")
    
    logger.info("Тест завершен")
