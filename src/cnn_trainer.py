"""
CNN тренер с расширенными метриками, визуализацией и мониторингом
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import json
import os
from datetime import datetime
import logging
from pathlib import Path

from cnn_models import CNNModelFactory, get_model_summary


class MetricsTracker:
    """Класс для отслеживания метрик обучения"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs_times = []
        
        # Дополнительные метрики
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1_scores = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
        self.val_aucs = []
    
    def update(self, epoch_metrics: Dict[str, float]):
        """Обновляет метрики для текущей эпохи"""
        for key, value in epoch_metrics.items():
            if hasattr(self, key):
                getattr(self, key).append(value)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Возвращает лучшие метрики"""
        if not self.val_accuracies:
            return {}
        
        best_epoch = np.argmax(self.val_accuracies)
        return {
            'best_epoch': best_epoch,
            'best_val_accuracy': self.val_accuracies[best_epoch],
            'best_val_loss': self.val_losses[best_epoch],
            'best_train_accuracy': self.train_accuracies[best_epoch],
            'best_train_loss': self.train_losses[best_epoch],
        }
    
    def save_to_file(self, filepath: str):
        """Сохраняет метрики в файл"""
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epochs_times': self.epochs_times,
            'train_precisions': self.train_precisions,
            'train_recalls': self.train_recalls,
            'train_f1_scores': self.train_f1_scores,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'val_f1_scores': self.val_f1_scores,
            'val_aucs': self.val_aucs,
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)


class CNNTrainer:
    """
    Расширенный тренер для CNN моделей с поддержкой различных архитектур
    """
    
    def __init__(self, 
                 model_type: str = 'resnet',
                 model_params: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None,
                 save_dir: str = 'cnn_experiments'):
        
        self.model_type = model_type
        self.model_params = model_params or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Инициализация модели
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Метрики и логирование
        self.metrics_tracker = MetricsTracker()
        self.logger = self._setup_logger()
        
        # Состояние обучения
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.training_start_time = None
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(f'CNNTrainer_{self.model_type}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build_model(self, input_length: int = 2000, num_classes: int = 2) -> nn.Module:
        """Создает и инициализирует модель"""
        self.model_params.update({
            'input_length': input_length,
            'num_classes': num_classes
        })
        
        self.model = CNNModelFactory.create_model(self.model_type, **self.model_params)
        self.model.to(self.device)
        
        # Логируем информацию о модели
        summary = get_model_summary(self.model, (1, input_length))
        self.logger.info(f"Создана модель {self.model_type}")
        self.logger.info(f"Параметры: {summary['total_parameters']:,}")
        self.logger.info(f"Размер: {summary['model_size_mb']:.2f} MB")
        
        return self.model
    
    def setup_optimizer(self, 
                       optimizer_type: str = 'adamw',
                       learning_rate: float = 1e-3,
                       weight_decay: float = 1e-4,
                       **optimizer_kwargs):
        """Настройка оптимизатора"""
        if self.model is None:
            raise ValueError("Модель должна быть создана перед настройкой оптимизатора")
        
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=optimizer_kwargs.get('momentum', 0.9),
                **{k: v for k, v in optimizer_kwargs.items() if k != 'momentum'}
            )
        else:
            raise ValueError(f"Неподдерживаемый тип оптимизатора: {optimizer_type}")
        
        self.logger.info(f"Настроен оптимизатор {optimizer_type} с lr={learning_rate}")
    
    def setup_scheduler(self, 
                       scheduler_type: str = 'cosine',
                       **scheduler_kwargs):
        """Настройка планировщика learning rate"""
        if self.optimizer is None:
            raise ValueError("Оптимизатор должен быть настроен перед планировщиком")
        
        if scheduler_type.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_kwargs.get('T_max', 100),
                **{k: v for k, v in scheduler_kwargs.items() if k != 'T_max'}
            )
        elif scheduler_type.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_kwargs.get('step_size', 30),
                gamma=scheduler_kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_kwargs.get('factor', 0.5),
                patience=scheduler_kwargs.get('patience', 10),
                verbose=True
            )
        else:
            raise ValueError(f"Неподдерживаемый тип планировщика: {scheduler_type}")
        
        self.logger.info(f"Настроен планировщик {scheduler_type}")
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Вычисляет метрики классификации"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # AUC для бинарной классификации
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_proba[:, 1])
                metrics['auc'] = auc
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Сохраняем предсказания для метрик
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        # Вычисляем метрики
        avg_loss = running_loss / len(train_loader)
        metrics = self.compute_metrics(
            np.array(all_targets), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': metrics['accuracy'],
            'train_precision': metrics['precision'],
            'train_recall': metrics['recall'],
            'train_f1_score': metrics['f1_score']
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Валидация на одной эпохе"""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                
                # Сохраняем предсказания для метрик
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Вычисляем метрики
        avg_loss = running_loss / len(val_loader)
        metrics = self.compute_metrics(
            np.array(all_targets), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        result = {
            'val_loss': avg_loss,
            'val_accuracy': metrics['accuracy'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'val_f1_score': metrics['f1_score']
        }
        
        if 'auc' in metrics:
            result['val_auc'] = metrics['auc']
        
        return result
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              save_best: bool = True,
              early_stopping_patience: int = 15,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Основной цикл обучения
        """
        self.training_start_time = time.time()
        self.logger.info(f"Начало обучения на {epochs} эпох")
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Обучение
            train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_metrics = self.validate_epoch(val_loader)
            
            # Обновление learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_accuracy'])
                else:
                    self.scheduler.step()
            
            # Время эпохи
            epoch_time = time.time() - epoch_start_time
            
            # Объединяем все метрики
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['learning_rates'] = current_lr
            epoch_metrics['epochs_times'] = epoch_time
            
            # Обновляем трекер метрик
            self.metrics_tracker.update(epoch_metrics)
            
            # Проверяем улучшение
            if val_metrics['val_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['val_accuracy']
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            # Логирование
            if verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping на эпохе {epoch+1}")
                break
        
        # Финальные результаты
        total_time = time.time() - self.training_start_time
        best_metrics = self.metrics_tracker.get_best_metrics()
        
        results = {
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'best_metrics': best_metrics,
            'final_metrics': epoch_metrics
        }
        
        self.logger.info(f"Обучение завершено за {total_time:.2f}s")
        self.logger.info(f"Лучшая точность: {best_val_accuracy:.4f}")
        
        return results
    
    def save_checkpoint(self, filename: str):
        """Сохранение чекпоинта модели"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'metrics': self.metrics_tracker.__dict__
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        self.logger.info(f"Чекпоинт сохранен: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Загрузка чекпоинта модели"""
        filepath = self.save_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Восстанавливаем метрики
        if 'metrics' in checkpoint:
            for key, value in checkpoint['metrics'].items():
                if hasattr(self.metrics_tracker, key):
                    setattr(self.metrics_tracker, key, value)
        
        self.logger.info(f"Чекпоинт загружен: {filepath}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Построение графиков обучения"""
        if not self.metrics_tracker.train_losses:
            self.logger.warning("Нет данных для построения графиков")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Curves - {self.model_type.upper()}', fontsize=16)
        
        epochs = range(1, len(self.metrics_tracker.train_losses) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.metrics_tracker.train_losses, 'b-', label='Train')
        axes[0, 0].plot(epochs, self.metrics_tracker.val_losses, 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.metrics_tracker.train_accuracies, 'b-', label='Train')
        axes[0, 1].plot(epochs, self.metrics_tracker.val_accuracies, 'r-', label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[0, 2].plot(epochs, self.metrics_tracker.learning_rates, 'g-')
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # F1 Score
        if self.metrics_tracker.train_f1_scores:
            axes[1, 0].plot(epochs, self.metrics_tracker.train_f1_scores, 'b-', label='Train')
            axes[1, 0].plot(epochs, self.metrics_tracker.val_f1_scores, 'r-', label='Validation')
            axes[1, 0].set_title('F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # AUC
        if self.metrics_tracker.val_aucs:
            axes[1, 1].plot(epochs, self.metrics_tracker.val_aucs, 'r-')
            axes[1, 1].set_title('Validation AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].grid(True)
        
        # Epoch Times
        if self.metrics_tracker.epochs_times:
            axes[1, 2].plot(epochs, self.metrics_tracker.epochs_times, 'purple')
            axes[1, 2].set_title('Epoch Time')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Time (seconds)')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Графики сохранены: {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Полная оценка модели на тестовых данных"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Вычисляем метрики
        metrics = self.compute_metrics(
            np.array(all_targets), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }


def create_synthetic_data(n_samples: int = 1000, 
                         input_length: int = 2000,
                         noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Создает синтетические данные для тестирования CNN
    """
    X = []
    y = []
    
    for i in range(n_samples):
        # Создаем базовый сигнал
        t = np.linspace(0, 10, input_length)
        
        if i % 2 == 0:  # Класс 0 - без транзита
            signal = np.ones_like(t) + np.random.normal(0, noise_level, len(t))
            label = 0
        else:  # Класс 1 - с транзитом
            signal = np.ones_like(t)
            # Добавляем транзит
            transit_start = np.random.randint(input_length // 4, 3 * input_length // 4)
            transit_duration = np.random.randint(20, 100)
            transit_depth = np.random.uniform(0.01, 0.05)
            
            signal[transit_start:transit_start + transit_duration] -= transit_depth
            signal += np.random.normal(0, noise_level, len(t))
            label = 1
        
        X.append(signal)
        y.append(label)
    
    X = torch.FloatTensor(X).unsqueeze(1)  # Добавляем канальное измерение
    y = torch.LongTensor(y)
    
    return X, y


if __name__ == "__main__":
    # Тестирование CNN тренера
    print("=== Тестирование CNN Trainer ===")
    
    # Создаем синтетические данные
    X_train, y_train = create_synthetic_data(800, 2000)
    X_val, y_val = create_synthetic_data(200, 2000)
    
    # Создаем DataLoader'ы
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Тестируем разные модели
    model_types = ['cnn', 'resnet', 'attention']
    
    for model_type in model_types:
        print(f"\n--- Тестирование {model_type.upper()} ---")
        
        # Создаем тренер
        trainer = CNNTrainer(
            model_type=model_type,
            save_dir=f'test_experiments_{model_type}'
        )
        
        # Строим модель
        trainer.build_model(input_length=2000, num_classes=2)
        
        # Настраиваем оптимизатор и планировщик
        trainer.setup_optimizer('adamw', learning_rate=1e-3)
        trainer.setup_scheduler('cosine', T_max=20)
        
        # Обучаем
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            verbose=False
        )
        
        print(f"Лучшая точность: {results['best_metrics']['best_val_accuracy']:.4f}")
        print(f"Время обучения: {results['total_time']:.2f}s")
        
        # Сохраняем метрики
        trainer.metrics_tracker.save_to_file(f'test_metrics_{model_type}.json')
