"""
Model Trainer for Transit Detection

Система обучения моделей с поддержкой:
- Transfer Learning (Kepler → TESS)
- Active Learning с пользовательской обратной связью
- Онлайн обучение на новых данных
- Мониторинг метрик и early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from pathlib import Path
import json
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available, plotting disabled")
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm
    def tqdm(iterable, **kwargs):
        return iterable
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("WandB not available")
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available")

from .models.base_model import BaseTransitModel
from .ensemble import EnsembleClassifier

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Универсальный тренер для моделей обнаружения транзитов
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 experiment_name: str = 'transit_detection',
                 use_wandb: bool = False,
                 use_mlflow: bool = False):
        
        self.model = model
        self.device = device
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        
        # Перемещаем модель на устройство
        self.model.to(device)
        
        # История обучения
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Лучшие метрики
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        # Инициализация логирования
        self._init_logging()
        
    def _init_logging(self):
        """Инициализация систем логирования"""
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="exoplanet-detection",
                name=self.experiment_name,
                config={
                    'model_type': self.model.__class__.__name__,
                    'device': self.device
                }
            )
        elif self.use_wandb:
            logger.warning("WandB requested but not available")
            self.use_wandb = False
        
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()
        elif self.use_mlflow:
            logger.warning("MLflow requested but not available")
            self.use_mlflow = False
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              optimizer_type: str = 'adamw',
              scheduler_type: str = 'cosine',
              early_stopping_patience: int = 10,
              gradient_clip_norm: float = 1.0,
              class_weights: Optional[torch.Tensor] = None,
              save_checkpoints: bool = True,
              checkpoint_dir: str = 'checkpoints') -> Dict[str, List[float]]:
        """
        Основной цикл обучения
        
        Args:
            train_loader: DataLoader для обучающих данных
            val_loader: DataLoader для валидационных данных
            num_epochs: Количество эпох
            learning_rate: Скорость обучения
            weight_decay: L2 регуляризация
            optimizer_type: Тип оптимизатора ('adam', 'adamw', 'sgd')
            scheduler_type: Тип планировщика ('cosine', 'step', 'plateau')
            early_stopping_patience: Терпение для early stopping
            gradient_clip_norm: Норма для gradient clipping
            class_weights: Веса классов для несбалансированных данных
            save_checkpoints: Сохранять ли чекпоинты
            checkpoint_dir: Директория для чекпоинтов
            
        Returns:
            История обучения
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Настройка оптимизатора
        optimizer = self._create_optimizer(optimizer_type, learning_rate, weight_decay)
        
        # Настройка планировщика
        scheduler = self._create_scheduler(scheduler_type, optimizer, num_epochs)
        
        # Функция потерь
        criterion = self._create_criterion(class_weights)
        
        # Директория для чекпоинтов
        if save_checkpoints:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Счетчики для early stopping
        patience_counter = 0
        
        # Основной цикл обучения
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Обучение
            train_metrics = self._train_epoch(
                train_loader, optimizer, criterion, gradient_clip_norm
            )
            
            # Валидация
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Обновление планировщика
            if scheduler_type == 'plateau' and val_loader is not None:
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            # Логирование
            epoch_time = time.time() - start_time
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time, optimizer)
            
            # Сохранение лучшей модели
            if val_loader is not None:
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    
                    if save_checkpoints:
                        self._save_checkpoint(
                            checkpoint_path / 'best_model.pth',
                            epoch, optimizer, scheduler, train_metrics, val_metrics
                        )
                else:
                    patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Сохранение регулярных чекпоинтов
            if save_checkpoints and (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    checkpoint_path / f'checkpoint_epoch_{epoch+1}.pth',
                    epoch, optimizer, scheduler, train_metrics, val_metrics
                )
        
        # Загружаем лучшую модель
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with val_loss: {self.best_val_loss:.4f}")
        
        # Финализация логирования
        self._finalize_logging()
        
        return self.training_history
    
    def transfer_learning(self,
                         source_model_path: str,
                         target_train_loader: DataLoader,
                         target_val_loader: Optional[DataLoader] = None,
                         freeze_layers: Optional[List[str]] = None,
                         fine_tune_epochs: int = 50,
                         fine_tune_lr: float = 1e-4) -> Dict[str, List[float]]:
        """
        Transfer Learning: обучение на Kepler, дообучение на TESS
        
        Args:
            source_model_path: Путь к предобученной модели
            target_train_loader: Данные целевого домена для обучения
            target_val_loader: Данные целевого домена для валидации
            freeze_layers: Слои для заморозки
            fine_tune_epochs: Количество эпох для fine-tuning
            fine_tune_lr: Скорость обучения для fine-tuning
            
        Returns:
            История fine-tuning
        """
        logger.info("Starting transfer learning")
        
        # Загружаем предобученную модель
        checkpoint = torch.load(source_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Замораживаем указанные слои
        if freeze_layers:
            self.model.freeze_layers(freeze_layers)
            logger.info(f"Frozen layers: {freeze_layers}")
        
        # Fine-tuning с меньшей скоростью обучения
        history = self.train(
            train_loader=target_train_loader,
            val_loader=target_val_loader,
            num_epochs=fine_tune_epochs,
            learning_rate=fine_tune_lr,
            optimizer_type='adamw',
            scheduler_type='cosine',
            early_stopping_patience=15
        )
        
        logger.info("Transfer learning completed")
        return history
    
    def active_learning_step(self,
                           unlabeled_data: torch.Tensor,
                           uncertainty_threshold: float = 0.5,
                           max_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Шаг активного обучения: выбор наиболее неопределенных образцов
        
        Args:
            unlabeled_data: Неразмеченные данные
            uncertainty_threshold: Порог неопределенности
            max_samples: Максимальное количество образцов для разметки
            
        Returns:
            Tuple из (selected_samples, uncertainties)
        """
        self.model.eval()
        
        uncertainties = []
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(unlabeled_data), 32):  # Батчи по 32
                batch = unlabeled_data[i:i+32].to(self.device)
                
                if isinstance(self.model, EnsembleClassifier):
                    # Для ансамбля используем встроенную оценку неопределенности
                    _, batch_uncertainties, _ = self.model.predict_with_uncertainty(batch)
                    uncertainties.extend(batch_uncertainties)
                else:
                    # Для одиночной модели используем энтропию
                    logits = self.model(batch)
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    uncertainties.extend(entropy.cpu().numpy())
                
                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        
        uncertainties = np.array(uncertainties)
        
        # Выбираем образцы с высокой неопределенностью
        uncertain_indices = np.where(uncertainties > uncertainty_threshold)[0]
        
        # Ограничиваем количество образцов
        if len(uncertain_indices) > max_samples:
            # Берем топ max_samples по неопределенности
            top_indices = np.argsort(uncertainties[uncertain_indices])[-max_samples:]
            uncertain_indices = uncertain_indices[top_indices]
        
        selected_samples = unlabeled_data[uncertain_indices]
        selected_uncertainties = uncertainties[uncertain_indices]
        
        logger.info(f"Selected {len(selected_samples)} samples for labeling")
        
        return selected_samples, selected_uncertainties
    
    def online_learning_update(self,
                             new_data: torch.Tensor,
                             new_labels: torch.Tensor,
                             learning_rate: float = 1e-4,
                             num_updates: int = 10):
        """
        Онлайн обучение на новых данных
        
        Args:
            new_data: Новые данные
            new_labels: Новые метки
            learning_rate: Скорость обучения
            num_updates: Количество обновлений
        """
        self.model.train()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for update in range(num_updates):
            optimizer.zero_grad()
            
            # Случайный батч из новых данных
            indices = torch.randperm(len(new_data))[:min(32, len(new_data))]
            batch_data = new_data[indices].to(self.device)
            batch_labels = new_labels[indices].to(self.device)
            
            # Forward pass
            logits = self.model(batch_data)
            loss = criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        logger.info(f"Online learning update completed with {len(new_data)} new samples")
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, gradient_clip_norm: float) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(data)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
            
            optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Обновление progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Валидация на одной эпохе"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation"):
                data, labels = data.to(self.device), labels.to(self.device)
                
                logits = self.model(data)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Вычисляем метрики
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Дополнительные метрики
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # AUC для бинарной классификации
        if len(np.unique(all_labels)) == 2:
            all_probs_array = np.array(all_probs)
            if all_probs_array.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probs_array[:, 1])
                metrics['auc'] = auc
        
        return metrics
    
    def _create_optimizer(self, optimizer_type: str, learning_rate: float, weight_decay: float):
        """Создание оптимизатора"""
        if optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self, scheduler_type: str, optimizer: optim.Optimizer, num_epochs: int):
        """Создание планировщика скорости обучения"""
        if scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
        elif scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _create_criterion(self, class_weights: Optional[torch.Tensor]):
        """Создание функции потерь"""
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float], epoch_time: float, optimizer: optim.Optimizer):
        """Логирование эпохи"""
        # Обновляем историю
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
        
        # Консольный вывод
        log_msg = f"Epoch {epoch+1}: "
        log_msg += f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}"
        
        if val_metrics:
            log_msg += f", Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
        
        log_msg += f", Time: {epoch_time:.2f}s"
        logger.info(log_msg)
        
        # Логирование в внешние системы
        if self.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            
            if val_metrics:
                log_dict.update({
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics.get('precision', 0),
                    'val_recall': val_metrics.get('recall', 0),
                    'val_f1': val_metrics.get('f1', 0)
                })
                
                if 'auc' in val_metrics:
                    log_dict['val_auc'] = val_metrics['auc']
            
            wandb.log(log_dict)
        
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            if val_metrics:
                mlflow.log_metrics({
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }, step=epoch)
    
    def _save_checkpoint(self, path: Path, epoch: int, optimizer: optim.Optimizer,
                        scheduler, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        
        torch.save(checkpoint, path)
    
    def _finalize_logging(self):
        """Финализация логирования"""
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Построение графиков обучения"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - matplotlib/seaborn not installed")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Acc')
        if self.training_history['val_acc']:
            axes[0, 1].plot(self.training_history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[1, 0].plot(self.training_history['learning_rates'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss difference (overfitting indicator)
        if self.training_history['val_loss']:
            loss_diff = np.array(self.training_history['val_loss']) - np.array(self.training_history['train_loss'])
            axes[1, 1].plot(loss_diff)
            axes[1, 1].set_title('Overfitting Indicator (Val - Train Loss)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class TransitDataset(Dataset):
    """
    Dataset для кривых блеска транзитов
    """
    
    def __init__(self, 
                 lightcurves: np.ndarray,
                 labels: np.ndarray,
                 transform: Optional[Callable] = None,
                 augment: bool = False):
        
        self.lightcurves = lightcurves
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.lightcurves)
    
    def __getitem__(self, idx):
        lightcurve = self.lightcurves[idx].copy()
        label = self.labels[idx]
        
        # Аугментация данных
        if self.augment:
            lightcurve = self._augment_lightcurve(lightcurve)
        
        # Применяем трансформации
        if self.transform:
            lightcurve = self.transform(lightcurve)
        
        return torch.FloatTensor(lightcurve), torch.LongTensor([label]).squeeze()
    
    def _augment_lightcurve(self, lightcurve: np.ndarray) -> np.ndarray:
        """Аугментация кривой блеска"""
        # Добавление шума
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.0001, 0.001)
            noise = np.random.normal(0, noise_level, lightcurve.shape)
            lightcurve += noise
        
        # Масштабирование
        if np.random.random() < 0.2:
            scale = np.random.uniform(0.98, 1.02)
            lightcurve *= scale
        
        # Сдвиг по времени (циклический)
        if np.random.random() < 0.3:
            shift = np.random.randint(1, len(lightcurve) // 10)
            lightcurve = np.roll(lightcurve, shift)
        
        return lightcurve
