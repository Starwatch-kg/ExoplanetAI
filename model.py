"""
Модуль определения и обучения нейронных сетей для поиска экзопланет.

Этот модуль содержит архитектуры нейронных сетей и функции обучения
для детекции транзитов экзопланет в кривых блеска.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, List
import pickle
from pathlib import Path

# Настройка логирования
logger = logging.getLogger(__name__)


class ExoplanetAutoencoder(nn.Module):
    """
    Автоэнкодер для детекции аномалий в кривых блеска.
    
    Архитектура: CNN энкодер -> LSTM -> CNN декодер
    """
    
    def __init__(self, input_length: int = 2000, 
                 latent_dim: int = 64,
                 hidden_dim: int = 128):
        """
        Инициализация автоэнкодера.
        
        Args:
            input_length: Длина входной последовательности.
            latent_dim: Размер латентного представления.
            hidden_dim: Размер скрытых слоев.
        """
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Энкодер
        self.encoder = nn.Sequential(
            # Первый блок CNN
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Второй блок CNN
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Третий блок CNN
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM для временных зависимостей
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Проекция в латентное пространство
        self.latent_projection = nn.Linear(hidden_dim, latent_dim)
        
        # Декодер
        self.decoder_projection = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Восстановление исходного размера
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose1d(32, 1, kernel_size=7, padding=3)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Кодирование входных данных.
        
        Args:
            x: Входные данные (batch_size, 1, sequence_length).
            
        Returns:
            torch.Tensor: Латентное представление.
        """
        # CNN энкодер
        encoded = self.encoder(x)  # (batch_size, 128, 1)
        encoded = encoded.squeeze(-1)  # (batch_size, 128)
        
        # LSTM обработка
        lstm_input = encoded.unsqueeze(1)  # (batch_size, 1, 128)
        lstm_out, _ = self.lstm(lstm_input)  # (batch_size, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (batch_size, hidden_dim)
        
        # Проекция в латентное пространство
        latent = self.latent_projection(lstm_out)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Декодирование латентного представления.
        
        Args:
            latent: Латентное представление.
            
        Returns:
            torch.Tensor: Восстановленные данные.
        """
        # Проекция обратно в скрытое пространство
        hidden = self.decoder_projection(latent)  # (batch_size, hidden_dim)
        
        # LSTM декодер
        lstm_input = hidden.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        lstm_out, _ = self.decoder_lstm(lstm_input)  # (batch_size, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (batch_size, hidden_dim)
        
        # Подготовка для CNN декодера
        cnn_input = lstm_out.unsqueeze(-1)  # (batch_size, hidden_dim, 1)
        
        # Проекция в размер для CNN декодера
        cnn_input = F.linear(cnn_input, 
                           torch.randn(128, self.latent_dim).to(latent.device))
        cnn_input = cnn_input.transpose(1, 2)  # (batch_size, 128, 1)
        
        # CNN декодер
        decoded = self.decoder(cnn_input)
        
        return decoded
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через автоэнкодер.
        
        Args:
            x: Входные данные.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Восстановленные данные и латентное представление.
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class ExoplanetClassifier(nn.Module):
    """
    Классификатор для детекции транзитов экзопланет.
    
    Архитектура: CNN + LSTM + Attention
    """
    
    def __init__(self, input_length: int = 2000,
                 num_classes: int = 2,
                 hidden_dim: int = 128):
        """
        Инициализация классификатора.
        
        Args:
            input_length: Длина входной последовательности.
            num_classes: Количество классов.
            hidden_dim: Размер скрытых слоев.
        """
        super().__init__()
        
        self.input_length = input_length
        
        # CNN для извлечения признаков
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM для временных зависимостей
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через классификатор.
        
        Args:
            x: Входные данные (batch_size, 1, sequence_length).
            
        Returns:
            torch.Tensor: Предсказания классов.
        """
        # CNN извлечение признаков
        cnn_features = self.cnn(x)  # (batch_size, 128, 1)
        cnn_features = cnn_features.squeeze(-1)  # (batch_size, 128)
        
        # LSTM обработка
        lstm_input = cnn_features.unsqueeze(1)  # (batch_size, 1, 128)
        lstm_out, _ = self.lstm(lstm_input)  # (batch_size, 1, hidden_dim*2)
        
        # Attention механизм
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Глобальное усреднение
        global_features = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim*2)
        
        # Классификация
        predictions = self.classifier(global_features)
        
        return predictions


class ModelTrainer:
    """Класс для обучения моделей."""
    
    def __init__(self, device: str = 'auto'):
        """
        Инициализация тренера.
        
        Args:
            device: Устройство для обучения ('auto', 'cpu', 'cuda').
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Используется устройство: {self.device}")
    
    def train_autoencoder(self, model: ExoplanetAutoencoder,
                         train_data: np.ndarray,
                         val_data: Optional[np.ndarray] = None,
                         epochs: int = 100,
                         batch_size: int = 32,
                         learning_rate: float = 1e-3,
                         save_path: Optional[str] = None) -> Dict:
        """
        Обучение автоэнкодера.
        
        Args:
            model: Модель автоэнкодера.
            train_data: Обучающие данные.
            val_data: Валидационные данные.
            epochs: Количество эпох.
            batch_size: Размер батча.
            learning_rate: Скорость обучения.
            save_path: Путь для сохранения модели.
            
        Returns:
            Dict: История обучения.
        """
        logger.info("Начинаем обучение автоэнкодера")
        
        model.to(self.device)
        model.train()
        
        # Подготовка данных
        train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data is not None:
            val_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1)
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # История обучения
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loss = 0.0
            
            for batch_data in train_loader:
                x = batch_data[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed, latent = model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Валидация
            val_loss = 0.0
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batch_data in val_loader:
                        x = batch_data[0].to(self.device)
                        reconstructed, latent = model(x)
                        loss = criterion(reconstructed, x)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                # Сохранение лучшей модели
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if save_path:
                        self.save_model(model, save_path)
            else:
                history['val_loss'].append(0.0)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss if val_loader else 0:.4f}")
        
        logger.info("Обучение автоэнкодера завершено")
        return history
    
    def train_classifier(self, model: ExoplanetClassifier,
                        train_data: np.ndarray,
                        train_labels: np.ndarray,
                        val_data: Optional[np.ndarray] = None,
                        val_labels: Optional[np.ndarray] = None,
                        epochs: int = 100,
                        batch_size: int = 32,
                        learning_rate: float = 1e-3,
                        save_path: Optional[str] = None) -> Dict:
        """
        Обучение классификатора.
        
        Args:
            model: Модель классификатора.
            train_data: Обучающие данные.
            train_labels: Метки обучающих данных.
            val_data: Валидационные данные.
            val_labels: Метки валидационных данных.
            epochs: Количество эпох.
            batch_size: Размер батча.
            learning_rate: Скорость обучения.
            save_path: Путь для сохранения модели.
            
        Returns:
            Dict: История обучения.
        """
        logger.info("Начинаем обучение классификатора")
        
        model.to(self.device)
        model.train()
        
        # Подготовка данных
        train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data is not None and val_labels is not None:
            val_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1)
            val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
            val_dataset = TensorDataset(val_tensor, val_labels_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # История обучения
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in train_loader:
                x = batch_data.to(self.device)
                labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(x)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # Валидация
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader is not None:
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_data, batch_labels in val_loader:
                        x = batch_data.to(self.device)
                        labels = batch_labels.to(self.device)
                        
                        predictions = model(x)
                        loss = criterion(predictions, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(predictions.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                # Сохранение лучшей модели
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if save_path:
                        self.save_model(model, save_path)
            else:
                history['val_loss'].append(0.0)
                history['val_acc'].append(0.0)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                           f"Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, "
                           f"Val Acc: {val_acc:.2f}%")
        
        logger.info("Обучение классификатора завершено")
        return history
    
    def save_model(self, model: nn.Module, path: str):
        """
        Сохранение модели.
        
        Args:
            model: Модель для сохранения.
            path: Путь для сохранения.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': {
                'input_length': getattr(model, 'input_length', None),
                'latent_dim': getattr(model, 'latent_dim', None),
                'hidden_dim': getattr(model, 'hidden_dim', None)
            }
        }, save_path)
        
        logger.info(f"Модель сохранена: {save_path}")
    
    def load_model(self, model: nn.Module, path: str):
        """
        Загрузка модели.
        
        Args:
            model: Модель для загрузки весов.
            path: Путь к сохраненной модели.
        """
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        logger.info(f"Модель загружена: {path}")


def train_model(train_data: np.ndarray,
               model_type: str = 'autoencoder',
               train_labels: Optional[np.ndarray] = None,
               val_data: Optional[np.ndarray] = None,
               val_labels: Optional[np.ndarray] = None,
               epochs: int = 100,
               batch_size: int = 32,
               learning_rate: float = 1e-3,
               save_path: Optional[str] = None) -> Tuple[nn.Module, Dict]:
    """
    Обучение модели для детекции экзопланет.
    
    Args:
        model_type: Тип модели ('autoencoder' или 'classifier').
        train_data: Обучающие данные.
        train_labels: Метки обучающих данных (для классификатора).
        val_data: Валидационные данные.
        val_labels: Метки валидационных данных.
        epochs: Количество эпох.
        batch_size: Размер батча.
        learning_rate: Скорость обучения.
        save_path: Путь для сохранения модели.
        
    Returns:
        Tuple[nn.Module, Dict]: Обученная модель и история обучения.
    """
    logger.info(f"Начинаем обучение модели типа: {model_type}")
    
    # Создание модели
    if model_type == 'autoencoder':
        model = ExoplanetAutoencoder(
            input_length=train_data.shape[1],
            latent_dim=64,
            hidden_dim=128
        )
    elif model_type == 'classifier':
        if train_labels is None:
            raise ValueError("Для классификатора необходимы метки")
        model = ExoplanetClassifier(
            input_length=train_data.shape[1],
            num_classes=len(np.unique(train_labels)),
            hidden_dim=128
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    # Обучение
    trainer = ModelTrainer()
    
    if model_type == 'autoencoder':
        history = trainer.train_autoencoder(
            model, train_data, val_data, epochs, batch_size, learning_rate, save_path
        )
    else:
        history = trainer.train_classifier(
            model, train_data, train_labels, val_data, val_labels, 
            epochs, batch_size, learning_rate, save_path
        )
    
    logger.info("Обучение модели завершено")
    return model, history
