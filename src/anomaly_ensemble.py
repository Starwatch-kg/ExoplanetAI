"""
Ансамбль вероятностной оценки аномалий для детекции экзопланет
Объединяет VAE, One-Class SVM и Bayesian Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class VariationalAutoencoder(nn.Module):
    """
    Вариационный автокодировщик для детекции аномалий
    """
    
    def __init__(self, input_dim: int = 128, latent_dim: int = 32,
                 hidden_dims: List[int] = [64, 32], dropout: float = 0.1):
        """
        Инициализация VAE
        
        Args:
            input_dim: Размер входного представления
            latent_dim: Размер латентного пространства
            hidden_dims: Размеры скрытых слоев
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Энкодер
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Среднее и логарифм дисперсии латентного представления
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Декодер
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Кодирование в латентное пространство
        
        Args:
            x: Входные данные
            
        Returns:
            Tuple[mu, logvar]: Среднее и логарифм дисперсии
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Репараметризация для обучения
        
        Args:
            mu: Среднее
            logvar: Логарифм дисперсии
            
        Returns:
            torch.Tensor: Сэмплированное латентное представление
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Декодирование из латентного пространства
        
        Args:
            z: Латентное представление
            
        Returns:
            torch.Tensor: Восстановленные данные
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход через VAE
        
        Args:
            x: Входные данные
            
        Returns:
            Tuple[reconstructed, mu, logvar]: Восстановленные данные и параметры
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет оценку аномальности
        
        Args:
            x: Входные данные
            
        Returns:
            torch.Tensor: Оценка аномальности
        """
        with torch.no_grad():
            reconstructed, mu, logvar = self.forward(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, x, reduction='none').sum(dim=1)
            
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Общая оценка аномальности
            anomaly_score = recon_loss + 0.1 * kl_div
            
        return anomaly_score


class BayesianNeuralNetwork(nn.Module):
    """
    Байесовская нейронная сеть для оценки неопределенности
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 output_dim: int = 1, num_samples: int = 100):
        """
        Инициализация BNN
        
        Args:
            input_dim: Размер входа
            hidden_dim: Размер скрытого слоя
            output_dim: Размер выхода
            num_samples: Количество сэмплов для MC Dropout
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        
        # Слои с dropout для байесовского вывода
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входные данные
            training: Режим обучения (влияет на dropout)
            
        Returns:
            torch.Tensor: Предсказания
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Предсказание с оценкой неопределенности
        
        Args:
            x: Входные данные
            
        Returns:
            Tuple[mean, std]: Среднее и стандартное отклонение предсказаний
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.forward(x, training=True)  # Dropout включен
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет оценку аномальности на основе неопределенности
        
        Args:
            x: Входные данные
            
        Returns:
            torch.Tensor: Оценка аномальности
        """
        mean, std = self.predict_with_uncertainty(x)
        
        # Аномальность пропорциональна неопределенности
        anomaly_score = std.squeeze()
        
        return anomaly_score


class AnomalyEnsemble:
    """
    Ансамбль методов детекции аномалий
    """
    
    def __init__(self, input_dim: int = 128, latent_dim: int = 32,
                 hidden_dim: int = 64, device: str = 'cpu'):
        """
        Инициализация ансамбля
        
        Args:
            input_dim: Размер входных представлений
            latent_dim: Размер латентного пространства VAE
            hidden_dim: Размер скрытых слоев
            device: Устройство для вычислений
        """
        self.device = torch.device(device)
        self.input_dim = input_dim
        
        # VAE
        self.vae = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim//2],
            dropout=0.1
        ).to(self.device)
        
        # Bayesian Neural Network
        self.bnn = BayesianNeuralNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_samples=100
        ).to(self.device)
        
        # One-Class SVM
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        
        # Нормализатор для SVM
        self.scaler = StandardScaler()
        
        # Веса для комбинирования методов
        self.vae_weight = 0.4
        self.bnn_weight = 0.3
        self.svm_weight = 0.3
        
        # Пороги для нормализации оценок
        self.vae_threshold = None
        self.bnn_threshold = None
        self.svm_threshold = None
    
    def train_vae(self, train_data: torch.Tensor, epochs: int = 100,
                  learning_rate: float = 1e-3) -> List[float]:
        """
        Обучение VAE
        
        Args:
            train_data: Данные для обучения
            epochs: Количество эпох
            learning_rate: Скорость обучения
            
        Returns:
            List[float]: История потерь
        """
        logger.info("Начинаем обучение VAE")
        
        self.vae.train()
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        loss_history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Случайная перестановка данных
            indices = torch.randperm(len(train_data))
            
            for i in range(0, len(train_data), 32):  # Батчи по 32
                batch_indices = indices[i:i+32]
                batch = train_data[batch_indices].to(self.device)
                
                optimizer.zero_grad()
                
                # Прямой проход
                reconstructed, mu, logvar = self.vae(batch)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, batch, reduction='sum')
                
                # KL divergence
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Общая функция потерь
                loss = recon_loss + 0.1 * kl_div
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            loss_history.append(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Обучение VAE завершено")
        return loss_history
    
    def train_bnn(self, train_data: torch.Tensor, epochs: int = 100,
                  learning_rate: float = 1e-3) -> List[float]:
        """
        Обучение Bayesian Neural Network
        
        Args:
            train_data: Данные для обучения
            epochs: Количество эпох
            learning_rate: Скорость обучения
            
        Returns:
            List[float]: История потерь
        """
        logger.info("Начинаем обучение BNN")
        
        self.bnn.train()
        optimizer = torch.optim.Adam(self.bnn.parameters(), lr=learning_rate)
        
        # Создание фиктивных меток (предполагаем, что все данные нормальные)
        dummy_labels = torch.zeros(len(train_data), 1)
        
        loss_history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Случайная перестановка данных
            indices = torch.randperm(len(train_data))
            
            for i in range(0, len(train_data), 32):
                batch_indices = indices[i:i+32]
                batch = train_data[batch_indices].to(self.device)
                labels = dummy_labels[batch_indices].to(self.device)
                
                optimizer.zero_grad()
                
                # Прямой проход
                outputs = self.bnn(batch, training=True)
                
                # MSE loss (предсказываем 0 для нормальных данных)
                loss = F.mse_loss(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            loss_history.append(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"BNN Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Обучение BNN завершено")
        return loss_history
    
    def train_svm(self, train_data: np.ndarray):
        """
        Обучение One-Class SVM
        
        Args:
            train_data: Данные для обучения
        """
        logger.info("Начинаем обучение One-Class SVM")
        
        # Нормализация данных
        train_data_scaled = self.scaler.fit_transform(train_data)
        
        # Обучение SVM
        self.ocsvm.fit(train_data_scaled)
        
        logger.info("Обучение One-Class SVM завершено")
    
    def train_ensemble(self, train_data: Union[torch.Tensor, np.ndarray],
                      epochs: int = 100, learning_rate: float = 1e-3):
        """
        Обучение всего ансамбля
        
        Args:
            train_data: Данные для обучения
            epochs: Количество эпох
            learning_rate: Скорость обучения
        """
        logger.info("Начинаем обучение ансамбля методов детекции аномалий")
        
        # Преобразование в тензоры если необходимо
        if isinstance(train_data, np.ndarray):
            train_tensor = torch.tensor(train_data, dtype=torch.float32)
        else:
            train_tensor = train_data
        
        # Обучение VAE
        vae_losses = self.train_vae(train_tensor, epochs, learning_rate)
        
        # Обучение BNN
        bnn_losses = self.train_bnn(train_tensor, epochs, learning_rate)
        
        # Обучение SVM
        self.train_svm(train_data if isinstance(train_data, np.ndarray) else train_data.numpy())
        
        # Вычисление порогов на обучающих данных
        self._compute_thresholds(train_tensor)
        
        logger.info("Обучение ансамбля завершено")
        
        return {
            'vae_losses': vae_losses,
            'bnn_losses': bnn_losses
        }
    
    def _compute_thresholds(self, train_data: torch.Tensor):
        """
        Вычисляет пороги для нормализации оценок аномальности
        
        Args:
            train_data: Обучающие данные
        """
        logger.info("Вычисляем пороги для нормализации оценок")
        
        # VAE оценки
        vae_scores = self.vae.compute_anomaly_score(train_data.to(self.device))
        self.vae_threshold = torch.quantile(vae_scores, 0.95).item()
        
        # BNN оценки
        bnn_scores = self.bnn.compute_anomaly_score(train_data.to(self.device))
        self.bnn_threshold = torch.quantile(bnn_scores, 0.95).item()
        
        # SVM оценки
        train_data_scaled = self.scaler.transform(train_data.numpy())
        svm_scores = -self.ocsvm.decision_function(train_data_scaled)  # Отрицательные расстояния
        self.svm_threshold = np.percentile(svm_scores, 95)
        
        logger.info(f"Пороги: VAE={self.vae_threshold:.4f}, BNN={self.bnn_threshold:.4f}, SVM={self.svm_threshold:.4f}")
    
    def predict_anomaly_scores(self, data: Union[torch.Tensor, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Предсказывает оценки аномальности для всех методов
        
        Args:
            data: Данные для предсказания
            
        Returns:
            Dict[str, np.ndarray]: Оценки аномальности для каждого метода
        """
        # Преобразование в тензоры если необходимо
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32)
        else:
            data_tensor = data
        
        scores = {}
        
        # VAE оценки
        self.vae.eval()
        with torch.no_grad():
            vae_scores = self.vae.compute_anomaly_score(data_tensor.to(self.device))
            scores['vae'] = vae_scores.cpu().numpy()
        
        # BNN оценки
        self.bnn.eval()
        with torch.no_grad():
            bnn_scores = self.bnn.compute_anomaly_score(data_tensor.to(self.device))
            scores['bnn'] = bnn_scores.cpu().numpy()
        
        # SVM оценки
        data_scaled = self.scaler.transform(data_tensor.numpy())
        svm_scores = -self.ocsvm.decision_function(data_scaled)
        scores['svm'] = svm_scores
        
        return scores
    
    def predict_combined_anomaly_score(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Предсказывает комбинированную оценку аномальности
        
        Args:
            data: Данные для предсказания
            
        Returns:
            np.ndarray: Комбинированные оценки аномальности
        """
        scores = self.predict_anomaly_scores(data)
        
        # Нормализация оценок
        vae_norm = np.clip(scores['vae'] / self.vae_threshold, 0, 1)
        bnn_norm = np.clip(scores['bnn'] / self.bnn_threshold, 0, 1)
        svm_norm = np.clip(scores['svm'] / self.svm_threshold, 0, 1)
        
        # Взвешенная комбинация
        combined_scores = (self.vae_weight * vae_norm + 
                          self.bnn_weight * bnn_norm + 
                          self.svm_weight * svm_norm)
        
        return combined_scores
    
    def predict_anomaly_probability(self, data: Union[torch.Tensor, np.ndarray],
                                   threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает вероятности аномальности
        
        Args:
            data: Данные для предсказания
            threshold: Порог для классификации
            
        Returns:
            Tuple[probabilities, predictions]: Вероятности и бинарные предсказания
        """
        combined_scores = self.predict_combined_anomaly_score(data)
        
        # Преобразование в вероятности (сигмоида)
        probabilities = 1 / (1 + np.exp(-5 * (combined_scores - threshold)))
        
        # Бинарные предсказания
        predictions = (probabilities > threshold).astype(int)
        
        return probabilities, predictions
    
    def evaluate_ensemble(self, test_data: Union[torch.Tensor, np.ndarray],
                         test_labels: np.ndarray) -> Dict[str, float]:
        """
        Оценивает качество ансамбля
        
        Args:
            test_data: Тестовые данные
            test_labels: Истинные метки (0 - нормальные, 1 - аномальные)
            
        Returns:
            Dict[str, float]: Метрики качества
        """
        logger.info("Оцениваем качество ансамбля")
        
        # Получение оценок
        scores = self.predict_anomaly_scores(test_data)
        combined_scores = self.predict_combined_anomaly_score(test_data)
        
        metrics = {}
        
        # Оценка каждого метода отдельно
        for method, method_scores in scores.items():
            try:
                auc = roc_auc_score(test_labels, method_scores)
                metrics[f'{method}_auc'] = auc
            except ValueError:
                metrics[f'{method}_auc'] = 0.5
        
        # Оценка комбинированного метода
        try:
            combined_auc = roc_auc_score(test_labels, combined_scores)
            metrics['combined_auc'] = combined_auc
        except ValueError:
            metrics['combined_auc'] = 0.5
        
        # Предсказания с разными порогами
        probabilities, predictions = self.predict_anomaly_probability(test_data, threshold=0.5)
        
        # Точность
        accuracy = np.mean(predictions == test_labels)
        metrics['accuracy'] = accuracy
        
        # Precision и Recall
        tp = np.sum((predictions == 1) & (test_labels == 1))
        fp = np.sum((predictions == 1) & (test_labels == 0))
        fn = np.sum((predictions == 0) & (test_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        logger.info(f"Результаты оценки: AUC={metrics['combined_auc']:.3f}, "
                   f"Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
        
        return metrics
    
    def save_ensemble(self, filepath: str):
        """Сохраняет обученный ансамбль"""
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'bnn_state_dict': self.bnn.state_dict(),
            'ocsvm': self.ocsvm,
            'scaler': self.scaler,
            'thresholds': {
                'vae': self.vae_threshold,
                'bnn': self.bnn_threshold,
                'svm': self.svm_threshold
            },
            'weights': {
                'vae': self.vae_weight,
                'bnn': self.bnn_weight,
                'svm': self.svm_weight
            }
        }, filepath)
        logger.info(f"Ансамбль сохранен: {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Загружает обученный ансамбль"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.bnn.load_state_dict(checkpoint['bnn_state_dict'])
        self.ocsvm = checkpoint['ocsvm']
        self.scaler = checkpoint['scaler']
        
        self.vae_threshold = checkpoint['thresholds']['vae']
        self.bnn_threshold = checkpoint['thresholds']['bnn']
        self.svm_threshold = checkpoint['thresholds']['svm']
        
        self.vae_weight = checkpoint['weights']['vae']
        self.bnn_weight = checkpoint['weights']['bnn']
        self.svm_weight = checkpoint['weights']['svm']
        
        logger.info(f"Ансамбль загружен: {filepath}")


def create_anomaly_dataset(normal_data: np.ndarray, anomaly_data: np.ndarray = None,
                          anomaly_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает датасет для обучения детекции аномалий
    
    Args:
        normal_data: Нормальные данные
        anomaly_data: Аномальные данные (опционально)
        anomaly_ratio: Доля аномальных данных для генерации
        
    Returns:
        Tuple[data, labels]: Данные и метки
    """
    if anomaly_data is None:
        # Генерация синтетических аномалий
        n_anomalies = int(len(normal_data) * anomaly_ratio)
        
        # Создание аномалий путем добавления шума и искажений
        anomaly_indices = np.random.choice(len(normal_data), n_anomalies, replace=False)
        anomaly_data = normal_data[anomaly_indices].copy()
        
        # Добавление различных типов аномалий
        for i in range(n_anomalies):
            anomaly_type = np.random.choice(['noise', 'shift', 'scale'])
            
            if anomaly_type == 'noise':
                anomaly_data[i] += np.random.normal(0, 0.1, anomaly_data[i].shape)
            elif anomaly_type == 'shift':
                anomaly_data[i] += np.random.uniform(-0.2, 0.2)
            elif anomaly_type == 'scale':
                anomaly_data[i] *= np.random.uniform(0.5, 1.5)
    
    # Объединение данных
    all_data = np.vstack([normal_data, anomaly_data])
    labels = np.concatenate([
        np.zeros(len(normal_data)),  # 0 - нормальные
        np.ones(len(anomaly_data))   # 1 - аномальные
    ])
    
    # Перемешивание
    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    labels = labels[indices]
    
    return all_data, labels


if __name__ == "__main__":
    # Пример использования
    logger.info("Тестирование ансамбля детекции аномалий")
    
    # Создание тестовых данных
    np.random.seed(42)
    
    # Нормальные данные (представления кривых блеска)
    normal_data = np.random.randn(500, 128)
    
    # Аномальные данные
    anomaly_data = np.random.randn(50, 128) + 2.0  # Сдвинутые данные
    
    # Создание датасета
    train_data, train_labels = create_anomaly_dataset(normal_data[:400], anomaly_data[:30])
    test_data, test_labels = create_anomaly_dataset(normal_data[400:], anomaly_data[30:])
    
    # Инициализация ансамбля
    ensemble = AnomalyEnsemble(input_dim=128, latent_dim=32, hidden_dim=64)
    
    # Обучение
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    ensemble.train_ensemble(train_tensor, epochs=50)
    
    # Оценка
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    metrics = ensemble.evaluate_ensemble(test_tensor, test_labels)
    
    # Вывод результатов
    print("Результаты оценки ансамбля:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Сохранение модели
    ensemble.save_ensemble('anomaly_ensemble.pth')
    
    logger.info("Тест завершен")
