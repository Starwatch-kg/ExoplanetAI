"""
Self-Supervised Representation Layer с контрастивным энкодером
Реализует SimCLR-подобный подход для временных рядов (кривых блеска)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from torch.utils.data import DataLoader, TensorDataset
import random

logger = logging.getLogger(__name__)

class TimeSeriesAugmentation:
    """
    Класс для аугментации временных рядов (кривых блеска)
    Создает позитивные пары для контрастивного обучения
    """
    
    def __init__(self, noise_std: float = 0.01, 
                 time_warp_strength: float = 0.1,
                 magnitude_scaling: float = 0.1,
                 dropout_rate: float = 0.1):
        """
        Инициализация аугментации
        
        Args:
            noise_std: Стандартное отклонение гауссовского шума
            time_warp_strength: Сила временного искажения
            magnitude_scaling: Сила масштабирования амплитуды
            dropout_rate: Вероятность dropout точек
        """
        self.noise_std = noise_std
        self.time_warp_strength = time_warp_strength
        self.magnitude_scaling = magnitude_scaling
        self.dropout_rate = dropout_rate
    
    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Добавляет гауссовский шум"""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет временное искажение"""
        seq_len = x.size(-1)
        
        # Создание случайной функции искажения времени
        warp_points = torch.linspace(0, 1, 5)
        warp_values = torch.randn(5) * self.time_warp_strength
        
        # Интерполяция для создания плавного искажения
        warp_function = torch.zeros(seq_len)
        for i in range(seq_len):
            t = i / (seq_len - 1)
            # Простая линейная интерполяция
            for j in range(len(warp_points) - 1):
                if warp_points[j] <= t <= warp_points[j + 1]:
                    alpha = (t - warp_points[j]) / (warp_points[j + 1] - warp_points[j])
                    warp_function[i] = warp_values[j] + alpha * (warp_values[j + 1] - warp_values[j])
                    break
        
        # Применение искажения
        indices = torch.arange(seq_len, dtype=torch.float32) + warp_function
        indices = torch.clamp(indices, 0, seq_len - 1)
        
        # Интерполяция для получения искаженного сигнала
        warped_x = torch.zeros_like(x)
        for i in range(seq_len):
            idx = int(indices[i])
            if idx < seq_len - 1:
                alpha = indices[i] - idx
                warped_x[..., i] = (1 - alpha) * x[..., idx] + alpha * x[..., idx + 1]
            else:
                warped_x[..., i] = x[..., idx]
        
        return warped_x
    
    def magnitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет масштабирование амплитуды"""
        scale_factor = 1 + torch.randn(1) * self.magnitude_scaling
        return x * scale_factor
    
    def random_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Случайно обнуляет некоторые точки"""
        mask = torch.rand_like(x) > self.dropout_rate
        return x * mask.float()
    
    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет случайную аугментацию"""
        # Выбор случайной комбинации аугментаций
        augmentations = [
            self.add_gaussian_noise,
            self.time_warp,
            self.magnitude_scaling,
            self.random_dropout
        ]
        
        # Применяем 2-3 случайные аугментации
        num_augmentations = random.randint(2, 3)
        selected_augmentations = random.sample(augmentations, num_augmentations)
        
        augmented_x = x.clone()
        for aug_func in selected_augmentations:
            if callable(aug_func):
                augmented_x = aug_func(augmented_x)
        
        return augmented_x
    
    def create_positive_pair(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Создает позитивную пару для контрастивного обучения"""
        x1 = self.apply_augmentation(x)
        x2 = self.apply_augmentation(x)
        return x1, x2


class ContrastiveEncoder(nn.Module):
    """
    Контрастивный энкодер для кривых блеска
    Создает 128-мерные представления, устойчивые к шуму и вариабельности звезды
    """
    
    def __init__(self, input_length: int = 2000, 
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """
        Инициализация контрастивного энкодера
        
        Args:
            input_length: Длина входной последовательности
            embedding_dim: Размер выходного представления
            hidden_dim: Размер скрытых слоев
            num_layers: Количество слоев энкодера
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        
        # CNN backbone для извлечения локальных признаков
        self.cnn_backbone = nn.Sequential(
            # Первый блок
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Второй блок
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Третий блок
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Четвертый блок
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Глобальное усреднение
        )
        
        # Вычисление размера после CNN
        cnn_output_size = 512
        
        # Transformer для анализа долгосрочных зависимостей
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cnn_output_size,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Проекционная головка для контрастивного обучения
        self.projection_head = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Нормализация представлений
        self.normalize = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через энкодер
        
        Args:
            x: Входные данные (batch_size, 1, sequence_length)
            
        Returns:
            torch.Tensor: Нормализованные представления (batch_size, embedding_dim)
        """
        # CNN извлечение признаков
        cnn_features = self.cnn_backbone(x)  # (batch_size, 512, 1)
        cnn_features = cnn_features.squeeze(-1)  # (batch_size, 512)
        
        # Transformer обработка
        # Добавляем размерность последовательности для transformer
        transformer_input = cnn_features.unsqueeze(1)  # (batch_size, 1, 512)
        transformer_output = self.transformer(transformer_input)  # (batch_size, 1, 512)
        transformer_output = transformer_output.squeeze(1)  # (batch_size, 512)
        
        # Проекция в пространство представлений
        embeddings = self.projection_head(transformer_output)
        
        # Нормализация
        normalized_embeddings = self.normalize(embeddings)
        
        return normalized_embeddings
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Кодирование без нормализации (для инференса)"""
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.squeeze(-1)
        
        transformer_input = cnn_features.unsqueeze(1)
        transformer_output = self.transformer(transformer_input)
        transformer_output = transformer_output.squeeze(1)
        
        embeddings = self.projection_head(transformer_output)
        return embeddings


class ContrastiveLoss(nn.Module):
    """
    Контрастивная функция потерь (SimCLR-style)
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Инициализация контрастивной функции потерь
        
        Args:
            temperature: Температурный параметр для softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет контрастивную функцию потерь
        
        Args:
            z1: Представления первого представления
            z2: Представления второго представления
            
        Returns:
            torch.Tensor: Значение функции потерь
        """
        batch_size = z1.size(0)
        
        # Объединяем представления
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, embedding_dim)
        
        # Нормализация
        z = F.normalize(z, dim=1)
        
        # Вычисление матрицы сходства
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Маска для позитивных пар
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2*batch_size,)
        
        # Удаляем диагональные элементы (самосходство)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Вычисление потерь
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class SelfSupervisedRepresentationLearner:
    """
    Основной класс для обучения представлений с самоконтролем
    """
    
    def __init__(self, input_length: int = 2000,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 temperature: float = 0.1):
        """
        Инициализация обучателя представлений
        
        Args:
            input_length: Длина входной последовательности
            embedding_dim: Размер представления
            hidden_dim: Размер скрытых слоев
            num_layers: Количество слоев
            dropout: Коэффициент dropout
            temperature: Температурный параметр
        """
        self.encoder = ContrastiveEncoder(
            input_length=input_length,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.augmentation = TimeSeriesAugmentation()
        self.criterion = ContrastiveLoss(temperature=temperature)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """
        Обучение на одной эпохе
        
        Args:
            dataloader: DataLoader с данными
            optimizer: Оптимизатор
            
        Returns:
            float: Средняя функция потерь
        """
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0]  # Предполагаем, что первый элемент - данные
            else:
                x = batch_data
            
            x = x.to(self.device)
            
            # Создание позитивных пар
            x1, x2 = self.augmentation.create_positive_pair(x)
            
            # Получение представлений
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            
            # Вычисление потерь
            loss = self.criterion(z1, z2)
            
            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, dataloader: DataLoader, epochs: int = 100, 
              learning_rate: float = 1e-3, weight_decay: float = 1e-4) -> List[float]:
        """
        Полное обучение представлений
        
        Args:
            dataloader: DataLoader с данными
            epochs: Количество эпох
            learning_rate: Скорость обучения
            weight_decay: Регуляризация весов
            
        Returns:
            List[float]: История потерь
        """
        logger.info(f"Начинаем обучение представлений на {epochs} эпох")
        
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        loss_history = []
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, optimizer)
            scheduler.step()
            
            loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Обучение представлений завершено")
        return loss_history
    
    def encode_dataset(self, dataloader: DataLoader) -> Tuple[np.ndarray, List]:
        """
        Кодирует весь датасет в представления
        
        Args:
            dataloader: DataLoader с данными
            
        Returns:
            Tuple[np.ndarray, List]: Представления и метаданные
        """
        self.encoder.eval()
        embeddings = []
        metadata = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                    meta = batch_data[1] if len(batch_data) > 1 else None
                else:
                    x = batch_data
                    meta = None
                
                x = x.to(self.device)
                
                # Получение представлений
                batch_embeddings = self.encoder.encode(x)
                embeddings.append(batch_embeddings.cpu().numpy())
                
                if meta is not None:
                    metadata.extend(meta)
        
        embeddings = np.vstack(embeddings)
        return embeddings, metadata
    
    def save_model(self, filepath: str):
        """Сохраняет обученную модель"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'model_config': {
                'input_length': self.encoder.input_length,
                'embedding_dim': self.encoder.embedding_dim
            }
        }, filepath)
        logger.info(f"Модель сохранена: {filepath}")
    
    def load_model(self, filepath: str):
        """Загружает обученную модель"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        logger.info(f"Модель загружена: {filepath}")


class RepresentationAnalyzer:
    """
    Анализатор представлений для понимания структуры данных
    """
    
    def __init__(self, embeddings: np.ndarray, metadata: List = None):
        """
        Инициализация анализатора
        
        Args:
            embeddings: Матрица представлений
            metadata: Метаданные для каждого представления
        """
        self.embeddings = embeddings
        self.metadata = metadata
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """Вычисляет матрицу сходства между представлениями"""
        # Нормализация представлений
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Матрица сходства (косинусное сходство)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def find_similar_samples(self, query_idx: int, top_k: int = 5) -> List[int]:
        """
        Находит наиболее похожие образцы
        
        Args:
            query_idx: Индекс запроса
            top_k: Количество похожих образцов
            
        Returns:
            List[int]: Индексы похожих образцов
        """
        similarity_matrix = self.compute_similarity_matrix()
        
        # Получение сходств для запроса
        query_similarities = similarity_matrix[query_idx]
        
        # Исключаем сам запрос
        query_similarities[query_idx] = -1
        
        # Находим топ-k наиболее похожих
        similar_indices = np.argsort(query_similarities)[-top_k:][::-1]
        
        return similar_indices.tolist()
    
    def cluster_analysis(self, n_clusters: int = 10) -> Dict:
        """
        Кластерный анализ представлений
        
        Args:
            n_clusters: Количество кластеров
            
        Returns:
            Dict: Результаты кластеризации
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # PCA для визуализации
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        return {
            'cluster_labels': cluster_labels,
            'embeddings_2d': embeddings_2d,
            'cluster_centers': kmeans.cluster_centers_,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }


def create_representation_dataset(lightcurves: List[np.ndarray], 
                                batch_size: int = 32) -> DataLoader:
    """
    Создает DataLoader для обучения представлений
    
    Args:
        lightcurves: Список кривых блеска
        batch_size: Размер батча
        
    Returns:
        DataLoader: DataLoader для обучения
    """
    # Преобразование в тензоры
    tensors = []
    for lc in lightcurves:
        # Нормализация
        lc_norm = (lc - np.mean(lc)) / (np.std(lc) + 1e-8)
        tensor = torch.tensor(lc_norm, dtype=torch.float32).unsqueeze(0)  # (1, length)
        tensors.append(tensor)
    
    # Создание датасета
    dataset = TensorDataset(torch.stack(tensors))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


if __name__ == "__main__":
    # Пример использования
    logger.info("Тестирование Self-Supervised Representation Learning")
    
    # Создание тестовых данных
    np.random.seed(42)
    lightcurves = []
    
    for i in range(100):
        # Создание синтетической кривой блеска
        times = np.linspace(0, 30, 2000)
        flux = np.ones_like(times) + 0.01 * np.random.randn(len(times))
        
        # Добавление периодических вариаций
        if i % 3 == 0:  # Каждая третья кривая имеет транзит
            period = np.random.uniform(5, 15)
            depth = np.random.uniform(0.01, 0.05)
            for t in times:
                phase = (t % period) / period
                if 0.45 <= phase <= 0.55:
                    flux[int(t * len(times) / 30)] -= depth
        
        lightcurves.append(flux)
    
    # Создание DataLoader
    dataloader = create_representation_dataset(lightcurves, batch_size=16)
    
    # Инициализация обучателя представлений
    learner = SelfSupervisedRepresentationLearner(
        input_length=2000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=4
    )
    
    # Обучение
    loss_history = learner.train(dataloader, epochs=50)
    
    # Кодирование датасета
    embeddings, metadata = learner.encode_dataset(dataloader)
    
    # Анализ представлений
    analyzer = RepresentationAnalyzer(embeddings)
    cluster_results = analyzer.cluster_analysis(n_clusters=5)
    
    logger.info(f"Обучение завершено. Размер представлений: {embeddings.shape}")
    logger.info(f"Количество кластеров: {len(np.unique(cluster_results['cluster_labels']))}")
    
    # Сохранение модели
    learner.save_model('contrastive_encoder.pth')
    
    logger.info("Тест завершен")
