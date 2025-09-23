"""
Embedding Manager for Transit Detection

Система управления embeddings для кэширования результатов анализа кривых блеска.
Позволяет избежать повторных вычислений для уже проанализированных звезд.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using sklearn for similarity search")
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRecord:
    """Запись embedding с метаданными"""
    target_name: str
    embedding: np.ndarray
    prediction_result: Any
    created_at: datetime
    model_version: str
    data_hash: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        return {
            'target_name': self.target_name,
            'embedding': self.embedding.tolist(),
            'prediction_result': asdict(self.prediction_result) if hasattr(self.prediction_result, '__dict__') else self.prediction_result,
            'created_at': self.created_at.isoformat(),
            'model_version': self.model_version,
            'data_hash': self.data_hash,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingRecord':
        """Создание из словаря"""
        return cls(
            target_name=data['target_name'],
            embedding=np.array(data['embedding']),
            prediction_result=data['prediction_result'],
            created_at=datetime.fromisoformat(data['created_at']),
            model_version=data['model_version'],
            data_hash=data['data_hash'],
            confidence=data['confidence']
        )

class EmbeddingManager:
    """
    Менеджер embeddings для кэширования и поиска похожих кривых блеска
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 similarity_threshold: float = 0.95,
                 max_cache_size: int = 10000,
                 cache_dir: str = 'data/embeddings',
                 use_faiss: bool = True):
        
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.cache_dir = Path(cache_dir)
        self.use_faiss = use_faiss
        
        # Создаем директорию
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Хранилище embeddings
        self.embeddings: Dict[str, EmbeddingRecord] = {}
        self.embedding_matrix: Optional[np.ndarray] = None
        self.target_names: List[str] = []
        
        # FAISS индекс для быстрого поиска
        if use_faiss and FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product для cosine similarity
        else:
            self.faiss_index = None
        
        # Загружаем существующий кэш
        self._load_cache()
        
    def cache_prediction(self,
                        target_name: str,
                        embedding: np.ndarray,
                        prediction_result: Any,
                        model_version: str = "1.0.0",
                        data_hash: Optional[str] = None) -> None:
        """
        Кэширование результата предсказания с embedding
        
        Args:
            target_name: Имя цели
            embedding: Вектор признаков
            prediction_result: Результат предсказания
            model_version: Версия модели
            data_hash: Хэш исходных данных
        """
        # Нормализуем embedding для cosine similarity
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        # Создаем запись
        record = EmbeddingRecord(
            target_name=target_name,
            embedding=embedding_norm,
            prediction_result=prediction_result,
            created_at=datetime.now(),
            model_version=model_version,
            data_hash=data_hash or self._compute_hash(embedding),
            confidence=getattr(prediction_result, 'confidence', 0.0)
        )
        
        # Добавляем в кэш
        self.embeddings[target_name] = record
        self.target_names.append(target_name)
        
        # Обновляем FAISS индекс
        if self.faiss_index is not None:
            self.faiss_index.add(embedding_norm.reshape(1, -1))
        
        # Обновляем матрицу embeddings
        self._update_embedding_matrix()
        
        # Проверяем размер кэша
        if len(self.embeddings) > self.max_cache_size:
            self._cleanup_cache()
        
        # Сохраняем кэш
        self._save_cache()
        
        logger.info(f"Cached prediction for {target_name}")
    
    def get_cached_prediction(self, target_name: str) -> Optional[Any]:
        """
        Получение кэшированного предсказания
        
        Args:
            target_name: Имя цели
            
        Returns:
            Кэшированный результат или None
        """
        if target_name in self.embeddings:
            record = self.embeddings[target_name]
            logger.info(f"Found cached prediction for {target_name}")
            return record.prediction_result
        
        return None
    
    def find_similar_targets(self,
                           embedding: np.ndarray,
                           top_k: int = 5,
                           min_similarity: float = None) -> List[Tuple[str, float, Any]]:
        """
        Поиск похожих целей по embedding
        
        Args:
            embedding: Вектор признаков для поиска
            top_k: Количество похожих целей
            min_similarity: Минимальная схожесть
            
        Returns:
            Список (target_name, similarity, prediction_result)
        """
        if len(self.embeddings) == 0:
            return []
        
        min_similarity = min_similarity or self.similarity_threshold
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        similar_targets = []
        
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            # Используем FAISS для быстрого поиска
            similarities, indices = self.faiss_index.search(
                embedding_norm.reshape(1, -1), 
                min(top_k, self.faiss_index.ntotal)
            )
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= min_similarity:
                    target_name = self.target_names[idx]
                    record = self.embeddings[target_name]
                    similar_targets.append((target_name, float(similarity), record.prediction_result))
        
        else:
            # Используем sklearn для поиска
            if self.embedding_matrix is not None:
                similarities = cosine_similarity(
                    embedding_norm.reshape(1, -1),
                    self.embedding_matrix
                )[0]
                
                # Сортируем по убыванию схожести
                sorted_indices = np.argsort(similarities)[::-1]
                
                for idx in sorted_indices[:top_k]:
                    similarity = similarities[idx]
                    if similarity >= min_similarity:
                        target_name = self.target_names[idx]
                        record = self.embeddings[target_name]
                        similar_targets.append((target_name, similarity, record.prediction_result))
        
        logger.info(f"Found {len(similar_targets)} similar targets")
        return similar_targets
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Получение статистики по embeddings"""
        if len(self.embeddings) == 0:
            return {'total_embeddings': 0}
        
        confidences = [record.confidence for record in self.embeddings.values()]
        model_versions = [record.model_version for record in self.embeddings.values()]
        
        # Кластеризация embeddings
        clusters_info = {}
        if self.embedding_matrix is not None and len(self.embeddings) > 5:
            try:
                n_clusters = min(5, len(self.embeddings) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(self.embedding_matrix)
                
                clusters_info = {
                    'n_clusters': n_clusters,
                    'cluster_sizes': np.bincount(cluster_labels).tolist(),
                    'inertia': float(kmeans.inertia_)
                }
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
        
        return {
            'total_embeddings': len(self.embeddings),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'model_versions': list(set(model_versions)),
            'embedding_dim': self.embedding_dim,
            'cache_size_mb': self._get_cache_size_mb(),
            'clusters': clusters_info
        }
    
    def cleanup_old_embeddings(self, days_threshold: int = 30) -> int:
        """
        Очистка старых embeddings
        
        Args:
            days_threshold: Количество дней для хранения
            
        Returns:
            Количество удаленных записей
        """
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        targets_to_remove = []
        for target_name, record in self.embeddings.items():
            if record.created_at < cutoff_date:
                targets_to_remove.append(target_name)
        
        # Удаляем старые записи
        for target_name in targets_to_remove:
            del self.embeddings[target_name]
            if target_name in self.target_names:
                self.target_names.remove(target_name)
        
        # Пересоздаем индексы
        self._rebuild_indices()
        
        logger.info(f"Removed {len(targets_to_remove)} old embeddings")
        return len(targets_to_remove)
    
    def export_embeddings(self, export_path: str, format: str = 'json') -> None:
        """
        Экспорт embeddings в файл
        
        Args:
            export_path: Путь для экспорта
            format: Формат файла ('json', 'pickle', 'npz')
        """
        export_path = Path(export_path)
        
        if format == 'json':
            data = {
                target_name: record.to_dict()
                for target_name, record in self.embeddings.items()
            }
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'pickle':
            with open(export_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
        
        elif format == 'npz':
            if self.embedding_matrix is not None:
                np.savez_compressed(
                    export_path,
                    embeddings=self.embedding_matrix,
                    target_names=self.target_names,
                    confidences=[record.confidence for record in self.embeddings.values()]
                )
        
        logger.info(f"Exported embeddings to {export_path}")
    
    def import_embeddings(self, import_path: str, format: str = 'json') -> int:
        """
        Импорт embeddings из файла
        
        Args:
            import_path: Путь к файлу
            format: Формат файла
            
        Returns:
            Количество импортированных записей
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"File not found: {import_path}")
        
        imported_count = 0
        
        if format == 'json':
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            for target_name, record_dict in data.items():
                record = EmbeddingRecord.from_dict(record_dict)
                self.embeddings[target_name] = record
                if target_name not in self.target_names:
                    self.target_names.append(target_name)
                imported_count += 1
        
        elif format == 'pickle':
            with open(import_path, 'rb') as f:
                imported_embeddings = pickle.load(f)
            
            for target_name, record in imported_embeddings.items():
                self.embeddings[target_name] = record
                if target_name not in self.target_names:
                    self.target_names.append(target_name)
                imported_count += 1
        
        # Пересоздаем индексы
        self._rebuild_indices()
        
        logger.info(f"Imported {imported_count} embeddings")
        return imported_count
    
    def _compute_hash(self, data: np.ndarray) -> str:
        """Вычисление хэша данных"""
        return hashlib.md5(data.tobytes()).hexdigest()
    
    def _update_embedding_matrix(self):
        """Обновление матрицы embeddings"""
        if len(self.embeddings) > 0:
            embeddings_list = []
            for target_name in self.target_names:
                if target_name in self.embeddings:
                    embeddings_list.append(self.embeddings[target_name].embedding)
            
            if embeddings_list:
                self.embedding_matrix = np.vstack(embeddings_list)
    
    def _cleanup_cache(self):
        """Очистка кэша при превышении размера"""
        # Удаляем записи с наименьшей уверенностью
        sorted_records = sorted(
            self.embeddings.items(),
            key=lambda x: x[1].confidence
        )
        
        # Удаляем 10% записей с наименьшей уверенностью
        to_remove = int(len(self.embeddings) * 0.1)
        
        for i in range(to_remove):
            target_name = sorted_records[i][0]
            del self.embeddings[target_name]
            if target_name in self.target_names:
                self.target_names.remove(target_name)
        
        # Пересоздаем индексы
        self._rebuild_indices()
        
        logger.info(f"Cleaned up {to_remove} cache entries")
    
    def _rebuild_indices(self):
        """Пересоздание индексов"""
        # Пересоздаем FAISS индекс
        if self.use_faiss and FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            
            if len(self.embeddings) > 0:
                embeddings_array = np.vstack([
                    self.embeddings[name].embedding 
                    for name in self.target_names 
                    if name in self.embeddings
                ])
                self.faiss_index.add(embeddings_array)
        
        # Обновляем матрицу
        self._update_embedding_matrix()
    
    def _save_cache(self):
        """Сохранение кэша на диск"""
        cache_file = self.cache_dir / 'embeddings_cache.pkl'
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'target_names': self.target_names,
                    'embedding_dim': self.embedding_dim
                }, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """Загрузка кэша с диска"""
        cache_file = self.cache_dir / 'embeddings_cache.pkl'
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.embeddings = cache_data.get('embeddings', {})
                self.target_names = cache_data.get('target_names', [])
                
                # Пересоздаем индексы
                self._rebuild_indices()
                
                logger.info(f"Loaded {len(self.embeddings)} cached embeddings")
                
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.embeddings = {}
                self.target_names = []
    
    def _get_cache_size_mb(self) -> float:
        """Получение размера кэша в МБ"""
        cache_file = self.cache_dir / 'embeddings_cache.pkl'
        if cache_file.exists():
            return cache_file.stat().st_size / (1024 * 1024)
        return 0.0
