"""
Database Manager for AI Transit Detection

Система управления базой данных PostgreSQL для хранения:
- Результатов анализа
- Embeddings
- Истории обучения
- Пользовательской обратной связи
"""

import asyncio
import asyncpg
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Результат анализа для БД"""
    target_name: str
    analysis_timestamp: datetime
    model_version: str
    is_transit: bool
    confidence: float
    transit_probability: float
    physical_parameters: Dict[str, Any]
    bls_parameters: Dict[str, Any]
    user_feedback: Optional[str] = None
    verified: Optional[bool] = None
    notes: Optional[str] = None

@dataclass
class UserFeedback:
    """Пользовательская обратная связь"""
    target_name: str
    user_id: str
    feedback_type: str  # 'correct', 'incorrect', 'uncertain'
    confidence_rating: int  # 1-5
    comments: Optional[str]
    timestamp: datetime

class DatabaseManager:
    """
    Менеджер базы данных для AI модуля
    """
    
    def __init__(self,
                 database_url: str = None,
                 max_connections: int = 10):
        
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://user:password@localhost/exoplanet_ai'
        )
        self.max_connections = max_connections
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Инициализация подключения к БД"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=self.max_connections
            )
            
            # Создаем таблицы если их нет
            await self._create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Закрытие подключения"""
        if self.pool:
            await self.pool.close()
    
    async def _create_tables(self):
        """Создание таблиц"""
        
        create_tables_sql = """
        -- Таблица результатов анализа
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            target_name VARCHAR(100) NOT NULL,
            analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            model_version VARCHAR(50) NOT NULL,
            is_transit BOOLEAN NOT NULL,
            confidence FLOAT NOT NULL,
            transit_probability FLOAT NOT NULL,
            physical_parameters JSONB,
            bls_parameters JSONB,
            user_feedback TEXT,
            verified BOOLEAN,
            notes TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Таблица embeddings
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            target_name VARCHAR(100) NOT NULL,
            embedding FLOAT[] NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            data_hash VARCHAR(64),
            confidence FLOAT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Таблица пользовательской обратной связи
        CREATE TABLE IF NOT EXISTS user_feedback (
            id SERIAL PRIMARY KEY,
            target_name VARCHAR(100) NOT NULL,
            user_id VARCHAR(100) NOT NULL,
            feedback_type VARCHAR(20) NOT NULL,
            confidence_rating INTEGER CHECK (confidence_rating >= 1 AND confidence_rating <= 5),
            comments TEXT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Таблица истории обучения
        CREATE TABLE IF NOT EXISTS training_history (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            dataset_name VARCHAR(100) NOT NULL,
            training_params JSONB,
            metrics JSONB,
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            status VARCHAR(20) DEFAULT 'running'
        );
        
        -- Таблица известных экзопланет для сравнения
        CREATE TABLE IF NOT EXISTS known_exoplanets (
            id SERIAL PRIMARY KEY,
            planet_name VARCHAR(100) NOT NULL,
            host_star VARCHAR(100) NOT NULL,
            discovery_method VARCHAR(50),
            orbital_period FLOAT,
            planet_radius FLOAT,
            equilibrium_temperature FLOAT,
            discovery_year INTEGER,
            confirmed BOOLEAN DEFAULT TRUE,
            source VARCHAR(100),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Индексы для производительности
        CREATE INDEX IF NOT EXISTS idx_analysis_target_name ON analysis_results(target_name);
        CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_results(analysis_timestamp);
        CREATE INDEX IF NOT EXISTS idx_embeddings_target_name ON embeddings(target_name);
        CREATE INDEX IF NOT EXISTS idx_feedback_target_name ON user_feedback(target_name);
        CREATE INDEX IF NOT EXISTS idx_known_planets_host_star ON known_exoplanets(host_star);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_tables_sql)
    
    async def save_analysis_result(self, result: AnalysisResult) -> int:
        """
        Сохранение результата анализа
        
        Returns:
            ID созданной записи
        """
        sql = """
        INSERT INTO analysis_results (
            target_name, analysis_timestamp, model_version, is_transit,
            confidence, transit_probability, physical_parameters, bls_parameters,
            user_feedback, verified, notes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                sql,
                result.target_name,
                result.analysis_timestamp,
                result.model_version,
                result.is_transit,
                result.confidence,
                result.transit_probability,
                json.dumps(result.physical_parameters),
                json.dumps(result.bls_parameters),
                result.user_feedback,
                result.verified,
                result.notes
            )
        
        logger.info(f"Saved analysis result for {result.target_name}, ID: {record_id}")
        return record_id
    
    async def get_analysis_history(self, 
                                  target_name: Optional[str] = None,
                                  limit: int = 100,
                                  days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Получение истории анализа
        
        Args:
            target_name: Фильтр по имени цели
            limit: Максимальное количество записей
            days_back: Количество дней назад
            
        Returns:
            Список результатов анализа
        """
        where_conditions = ["analysis_timestamp >= $1"]
        params = [datetime.now() - timedelta(days=days_back)]
        
        if target_name:
            where_conditions.append("target_name = $2")
            params.append(target_name)
        
        sql = f"""
        SELECT * FROM analysis_results 
        WHERE {' AND '.join(where_conditions)}
        ORDER BY analysis_timestamp DESC 
        LIMIT {limit}
        """
        
        async with self.pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        
        results = []
        for record in records:
            result_dict = dict(record)
            # Парсим JSON поля
            result_dict['physical_parameters'] = json.loads(result_dict['physical_parameters'] or '{}')
            result_dict['bls_parameters'] = json.loads(result_dict['bls_parameters'] or '{}')
            results.append(result_dict)
        
        return results
    
    async def save_embedding(self,
                           target_name: str,
                           embedding: np.ndarray,
                           model_version: str,
                           confidence: float,
                           data_hash: str) -> int:
        """Сохранение embedding"""
        sql = """
        INSERT INTO embeddings (target_name, embedding, model_version, confidence, data_hash)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                sql,
                target_name,
                embedding.tolist(),
                model_version,
                confidence,
                data_hash
            )
        
        return record_id
    
    async def get_similar_embeddings(self,
                                   embedding: np.ndarray,
                                   similarity_threshold: float = 0.9,
                                   limit: int = 10) -> List[Tuple[str, float]]:
        """
        Поиск похожих embeddings (упрощенная версия)
        В продакшене лучше использовать специализированные векторные БД
        """
        sql = """
        SELECT target_name, embedding, confidence
        FROM embeddings
        ORDER BY created_at DESC
        LIMIT 1000
        """
        
        async with self.pool.acquire() as conn:
            records = await conn.fetch(sql)
        
        similar_targets = []
        target_embedding = embedding / np.linalg.norm(embedding)
        
        for record in records:
            stored_embedding = np.array(record['embedding'])
            stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
            
            # Косинусное сходство
            similarity = np.dot(target_embedding, stored_embedding)
            
            if similarity >= similarity_threshold:
                similar_targets.append((record['target_name'], float(similarity)))
        
        # Сортируем по убыванию сходства
        similar_targets.sort(key=lambda x: x[1], reverse=True)
        
        return similar_targets[:limit]
    
    async def save_user_feedback(self, feedback: UserFeedback) -> int:
        """Сохранение пользовательской обратной связи"""
        sql = """
        INSERT INTO user_feedback (
            target_name, user_id, feedback_type, confidence_rating, comments, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                sql,
                feedback.target_name,
                feedback.user_id,
                feedback.feedback_type,
                feedback.confidence_rating,
                feedback.comments,
                feedback.timestamp
            )
        
        logger.info(f"Saved user feedback for {feedback.target_name}")
        return record_id
    
    async def get_feedback_for_active_learning(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Получение обратной связи для активного обучения
        """
        sql = """
        SELECT uf.*, ar.confidence, ar.is_transit, ar.physical_parameters
        FROM user_feedback uf
        JOIN analysis_results ar ON uf.target_name = ar.target_name
        WHERE uf.feedback_type IN ('correct', 'incorrect')
        ORDER BY uf.timestamp DESC
        LIMIT $1
        """
        
        async with self.pool.acquire() as conn:
            records = await conn.fetch(sql, limit)
        
        feedback_data = []
        for record in records:
            feedback_dict = dict(record)
            feedback_dict['physical_parameters'] = json.loads(
                feedback_dict['physical_parameters'] or '{}'
            )
            feedback_data.append(feedback_dict)
        
        return feedback_data
    
    async def save_training_session(self,
                                  model_name: str,
                                  model_version: str,
                                  dataset_name: str,
                                  training_params: Dict[str, Any],
                                  metrics: Dict[str, Any],
                                  started_at: datetime,
                                  completed_at: Optional[datetime] = None,
                                  status: str = 'completed') -> int:
        """Сохранение сессии обучения"""
        sql = """
        INSERT INTO training_history (
            model_name, model_version, dataset_name, training_params,
            metrics, started_at, completed_at, status
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                sql,
                model_name,
                model_version,
                dataset_name,
                json.dumps(training_params),
                json.dumps(metrics),
                started_at,
                completed_at,
                status
            )
        
        return record_id
    
    async def get_known_exoplanet(self, host_star: str) -> Optional[Dict[str, Any]]:
        """
        Поиск известной экзопланеты по имени звезды-хозяина
        """
        sql = """
        SELECT * FROM known_exoplanets 
        WHERE host_star ILIKE $1 OR planet_name ILIKE $1
        ORDER BY confirmed DESC, discovery_year DESC
        LIMIT 1
        """
        
        async with self.pool.acquire() as conn:
            record = await conn.fetchrow(sql, f"%{host_star}%")
        
        if record:
            return dict(record)
        return None
    
    async def add_known_exoplanet(self,
                                planet_name: str,
                                host_star: str,
                                discovery_method: str = 'Transit',
                                orbital_period: Optional[float] = None,
                                planet_radius: Optional[float] = None,
                                equilibrium_temperature: Optional[float] = None,
                                discovery_year: Optional[int] = None,
                                source: str = 'Manual') -> int:
        """Добавление известной экзопланеты"""
        sql = """
        INSERT INTO known_exoplanets (
            planet_name, host_star, discovery_method, orbital_period,
            planet_radius, equilibrium_temperature, discovery_year, source
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                sql,
                planet_name,
                host_star,
                discovery_method,
                orbital_period,
                planet_radius,
                equilibrium_temperature,
                discovery_year,
                source
            )
        
        return record_id
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по базе данных"""
        stats_queries = {
            'total_analyses': "SELECT COUNT(*) FROM analysis_results",
            'transit_detections': "SELECT COUNT(*) FROM analysis_results WHERE is_transit = TRUE",
            'avg_confidence': "SELECT AVG(confidence) FROM analysis_results",
            'total_embeddings': "SELECT COUNT(*) FROM embeddings",
            'total_feedback': "SELECT COUNT(*) FROM user_feedback",
            'known_planets': "SELECT COUNT(*) FROM known_exoplanets",
            'recent_analyses': """
                SELECT COUNT(*) FROM analysis_results 
                WHERE analysis_timestamp >= NOW() - INTERVAL '7 days'
            """
        }
        
        stats = {}
        async with self.pool.acquire() as conn:
            for stat_name, query in stats_queries.items():
                result = await conn.fetchval(query)
                stats[stat_name] = result
        
        # Дополнительная статистика
        async with self.pool.acquire() as conn:
            # Распределение по типам обратной связи
            feedback_dist = await conn.fetch("""
                SELECT feedback_type, COUNT(*) as count 
                FROM user_feedback 
                GROUP BY feedback_type
            """)
            stats['feedback_distribution'] = {
                record['feedback_type']: record['count'] 
                for record in feedback_dist
            }
            
            # Топ целей по количеству анализов
            top_targets = await conn.fetch("""
                SELECT target_name, COUNT(*) as analysis_count
                FROM analysis_results
                GROUP BY target_name
                ORDER BY analysis_count DESC
                LIMIT 10
            """)
            stats['top_analyzed_targets'] = [
                {'target': record['target_name'], 'count': record['analysis_count']}
                for record in top_targets
            ]
        
        return stats
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Очистка старых данных
        
        Returns:
            Количество удаленных записей по таблицам
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleanup_queries = {
            'analysis_results': """
                DELETE FROM analysis_results 
                WHERE analysis_timestamp < $1 AND verified IS NULL
            """,
            'embeddings': """
                DELETE FROM embeddings 
                WHERE created_at < $1
            """,
            'user_feedback': """
                DELETE FROM user_feedback 
                WHERE timestamp < $1
            """,
            'training_history': """
                DELETE FROM training_history 
                WHERE completed_at < $1 AND status = 'completed'
            """
        }
        
        deleted_counts = {}
        async with self.pool.acquire() as conn:
            for table_name, query in cleanup_queries.items():
                result = await conn.execute(query, cutoff_date)
                # Извлекаем количество удаленных строк из результата
                deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                deleted_counts[table_name] = deleted_count
        
        logger.info(f"Cleaned up old data: {deleted_counts}")
        return deleted_counts
