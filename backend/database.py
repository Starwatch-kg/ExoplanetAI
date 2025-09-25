import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from databases import Database
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Настройки базы данных - используем SQLite по умолчанию
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./exoplanet_ai.db")
SQLALCHEMY_DATABASE_URL = DATABASE_URL

# Создание движка SQLAlchemy
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Создание базового класса для моделей
Base = declarative_base()

# Асинхронная база данных
database = Database(DATABASE_URL)

# Метаданные для создания таблиц
metadata = MetaData()

# Таблица для результатов анализа
analysis_results = Table(
    "analysis_results",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("target_name", String(100), index=True),
    Column("catalog", String(20)),
    Column("mission", String(20)),
    Column("analysis_timestamp", DateTime, default=datetime.utcnow),
    Column("lightcurve_data", JSON),
    Column("bls_results", JSON),
    Column("candidates", JSON),
    Column("ai_analysis", JSON, nullable=True),
    Column("physical_parameters", JSON, nullable=True),
    Column("status", String(20), default="success"),
    Column("message", Text, nullable=True),
    Column("user_feedback", JSON, nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)

# Таблица для целей (звезд)
targets = Table(
    "targets",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("name", String(100), unique=True, index=True),
    Column("catalog_id", String(50)),
    Column("ra", Float),
    Column("dec", Float),
    Column("magnitude", Float),
    Column("stellar_type", String(20), nullable=True),
    Column("temperature", Float, nullable=True),
    Column("radius", Float, nullable=True),
    Column("mass", Float, nullable=True),
    Column("distance", Float, nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# Таблица для кандидатов экзопланет
planet_candidates = Table(
    "planet_candidates",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("analysis_id", Integer, index=True),
    Column("target_name", String(100), index=True),
    Column("period", Float),
    Column("epoch", Float),
    Column("duration", Float),
    Column("depth", Float),
    Column("snr", Float),
    Column("significance", Float),
    Column("is_planet_candidate", Boolean, default=False),
    Column("confidence", Float),
    Column("ai_confidence", Float, nullable=True),
    Column("validation_status", String(20), default="pending"),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# Таблица для пользовательской обратной связи
user_feedback = Table(
    "user_feedback",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("analysis_id", Integer, index=True),
    Column("target_name", String(100)),
    Column("feedback_type", String(20)),  # 'positive', 'negative', 'correction'
    Column("is_correct", Boolean),
    Column("user_classification", String(50), nullable=True),
    Column("comments", Text, nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# Таблица для кэширования embeddings
embeddings_cache = Table(
    "embeddings_cache",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("target_name", String(100), unique=True, index=True),
    Column("embedding_vector", JSON),
    Column("model_version", String(20)),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)

# Функции для работы с базой данных
async def connect_db():
    """Подключение к базе данных"""
    try:
        await database.connect()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

async def disconnect_db():
    """Отключение от базы данных"""
    try:
        await database.disconnect()
        logger.info("Database disconnected")
    except Exception as e:
        logger.error(f"Database disconnection failed: {e}")

def create_tables():
    """Создание всех таблиц"""
    try:
        metadata.create_all(bind=engine)
        logger.info("All tables created successfully")
    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        raise

# Функции для работы с данными
async def save_analysis_result(result_data: dict):
    """Сохранение результата анализа"""
    query = analysis_results.insert().values(**result_data)
    result_id = await database.execute(query)
    return result_id

async def get_analysis_results(limit: int = 100, offset: int = 0):
    """Получение результатов анализа"""
    query = analysis_results.select().limit(limit).offset(offset).order_by(analysis_results.c.created_at.desc())
    results = await database.fetch_all(query)
    return results

async def get_analysis_by_target(target_name: str):
    """Получение анализов по имени цели"""
    query = analysis_results.select().where(analysis_results.c.target_name == target_name)
    results = await database.fetch_all(query)
    return results

async def save_user_feedback(feedback_data: dict):
    """Сохранение пользовательской обратной связи"""
    query = user_feedback.insert().values(**feedback_data)
    feedback_id = await database.execute(query)
    return feedback_id

async def get_similar_targets(embedding_vector: list, limit: int = 5):
    """Поиск похожих целей по embeddings (упрощенная версия)"""
    # В реальной реализации здесь будет векторный поиск
    query = embeddings_cache.select().limit(limit)
    results = await database.fetch_all(query)
    return results

async def save_embedding(target_name: str, embedding_vector: list, model_version: str):
    """Сохранение embedding вектора"""
    query = embeddings_cache.insert().values(
        target_name=target_name,
        embedding_vector=embedding_vector,
        model_version=model_version
    )
    await database.execute(query)

# Dependency для получения сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
