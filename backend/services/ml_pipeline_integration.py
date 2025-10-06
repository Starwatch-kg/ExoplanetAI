"""
Интеграция улучшенной системы автообучения с существующим ML pipeline
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from core.logging import get_logger
from services.enhanced_auto_trainer import get_enhanced_trainer
from services.auto_ml_trainer import get_auto_trainer
from ml.exoplanet_classifier import ExoplanetEnsembleClassifier
from api.routes.ml_classification import get_classifier, model_status

logger = get_logger(__name__)

class MLPipelineIntegrator:
    """
    Интегратор для связи автообучения с основным ML pipeline
    """
    
    def __init__(self):
        self.enhanced_trainer = get_enhanced_trainer()
        self.legacy_trainer = get_auto_trainer()
        self.is_integrated = False
        
    async def initialize_integration(self):
        """Инициализация интеграции"""
        try:
            logger.info("Initializing ML pipeline integration")
            
            # Проверяем состояние существующей модели
            current_classifier = get_classifier()
            
            if current_classifier and hasattr(current_classifier, 'is_trained'):
                # Синхронизируем состояние с автотренером
                await self.sync_model_state(current_classifier)
            
            # Запускаем мониторинг интеграции
            asyncio.create_task(self.integration_monitor())
            
            self.is_integrated = True
            logger.info("ML pipeline integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML pipeline integration: {e}")
            raise
    
    async def sync_model_state(self, classifier: ExoplanetEnsembleClassifier):
        """Синхронизация состояния модели между системами"""
        try:
            # Обновляем статус в model_status
            if hasattr(classifier, 'model_version'):
                model_status["model_version"] = classifier.model_version
            
            model_status["is_trained"] = True
            model_status["last_sync_time"] = datetime.now().isoformat()
            
            # Синхронизируем с enhanced trainer
            if hasattr(classifier, 'model_version'):
                self.enhanced_trainer.model_version = classifier.model_version
            
            self.enhanced_trainer.classifier = classifier
            
            logger.info("Model state synchronized successfully")
            
        except Exception as e:
            logger.error(f"Failed to sync model state: {e}")
    
    async def integration_monitor(self):
        """Мониторинг интеграции между системами"""
        while self.is_integrated:
            try:
                await asyncio.sleep(300)  # Проверяем каждые 5 минут
                
                # Проверяем консистентность между системами
                await self.check_consistency()
                
                # Обновляем метрики интеграции
                await self.update_integration_metrics()
                
            except Exception as e:
                logger.error(f"Error in integration monitor: {e}")
                await asyncio.sleep(60)  # При ошибке ждем меньше
    
    async def check_consistency(self):
        """Проверка консистентности между системами"""
        try:
            # Сравниваем версии моделей
            ml_classifier = get_classifier()
            enhanced_classifier = self.enhanced_trainer.classifier
            
            if (hasattr(ml_classifier, 'model_version') and 
                hasattr(enhanced_classifier, 'model_version')):
                
                if ml_classifier.model_version != enhanced_classifier.model_version:
                    logger.warning(
                        f"Model version mismatch: ML={ml_classifier.model_version}, "
                        f"Enhanced={enhanced_classifier.model_version}"
                    )
                    
                    # Синхронизируем к более новой версии
                    await self.resolve_version_conflict(ml_classifier, enhanced_classifier)
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
    
    async def resolve_version_conflict(self, ml_classifier, enhanced_classifier):
        """Разрешение конфликта версий моделей"""
        try:
            ml_version = getattr(ml_classifier, 'model_version', 0)
            enhanced_version = getattr(enhanced_classifier, 'model_version', 0)
            
            if enhanced_version > ml_version:
                # Enhanced trainer имеет более новую модель
                logger.info(f"Updating ML classifier to version {enhanced_version}")
                
                # Обновляем глобальный классификатор
                global classifier
                classifier = enhanced_classifier
                
                # Обновляем статус
                model_status["model_version"] = enhanced_version
                model_status["last_update"] = datetime.now().isoformat()
                
            elif ml_version > enhanced_version:
                # ML classifier имеет более новую модель
                logger.info(f"Updating enhanced trainer to version {ml_version}")
                self.enhanced_trainer.classifier = ml_classifier
                self.enhanced_trainer.model_version = ml_version
            
        except Exception as e:
            logger.error(f"Error resolving version conflict: {e}")
    
    async def update_integration_metrics(self):
        """Обновление метрик интеграции"""
        try:
            metrics = {
                'integration_active': self.is_integrated,
                'last_check': datetime.now().isoformat(),
                'ml_classifier_ready': get_classifier() is not None,
                'enhanced_trainer_ready': self.enhanced_trainer is not None,
                'model_versions_synced': await self.check_versions_synced()
            }
            
            # Сохраняем метрики в model_status для доступа через API
            model_status["integration_metrics"] = metrics
            
        except Exception as e:
            logger.error(f"Error updating integration metrics: {e}")
    
    async def check_versions_synced(self) -> bool:
        """Проверка синхронизации версий"""
        try:
            ml_classifier = get_classifier()
            enhanced_classifier = self.enhanced_trainer.classifier
            
            ml_version = getattr(ml_classifier, 'model_version', 0)
            enhanced_version = getattr(enhanced_classifier, 'model_version', 0)
            
            return ml_version == enhanced_version
            
        except Exception as e:
            logger.error(f"Error checking version sync: {e}")
            return False
    
    async def trigger_integrated_training(self, use_enhanced: bool = True):
        """Запуск интегрированного обучения"""
        try:
            if use_enhanced:
                logger.info("Triggering enhanced training")
                await self.enhanced_trainer.enhanced_model_training()
            else:
                logger.info("Triggering legacy training")
                await self.legacy_trainer.retrain_model()
            
            # После обучения синхронизируем состояние
            await self.sync_model_state(self.enhanced_trainer.classifier)
            
        except Exception as e:
            logger.error(f"Error in integrated training: {e}")
            raise
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Получение статуса интеграции"""
        return {
            'is_integrated': self.is_integrated,
            'enhanced_trainer_active': self.enhanced_trainer is not None,
            'legacy_trainer_active': self.legacy_trainer is not None,
            'integration_metrics': model_status.get('integration_metrics', {}),
            'last_sync': model_status.get('last_sync_time'),
            'model_version': model_status.get('model_version', 0)
        }

# Глобальный интегратор
_integrator = None

def get_ml_integrator() -> MLPipelineIntegrator:
    """Получение глобального интегратора"""
    global _integrator
    if _integrator is None:
        _integrator = MLPipelineIntegrator()
    return _integrator

async def initialize_ml_integration():
    """Инициализация интеграции ML pipeline"""
    integrator = get_ml_integrator()
    await integrator.initialize_integration()
    return integrator
