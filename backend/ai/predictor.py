"""
Transit Predictor with AI Assistant

Система предсказания транзитов с ИИ-ассистентом для интерпретации результатов.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

from .models.base_model import BaseTransitModel
from .ensemble import EnsembleClassifier
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class TransitPrediction:
    """Результат предсказания транзита"""
    is_transit: bool
    confidence: float
    confidence_level: ConfidenceLevel
    transit_probability: float
    physical_parameters: Dict[str, Optional[float]]
    explanation: str
    recommendations: List[str]
    uncertainty_sources: List[str]

class TransitPredictor:
    """
    ИИ-предиктор для анализа транзитов экзопланет
    """
    
    def __init__(self,
                 model: BaseTransitModel,
                 embedding_manager: EmbeddingManager,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 confidence_threshold: float = 0.7):
        
        self.model = model
        self.embedding_manager = embedding_manager
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        self.model.to(device)
        self.model.eval()
        
    def predict(self, 
                lightcurve: np.ndarray,
                target_name: str,
                stellar_params: Optional[Dict[str, float]] = None,
                return_embeddings: bool = False) -> TransitPrediction:
        """
        Основное предсказание транзита
        
        Args:
            lightcurve: Кривая блеска
            target_name: Имя цели
            stellar_params: Параметры звезды
            return_embeddings: Возвращать ли embeddings
            
        Returns:
            Результат предсказания
        """
        # Проверяем кэш embeddings
        cached_result = self.embedding_manager.get_cached_prediction(target_name)
        if cached_result:
            logger.info(f"Found cached prediction for {target_name}")
            return cached_result
        
        # Подготавливаем данные
        x = torch.FloatTensor(lightcurve).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Получаем предсказание
            if isinstance(self.model, EnsembleClassifier):
                predictions, uncertainties, individual_preds = self.model.predict_with_uncertainty(x)
                transit_prob = individual_preds[list(individual_preds.keys())[0]][0, 1]
                uncertainty = uncertainties[0]
            else:
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                transit_prob = probs[0, 1].item()
                uncertainty = self._calculate_uncertainty(probs[0])
            
            # Извлекаем признаки для embeddings
            features = self.model.extract_features(x)
            
        # Определяем уровень уверенности
        confidence_level = self._get_confidence_level(transit_prob, uncertainty)
        
        # Вычисляем физические параметры
        physical_params = self._estimate_physical_parameters(
            lightcurve, transit_prob, stellar_params
        )
        
        # Генерируем объяснение
        explanation = self._generate_explanation(
            transit_prob, confidence_level, physical_params, uncertainty
        )
        
        # Генерируем рекомендации
        recommendations = self._generate_recommendations(
            transit_prob, confidence_level, uncertainty
        )
        
        # Определяем источники неопределенности
        uncertainty_sources = self._identify_uncertainty_sources(
            lightcurve, uncertainty, transit_prob
        )
        
        # Создаем результат
        result = TransitPrediction(
            is_transit=transit_prob > self.confidence_threshold,
            confidence=transit_prob,
            confidence_level=confidence_level,
            transit_probability=transit_prob,
            physical_parameters=physical_params,
            explanation=explanation,
            recommendations=recommendations,
            uncertainty_sources=uncertainty_sources
        )
        
        # Сохраняем в кэш
        self.embedding_manager.cache_prediction(
            target_name, features.cpu().numpy(), result
        )
        
        return result
    
    def _calculate_uncertainty(self, probs: torch.Tensor) -> float:
        """Вычисление неопределенности через энтропию"""
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()
    
    def _get_confidence_level(self, prob: float, uncertainty: float) -> ConfidenceLevel:
        """Определение уровня уверенности"""
        if prob > 0.9 and uncertainty < 0.1:
            return ConfidenceLevel.VERY_HIGH
        elif prob > 0.8 and uncertainty < 0.2:
            return ConfidenceLevel.HIGH
        elif prob > 0.6 and uncertainty < 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _estimate_physical_parameters(self,
                                    lightcurve: np.ndarray,
                                    transit_prob: float,
                                    stellar_params: Optional[Dict[str, float]]) -> Dict[str, Optional[float]]:
        """Оценка физических параметров планеты"""
        if transit_prob < 0.5:
            return {
                'planet_radius': None,
                'orbital_period': None,
                'transit_depth': None,
                'transit_duration': None,
                'equilibrium_temperature': None
            }
        
        # Простая оценка параметров
        # В реальной системе здесь был бы более сложный анализ
        
        # Глубина транзита
        baseline = np.median(lightcurve)
        min_flux = np.min(lightcurve)
        transit_depth = (baseline - min_flux) / baseline * 1e6  # в ppm
        
        # Радиус планеты (приблизительно)
        stellar_radius = stellar_params.get('radius', 1.0) if stellar_params else 1.0
        planet_radius = np.sqrt(transit_depth / 1e6) * stellar_radius * 109.2  # в радиусах Земли
        
        # Температура (очень приблизительно)
        stellar_temp = stellar_params.get('temperature', 5778) if stellar_params else 5778
        equilibrium_temp = stellar_temp * 0.5  # Очень грубая оценка
        
        return {
            'planet_radius': planet_radius if planet_radius > 0 else None,
            'orbital_period': None,  # Требует более сложного анализа
            'transit_depth': transit_depth,
            'transit_duration': None,  # Требует анализа формы транзита
            'equilibrium_temperature': equilibrium_temp
        }
    
    def _generate_explanation(self,
                            transit_prob: float,
                            confidence_level: ConfidenceLevel,
                            physical_params: Dict[str, Optional[float]],
                            uncertainty: float) -> str:
        """Генерация объяснения результата"""
        
        if transit_prob > 0.8:
            base_text = "🎯 Обнаружен сильный транзитный сигнал! "
        elif transit_prob > 0.6:
            base_text = "🔍 Найден вероятный транзитный сигнал. "
        elif transit_prob > 0.4:
            base_text = "❓ Слабый сигнал, возможно транзит. "
        else:
            base_text = "❌ Транзитный сигнал не обнаружен. "
        
        # Добавляем информацию об уверенности
        if confidence_level == ConfidenceLevel.VERY_HIGH:
            confidence_text = "Очень высокая уверенность в результате."
        elif confidence_level == ConfidenceLevel.HIGH:
            confidence_text = "Высокая уверенность в результате."
        elif confidence_level == ConfidenceLevel.MEDIUM:
            confidence_text = "Средняя уверенность, требуется дополнительная проверка."
        else:
            confidence_text = "Низкая уверенность, результат сомнительный."
        
        # Добавляем физические параметры
        params_text = ""
        if physical_params.get('planet_radius'):
            params_text += f" Оценочный радиус планеты: {physical_params['planet_radius']:.1f} R⊕."
        
        if physical_params.get('transit_depth'):
            params_text += f" Глубина транзита: {physical_params['transit_depth']:.0f} ppm."
        
        return base_text + confidence_text + params_text
    
    def _generate_recommendations(self,
                                transit_prob: float,
                                confidence_level: ConfidenceLevel,
                                uncertainty: float) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        if transit_prob > 0.7:
            recommendations.append("✅ Рекомендуется дальнейшее наблюдение для подтверждения")
            recommendations.append("📊 Проведите анализ с другими методами (RV, TTV)")
        
        if confidence_level == ConfidenceLevel.LOW:
            recommendations.append("⚠️ Низкая уверенность - проверьте качество данных")
            recommendations.append("🔧 Попробуйте другие параметры BLS алгоритма")
        
        if uncertainty > 0.5:
            recommendations.append("❓ Высокая неопределенность - нужны дополнительные данные")
        
        if transit_prob > 0.5:
            recommendations.append("🔍 Проверьте на ложные позитивы (затменные двойные)")
            recommendations.append("📈 Сравните с каталогами известных экзопланет")
        
        return recommendations
    
    def _identify_uncertainty_sources(self,
                                    lightcurve: np.ndarray,
                                    uncertainty: float,
                                    transit_prob: float) -> List[str]:
        """Определение источников неопределенности"""
        sources = []
        
        # Анализ качества данных
        noise_level = np.std(lightcurve)
        if noise_level > 0.01:
            sources.append("Высокий уровень шума в данных")
        
        # Анализ пропусков данных
        if len(lightcurve) < 1000:
            sources.append("Недостаточно точек данных")
        
        # Неопределенность модели
        if uncertainty > 0.3:
            sources.append("Модель не уверена в классификации")
        
        # Слабый сигнал
        if 0.4 < transit_prob < 0.6:
            sources.append("Сигнал находится в пограничной области")
        
        return sources


class AIAssistant:
    """
    ИИ-ассистент для объяснения результатов простыми словами
    """
    
    def __init__(self):
        self.educational_mode = True
    
    def explain_for_beginners(self, prediction: TransitPrediction, target_name: str) -> str:
        """Объяснение для начинающих"""
        
        explanation = f"🌟 Анализ звезды {target_name}:\n\n"
        
        if prediction.is_transit:
            explanation += "🎉 Отличные новости! Мы нашли признаки планеты у этой звезды!\n\n"
            
            explanation += "🔬 Что это значит?\n"
            explanation += "• Планета проходит перед своей звездой (это называется 'транзит')\n"
            explanation += "• Когда планета закрывает часть света звезды, мы видим небольшое потемнение\n"
            explanation += "• Наш ИИ обнаружил этот периодический сигнал в данных\n\n"
            
            if prediction.physical_parameters.get('planet_radius'):
                radius = prediction.physical_parameters['planet_radius']
                if radius < 1.5:
                    size_desc = "размером примерно как Земля"
                elif radius < 4:
                    size_desc = "суперземля (больше Земли, но меньше Нептуна)"
                else:
                    size_desc = "газовый гигант (как Юпитер)"
                
                explanation += f"🪐 Планета {size_desc} ({radius:.1f} радиусов Земли)\n\n"
        
        else:
            explanation += "🔍 Мы не нашли убедительных признаков планеты у этой звезды.\n\n"
            explanation += "Это не значит, что планет точно нет - возможно:\n"
            explanation += "• Планеты есть, но их орбиты не проходят перед звездой\n"
            explanation += "• Сигнал слишком слабый для обнаружения\n"
            explanation += "• Нужны более качественные данные\n\n"
        
        # Уровень уверенности
        confidence_emoji = {
            ConfidenceLevel.VERY_HIGH: "🎯",
            ConfidenceLevel.HIGH: "✅", 
            ConfidenceLevel.MEDIUM: "🤔",
            ConfidenceLevel.LOW: "❓"
        }
        
        explanation += f"{confidence_emoji[prediction.confidence_level]} Уверенность: "
        explanation += f"{prediction.confidence_level.value} ({prediction.confidence:.1%})\n\n"
        
        # Рекомендации простыми словами
        if prediction.recommendations:
            explanation += "💡 Что делать дальше:\n"
            for rec in prediction.recommendations[:3]:  # Только первые 3
                explanation += f"• {rec}\n"
        
        return explanation
    
    def compare_with_known_planets(self, prediction: TransitPrediction) -> str:
        """Сравнение с известными планетами"""
        
        if not prediction.is_transit:
            return "Сравнение невозможно - транзит не обнаружен."
        
        radius = prediction.physical_parameters.get('planet_radius')
        if not radius:
            return "Недостаточно данных для сравнения."
        
        comparisons = []
        
        if radius < 1.2:
            comparisons.append("Похожа на Землю по размеру")
        elif radius < 1.8:
            comparisons.append("Как Kepler-452b ('кузина Земли')")
        elif radius < 4:
            comparisons.append("Как Нептун - суперземля")
        else:
            comparisons.append("Как Юпитер - газовый гигант")
        
        temp = prediction.physical_parameters.get('equilibrium_temperature')
        if temp:
            if temp < 200:
                comparisons.append("Очень холодная (как Плутон)")
            elif temp < 400:
                comparisons.append("Умеренная температура (возможна жизнь)")
            elif temp < 1000:
                comparisons.append("Горячая (как Венера)")
            else:
                comparisons.append("Очень горячая (расплавленная поверхность)")
        
        return "🌍 Сравнение: " + ", ".join(comparisons)
    
    def explain_habitability(self, prediction: TransitPrediction) -> str:
        """Объяснение возможности жизни"""
        
        if not prediction.is_transit:
            return "Планета не обнаружена - оценка обитаемости невозможна."
        
        radius = prediction.physical_parameters.get('planet_radius', 0)
        temp = prediction.physical_parameters.get('equilibrium_temperature', 0)
        
        habitability_score = 0
        factors = []
        
        # Размер
        if 0.5 < radius < 2.0:
            habitability_score += 2
            factors.append("✅ Подходящий размер для каменистой планеты")
        elif radius > 4:
            factors.append("❌ Слишком большая - вероятно газовый гигант")
        else:
            factors.append("⚠️ Необычный размер")
        
        # Температура
        if 200 < temp < 400:
            habitability_score += 3
            factors.append("✅ Температура подходит для жидкой воды")
        elif temp > 1000:
            factors.append("❌ Слишком горячая для жизни")
        elif temp < 100:
            factors.append("❌ Слишком холодная")
        else:
            factors.append("⚠️ Пограничная температура")
        
        # Итоговая оценка
        if habitability_score >= 4:
            result = "🌍 Высокий потенциал обитаемости!"
        elif habitability_score >= 2:
            result = "🤔 Возможно пригодна для жизни"
        else:
            result = "❌ Вероятно не пригодна для жизни"
        
        explanation = f"{result}\n\nФакторы:\n"
        for factor in factors:
            explanation += f"• {factor}\n"
        
        return explanation
