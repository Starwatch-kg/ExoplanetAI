"""
Координатор ансамблевого поиска - замена монолитного EnsembleSearchService
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from core.constants import MLConstants, TransitConstants
from core.exceptions import AnalysisError, handle_errors
from ml.transit_analyzer import TransitAnalyzer
from ml.period_detector import PeriodDetector
from ml.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class EnsembleCoordinator:
    """
    Координатор ансамблевого анализа экзопланет
    
    Объединяет специализированные анализаторы:
    - TransitAnalyzer: анализ транзитных сигналов
    - PeriodDetector: детекция периодических сигналов  
    - FeatureExtractor: извлечение ML признаков
    """
    
    def __init__(self):
        self.name = "Ensemble Coordinator"
        self.version = "2.0.0"
        
        # Инициализация компонентов
        self.transit_analyzer = TransitAnalyzer()
        self.period_detector = PeriodDetector()
        self.feature_extractor = FeatureExtractor()
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    @handle_errors("ensemble_analysis")
    async def analyze_lightcurve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        target_name: str = "Unknown"
    ) -> Dict[str, any]:
        """
        Полный ансамблевый анализ кривой блеска
        
        Args:
            time: Временные метки
            flux: Значения потока
            flux_err: Ошибки потока (опционально)
            target_name: Название цели
            
        Returns:
            Комплексные результаты анализа
        """
        
        logger.info(f"Starting ensemble analysis for {target_name}")
        
        # Валидация входных данных
        self._validate_input_data(time, flux, target_name)
        
        # Параллельный запуск всех анализаторов
        results = {}
        
        try:
            # 1. Анализ транзитов (BLS)
            logger.info("Running transit analysis...")
            transit_results = await self.transit_analyzer.detect_transits(
                time, flux, flux_err
            )
            results['transit_analysis'] = transit_results
            
            # 2. Детекция периодов (множественные методы)
            logger.info("Running period detection...")
            period_results = await self.period_detector.detect_periods(time, flux)
            results['period_analysis'] = period_results
            
            # 3. Извлечение ML признаков
            logger.info("Extracting ML features...")
            feature_results = await self.feature_extractor.extract_all_features(time, flux)
            results['feature_analysis'] = feature_results
            
            # 4. Интеграция результатов
            logger.info("Integrating analysis results...")
            integrated_results = await self._integrate_results(
                results, time, flux, target_name
            )
            
            # 5. Финальная оценка
            final_assessment = await self._assess_planet_candidate(
                integrated_results, target_name
            )
            
            return {
                'target_name': target_name,
                'individual_analyses': results,
                'integrated_results': integrated_results,
                'final_assessment': final_assessment,
                'metadata': {
                    'analyzer_versions': {
                        'coordinator': self.version,
                        'transit_analyzer': self.transit_analyzer.version,
                        'period_detector': self.period_detector.version,
                        'feature_extractor': self.feature_extractor.version
                    },
                    'data_points': len(time),
                    'time_span_days': float(time.max() - time.min()),
                    'analysis_timestamp': np.datetime64('now').astype(str)
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble analysis failed for {target_name}: {e}")
            raise AnalysisError(f"Ensemble analysis failed: {str(e)}")
    
    def _validate_input_data(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        target_name: str
    ) -> None:
        """Валидация входных данных"""
        
        if len(time) != len(flux):
            raise ValueError("Time and flux arrays must have the same length")
        
        if len(time) < MLConstants.MIN_DATA_POINTS:
            raise ValueError(
                f"Insufficient data points: {len(time)} < {MLConstants.MIN_DATA_POINTS}"
            )
        
        if not target_name or len(target_name.strip()) == 0:
            raise ValueError("Target name cannot be empty")
        
        # Проверка на NaN и Inf
        if not np.all(np.isfinite(time)):
            raise ValueError("Time array contains NaN or Inf values")
        
        if not np.all(np.isfinite(flux)):
            logger.warning("Flux array contains NaN or Inf values - will be cleaned")
    
    async def _integrate_results(
        self, 
        results: Dict[str, any], 
        time: np.ndarray, 
        flux: np.ndarray,
        target_name: str
    ) -> Dict[str, any]:
        """Интеграция результатов различных анализаторов"""
        
        integrated = {
            'consensus_period': None,
            'consensus_depth': None,
            'detection_confidence': 0.0,
            'method_agreement': {},
            'combined_score': 0.0
        }
        
        try:
            # Извлечение периодов из разных методов
            periods = []
            confidences = []
            
            # Период из транзитного анализа
            transit_results = results.get('transit_analysis', {})
            if transit_results.get('is_valid_candidate'):
                periods.append(transit_results['period'])
                confidences.append(transit_results['confidence'])
            
            # Период из детекции периодов
            period_results = results.get('period_analysis', {})
            consensus_period_data = period_results.get('consensus', {})
            if consensus_period_data.get('consensus_period'):
                periods.append(consensus_period_data['consensus_period'])
                confidences.append(consensus_period_data['confidence'])
            
            # Консенсус по периодам
            if periods:
                integrated['consensus_period'] = await self._find_period_consensus(
                    periods, confidences
                )
                
                # Средняя уверенность
                integrated['detection_confidence'] = np.mean(confidences)
            
            # Глубина транзита
            if transit_results.get('depth'):
                integrated['consensus_depth'] = transit_results['depth']
            
            # Согласованность методов
            integrated['method_agreement'] = await self._calculate_method_agreement(
                results
            )
            
            # Комбинированный скор
            integrated['combined_score'] = await self._calculate_combined_score(
                results, integrated
            )
            
        except Exception as e:
            logger.warning(f"Results integration failed: {e}")
        
        return integrated
    
    async def _find_period_consensus(
        self, 
        periods: List[float], 
        confidences: List[float]
    ) -> Optional[float]:
        """Поиск консенсуса по периодам"""
        
        if not periods:
            return None
        
        # Взвешенное среднее по уверенности
        weights = np.array(confidences)
        weighted_period = np.average(periods, weights=weights)
        
        # Проверка на разумность
        if (TransitConstants.MIN_PERIOD_DAYS <= weighted_period <= 
            TransitConstants.MAX_PERIOD_DAYS):
            return float(weighted_period)
        
        # Если взвешенное среднее неразумно, берем медиану
        return float(np.median(periods))
    
    async def _calculate_method_agreement(self, results: Dict[str, any]) -> Dict[str, float]:
        """Расчет согласованности между методами"""
        
        agreement = {
            'period_agreement': 0.0,
            'detection_agreement': 0.0,
            'overall_agreement': 0.0
        }
        
        try:
            # Согласованность по периодам
            transit_period = results.get('transit_analysis', {}).get('period')
            consensus_period = results.get('period_analysis', {}).get('consensus', {}).get('consensus_period')
            
            if transit_period and consensus_period:
                period_diff = abs(transit_period - consensus_period) / max(transit_period, consensus_period)
                agreement['period_agreement'] = max(0.0, 1.0 - period_diff)
            
            # Согласованность по детекции
            transit_valid = results.get('transit_analysis', {}).get('is_valid_candidate', False)
            period_confident = results.get('period_analysis', {}).get('consensus', {}).get('confidence', 0) > 0.5
            
            if transit_valid and period_confident:
                agreement['detection_agreement'] = 1.0
            elif not transit_valid and not period_confident:
                agreement['detection_agreement'] = 1.0
            else:
                agreement['detection_agreement'] = 0.0
            
            # Общая согласованность
            agreement['overall_agreement'] = np.mean([
                agreement['period_agreement'],
                agreement['detection_agreement']
            ])
            
        except Exception as e:
            logger.warning(f"Method agreement calculation failed: {e}")
        
        return agreement
    
    async def _calculate_combined_score(
        self, 
        results: Dict[str, any], 
        integrated: Dict[str, any]
    ) -> float:
        """Расчет комбинированного скора качества"""
        
        try:
            scores = []
            
            # Скор от транзитного анализа
            transit_results = results.get('transit_analysis', {})
            if transit_results.get('quality_score'):
                scores.append(transit_results['quality_score'])
            
            # Скор от детекции периодов
            period_results = results.get('period_analysis', {})
            data_quality = period_results.get('data_quality', {})
            if data_quality.get('quality_score'):
                scores.append(data_quality['quality_score'])
            
            # Скор от извлечения признаков
            feature_results = results.get('feature_analysis', {})
            feature_quality = feature_results.get('data_quality', {})
            if feature_quality.get('overall_quality'):
                scores.append(feature_quality['overall_quality'])
            
            # Бонус за согласованность методов
            method_agreement = integrated.get('method_agreement', {})
            if method_agreement.get('overall_agreement'):
                scores.append(method_agreement['overall_agreement'])
            
            # Комбинированный скор
            if scores:
                return float(np.mean(scores))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Combined score calculation failed: {e}")
            return 0.0
    
    async def _assess_planet_candidate(
        self, 
        integrated_results: Dict[str, any], 
        target_name: str
    ) -> Dict[str, any]:
        """Финальная оценка кандидата в экзопланеты"""
        
        assessment = {
            'is_planet_candidate': False,
            'confidence_level': 'low',
            'classification': 'unknown',
            'recommendations': [],
            'risk_factors': []
        }
        
        try:
            combined_score = integrated_results.get('combined_score', 0.0)
            detection_confidence = integrated_results.get('detection_confidence', 0.0)
            method_agreement = integrated_results.get('method_agreement', {}).get('overall_agreement', 0.0)
            
            # Критерии для классификации
            high_confidence_threshold = 0.8
            medium_confidence_threshold = 0.6
            
            # Комбинированная метрика
            final_confidence = (combined_score * 0.4 + 
                              detection_confidence * 0.4 + 
                              method_agreement * 0.2)
            
            # Классификация
            if final_confidence >= high_confidence_threshold:
                assessment['is_planet_candidate'] = True
                assessment['confidence_level'] = 'high'
                assessment['classification'] = 'strong_candidate'
                assessment['recommendations'].append(
                    "Excellent planet candidate - recommend follow-up observations"
                )
            elif final_confidence >= medium_confidence_threshold:
                assessment['is_planet_candidate'] = True
                assessment['confidence_level'] = 'medium'
                assessment['classification'] = 'moderate_candidate'
                assessment['recommendations'].append(
                    "Promising candidate - additional analysis recommended"
                )
            else:
                assessment['is_planet_candidate'] = False
                assessment['confidence_level'] = 'low'
                assessment['classification'] = 'unlikely_candidate'
                assessment['recommendations'].append(
                    "Low probability - likely false positive or insufficient data"
                )
            
            # Факторы риска
            if method_agreement < 0.5:
                assessment['risk_factors'].append("Low agreement between detection methods")
            
            if detection_confidence < 0.5:
                assessment['risk_factors'].append("Low detection confidence")
            
            consensus_period = integrated_results.get('consensus_period')
            if consensus_period:
                if consensus_period < TransitConstants.MIN_PERIOD_DAYS:
                    assessment['risk_factors'].append("Period too short for stable orbit")
                elif consensus_period > TransitConstants.MAX_PERIOD_DAYS:
                    assessment['risk_factors'].append("Period very long - may be stellar variability")
            
            # Дополнительные рекомендации
            if not assessment['risk_factors']:
                assessment['recommendations'].append("No significant risk factors identified")
            
            assessment['final_confidence_score'] = float(final_confidence)
            
        except Exception as e:
            logger.warning(f"Planet candidate assessment failed: {e}")
        
        return assessment
    
    async def get_analysis_summary(self, analysis_results: Dict[str, any]) -> str:
        """Генерация краткого резюме анализа"""
        
        try:
            target_name = analysis_results.get('target_name', 'Unknown')
            final_assessment = analysis_results.get('final_assessment', {})
            integrated_results = analysis_results.get('integrated_results', {})
            
            # Основная информация
            is_candidate = final_assessment.get('is_planet_candidate', False)
            confidence_level = final_assessment.get('confidence_level', 'unknown')
            consensus_period = integrated_results.get('consensus_period')
            consensus_depth = integrated_results.get('consensus_depth')
            
            # Формирование резюме
            summary_parts = [
                f"Analysis Summary for {target_name}:",
                f"Planet Candidate: {'Yes' if is_candidate else 'No'}",
                f"Confidence Level: {confidence_level.title()}"
            ]
            
            if consensus_period:
                summary_parts.append(f"Orbital Period: {consensus_period:.2f} days")
            
            if consensus_depth:
                summary_parts.append(f"Transit Depth: {consensus_depth*100:.3f}%")
            
            # Рекомендации
            recommendations = final_assessment.get('recommendations', [])
            if recommendations:
                summary_parts.append(f"Recommendation: {recommendations[0]}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return f"Analysis completed for {analysis_results.get('target_name', 'Unknown')}"
