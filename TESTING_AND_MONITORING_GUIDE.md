# 🧪 ExoplanetAI - Руководство по тестированию и мониторингу

## 📊 Тестирование моделей ИИ

### 1. Unit тесты для ML компонентов

```python
# backend/tests/test_enhanced_classifier.py
import pytest
import numpy as np
from ml.enhanced_classifier import OptimizedEnsemble, EnhancedFeatureExtractor

class TestEnhancedClassifier:
    
    @pytest.fixture
    def sample_data(self):
        """Генерация тестовых данных"""
        np.random.seed(42)
        time = np.linspace(0, 27.4, 1000)  # TESS sector
        flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
        
        # Добавляем транзит
        transit_mask = (time % 2.5 < 0.1)  # Период 2.5 дня
        flux[transit_mask] -= 0.01  # Глубина 1%
        
        flux_err = np.full_like(flux, 0.001)
        return time, flux, flux_err
    
    def test_feature_extraction(self, sample_data):
        """Тест извлечения признаков"""
        time, flux, flux_err = sample_data
        extractor = EnhancedFeatureExtractor()
        
        features = extractor.extract_features(time, flux, flux_err)
        
        assert len(features) > 0
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_astrophysical_features(self, sample_data):
        """Тест астрофизических признаков"""
        time, flux, flux_err = sample_data
        extractor = EnhancedFeatureExtractor()
        
        transit_params = {
            'primary_depth': 0.01,
            'secondary_depth': 0.0001,
            'ingress_duration': 0.1,
            'egress_duration': 0.1,
            'transit_duration': 1.0
        }
        
        features = extractor.extract_astrophysical_features(time, flux, transit_params)
        
        assert 'stellar_contamination_ratio' in features
        assert 'limb_darkening_coefficient' in features
        assert features['stellar_contamination_ratio'] >= 0
    
    def test_model_prediction_consistency(self, sample_data):
        """Тест консистентности предсказаний"""
        time, flux, flux_err = sample_data
        classifier = OptimizedEnsemble()
        
        # Создаем фиктивные признаки
        features = np.random.random(50)
        
        # Делаем несколько предсказаний
        predictions = []
        for _ in range(5):
            pred = classifier.predict_with_uncertainty(features)
            predictions.append(pred['confidence'])
        
        # Проверяем консистентность (разброс < 5%)
        confidence_std = np.std(predictions)
        assert confidence_std < 0.05, "Предсказания модели нестабильны"

    def test_uncertainty_calibration(self, sample_data):
        """Тест калибровки неопределенности"""
        time, flux, flux_err = sample_data
        classifier = OptimizedEnsemble()
        
        features = np.random.random(50)
        prediction = classifier.predict_with_uncertainty(features)
        
        # Проверяем, что неопределенность в разумных пределах
        assert 0 <= prediction['uncertainty'] <= 1
        assert len(prediction['uncertainty_bounds']) == 2
        assert prediction['uncertainty_bounds'][0] <= prediction['uncertainty_bounds'][1]
```

### 2. Integration тесты для API

```python
# backend/tests/test_smart_search_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestSmartSearchAPI:
    
    def test_smart_analysis_endpoint(self):
        """Тест эндпоинта умного анализа"""
        request_data = {
            "target_name": "TOI-715",
            "mission": "TESS",
            "include_uncertainty": True,
            "explain_prediction": True
        }
        
        response = client.post("/api/v1/ai/analyze_lightcurve", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
        assert "predicted_class" in data
        assert "confidence_score" in data
        assert "uncertainty_bounds" in data
        assert "feature_importance" in data
        assert "decision_reasoning" in data
        
        # Проверяем валидность данных
        assert 0 <= data["confidence_score"] <= 1
        assert len(data["uncertainty_bounds"]) == 2
        assert isinstance(data["decision_reasoning"], list)
    
    def test_smart_search_endpoint(self):
        """Тест эндпоинта умного поиска"""
        request_data = {
            "query": "TOI-715",
            "filters": {
                "confidence_min": 0.7,
                "snr_min": 5.0,
                "missions": ["TESS"]
            },
            "use_ai_ranking": True,
            "max_results": 10
        }
        
        response = client.post("/api/v1/ai/smart_search", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "recommendations" in data
        assert "search_time_ms" in data
        assert data["search_time_ms"] > 0
    
    def test_batch_analysis_endpoint(self):
        """Тест пакетного анализа"""
        request_data = {
            "targets": ["TOI-715", "Kepler-452b"],
            "analysis_params": {
                "mission": "TESS",
                "include_uncertainty": True
            },
            "parallel_limit": 2
        }
        
        response = client.post("/api/v1/ai/batch_analyze", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "successful_analyses" in data
        assert "failed_analyses" in data
        assert "success_rate" in data
        assert 0 <= data["success_rate"] <= 1
```

### 3. Performance тесты

```python
# backend/tests/test_performance.py
import time
import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ml.enhanced_classifier import OptimizedEnsemble

class TestPerformance:
    
    def test_analysis_speed(self):
        """Тест скорости анализа"""
        classifier = OptimizedEnsemble()
        features = np.random.random(50)
        
        start_time = time.time()
        prediction = classifier.predict_with_uncertainty(features)
        end_time = time.time()
        
        analysis_time = (end_time - start_time) * 1000  # мс
        
        # Анализ должен занимать менее 500мс
        assert analysis_time < 500, f"Анализ слишком медленный: {analysis_time:.1f}мс"
    
    def test_concurrent_analysis(self):
        """Тест параллельного анализа"""
        classifier = OptimizedEnsemble()
        
        def analyze_sample():
            features = np.random.random(50)
            return classifier.predict_with_uncertainty(features)
        
        start_time = time.time()
        
        # Запускаем 10 параллельных анализов
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(analyze_sample) for _ in range(10)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Все анализы должны завершиться менее чем за 2 секунды
        assert total_time < 2000, f"Параллельный анализ слишком медленный: {total_time:.1f}мс"
        assert len(results) == 10
        
    def test_memory_usage(self):
        """Тест использования памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        classifier = OptimizedEnsemble()
        
        # Выполняем много анализов
        for _ in range(100):
            features = np.random.random(50)
            classifier.predict_with_uncertainty(features)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Увеличение памяти не должно превышать 100MB
        assert memory_increase < 100, f"Утечка памяти: +{memory_increase:.1f}MB"
```

## 📈 Система мониторинга

### 1. Мониторинг качества модели

```python
# backend/monitoring/model_monitor.py
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis
import json

logger = logging.getLogger(__name__)

class ModelPerformanceMonitor:
    """Мониторинг производительности ML модели"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.metrics_history = []
        
    async def track_prediction(self, 
                             prediction: Dict,
                             ground_truth: Optional[str] = None,
                             user_feedback: Optional[Dict] = None):
        """Отслеживание предсказания"""
        
        timestamp = datetime.now()
        
        # Базовые метрики
        metrics = {
            'timestamp': timestamp.isoformat(),
            'predicted_class': prediction['class_name'],
            'confidence': prediction['confidence'],
            'uncertainty': prediction.get('uncertainty', 0),
            'processing_time_ms': prediction.get('processing_time_ms', 0)
        }
        
        # Добавляем ground truth если доступен
        if ground_truth:
            metrics['ground_truth'] = ground_truth
            metrics['correct_prediction'] = (prediction['class_name'] == ground_truth)
            
        # Добавляем обратную связь пользователя
        if user_feedback:
            metrics['user_rating'] = user_feedback.get('rating')
            metrics['user_comment'] = user_feedback.get('comment')
            
        # Сохраняем в Redis
        if self.redis_client:
            key = f"model_metrics:{timestamp.strftime('%Y%m%d')}"
            self.redis_client.lpush(key, json.dumps(metrics))
            self.redis_client.expire(key, 86400 * 30)  # 30 дней
            
        self.metrics_history.append(metrics)
        
        # Проверяем на аномалии
        await self._check_anomalies(metrics)
        
    async def _check_anomalies(self, metrics: Dict):
        """Проверка на аномалии в производительности"""
        
        # Проверка времени обработки
        if metrics['processing_time_ms'] > 5000:  # 5 секунд
            await self._send_alert(
                "Slow prediction detected",
                f"Processing time: {metrics['processing_time_ms']}ms"
            )
            
        # Проверка низкой уверенности
        if metrics['confidence'] < 0.3:
            await self._send_alert(
                "Low confidence prediction",
                f"Confidence: {metrics['confidence']:.2f}"
            )
            
        # Проверка высокой неопределенности
        if metrics['uncertainty'] > 0.8:
            await self._send_alert(
                "High uncertainty detected",
                f"Uncertainty: {metrics['uncertainty']:.2f}"
            )
    
    async def calculate_drift_metrics(self, window_days: int = 7) -> Dict:
        """Расчет метрик дрейфа модели"""
        
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        
        if len(recent_metrics) < 10:
            return {"error": "Insufficient data for drift calculation"}
            
        # Дрейф уверенности
        confidences = [m['confidence'] for m in recent_metrics]
        confidence_drift = np.std(confidences)
        
        # Дрейф времени обработки
        processing_times = [m['processing_time_ms'] for m in recent_metrics]
        time_drift = np.std(processing_times)
        
        # Распределение классов
        class_distribution = {}
        for m in recent_metrics:
            cls = m['predicted_class']
            class_distribution[cls] = class_distribution.get(cls, 0) + 1
            
        return {
            'confidence_drift': confidence_drift,
            'processing_time_drift': time_drift,
            'class_distribution': class_distribution,
            'total_predictions': len(recent_metrics),
            'window_days': window_days
        }
    
    async def _send_alert(self, title: str, message: str):
        """Отправка алерта"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'message': message,
            'severity': 'warning'
        }
        
        logger.warning(f"Model Alert: {title} - {message}")
        
        # Здесь можно добавить отправку в Slack, email и т.д.
        if self.redis_client:
            self.redis_client.lpush("model_alerts", json.dumps(alert))
```

### 2. Система метрик и дашборд

```python
# backend/monitoring/metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Prometheus метрики
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total ML predictions', ['model', 'class'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'ML prediction latency')
CONFIDENCE_GAUGE = Gauge('ml_confidence_score', 'Current confidence score')
ACCURACY_GAUGE = Gauge('ml_accuracy_score', 'Current accuracy score')
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')

def track_prediction_metrics(func):
    """Декоратор для отслеживания метрик предсказаний"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Обновляем метрики
            PREDICTION_COUNTER.labels(
                model='enhanced_ensemble',
                class=result.get('predicted_class', 'unknown')
            ).inc()
            
            CONFIDENCE_GAUGE.set(result.get('confidence_score', 0))
            
            return result
            
        finally:
            # Записываем время выполнения
            PREDICTION_LATENCY.observe(time.time() - start_time)
            
    return wrapper

class MetricsCollector:
    """Сборщик метрик для мониторинга"""
    
    def __init__(self):
        self.start_prometheus_server()
        
    def start_prometheus_server(self, port: int = 8000):
        """Запуск Prometheus сервера метрик"""
        start_http_server(port)
        
    def update_accuracy_metrics(self, accuracy: float):
        """Обновление метрик точности"""
        ACCURACY_GAUGE.set(accuracy)
        
    def track_user_activity(self, user_count: int):
        """Отслеживание активности пользователей"""
        ACTIVE_USERS.set(user_count)
```

### 3. A/B тестирование моделей

```python
# backend/monitoring/ab_testing.py
import random
import hashlib
from typing import Dict, Any
from enum import Enum

class ModelVersion(Enum):
    CURRENT = "current"
    ENHANCED = "enhanced"
    EXPERIMENTAL = "experimental"

class ABTestManager:
    """Менеджер A/B тестирования моделей"""
    
    def __init__(self):
        self.model_weights = {
            ModelVersion.CURRENT: 0.7,      # 70% трафика
            ModelVersion.ENHANCED: 0.25,     # 25% трафика  
            ModelVersion.EXPERIMENTAL: 0.05  # 5% трафика
        }
        
        self.results = {version: [] for version in ModelVersion}
        
    def get_model_version(self, user_id: str) -> ModelVersion:
        """Определение версии модели для пользователя"""
        
        # Используем hash для консистентного распределения
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 1000) / 1000.0
        
        cumulative_weight = 0
        for version, weight in self.model_weights.items():
            cumulative_weight += weight
            if normalized_hash <= cumulative_weight:
                return version
                
        return ModelVersion.CURRENT
    
    def record_result(self, 
                     version: ModelVersion,
                     prediction: Dict[str, Any],
                     user_feedback: Dict[str, Any] = None):
        """Запись результата для A/B теста"""
        
        result = {
            'timestamp': time.time(),
            'confidence': prediction.get('confidence', 0),
            'processing_time': prediction.get('processing_time_ms', 0),
            'user_satisfaction': user_feedback.get('rating', 0) if user_feedback else None
        }
        
        self.results[version].append(result)
        
    def calculate_test_results(self) -> Dict[str, Dict]:
        """Расчет результатов A/B теста"""
        
        results = {}
        
        for version, data in self.results.items():
            if not data:
                continue
                
            confidences = [r['confidence'] for r in data]
            processing_times = [r['processing_time'] for r in data]
            satisfactions = [r['user_satisfaction'] for r in data if r['user_satisfaction'] is not None]
            
            results[version.value] = {
                'sample_size': len(data),
                'avg_confidence': np.mean(confidences),
                'avg_processing_time': np.mean(processing_times),
                'avg_user_satisfaction': np.mean(satisfactions) if satisfactions else None,
                'confidence_std': np.std(confidences),
                'processing_time_std': np.std(processing_times)
            }
            
        return results
    
    def should_promote_model(self, 
                           test_version: ModelVersion,
                           control_version: ModelVersion = ModelVersion.CURRENT,
                           confidence_threshold: float = 0.95) -> bool:
        """Определение необходимости продвижения модели"""
        
        results = self.calculate_test_results()
        
        if test_version.value not in results or control_version.value not in results:
            return False
            
        test_data = results[test_version.value]
        control_data = results[control_version.value]
        
        # Проверяем статистическую значимость улучшений
        confidence_improvement = test_data['avg_confidence'] - control_data['avg_confidence']
        speed_improvement = control_data['avg_processing_time'] - test_data['avg_processing_time']
        
        # Простая эвристика для принятия решения
        if (confidence_improvement > 0.05 and  # +5% уверенности
            speed_improvement > 0 and          # Не медленнее
            test_data['sample_size'] > 100):   # Достаточно данных
            return True
            
        return False
```

## 🚨 Алерты и уведомления

### 1. Система алертов

```python
# backend/monitoring/alerting.py
import asyncio
import aiohttp
from typing import List, Dict
from enum import Enum

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertManager:
    """Менеджер системы алертов"""
    
    def __init__(self, slack_webhook: str = None, email_config: Dict = None):
        self.slack_webhook = slack_webhook
        self.email_config = email_config
        self.alert_rules = self._setup_alert_rules()
        
    def _setup_alert_rules(self) -> List[Dict]:
        """Настройка правил алертов"""
        return [
            {
                'name': 'High Error Rate',
                'condition': lambda metrics: metrics.get('error_rate', 0) > 0.05,
                'severity': AlertSeverity.ERROR,
                'message': 'Error rate exceeded 5%'
            },
            {
                'name': 'Low Model Confidence',
                'condition': lambda metrics: metrics.get('avg_confidence', 1) < 0.5,
                'severity': AlertSeverity.WARNING,
                'message': 'Average model confidence below 50%'
            },
            {
                'name': 'Slow Response Time',
                'condition': lambda metrics: metrics.get('avg_response_time', 0) > 2000,
                'severity': AlertSeverity.WARNING,
                'message': 'Average response time above 2 seconds'
            },
            {
                'name': 'Model Drift Detected',
                'condition': lambda metrics: metrics.get('confidence_drift', 0) > 0.3,
                'severity': AlertSeverity.ERROR,
                'message': 'Significant model drift detected'
            }
        ]
    
    async def check_alerts(self, metrics: Dict):
        """Проверка условий алертов"""
        
        for rule in self.alert_rules:
            if rule['condition'](metrics):
                await self.send_alert(
                    title=rule['name'],
                    message=rule['message'],
                    severity=rule['severity'],
                    metrics=metrics
                )
    
    async def send_alert(self, 
                        title: str,
                        message: str,
                        severity: AlertSeverity,
                        metrics: Dict = None):
        """Отправка алерта"""
        
        alert_data = {
            'title': title,
            'message': message,
            'severity': severity.value,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
        
        # Отправка в Slack
        if self.slack_webhook:
            await self._send_slack_alert(alert_data)
            
        # Отправка по email
        if self.email_config:
            await self._send_email_alert(alert_data)
            
        # Логирование
        logger.error(f"ALERT: {title} - {message}")
    
    async def _send_slack_alert(self, alert_data: Dict):
        """Отправка алерта в Slack"""
        
        color_map = {
            'info': '#36a64f',
            'warning': '#ff9500', 
            'error': '#ff0000',
            'critical': '#8b0000'
        }
        
        slack_message = {
            'attachments': [{
                'color': color_map.get(alert_data['severity'], '#ff0000'),
                'title': f"🚨 {alert_data['title']}",
                'text': alert_data['message'],
                'fields': [
                    {
                        'title': 'Severity',
                        'value': alert_data['severity'].upper(),
                        'short': True
                    },
                    {
                        'title': 'Timestamp',
                        'value': alert_data['timestamp'],
                        'short': True
                    }
                ]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(self.slack_webhook, json=slack_message)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
```

## 📋 Чек-лист для production

### Перед развертыванием:

- [ ] Все unit тесты проходят (>95% покрытие)
- [ ] Integration тесты API проходят
- [ ] Performance тесты показывают <500ms время ответа
- [ ] Нагрузочные тесты с 100+ пользователями
- [ ] Мониторинг настроен и работает
- [ ] Алерты настроены для критических метрик
- [ ] A/B тестирование готово к запуску
- [ ] Логирование настроено на production уровне
- [ ] Резервное копирование моделей настроено
- [ ] Rollback план подготовлен

### После развертывания:

- [ ] Мониторинг метрик в реальном времени
- [ ] Проверка алертов на тестовых данных
- [ ] Валидация A/B тестов
- [ ] Проверка производительности под нагрузкой
- [ ] Мониторинг пользовательской обратной связи
- [ ] Отслеживание дрейфа модели
- [ ] Регулярные проверки качества данных
