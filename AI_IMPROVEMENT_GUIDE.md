# 🚀 ExoplanetAI - Руководство по улучшению ИИ и интеграции умного поиска

## 📊 Анализ текущей архитектуры

### Текущие модели:
- **LightGBM**: Gradient boosting для табличных данных
- **Random Forest**: Устойчивость к шуму и переобучению  
- **1D-CNN**: Анализ временных рядов кривых блеска
- **Ensemble**: Voting classifier для финального решения

### Извлечение признаков (71 признак):
- Статистические: mean, std, skewness, kurtosis, перцентили
- Транзитные: depth, duration, SNR, форма (V/U/Box)
- Частотные: FFT, Lomb-Scargle, спектральные характеристики
- Морфологические: ingress/egress, вторичные затмения
- Качество данных: каденция, пропуски, шум

## 🎯 Стратегия улучшения точности

### 1. Новые признаки (Feature Engineering)

#### Астрофизические признаки:
```python
# Stellar contamination ratio
stellar_contamination = secondary_eclipse_depth / primary_transit_depth

# Limb darkening coefficient
limb_darkening = (ingress_duration + egress_duration) / transit_duration

# Orbital eccentricity indicator  
eccentricity_proxy = abs(secondary_eclipse_phase - 0.5)

# Multi-planet system indicators
ttv_amplitude = std(transit_timing_variations)
ttvs_periodicity = dominant_frequency(ttv_signal)
```

#### Признаки качества данных:
```python
# Photometric precision
photometric_precision = median(flux_err) / median(flux)

# Systematic noise level
systematic_noise = correlation(flux, time_trends)

# Data completeness
completeness = 1.0 - gap_fraction

# Instrumental effects
instrumental_score = correlation(flux, spacecraft_position)
```

### 2. Улучшение архитектуры моделей

#### Замена LightGBM на XGBoost с оптимизацией:
```python
xgb_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 3.0  # Для несбалансированных классов
}
```

#### Улучшенная CNN архитектура:
```python
def build_advanced_cnn():
    model = Sequential([
        # Attention mechanism
        Conv1D(64, 5, activation='relu'),
        BatchNormalization(),
        Attention(),
        
        # Residual blocks
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # LSTM для временных зависимостей
        LSTM(64, return_sequences=True),
        LSTM(32),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model
```

### 3. AutoML и гиперпараметрическая оптимизация

#### Optuna для поиска гиперпараметров:
```python
import optuna

def objective(trial):
    # XGBoost параметры
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    # CNN параметры
    cnn_params = {
        'filters': trial.suggest_categorical('filters', [32, 64, 128]),
        'kernel_size': trial.suggest_int('kernel_size', 3, 7),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'lstm_units': trial.suggest_int('lstm_units', 32, 128)
    }
    
    # Обучение и валидация
    model = train_ensemble(xgb_params, cnn_params)
    score = cross_validate(model, X_val, y_val)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 🔍 Архитектура умного поиска

### 1. Новый API эндпоинт для анализа

```python
@router.post("/ai/analyze_lightcurve")
async def smart_lightcurve_analysis(request: SmartAnalysisRequest):
    """
    Умный анализ кривой блеска с улучшенным ИИ
    """
    # Предобработка с адаптивными параметрами
    preprocessor = AdaptivePreprocessor()
    processed_data = preprocessor.smart_preprocess(
        time=request.time_data,
        flux=request.flux_data,
        auto_detect_cadence=True,
        adaptive_detrending=True
    )
    
    # Извлечение расширенного набора признаков
    feature_extractor = EnhancedFeatureExtractor()
    features = feature_extractor.extract_all_features(
        processed_data,
        include_astrophysical=True,
        include_quality_metrics=True
    )
    
    # Ensemble предсказание с uncertainty quantification
    ensemble = OptimizedEnsemble()
    prediction = ensemble.predict_with_uncertainty(features)
    
    # Интерпретируемость через SHAP
    explanation = ensemble.explain_prediction(features)
    
    return SmartAnalysisResult(
        predicted_class=prediction.class_name,
        confidence_score=prediction.confidence,
        uncertainty_bounds=prediction.uncertainty,
        transit_probability=prediction.transit_prob,
        signal_characteristics=prediction.signal_params,
        feature_importance=explanation.feature_importance,
        decision_reasoning=explanation.reasoning,
        recommendations=generate_recommendations(prediction)
    )
```

### 2. Интеллектуальные фильтры поиска

```python
class SmartSearchFilters:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.snr_threshold = 5.0
        self.data_quality_threshold = 0.8
        
    def apply_smart_filters(self, results: List[AnalysisResult]) -> List[AnalysisResult]:
        filtered = []
        
        for result in results:
            # Адаптивные пороги на основе качества данных
            adaptive_confidence = self.calculate_adaptive_threshold(result)
            
            if (result.confidence_score >= adaptive_confidence and
                result.signal_characteristics.snr >= self.snr_threshold and
                result.data_quality_score >= self.data_quality_threshold):
                
                # Дополнительная проверка на ложные позитивы
                if not self.is_likely_false_positive(result):
                    filtered.append(result)
                    
        return self.rank_by_discovery_potential(filtered)
```

### 3. Кэширование и оптимизация

```python
# Redis кэширование результатов
@cache_result(ttl=3600)  # 1 час
async def cached_analysis(target_id: str, analysis_params: dict):
    return await perform_analysis(target_id, analysis_params)

# Batch обработка для множественных целей
@router.post("/ai/batch_analyze")
async def batch_lightcurve_analysis(targets: List[str]):
    tasks = [smart_lightcurve_analysis(target) for target in targets]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return BatchAnalysisResult(results=results)
```

## 🔒 Безопасность и валидация

### JWT авторизация:
```python
@router.post("/ai/analyze_lightcurve")
@require_auth(roles=["researcher", "admin"])
@rate_limit(requests_per_minute=10)
async def protected_analysis(
    request: SmartAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    # Валидация входных данных
    validator = InputValidator()
    validated_data = validator.validate_lightcurve_data(request)
    
    # Логирование для аудита
    audit_logger.info(f"Analysis requested by {current_user.username}")
    
    return await smart_lightcurve_analysis(validated_data)
```

## 🌐 Фронтенд интеграция

### React компонент умного поиска:
```typescript
const SmartSearch: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({
    confidenceMin: 0.7,
    snrMin: 5.0,
    dataQualityMin: 0.8,
    missions: ['TESS', 'Kepler'],
    planetTypes: ['confirmed', 'candidate']
  });

  const handleSmartSearch = async () => {
    const response = await apiClient.post('/ai/smart_search', {
      query: searchQuery,
      filters: filters,
      use_ai_ranking: true
    });
    
    setResults(response.data.results);
    setRecommendations(response.data.recommendations);
  };

  return (
    <div className="smart-search-container">
      <SearchInput 
        value={searchQuery}
        onChange={setSearchQuery}
        placeholder="Введите TIC ID, координаты или параметры транзита..."
      />
      
      <AdvancedFilters 
        filters={filters}
        onChange={setFilters}
      />
      
      <SearchResults 
        results={results}
        onAnalyze={handleDetailedAnalysis}
      />
      
      <AIRecommendations 
        recommendations={recommendations}
      />
    </div>
  );
};
```

## 📈 Мониторинг и метрики

### Система мониторинга точности:
```python
class ModelMonitor:
    def __init__(self):
        self.metrics_tracker = MetricsTracker()
        
    async def track_prediction_quality(self, prediction: Prediction, ground_truth: Optional[str] = None):
        # Отслеживание drift модели
        feature_drift = self.detect_feature_drift(prediction.features)
        
        # Uncertainty calibration
        uncertainty_quality = self.assess_uncertainty_calibration(prediction)
        
        # Performance metrics
        if ground_truth:
            accuracy = self.calculate_accuracy(prediction.class_name, ground_truth)
            self.metrics_tracker.log_accuracy(accuracy)
            
        # Алерты при деградации
        if feature_drift > 0.1 or uncertainty_quality < 0.8:
            await self.send_alert("Model performance degradation detected")
```

## 🚀 План внедрения

### Фаза 1 (2 недели):
1. Реализация новых признаков
2. Оптимизация XGBoost с Optuna
3. Создание базового умного поиска API

### Фаза 2 (3 недели):
1. Улучшенная CNN архитектура
2. Интеграция фронтенд компонентов
3. Система кэширования Redis

### Фаза 3 (2 недели):
1. Мониторинг и алерты
2. A/B тестирование новых моделей
3. Production развертывание

## 📊 Ожидаемые результаты

- **Точность**: +15-20% (с 85% до 95%+)
- **Скорость**: <500ms для анализа
- **Ложные позитивы**: -50% (с 10% до 5%)
- **Пользовательский опыт**: Интуитивный поиск с ИИ рекомендациями
