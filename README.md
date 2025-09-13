# Exoplanet Transit Detection AI

Система искусственного интеллекта для обнаружения транзитов экзопланет в данных космических телескопов.

## 🌟 Возможности

- Поддержка данных с телескопов Kepler, K2 и TESS
- Автоматическое кэширование загруженных данных
- Продвинутые методы предобработки данных
- Валидация качества данных
- Нейронные сети CNN и LSTM для детекции

## 🚀 Быстрый старт

```python
from src import preprocess, model, detect, visualize

# Загрузка данных с кэшированием
times, flux = preprocess.load_lightcurve(
    "Kepler-10",
    mission=preprocess.DataSource.KEPLER,
    use_cache=True
)

# Валидация данных
validation = preprocess.validate_lightcurve(times, flux)
if not preprocess.check_data_quality(validation):
    print("Предупреждение: данные не соответствуют критериям качества")
    print(validation)

# Предобработка
flux = preprocess.remove_outliers(flux, method='mad')  # Робастное удаление выбросов
flux = preprocess.detrend(flux, times, method='polynomial')  # Удаление тренда
flux = preprocess.smooth_lightcurve(flux, method='savgol')  # Сглаживание
flux = preprocess.normalize_array(flux, method='robust')  # Нормализация

# Обучение модели
X_train, y_train = preprocess.generate_training_dataset(
    num_samples=5000,
    sequence_length=2000,
    transit_probability=0.3
)

model, history = detect.train_on_windows(
    X_train,
    y_train,
    model_type='cnn',
    epochs=30,
    batch_size=64,
    lr=0.0005
)

# Визуализация обучения
visualize.plot_training_history(history)

# Поиск транзитов
probs = detect.sliding_prediction_full(model, flux)
candidates = detect.extract_candidates(times, probs, threshold=0.7)

# Визуализация результатов
visualize.plot_lightcurve(times, flux, probs, candidates)
```

## 📊 Предобработка данных

### Нормализация

```python
# Доступны разные методы нормализации
flux_minmax = preprocess.normalize_array(flux, method='minmax')  # [0, 1]
flux_zscore = preprocess.normalize_array(flux, method='zscore')  # μ=0, σ=1
flux_robust = preprocess.normalize_array(flux, method='robust')  # На основе MAD
flux_percent = preprocess.normalize_array(flux, method='percent')  # Процентное отклонение
```

### Удаление выбросов

```python
# Различные методы детекции выбросов
flux_clean = preprocess.remove_outliers(flux, method='sigma')  # Сигма-клиппинг
flux_clean = preprocess.remove_outliers(flux, method='mad')    # MAD
flux_clean = preprocess.remove_outliers(flux, method='iqr')    # Межквартильный размах
flux_clean = preprocess.remove_outliers(flux, method='local')  # Локальный метод
```

### Сглаживание

```python
# Методы сглаживания
flux_smooth = preprocess.smooth_lightcurve(flux, method='savgol')    # Савицкий-Голей
flux_smooth = preprocess.smooth_lightcurve(flux, method='gaussian')  # Гауссово ядро
flux_smooth = preprocess.smooth_lightcurve(flux, method='median')    # Медианный фильтр
flux_smooth = preprocess.smooth_lightcurve(flux, method='lowess')    # LOWESS
```

### Удаление тренда

```python
# Методы удаления тренда
flux_detrend = preprocess.detrend(flux, times, method='polynomial')  # Полином
flux_detrend = preprocess.detrend(flux, times, method='spline')     # Сплайн
flux_detrend = preprocess.detrend(flux, times, method='median')     # Медианный тренд
```

## 🔍 Валидация данных

```python
# Проверка качества данных
validation = preprocess.validate_lightcurve(times, flux)

# Строгие критерии качества
is_good = preprocess.check_data_quality(validation, strict=True)

# Доступная информация
print(f"Временной интервал: {validation['time_span']} дней")
print(f"Медианный интервал: {validation['median_cadence']} дней")
print(f"Количество пропусков: {validation['n_gaps']}")
print(f"Количество выбросов: {validation['n_outliers']}")
```

## 🗄️ Кэширование

```python
# Загрузка с кэшированием
times, flux = preprocess.load_lightcurve(
    "Kepler-10",
    mission=preprocess.DataSource.KEPLER,
    use_cache=True  # Использовать кэш
)

# Кэш автоматически сохраняется в data/cache/
# Для каждого набора параметров создается уникальный ключ
```

## 📈 Визуализация

```python
# График процесса обучения
visualize.plot_training_history(history)

# Кривая блеска с детекциями
visualize.plot_lightcurve(times, flux, probs, candidates)

# Детальный просмотр кандидата
visualize.plot_candidate_details(times, flux, candidates[0])
```

## 🔧 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/username/exoplanet-ai.git
cd exoplanet-ai
```

2. Создайте виртуальное окружение:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## 📚 Зависимости

- PyTorch >= 2.0.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- lightkurve >= 2.0.0
- scipy >= 1.7.0
- astropy >= 5.0.0
- statsmodels >= 0.13.0

## 📝 Лицензия

MIT License. See [LICENSE](LICENSE) for more information.