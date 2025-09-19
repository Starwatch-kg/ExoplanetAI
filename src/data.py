import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def add_noise(signal, noise_level=0.01):
    """Добавляет гауссовский шум к сигналу."""
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def random_shift(signal, max_shift=100):
    """Случайно сдвигает сигнал."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(signal, shift)

def generate_transit_signal(length, period, depth, duration, phase=0):
    """Генерирует сигнал транзита экзопланеты"""
    times = np.arange(length)
    signal = np.ones(length)
    
    # Добавляем транзиты с заданным периодом
    transit_times = np.arange(phase, length, period)
    
    for t_center in transit_times:
        # Создаем трапециевидный транзит
        ingress_duration = duration * 0.1
        egress_duration = duration * 0.1
        flat_duration = duration * 0.8
        
        t_start = t_center - duration / 2
        t_end = t_center + duration / 2
        
        # Вход в транзит
        ingress_mask = (times >= t_start) & (times < t_start + ingress_duration)
        signal[ingress_mask] = 1 - depth * (times[ingress_mask] - t_start) / ingress_duration
        
        # Плоская часть транзита
        flat_mask = (times >= t_start + ingress_duration) & (times < t_end - egress_duration)
        signal[flat_mask] = 1 - depth
        
        # Выход из транзита
        egress_mask = (times >= t_end - egress_duration) & (times < t_end)
        signal[egress_mask] = 1 - depth * (1 - (times[egress_mask] - (t_end - egress_duration)) / egress_duration)
    
    return signal

def generate_stellar_variability(length, amplitude=0.01, period=None):
    """Генерирует звездную переменность"""
    times = np.arange(length)
    
    if period is None:
        period = np.random.uniform(50, 500)  # Период ротации звезды
    
    # Основная гармоника
    variability = amplitude * np.sin(2 * np.pi * times / period)
    
    # Добавляем высшие гармоники
    variability += amplitude * 0.3 * np.sin(4 * np.pi * times / period)
    variability += amplitude * 0.1 * np.sin(6 * np.pi * times / period)
    
    return variability

def generate_realistic_lightcurves(n_samples, length):
    """
    Генерирует реалистичные кривые блеска с транзитами и без них
    
    Args:
        n_samples: количество образцов
        length: длина временного ряда
    
    Returns:
        X: массив кривых блеска
        y: метки классов (0 - без транзита, 1 - с транзитом)
    """
    X = np.zeros((n_samples, length))
    y = np.zeros(n_samples)
    
    # 20% образцов с транзитами (реалистичная статистика)
    n_with_transits = int(n_samples * 0.2)
    transit_indices = np.random.choice(n_samples, n_with_transits, replace=False)
    
    for i in range(n_samples):
        # Базовый поток
        lightcurve = np.ones(length)
        
        # Добавляем реалистичный шум (зависит от яркости звезды)
        magnitude = np.random.uniform(8, 16)  # TESS магнитуда
        noise_level = 10**(0.4 * (magnitude - 10)) * 1e-4
        noise = np.random.normal(0, noise_level, length)
        lightcurve += noise
        
        # Добавляем звездную переменность
        if np.random.random() < 0.7:  # 70% звезд показывают переменность
            variability_amplitude = np.random.uniform(0.001, 0.01)
            variability = generate_stellar_variability(length, variability_amplitude)
            lightcurve += variability
        
        # Добавляем инструментальные эффекты
        # Дрейф
        drift = np.random.uniform(-1e-5, 1e-5) * np.arange(length)
        lightcurve += drift
        
        # Периодические систематики
        systematic_period = np.random.uniform(10, 100)
        systematic_amp = np.random.uniform(1e-5, 1e-4)
        systematic = systematic_amp * np.sin(2 * np.pi * np.arange(length) / systematic_period)
        lightcurve += systematic
        
        # Добавляем транзит если нужно
        if i in transit_indices:
            # Реалистичные параметры транзита
            period = np.random.uniform(1, 50)  # Период в кадрах
            depth = np.random.uniform(0.0005, 0.02)  # Глубина транзита
            duration = np.random.uniform(5, 30)  # Длительность в кадрах
            phase = np.random.uniform(0, period)
            
            transit_signal = generate_transit_signal(length, period, depth, duration, phase)
            lightcurve *= transit_signal
            y[i] = 1
        
        # Добавляем случайные выбросы (космические лучи)
        n_outliers = np.random.poisson(2)
        for _ in range(n_outliers):
            outlier_pos = np.random.randint(0, length)
            outlier_amp = np.random.uniform(0.01, 0.1)
            lightcurve[outlier_pos] += outlier_amp
        
        X[i] = lightcurve
    
    return X, y

class ExoplanetDataset(Dataset):
    """
    Кастомный Dataset для кривых блеска.
    Выполняет нормализацию и аугментацию на лету.
    """
    def __init__(self, features, labels, augmentations=False):
        self.features = features
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        signal = self.features[idx].copy()
        label = self.labels[idx]

        # Аугментация (только для обучающей выборки)
        if self.augmentations:
            if np.random.rand() > 0.5:
                signal = add_noise(signal, noise_level=0.01)
            if np.random.rand() > 0.5:
                signal = random_shift(signal, max_shift=100)

        # Нормализация (mean=0, std=1) для каждого сэмпла
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        # Преобразование в тензоры PyTorch
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0) # (1, input_length)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor

def get_dataloaders(batch_size, input_length):
    """
    Создает и возвращает DataLoader'ы для обучения и валидации.
    Использует WeightedRandomSampler для балансировки классов в обучающей выборке.
    Генерирует реалистичные кривые блеска с транзитами экзопланет.
    """
    # Генерация реалистичных данных кривых блеска
    num_train_samples = 1000
    num_val_samples = 200
    
    # Создаем реалистичные кривые блеска
    X_train, y_train = generate_realistic_lightcurves(num_train_samples, input_length)
    X_val, y_val = generate_realistic_lightcurves(num_val_samples, input_length)

    train_dataset = ExoplanetDataset(X_train, y_train, augmentations=True)
    val_dataset = ExoplanetDataset(X_val, y_val, augmentations=False)

    # --- Балансировка классов для обучающей выборки ---
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for t in y_train.astype(int)])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
