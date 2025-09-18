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
    """
    # --- Заглушка для данных (замените на загрузку ваших реальных данных) ---
    # Создадим несбалансированную обучающую выборку: 95% класс 0, 5% класс 1
    num_train_samples = 1000
    num_val_samples = 200
    
    X_train = np.random.randn(num_train_samples, input_length)
    y_train = np.zeros(num_train_samples)
    y_train[:int(num_train_samples * 0.05)] = 1 # 5% - экзопланеты
    np.random.shuffle(y_train) # Перемешаем для случайности

    X_val = np.random.randn(num_val_samples, input_length)
    y_val = np.random.randint(0, 2, num_val_samples)
    # --------------------------------------------------------------------

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
