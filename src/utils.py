import torch
import numpy as np
import random
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Reproducibility ---
def set_seed(seed):
    """
    Устанавливает seed для всех генераторов случайных чисел для воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")

# --- File Handling ---
def load_config(path):
    """
    Загружает конфигурационный файл в формате YAML.
    """
    logging.info(f"Loading config from {path}...")
    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            logging.info("Config loaded successfully.")
            return config
        except yaml.YAMLError as e:
            logging.error(f"Error loading YAML file: {e}")
            raise
    
def save_config(config, path):
    """
    Сохраняет конфигурационный словарь в файл YAML.
    """
    logging.info(f"Saving config to {path}...")
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logging.info("Config saved successfully.")

def load_checkpoint(filepath, model, optimizer=None):
    """
    Загружает чекпоинт модели и состояние оптимизатора.
    """
    logging.info(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Optimizer state loaded.")
    epoch = checkpoint.get('epoch', -1)
    logging.info(f"Checkpoint loaded. Resuming from epoch {epoch}.")
    return model, optimizer, epoch

def save_checkpoint(state, filepath):
    """
    Сохраняет состояние (модель, оптимизатор, эпоха) в файл.
    """
    logging.info(f"Saving checkpoint to {filepath}...")
    torch.save(state, filepath)
    logging.info("Checkpoint saved successfully.")

# --- Metrics ---
def calculate_metrics(y_true, y_pred_logits, threshold=0.5):
    """
    Вычисляет и возвращает словарь с метриками: accuracy, precision, recall, f1-score.
    
    Args:
        y_true (torch.Tensor or np.array): Истинные метки.
        y_pred_logits (torch.Tensor or np.array): Логиты или вероятности от модели.
        threshold (float): Порог для бинарной классификации.

    Returns:
        dict: Словарь с метриками.
    """
    # Преобразуем в numpy, если это тензоры
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_logits, torch.Tensor):
        # Если на входе логиты, применяем softmax/sigmoid
        if y_pred_logits.ndim > 1 and y_pred_logits.shape[1] > 1: # Logits for multi-class
            y_pred_probs = torch.softmax(y_pred_logits, dim=1)
            y_pred_labels = torch.argmax(y_pred_probs, dim=1).cpu().numpy()
        else: # Logits for binary
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_labels = (y_pred_probs > threshold).cpu().numpy().astype(int)
    else: # Если на входе уже numpy
        y_pred_labels = (y_pred_logits > threshold).astype(int) if y_pred_logits.ndim == 1 else np.argmax(y_pred_logits, axis=1)

    accuracy = accuracy_score(y_true, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
