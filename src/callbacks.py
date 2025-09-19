import numpy as np
import torch
import os
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Callback:
    """Абстрактный базовый класс для всех коллбэков."""
    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

class EarlyStopping(Callback):
    """
    Останавливает обучение, если отслеживаемый показатель не улучшается в течение `patience` эпох.
    """
    def __init__(self, monitor='val_loss', patience=5, min_delta=1e-4, mode='min', verbose=1):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.wait = 0
        self.best_score = np.inf if mode == 'min' else -np.inf
        
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

    def on_epoch_end(self, epoch, logs=None):
        score = logs.get(self.monitor)
        if score is None:
            logging.warning(f"EarlyStopping requires {self.monitor} in logs. Skipping.")
            return

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.trainer.stop_training = True
                if self.verbose > 0:
                    logging.info(f"Epoch {epoch+1}: Early stopping triggered after {self.patience} epochs with no improvement.")

class ModelCheckpoint(Callback):
    """
    Сохраняет модель после каждой эпохи, если отслеживаемый показатель улучшился.
    """
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_score = np.inf if mode == 'min' else -np.inf
        
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
            
        # Создаем директорию, если ее нет
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        score = logs.get(self.monitor)
        if score is None:
            logging.warning(f"ModelCheckpoint requires {self.monitor} in logs. Skipping.")
            return

        if self.save_best_only:
            if self.mode == 'min':
                improved = score < self.best_score
            else:
                improved = score > self.best_score

            if improved:
                if self.verbose > 0:
                    logging.info(f"Epoch {epoch+1}: {self.monitor} improved from {self.best_score:.4f} to {score:.4f}. Saving model to {self.filepath}")
                self.best_score = score
                torch.save(self.trainer.model.state_dict(), self.filepath)
        else:
            # Сохраняем модель в конце каждой эпохи
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.verbose > 0:
                logging.info(f"Epoch {epoch+1}: saving model to {filepath}")
            torch.save(self.trainer.model.state_dict(), filepath)


class ReduceLROnPlateau(Callback):
    """
    Уменьшает learning rate, когда метрика перестает улучшаться.
    """
    def __init__(self, monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, mode='min', verbose=1):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        self.wait = 0
        self.best_score = np.inf if mode == 'min' else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        score = logs.get(self.monitor)
        if score is None:
            logging.warning(f"ReduceLROnPlateau requires {self.monitor} in logs. Skipping.")
            return

        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score

        if improved:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                old_lr = float(self.trainer.optimizer.param_groups[0]['lr'])
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if self.verbose > 0:
                        logging.info(f"Epoch {epoch+1}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}.")
                    self.trainer.optimizer.param_groups[0]['lr'] = new_lr
