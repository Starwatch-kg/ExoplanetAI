import torch
from torch.cuda.amp import GradScaler, autocast
import logging
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    """
    Гибкий класс для обучения моделей PyTorch с поддержкой коллбэков, 
    смешанной точности и гибкой настройки.
    """
    def __init__(self, model, optimizer, criterion, device, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks if callbacks else []
        self.stop_training = False
        self.scaler = GradScaler()

        # Привязываем трейнер к коллбэкам
        for cb in self.callbacks:
            cb.set_trainer(self)

    def _run_epoch(self, data_loader, is_training=True):
        """Запускает одну эпоху обучения или валидации."""
        self.model.train(is_training)
        total_loss = 0.0
        
        progress_bar = tqdm(data_loader, desc="Training" if is_training else "Validation")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(is_training):
                with autocast():  # Mixed precision
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(data_loader)

    def fit(self, train_loader, val_loader, epochs, scheduler=None):
        """Основной метод для запуска процесса обучения."""
        logs = {}
        self._call_callbacks('on_train_begin', logs=logs)

        for epoch in range(epochs):
            self._call_callbacks('on_epoch_begin', epoch=epoch, logs=logs)

            train_loss = self._run_epoch(train_loader, is_training=True)
            val_loss = self._run_epoch(val_loader, is_training=False)
            
            lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")

            logs = {'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr}
            
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Этот шедулер использует метрику, а не шаг
                    scheduler.step(val_loss)
                else:
                    # Большинство шедулеров делают шаг без аргументов
                    scheduler.step()

            self._call_callbacks('on_epoch_end', epoch=epoch, logs=logs)

            if self.stop_training:
                logging.info("Training stopped by a callback.")
                break
        
        self._call_callbacks('on_train_end', logs=logs)

    def _call_callbacks(self, method_name, **kwargs):
        """Вызывает указанный метод у всех коллбэков."""
        for cb in self.callbacks:
            method = getattr(cb, method_name, None)
            if method:
                method(**kwargs)
