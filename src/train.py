import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml

from model import ExoplanetNet
from data import get_dataloaders
from trainer import Trainer
from callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(config):
    """
    Основная функция для обучения модели с использованием Trainer и коллбэков.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataloaders ---
    train_loader, val_loader = get_dataloaders(
        batch_size=config['batch_size'], 
        num_workers=config.get('num_workers', 0),
        path=config['data_path']
    )

    # --- Model ---
    model = ExoplanetNet(
        input_length=config['input_length'],
        num_blocks=config['num_blocks'],
        base_filters=config['base_filters'],
        kernel_size=config['kernel_size'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # --- Loss Function ---
    criterion = nn.CrossEntropyLoss()

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['callbacks']['early_stopping']['patience'],
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=os.path.join(config['log_dir'], 'best_model.pth'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['callbacks']['reduce_lr']['factor'],
            patience=config['callbacks']['reduce_lr']['patience'],
            verbose=1,
            mode='min'
        )
    ]

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs']
    )
    print("Training finished.")

if __name__ == '__main__':
    # Загрузка конфигурации
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создание директории для логов, если ее нет
    os.makedirs(config['log_dir'], exist_ok=True)

    train_model(config)

