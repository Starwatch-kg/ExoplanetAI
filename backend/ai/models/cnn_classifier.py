"""
CNN Classifier for Transit Detection

Сверточная нейронная сеть для выделения транзитных сигналов от шума.
Использует 1D свертки для анализа кривых блеска.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

from .base_model import BaseTransitModel

logger = logging.getLogger(__name__)

class CNNClassifier(BaseTransitModel):
    """
    CNN для классификации транзитов экзопланет
    
    Архитектура:
    - Несколько сверточных блоков с batch normalization
    - Residual connections для глубоких сетей
    - Attention механизм для фокусировки на важных участках
    - Fully connected слои для финальной классификации
    """
    
    def __init__(self,
                 input_size: int = 1024,
                 num_classes: int = 2,
                 num_filters: Tuple[int, ...] = (32, 64, 128, 256),
                 kernel_sizes: Tuple[int, ...] = (7, 5, 3, 3),
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 use_residual: bool = True,
                 **kwargs):
        
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        super().__init__(input_size, num_classes, dropout, **kwargs)
        
    def _build_model(self):
        """Построение CNN архитектуры"""
        
        # Входной слой
        self.input_conv = nn.Conv1d(1, self.num_filters[0], 
                                   kernel_size=self.kernel_sizes[0], 
                                   padding=self.kernel_sizes[0]//2)
        self.input_bn = nn.BatchNorm1d(self.num_filters[0])
        
        # Сверточные блоки
        self.conv_blocks = nn.ModuleList()
        
        for i in range(len(self.num_filters) - 1):
            block = self._make_conv_block(
                self.num_filters[i],
                self.num_filters[i + 1],
                self.kernel_sizes[i + 1] if i + 1 < len(self.kernel_sizes) else 3
            )
            self.conv_blocks.append(block)
        
        # Attention механизм
        if self.use_attention:
            self.attention = SelfAttention1D(self.num_filters[-1])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier head
        feature_size = self.num_filters[-1] * 2  # avg + max pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )
        
        # Для извлечения признаков
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int):
        """Создание сверточного блока"""
        
        layers = []
        
        # Основная свертка
        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ])
        
        # Дополнительная свертка в блоке
        layers.extend([
            nn.Conv1d(out_channels, out_channels, kernel_size, 
                     padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
        ])
        
        block = nn.Sequential(*layers)
        
        # Residual connection
        if self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        
        return block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход CNN
        
        Args:
            x: Входные данные [batch_size, sequence_length] или [batch_size, 1, sequence_length]
            
        Returns:
            Логиты классов [batch_size, num_classes]
        """
        # Приведение к нужной размерности
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, sequence_length]
        
        # Входной слой
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # Сверточные блоки с residual connections
        for i, block in enumerate(self.conv_blocks):
            residual = x
            
            x = block(x)
            
            # Residual connection
            if self.use_residual:
                if hasattr(self, 'residual_conv') and residual.shape[1] != x.shape[1]:
                    residual = self.residual_conv(residual)
                
                if residual.shape == x.shape:
                    x = x + residual
            
            x = F.relu(x)
            
            # Downsampling через max pooling
            if i < len(self.conv_blocks) - 1:
                x = F.max_pool1d(x, kernel_size=2)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        # Global pooling
        avg_pool = self.global_pool(x).squeeze(-1)  # [batch_size, channels]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [batch_size, channels]
        
        # Объединяем avg и max pooling
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Классификация
        logits = self.classifier(features)
        
        return logits
    
    def _extract_features_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков из CNN"""
        # Повторяем forward до pooling
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        for i, block in enumerate(self.conv_blocks):
            residual = x
            x = block(x)
            
            if self.use_residual and hasattr(self, 'residual_conv') and residual.shape[1] != x.shape[1]:
                residual = self.residual_conv(residual)
            
            if self.use_residual and residual.shape == x.shape:
                x = x + residual
            
            x = F.relu(x)
            
            if i < len(self.conv_blocks) - 1:
                x = F.max_pool1d(x, kernel_size=2)
        
        if self.use_attention:
            x = self.attention(x)
        
        # Global pooling
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Применяем feature extractor
        features = self.feature_extractor(features)
        
        return features
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Получение весов attention для интерпретации
        
        Args:
            x: Входные данные
            
        Returns:
            Веса attention или None если attention не используется
        """
        if not self.use_attention:
            return None
        
        self.eval()
        with torch.no_grad():
            # Прогоняем до attention слоя
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            x = self.input_conv(x)
            x = self.input_bn(x)
            x = F.relu(x)
            
            for i, block in enumerate(self.conv_blocks):
                residual = x
                x = block(x)
                
                if self.use_residual and hasattr(self, 'residual_conv') and residual.shape[1] != x.shape[1]:
                    residual = self.residual_conv(residual)
                
                if self.use_residual and residual.shape == x.shape:
                    x = x + residual
                
                x = F.relu(x)
                
                if i < len(self.conv_blocks) - 1:
                    x = F.max_pool1d(x, kernel_size=2)
            
            # Получаем attention веса
            _, attention_weights = self.attention(x, return_attention=True)
            
        return attention_weights


class SelfAttention1D(nn.Module):
    """
    Self-Attention механизм для 1D последовательностей
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Query, Key, Value проекции
        self.query_conv = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, 1)
        
        # Выходная проекция
        self.output_conv = nn.Conv1d(in_channels, in_channels, 1)
        
        # Инициализация
        nn.init.kaiming_normal_(self.query_conv.weight)
        nn.init.kaiming_normal_(self.key_conv.weight)
        nn.init.kaiming_normal_(self.value_conv.weight)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: [batch_size, channels, length]
            return_attention: Возвращать ли веса attention
            
        Returns:
            Выход attention + остаточное соединение
        """
        batch_size, channels, length = x.shape
        
        # Вычисляем Q, K, V
        query = self.query_conv(x)  # [B, C//r, L]
        key = self.key_conv(x)      # [B, C//r, L]
        value = self.value_conv(x)  # [B, C, L]
        
        # Reshape для matrix multiplication
        query = query.view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C//r]
        key = key.view(batch_size, -1, length)                       # [B, C//r, L]
        value = value.view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C]
        
        # Attention scores
        attention_scores = torch.bmm(query, key)  # [B, L, L]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Применяем attention к values
        attended = torch.bmm(attention_weights, value)  # [B, L, C]
        attended = attended.permute(0, 2, 1)  # [B, C, L]
        
        # Выходная проекция
        output = self.output_conv(attended)
        
        # Residual connection с learnable weight
        output = self.gamma * output + x
        
        if return_attention:
            return output, attention_weights
        
        return output
