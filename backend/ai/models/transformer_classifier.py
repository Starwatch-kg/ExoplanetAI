"""
Transformer Classifier for Transit Detection

Современная архитектура Transformer для анализа временных рядов кривых блеска.
Использует self-attention для захвата долгосрочных зависимостей.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import logging

from .base_model import BaseTransitModel

logger = logging.getLogger(__name__)

class TransformerClassifier(BaseTransitModel):
    """
    Transformer для классификации транзитов экзопланет
    
    Архитектура:
    - Positional encoding для временных последовательностей
    - Multi-head self-attention слои
    - Feed-forward networks с residual connections
    - Layer normalization
    - Classification head с global pooling
    """
    
    def __init__(self,
                 input_size: int = 1024,
                 num_classes: int = 2,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 2048,
                 use_positional_encoding: bool = True,
                 pooling_strategy: str = 'cls',  # 'cls', 'mean', 'max', 'attention'
                 **kwargs):
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        self.use_positional_encoding = use_positional_encoding
        self.pooling_strategy = pooling_strategy
        
        super().__init__(input_size, num_classes, dropout, **kwargs)
        
    def _build_model(self):
        """Построение Transformer архитектуры"""
        
        # Входная проекция
        self.input_projection = nn.Linear(1, self.d_model)
        
        # Positional encoding
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                self.d_model, 
                dropout=self.dropout,
                max_len=self.max_seq_length
            )
        
        # CLS token для классификации (если используется)
        if self.pooling_strategy == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # Pooling layer для attention pooling
        if self.pooling_strategy == 'attention':
            self.attention_pool = AttentionPooling(self.d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Прямой проход Transformer
        
        Args:
            x: Входные данные [batch_size, sequence_length] или [batch_size, 1, sequence_length]
            src_key_padding_mask: Маска для padding токенов [batch_size, sequence_length]
            
        Returns:
            Логиты классов [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Приведение к нужной размерности
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Убираем канал если есть
        
        seq_length = x.shape[1]
        
        # Обрезаем или дополняем последовательность
        if seq_length > self.max_seq_length:
            x = x[:, :self.max_seq_length]
            seq_length = self.max_seq_length
        
        # Reshape для transformer: [batch_size, seq_len, 1]
        x = x.unsqueeze(-1)
        
        # Входная проекция
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Добавляем CLS token если нужно
        if self.pooling_strategy == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len + 1, d_model]
            
            # Обновляем маску если есть
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        
        # Positional encoding
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        
        # Transformer encoder
        transformer_output = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_len, d_model]
        
        # Pooling strategy
        if self.pooling_strategy == 'cls':
            # Используем CLS token
            pooled_output = transformer_output[:, 0]  # [batch_size, d_model]
        elif self.pooling_strategy == 'mean':
            # Среднее по последовательности
            if src_key_padding_mask is not None:
                # Учитываем маску при усреднении
                mask = ~src_key_padding_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
                masked_output = transformer_output * mask
                pooled_output = masked_output.sum(dim=1) / mask.sum(dim=1)
            else:
                pooled_output = transformer_output.mean(dim=1)
        elif self.pooling_strategy == 'max':
            # Максимум по последовательности
            pooled_output, _ = transformer_output.max(dim=1)
        elif self.pooling_strategy == 'attention':
            # Attention pooling
            pooled_output = self.attention_pool(transformer_output, src_key_padding_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Классификация
        logits = self.classifier(pooled_output)
        
        return logits
    
    def _extract_features_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков из Transformer"""
        batch_size = x.shape[0]
        
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        seq_length = x.shape[1]
        
        if seq_length > self.max_seq_length:
            x = x[:, :self.max_seq_length]
        
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        
        if self.pooling_strategy == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        
        transformer_output = self.transformer_encoder(x)
        
        # Pooling
        if self.pooling_strategy == 'cls':
            features = transformer_output[:, 0]
        elif self.pooling_strategy == 'mean':
            features = transformer_output.mean(dim=1)
        elif self.pooling_strategy == 'max':
            features, _ = transformer_output.max(dim=1)
        elif self.pooling_strategy == 'attention':
            features = self.attention_pool(transformer_output)
        
        # Применяем feature extractor
        features = self.feature_extractor(features)
        
        return features
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Получение attention карт для интерпретации
        
        Args:
            x: Входные данные
            
        Returns:
            Список attention карт для каждого слоя
        """
        self.eval()
        attention_maps = []
        
        # Модифицируем forward hook для сбора attention
        def hook_fn(module, input, output):
            if hasattr(output, 'attention_weights'):
                attention_maps.append(output.attention_weights)
        
        # Регистрируем hooks
        hooks = []
        for layer in self.transformer_encoder.layers:
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = self.forward(x)
        finally:
            # Удаляем hooks
            for hook in hooks:
                hook.remove()
        
        return attention_maps


class PositionalEncoding(nn.Module):
    """
    Positional Encoding для Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Создаем positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :, :].transpose(0, 1)  # [batch_size, seq_len, d_model]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling для агрегации последовательности
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] - True для padding токенов
            
        Returns:
            Pooled representation [batch_size, d_model]
        """
        # Вычисляем attention scores
        attention_scores = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Применяем маску если есть
        if mask is not None:
            attention_scores.masked_fill_(mask, float('-inf'))
        
        # Softmax для получения весов
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Взвешенная сумма
        pooled = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            x                                # [batch_size, seq_len, d_model]
        ).squeeze(1)  # [batch_size, d_model]
        
        return pooled


class TimeSeriesTransformer(TransformerClassifier):
    """
    Специализированная версия Transformer для временных рядов
    с дополнительными возможностями для анализа кривых блеска
    """
    
    def __init__(self,
                 input_size: int = 1024,
                 num_classes: int = 2,
                 patch_size: int = 16,
                 **kwargs):
        
        self.patch_size = patch_size
        super().__init__(input_size, num_classes, **kwargs)
        
    def _build_model(self):
        """Построение архитектуры с patch embedding"""
        
        # Patch embedding вместо простой проекции
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size,
            d_model=self.d_model
        )
        
        # Остальная архитектура такая же
        super()._build_model()
        
        # Переопределяем input_projection
        self.input_projection = self.patch_embedding
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward с patch embedding"""
        # Patch embedding уже делает нужные преобразования
        return super().forward(x, **kwargs)


class PatchEmbedding(nn.Module):
    """
    Patch Embedding для временных рядов
    Разбивает последовательность на патчи и проецирует их
    """
    
    def __init__(self, patch_size: int = 16, d_model: int = 256):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Проекция патча в d_model размерность
        self.projection = nn.Linear(patch_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] или [batch_size, 1, seq_len]
            
        Returns:
            Patch embeddings [batch_size, num_patches, d_model]
        """
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        batch_size, seq_len = x.shape
        
        # Обрезаем до кратного patch_size
        num_patches = seq_len // self.patch_size
        if num_patches == 0:
            # Если последовательность короче patch_size, дополняем нулями
            padding = self.patch_size - seq_len
            x = F.pad(x, (0, padding))
            num_patches = 1
        else:
            x = x[:, :num_patches * self.patch_size]
        
        # Разбиваем на патчи
        patches = x.view(batch_size, num_patches, self.patch_size)
        
        # Проецируем каждый патч
        embeddings = self.projection(patches)
        
        return embeddings
