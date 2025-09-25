"""
LSTM Classifier for Transit Detection

Рекуррентная нейронная сеть для анализа временных рядов кривых блеска.
Учитывает временную динамику и долгосрочные зависимости в данных.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

from .base_model import BaseTransitModel

logger = logging.getLogger(__name__)

class LSTMClassifier(BaseTransitModel):
    """
    LSTM для классификации транзитов экзопланет
    
    Архитектура:
    - Bidirectional LSTM слои для захвата паттернов в обоих направлениях
    - Attention механизм для фокусировки на важных временных моментах
    - Dropout для регуляризации
    - Fully connected слои для классификации
    """
    
    def __init__(self,
                 input_size: int = 1024,
                 num_classes: int = 2,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 use_layer_norm: bool = True,
                 **kwargs):
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        
        super().__init__(input_size, num_classes, dropout, **kwargs)
        
    def _build_model(self):
        """Построение LSTM архитектуры"""
        
        # Входная проекция (если нужно)
        self.input_projection = nn.Linear(1, 32)
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Размер выхода LSTM
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Layer normalization
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Attention механизм
        if self.use_attention:
            self.attention = TemporalAttention(lstm_output_size)
            attention_output_size = lstm_output_size
        else:
            attention_output_size = lstm_output_size
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(attention_output_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes)
        )
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(attention_output_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов модели"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Забыть гейт bias = 1 (стандартная практика для LSTM)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход LSTM
        
        Args:
            x: Входные данные [batch_size, sequence_length] или [batch_size, 1, sequence_length]
            
        Returns:
            Логиты классов [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Приведение к нужной размерности
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Убираем канал если есть
        
        # Reshape для LSTM: [batch_size, seq_len, features]
        x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Входная проекция
        x = self.input_projection(x)  # [batch_size, seq_len, 32]
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, hidden_size * directions]
        
        # Layer normalization
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        # Attention или pooling
        if self.use_attention:
            # Attention по временной оси
            attended_output = self.attention(lstm_out)  # [batch_size, hidden_size * directions]
        else:
            # Простое усреднение по времени
            attended_output = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_size * directions]
        
        # Классификация
        logits = self.classifier(attended_output)
        
        return logits
    
    def _extract_features_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков из LSTM"""
        batch_size = x.shape[0]
        
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        
        lstm_out, _ = self.lstm(x)
        
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        if self.use_attention:
            features = self.attention(lstm_out)
        else:
            features = torch.mean(lstm_out, dim=1)
        
        # Применяем feature extractor
        features = self.feature_extractor(features)
        
        return features
    
    def get_hidden_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получение скрытых состояний LSTM
        
        Args:
            x: Входные данные
            
        Returns:
            Tuple из (lstm_output, final_hidden_state)
        """
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 3:
                x = x.squeeze(1)
            
            x = x.unsqueeze(-1)
            x = self.input_projection(x)
            
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Финальное скрытое состояние
            if self.bidirectional:
                # Объединяем forward и backward состояния
                final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                final_hidden = hidden[-1]
        
        return lstm_out, final_hidden
    
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
            if len(x.shape) == 3:
                x = x.squeeze(1)
            
            x = x.unsqueeze(-1)
            x = self.input_projection(x)
            
            lstm_out, _ = self.lstm(x)
            
            if self.use_layer_norm:
                lstm_out = self.layer_norm(lstm_out)
            
            # Получаем attention веса
            _, attention_weights = self.attention(lstm_out, return_attention=True)
        
        return attention_weights


class TemporalAttention(nn.Module):
    """
    Temporal Attention механизм для LSTM выходов
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Attention layers
        self.attention_linear = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        
        # Инициализация
        nn.init.xavier_uniform_(self.attention_linear.weight)
        nn.init.xavier_uniform_(self.context_vector.weight)
        
    def forward(self, lstm_output: torch.Tensor, return_attention: bool = False):
        """
        Args:
            lstm_output: [batch_size, seq_len, hidden_size]
            return_attention: Возвращать ли веса attention
            
        Returns:
            Взвешенное представление последовательности
        """
        batch_size, seq_len, hidden_size = lstm_output.shape
        
        # Вычисляем attention scores
        attention_hidden = torch.tanh(self.attention_linear(lstm_output))  # [B, L, H]
        attention_scores = self.context_vector(attention_hidden).squeeze(-1)  # [B, L]
        
        # Softmax для получения весов
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, L]
        
        # Взвешенная сумма
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, L]
            lstm_output                      # [B, L, H]
        ).squeeze(1)  # [B, H]
        
        if return_attention:
            return attended_output, attention_weights
        
        return attended_output


class GRUClassifier(BaseTransitModel):
    """
    Альтернативная реализация с GRU вместо LSTM
    GRU быстрее и иногда показывает лучшие результаты на коротких последовательностях
    """
    
    def __init__(self,
                 input_size: int = 1024,
                 num_classes: int = 2,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 **kwargs):
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        super().__init__(input_size, num_classes, dropout, **kwargs)
        
    def _build_model(self):
        """Построение GRU архитектуры"""
        
        self.input_projection = nn.Linear(1, 32)
        
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        gru_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        if self.use_attention:
            self.attention = TemporalAttention(gru_output_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes)
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход GRU"""
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        
        gru_out, hidden = self.gru(x)
        
        if self.use_attention:
            output = self.attention(gru_out)
        else:
            output = torch.mean(gru_out, dim=1)
        
        logits = self.classifier(output)
        return logits
    
    def _extract_features_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков из GRU"""
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        
        gru_out, _ = self.gru(x)
        
        if self.use_attention:
            features = self.attention(gru_out)
        else:
            features = torch.mean(gru_out, dim=1)
        
        features = self.feature_extractor(features)
        return features
