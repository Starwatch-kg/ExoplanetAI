"""
Расширенные CNN модели для детекции экзопланет
Поддерживает различные архитектуры: ResNet, DenseNet, EfficientNet, Attention-based CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any


class SEBlock(nn.Module):
    """Squeeze-and-Excitation блок для улучшения представлений"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock1D(nn.Module):
    """Улучшенный ResNet блок с SE-attention и dropout"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 11, 
                 stride: int = 1, use_se: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout1d(dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += self.shortcut(residual)
        return F.relu(out)


class DenseBlock1D(nn.Module):
    """DenseNet блок для 1D данных"""
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, 
                 kernel_size: int = 11, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size, padding=kernel_size//2, bias=False),
                nn.Dropout1d(dropout_rate)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class AttentionBlock1D(nn.Module):
    """Self-attention блок для временных рядов"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        # x shape: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        B, L, C = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, C)
        out = self.out(out) + residual
        
        # Feed-forward
        residual = out
        out = self.norm2(out)
        out = self.ffn(out) + residual
        
        # (B, L, C) -> (B, C, L)
        return out.transpose(1, 2)


class ExoplanetCNN(nn.Module):
    """
    Базовая CNN модель для детекции экзопланет
    """
    def __init__(self, input_length: int = 2000, num_classes: int = 2, 
                 base_filters: int = 32, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Первый блок
            nn.Conv1d(1, base_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_rate),
            
            # Второй блок
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_rate),
            
            # Третий блок
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_rate),
            
            # Четвертый блок
            nn.Conv1d(base_filters * 4, base_filters * 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


class ExoplanetResNet(nn.Module):
    """
    ResNet-based модель для детекции экзопланет
    """
    def __init__(self, input_length: int = 2000, num_classes: int = 2,
                 layers: List[int] = [2, 2, 2, 2], base_filters: int = 32,
                 use_se: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        
        self.in_channels = base_filters
        
        # Входной слой
        self.conv1 = nn.Conv1d(1, base_filters, kernel_size=15, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        
        # ResNet блоки
        self.layer1 = self._make_layer(base_filters, layers[0], use_se, dropout_rate)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], use_se, dropout_rate, stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], use_se, dropout_rate, stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], use_se, dropout_rate, stride=2)
        
        # Глобальный пулинг и классификатор
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_filters * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, out_channels: int, blocks: int, use_se: bool, 
                   dropout_rate: float, stride: int = 1):
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, 
                                     stride=stride, use_se=use_se, dropout_rate=dropout_rate))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 
                                         use_se=use_se, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ExoplanetDenseNet(nn.Module):
    """
    DenseNet-based модель для детекции экзопланет
    """
    def __init__(self, input_length: int = 2000, num_classes: int = 2,
                 growth_rate: int = 12, block_config: List[int] = [6, 12, 24, 16],
                 base_filters: int = 32, dropout_rate: float = 0.3):
        super().__init__()
        
        # Входной слой
        self.features = nn.Sequential(
            nn.Conv1d(1, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # DenseNet блоки
        num_features = base_filters
        for i, num_layers in enumerate(block_config):
            block = DenseBlock1D(num_features, growth_rate, num_layers, dropout_rate=dropout_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                # Transition layer
                trans = nn.Sequential(
                    nn.BatchNorm1d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(num_features, num_features // 2, kernel_size=1, bias=False),
                    nn.AvgPool1d(kernel_size=2, stride=2)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Финальные слои
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool', nn.AdaptiveAvgPool1d(1))
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        return self.classifier(out)


class ExoplanetAttentionCNN(nn.Module):
    """
    CNN с self-attention для детекции экзопланет
    """
    def __init__(self, input_length: int = 2000, num_classes: int = 2,
                 base_filters: int = 64, num_heads: int = 8, dropout_rate: float = 0.3):
        super().__init__()
        
        # CNN backbone
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, base_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True),
        )
        
        # Attention блоки
        self.attention1 = AttentionBlock1D(base_filters * 4, num_heads)
        self.attention2 = AttentionBlock1D(base_filters * 4, num_heads)
        
        # Финальные слои
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_filters * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # CNN features
        x = self.conv_layers(x)
        
        # Self-attention
        x = self.attention1(x)
        x = self.attention2(x)
        
        # Global pooling и классификация
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CNNModelFactory:
    """
    Фабрика для создания различных CNN моделей
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> nn.Module:
        """
        Создает модель указанного типа
        
        Args:
            model_type: Тип модели ('cnn', 'resnet', 'densenet', 'attention')
            **kwargs: Параметры модели
        
        Returns:
            Инициализированная модель
        """
        model_type = model_type.lower()
        
        if model_type == 'cnn':
            return ExoplanetCNN(**kwargs)
        elif model_type == 'resnet':
            return ExoplanetResNet(**kwargs)
        elif model_type == 'densenet':
            return ExoplanetDenseNet(**kwargs)
        elif model_type == 'attention':
            return ExoplanetAttentionCNN(**kwargs)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        Возвращает информацию о модели
        """
        info = {
            'cnn': {
                'name': 'Basic CNN',
                'description': 'Базовая сверточная нейронная сеть',
                'complexity': 'Low',
                'parameters': ['input_length', 'num_classes', 'base_filters', 'dropout_rate']
            },
            'resnet': {
                'name': 'ResNet CNN',
                'description': 'CNN с остаточными связями и SE-блоками',
                'complexity': 'Medium',
                'parameters': ['input_length', 'num_classes', 'layers', 'base_filters', 'use_se', 'dropout_rate']
            },
            'densenet': {
                'name': 'DenseNet CNN',
                'description': 'CNN с плотными связями между слоями',
                'complexity': 'High',
                'parameters': ['input_length', 'num_classes', 'growth_rate', 'block_config', 'base_filters', 'dropout_rate']
            },
            'attention': {
                'name': 'Attention CNN',
                'description': 'CNN с механизмом self-attention',
                'complexity': 'High',
                'parameters': ['input_length', 'num_classes', 'base_filters', 'num_heads', 'dropout_rate']
            }
        }
        
        return info.get(model_type.lower(), {})


def count_parameters(model: nn.Module) -> int:
    """Подсчитывает количество обучаемых параметров в модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: tuple) -> Dict[str, Any]:
    """
    Возвращает сводку о модели
    """
    model.eval()
    
    # Создаем тестовый вход
    x = torch.randn(1, *input_size)
    
    # Прямой проход
    with torch.no_grad():
        output = model(x)
    
    return {
        'total_parameters': count_parameters(model),
        'input_shape': input_size,
        'output_shape': tuple(output.shape[1:]),
        'model_size_mb': count_parameters(model) * 4 / (1024 * 1024),  # Примерный размер в MB
    }


if __name__ == "__main__":
    # Тестирование моделей
    input_length = 2000
    batch_size = 32
    
    models = {
        'cnn': CNNModelFactory.create_model('cnn', input_length=input_length),
        'resnet': CNNModelFactory.create_model('resnet', input_length=input_length),
        'densenet': CNNModelFactory.create_model('densenet', input_length=input_length),
        'attention': CNNModelFactory.create_model('attention', input_length=input_length)
    }
    
    # Тестовые данные
    x = torch.randn(batch_size, 1, input_length)
    
    print("=== Тестирование CNN моделей ===")
    for name, model in models.items():
        try:
            output = model(x)
            summary = get_model_summary(model, (1, input_length))
            
            print(f"\n{name.upper()}:")
            print(f"  Выход: {output.shape}")
            print(f"  Параметры: {summary['total_parameters']:,}")
            print(f"  Размер: {summary['model_size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"\n{name.upper()}: ОШИБКА - {e}")
