"""
Классификатор изображений с использованием CNN
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

from cnn_models import CNNModelFactory


class ImageCNNClassifier:
    """
    Классификатор изображений на основе CNN для различных задач
    """
    
    def __init__(self, 
                 model_type: str = 'resnet',
                 num_classes: int = 10,
                 class_names: Optional[List[str]] = None,
                 device: Optional[torch.device] = None):
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Названия классов (по умолчанию CIFAR-10)
        self.class_names = class_names or [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Инициализация модели
        self.model = None
        self.is_trained = False
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Стандартный размер для CNN
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet нормализация
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Логгер
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(f'ImageClassifier_{self.model_type}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build_model(self) -> nn.Module:
        """Создает модель CNN для классификации изображений"""
        
        # Адаптируем CNN модели для изображений (3 канала вместо 1)
        if self.model_type == 'cnn':
            self.model = self._create_image_cnn()
        elif self.model_type == 'resnet':
            self.model = self._create_image_resnet()
        elif self.model_type == 'densenet':
            self.model = self._create_image_densenet()
        elif self.model_type == 'attention':
            self.model = self._create_image_attention_cnn()
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
        
        self.model.to(self.device)
        self.logger.info(f"Создана модель {self.model_type} для классификации изображений")
        
        return self.model
    
    def _create_image_cnn(self) -> nn.Module:
        """Создает базовую CNN для изображений"""
        return nn.Sequential(
            # Первый блок
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Второй блок
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Третий блок
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Четвертый блок
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Классификатор
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )
    
    def _create_image_resnet(self) -> nn.Module:
        """Создает ResNet-подобную модель для изображений"""
        class ResBlock2D(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                return torch.relu(out)
        
        return nn.Sequential(
            # Входной слой
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet блоки
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
            ResBlock2D(64, 128, stride=2),
            ResBlock2D(128, 128),
            ResBlock2D(128, 256, stride=2),
            ResBlock2D(256, 256),
            ResBlock2D(256, 512, stride=2),
            ResBlock2D(512, 512),
            
            # Классификатор
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
    
    def _create_image_densenet(self) -> nn.Module:
        """Создает DenseNet-подобную модель для изображений"""
        class DenseBlock2D(nn.Module):
            def __init__(self, in_channels, growth_rate, num_layers):
                super().__init__()
                self.layers = nn.ModuleList()
                
                for i in range(num_layers):
                    layer = nn.Sequential(
                        nn.BatchNorm2d(in_channels + i * growth_rate),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False),
                        nn.Dropout2d(0.2)
                    )
                    self.layers.append(layer)
            
            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_feature = layer(torch.cat(features, 1))
                    features.append(new_feature)
                return torch.cat(features, 1)
        
        return nn.Sequential(
            # Входной слой
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Dense блоки
            DenseBlock2D(64, 32, 6),
            nn.BatchNorm2d(64 + 6 * 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 + 6 * 32, 128, 1, bias=False),
            nn.AvgPool2d(2),
            
            DenseBlock2D(128, 32, 12),
            nn.BatchNorm2d(128 + 12 * 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 + 12 * 32, 256, 1, bias=False),
            nn.AvgPool2d(2),
            
            # Классификатор
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def _create_image_attention_cnn(self) -> nn.Module:
        """Создает CNN с attention для изображений"""
        class SelfAttention2D(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
                self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
                self.value = nn.Conv2d(in_channels, in_channels, 1)
                self.gamma = nn.Parameter(torch.zeros(1))
                
            def forward(self, x):
                batch_size, channels, height, width = x.size()
                
                query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
                key = self.key(x).view(batch_size, -1, height * width)
                value = self.value(x).view(batch_size, -1, height * width)
                
                attention = torch.bmm(query, key)
                attention = torch.softmax(attention, dim=-1)
                
                out = torch.bmm(value, attention.permute(0, 2, 1))
                out = out.view(batch_size, channels, height, width)
                
                return self.gamma * out + x
        
        return nn.Sequential(
            # CNN backbone
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Attention блок
            SelfAttention2D(128),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Attention блок
            SelfAttention2D(256),
            
            # Классификатор
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )
    
    def load_pretrained_weights(self, weights_path: str):
        """Загружает предобученные веса"""
        if self.model is None:
            raise ValueError("Модель должна быть создана перед загрузкой весов")
        
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.is_trained = True
            self.logger.info(f"Загружены веса из {weights_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки весов: {e}")
            # Если нет предобученных весов, используем случайную инициализацию
            self.logger.warning("Используется случайная инициализация весов")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Предобработка изображения"""
        # Конвертируем в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Применяем трансформации
        tensor = self.transform(image)
        
        # Добавляем batch dimension
        return tensor.unsqueeze(0)
    
    def classify_image(self, image: Image.Image) -> Dict:
        """Классифицирует одно изображение"""
        if self.model is None:
            raise ValueError("Модель не создана")
        
        start_time = time.time()
        
        # Предобработка
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Инференс
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        processing_time = time.time() - start_time
        
        # Формируем результат
        result = {
            'class_name': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'processing_time': processing_time
        }
        
        self.logger.info(
            f"Классификация: {result['class_name']} "
            f"({result['confidence']:.3f}) за {processing_time:.3f}s"
        )
        
        return result
    
    def classify_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Классифицирует пакет изображений"""
        if self.model is None:
            raise ValueError("Модель не создана")
        
        start_time = time.time()
        
        # Предобработка всех изображений
        batch_tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            batch_tensors.append(tensor)
        
        # Объединяем в batch
        batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
        
        # Инференс
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        processing_time = time.time() - start_time
        
        # Формируем результаты
        results = []
        for i in range(len(images)):
            predicted_class = predicted_classes[i].item()
            confidence = probabilities[i][predicted_class].item()
            
            result = {
                'class_name': self.class_names[predicted_class],
                'class_id': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    self.class_names[j]: prob.item() 
                    for j, prob in enumerate(probabilities[i])
                },
                'processing_time': processing_time / len(images)  # Среднее время на изображение
            }
            results.append(result)
        
        self.logger.info(f"Классифицировано {len(images)} изображений за {processing_time:.3f}s")
        
        return results
    
    def get_model_info(self) -> Dict:
        """Возвращает информацию о модели"""
        if self.model is None:
            return {'error': 'Модель не создана'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_trained': self.is_trained
        }


# Глобальные экземпляры классификаторов
_classifiers = {}

def get_classifier(model_type: str = 'resnet') -> ImageCNNClassifier:
    """Получает или создает классификатор"""
    if model_type not in _classifiers:
        classifier = ImageCNNClassifier(model_type=model_type)
        classifier.build_model()
        
        # Пытаемся загрузить предобученные веса
        weights_path = Path(f'models/image_classifier_{model_type}.pth')
        if weights_path.exists():
            classifier.load_pretrained_weights(str(weights_path))
        
        _classifiers[model_type] = classifier
    
    return _classifiers[model_type]


if __name__ == "__main__":
    # Тестирование классификатора
    print("=== Тестирование Image CNN Classifier ===")
    
    # Создаем тестовое изображение
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Тестируем разные модели
    for model_type in ['cnn', 'resnet', 'attention']:
        print(f"\n--- Тестирование {model_type.upper()} ---")
        
        classifier = ImageCNNClassifier(model_type=model_type)
        classifier.build_model()
        
        # Классификация
        result = classifier.classify_image(test_image)
        
        print(f"Класс: {result['class_name']}")
        print(f"Уверенность: {result['confidence']:.3f}")
        print(f"Время: {result['processing_time']:.3f}s")
        
        # Информация о модели
        info = classifier.get_model_info()
        print(f"Параметры: {info['total_parameters']:,}")
