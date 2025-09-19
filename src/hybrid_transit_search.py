"""
Модуль гибридного поиска транзитов экзопланет
Объединяет классический Box Least Squares (BLS) с Neural Periodogram
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import logging
from scipy import signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Настройка логирования
logger = logging.getLogger(__name__)

class BoxLeastSquares:
    """
    Реализация алгоритма Box Least Squares для поиска периодических транзитов
    """
    
    def __init__(self, period_range: Tuple[float, float] = (0.5, 50.0),
                 nperiods: int = 1000, oversample_factor: int = 5):
        """
        Инициализация BLS
        
        Args:
            period_range: Диапазон периодов для поиска (дни)
            nperiods: Количество периодов для тестирования
            oversample_factor: Коэффициент передискретизации
        """
        self.period_range = period_range
        self.nperiods = nperiods
        self.oversample_factor = oversample_factor
        
        # Создание сетки периодов
        self.periods = np.logspace(
            np.log10(period_range[0]), 
            np.log10(period_range[1]), 
            nperiods
        )
    
    def compute_periodogram(self, times: np.ndarray, fluxes: np.ndarray, 
                           errors: Optional[np.ndarray] = None) -> Dict:
        """
        Вычисляет периодограмму BLS
        
        Args:
            times: Временные метки
            fluxes: Потоки
            errors: Ошибки измерений (опционально)
            
        Returns:
            Dict: Результаты BLS с периодами, мощностями и параметрами
        """
        logger.info("Вычисление BLS периодограммы")
        
        if errors is None:
            errors = np.ones_like(fluxes) * np.std(fluxes)
        
        # Нормализация данных
        fluxes_norm = (fluxes - np.mean(fluxes)) / np.std(fluxes)
        
        # Вычисление периодограммы
        powers = []
        best_params = []
        
        for period in self.periods:
            power, params = self._compute_power_at_period(
                times, fluxes_norm, errors, period
            )
            powers.append(power)
            best_params.append(params)
        
        powers = np.array(powers)
        best_params = np.array(best_params)
        
        # Поиск лучшего периода
        best_idx = np.argmax(powers)
        best_period = self.periods[best_idx]
        best_power = powers[best_idx]
        best_param = best_params[best_idx]
        
        logger.info(f"Лучший период BLS: {best_period:.3f} дней, мощность: {best_power:.3f}")
        
        return {
            'periods': self.periods,
            'powers': powers,
            'best_period': best_period,
            'best_power': best_power,
            'best_params': best_param,
            'all_params': best_params
        }
    
    def _compute_power_at_period(self, times: np.ndarray, fluxes: np.ndarray,
                                errors: np.ndarray, period: float) -> Tuple[float, Dict]:
        """
        Вычисляет мощность BLS для конкретного периода
        
        Args:
            times: Временные метки
            fluxes: Нормализованные потоки
            errors: Ошибки измерений
            period: Период для тестирования
            
        Returns:
            Tuple[power, params]: Мощность и параметры транзита
        """
        # Фазирование данных
        phases = (times % period) / period
        
        # Сетка параметров транзита
        durations = np.linspace(0.01, 0.3, 20)  # От 1% до 30% периода
        depths = np.linspace(0.001, 0.1, 20)    # Глубина транзита
        
        best_power = 0
        best_params = {}
        
        for duration in durations:
            for depth in depths:
                # Создание модели транзита
                transit_model = self._create_transit_model(phases, duration, depth)
                
                # Вычисление мощности
                power = self._compute_transit_power(fluxes, transit_model, errors)
                
                if power > best_power:
                    best_power = power
                    best_params = {
                        'period': period,
                        'duration': duration,
                        'depth': depth,
                        'transit_model': transit_model
                    }
        
        return best_power, best_params
    
    def _create_transit_model(self, phases: np.ndarray, duration: float, 
                             depth: float) -> np.ndarray:
        """
        Создает модель транзита
        
        Args:
            phases: Фазы (0-1)
            duration: Длительность транзита (доля периода)
            depth: Глубина транзита
            
        Returns:
            np.ndarray: Модель транзита
        """
        model = np.ones_like(phases)
        
        # Определение областей транзита
        transit_start = 0.5 - duration / 2
        transit_end = 0.5 + duration / 2
        
        # Обработка случая, когда транзит пересекает фазу 0
        if transit_start < 0:
            mask1 = (phases >= 1 + transit_start) | (phases <= transit_end)
            mask2 = (phases >= transit_start) & (phases <= transit_end)
            mask = mask1 | mask2
        else:
            mask = (phases >= transit_start) & (phases <= transit_end)
        
        model[mask] -= depth
        
        return model
    
    def _compute_transit_power(self, fluxes: np.ndarray, model: np.ndarray,
                              errors: np.ndarray) -> float:
        """
        Вычисляет мощность транзита
        
        Args:
            fluxes: Наблюдаемые потоки
            model: Модель транзита
            errors: Ошибки измерений
            
        Returns:
            float: Мощность транзита
        """
        # Вычисление остатков
        residuals = fluxes - model
        
        # Мощность как отношение дисперсии модели к общей дисперсии
        model_var = np.var(model)
        total_var = np.var(fluxes)
        
        if total_var > 0:
            power = model_var / total_var
        else:
            power = 0
        
        return power


class NeuralPeriodogram(nn.Module):
    """
    Нейронная сеть для анализа периодограмм и поиска нестандартных форм транзитов
    """
    
    def __init__(self, input_length: int = 2000, hidden_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.3):
        """
        Инициализация Neural Periodogram
        
        Args:
            input_length: Длина входной последовательности
            hidden_dim: Размер скрытого слоя
            num_layers: Количество LSTM слоев
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        
        # CNN для извлечения локальных признаков
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Вычисление размера после сверточных слоев
        conv_output_size = input_length // 8 * 128
        
        # LSTM для анализа временных зависимостей
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Классификатор периодов
        self.period_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Вероятность наличия транзита
            nn.Sigmoid()
        )
        
        # Регрессор параметров транзита
        self.transit_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # period, duration, depth
        )
    
    def forward(self, x):
        """
        Прямой проход через нейронную сеть
        
        Args:
            x: Входные данные (batch_size, 1, sequence_length)
            
        Returns:
            Tuple[transit_prob, transit_params]: Вероятность транзита и его параметры
        """
        batch_size = x.size(0)
        
        # CNN извлечение признаков
        conv_out = self.conv_layers(x)  # (batch_size, 128, sequence_length/8)
        
        # Перестановка для LSTM
        conv_out = conv_out.transpose(1, 2)  # (batch_size, sequence_length/8, 128)
        
        # LSTM обработка
        lstm_out, _ = self.lstm(conv_out)  # (batch_size, sequence_length/8, hidden_dim*2)
        
        # Attention механизм
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Глобальное усреднение
        global_features = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim*2)
        
        # Классификация и регрессия
        transit_prob = self.period_classifier(global_features)
        transit_params = self.transit_regressor(global_features)
        
        return transit_prob, transit_params
    
    def predict_periodogram(self, times: np.ndarray, fluxes: np.ndarray,
                           device: str = 'cpu') -> Dict:
        """
        Предсказывает периодограмму для кривой блеска
        
        Args:
            times: Временные метки
            fluxes: Потоки
            device: Устройство для вычислений
            
        Returns:
            Dict: Результаты нейронной периодограммы
        """
        self.eval()
        self.to(device)
        
        # Нормализация данных
        fluxes_norm = (fluxes - np.mean(fluxes)) / np.std(fluxes)
        
        # Создание окон для анализа
        window_size = self.input_length
        stride = window_size // 4
        
        predictions = []
        periods = []
        
        with torch.no_grad():
            for start in range(0, len(fluxes_norm) - window_size + 1, stride):
                window = fluxes_norm[start:start + window_size]
                
                # Преобразование в тензор
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Предсказание
                prob, params = self.forward(x)
                
                predictions.append({
                    'prob': prob.cpu().numpy()[0],
                    'params': params.cpu().numpy()[0],
                    'start_time': times[start],
                    'end_time': times[start + window_size - 1]
                })
                
                periods.append(times[start + window_size // 2])
        
        # Агрегация результатов
        probs = np.array([p['prob'] for p in predictions])
        param_means = np.mean([p['params'] for p in predictions], axis=0)
        
        return {
            'periods': periods,
            'probabilities': probs,
            'mean_transit_prob': np.mean(probs),
            'transit_params': param_means,
            'detailed_predictions': predictions
        }


class HybridTransitSearch:
    """
    Гибридный поиск транзитов, объединяющий BLS и Neural Periodogram
    """
    
    def __init__(self, bls_config: Dict = None, neural_config: Dict = None):
        """
        Инициализация гибридного поиска
        
        Args:
            bls_config: Конфигурация BLS
            neural_config: Конфигурация нейронной сети
        """
        # Конфигурация BLS
        bls_config = bls_config or {}
        self.bls = BoxLeastSquares(
            period_range=bls_config.get('period_range', (0.5, 50.0)),
            nperiods=bls_config.get('nperiods', 1000),
            oversample_factor=bls_config.get('oversample_factor', 5)
        )
        
        # Конфигурация нейронной сети
        neural_config = neural_config or {}
        self.neural_periodogram = NeuralPeriodogram(
            input_length=neural_config.get('input_length', 2000),
            hidden_dim=neural_config.get('hidden_dim', 256),
            num_layers=neural_config.get('num_layers', 3),
            dropout=neural_config.get('dropout', 0.3)
        )
        
        # Веса для комбинирования результатов
        self.bls_weight = 0.6
        self.neural_weight = 0.4
    
    def search_transits(self, times: np.ndarray, fluxes: np.ndarray,
                       errors: Optional[np.ndarray] = None,
                       device: str = 'cpu') -> Dict:
        """
        Выполняет гибридный поиск транзитов
        
        Args:
            times: Временные метки
            fluxes: Потоки
            errors: Ошибки измерений
            device: Устройство для нейронной сети
            
        Returns:
            Dict: Результаты гибридного поиска
        """
        logger.info("Начинаем гибридный поиск транзитов")
        
        # BLS анализ
        logger.info("Выполняем BLS анализ")
        bls_results = self.bls.compute_periodogram(times, fluxes, errors)
        
        # Neural Periodogram анализ
        logger.info("Выполняем Neural Periodogram анализ")
        neural_results = self.neural_periodogram.predict_periodogram(times, fluxes, device)
        
        # Комбинирование результатов
        logger.info("Комбинируем результаты BLS и Neural Periodogram")
        combined_results = self._combine_results(bls_results, neural_results)
        
        # Поиск кандидатов
        candidates = self._find_candidates(combined_results, times, fluxes)
        
        logger.info(f"Найдено {len(candidates)} кандидатов в транзиты")
        
        return {
            'bls_results': bls_results,
            'neural_results': neural_results,
            'combined_results': combined_results,
            'candidates': candidates,
            'summary': self._create_summary(candidates)
        }
    
    def _combine_results(self, bls_results: Dict, neural_results: Dict) -> Dict:
        """
        Комбинирует результаты BLS и Neural Periodogram
        
        Args:
            bls_results: Результаты BLS
            neural_results: Результаты нейронной сети
            
        Returns:
            Dict: Комбинированные результаты
        """
        # Нормализация мощностей BLS
        bls_power_norm = bls_results['best_power'] / np.max(bls_results['powers'])
        
        # Нормализация вероятностей нейронной сети
        neural_prob_norm = neural_results['mean_transit_prob']
        
        # Взвешенная комбинация
        combined_score = (self.bls_weight * bls_power_norm + 
                         self.neural_weight * neural_prob_norm)
        
        # Комбинированные параметры транзита
        bls_params = bls_results['best_params']
        neural_params = neural_results['transit_params']
        
        combined_params = {
            'period': (self.bls_weight * bls_params['period'] + 
                     self.neural_weight * neural_params[0]),
            'duration': (self.bls_weight * bls_params['duration'] + 
                        self.neural_weight * neural_params[1]),
            'depth': (self.bls_weight * bls_params['depth'] + 
                     self.neural_weight * neural_params[2])
        }
        
        return {
            'combined_score': combined_score,
            'bls_contribution': bls_power_norm,
            'neural_contribution': neural_prob_norm,
            'combined_params': combined_params,
            'confidence': min(combined_score * 2, 1.0)  # Нормализация до [0,1]
        }
    
    def _find_candidates(self, combined_results: Dict, times: np.ndarray, 
                        fluxes: np.ndarray, threshold: float = 0.3) -> List[Dict]:
        """
        Находит кандидатов в транзиты на основе комбинированных результатов
        
        Args:
            combined_results: Комбинированные результаты
            times: Временные метки
            fluxes: Потоки
            threshold: Порог для отбора кандидатов
            
        Returns:
            List[Dict]: Список кандидатов
        """
        candidates = []
        
        if combined_results['combined_score'] > threshold:
            params = combined_results['combined_params']
            
            # Создание модели транзита
            transit_model = self._create_transit_model(
                times, params['period'], params['duration'], params['depth']
            )
            
            # Поиск областей транзита
            transit_regions = self._find_transit_regions(fluxes, transit_model)
            
            for region in transit_regions:
                candidate = {
                    'period': params['period'],
                    'duration': params['duration'],
                    'depth': params['depth'],
                    'start_time': times[region['start_idx']],
                    'end_time': times[region['end_idx']],
                    'start_idx': region['start_idx'],
                    'end_idx': region['end_idx'],
                    'confidence': combined_results['confidence'],
                    'combined_score': combined_results['combined_score'],
                    'transit_model': transit_model[region['start_idx']:region['end_idx']]
                }
                candidates.append(candidate)
        
        return candidates
    
    def _create_transit_model(self, times: np.ndarray, period: float,
                             duration: float, depth: float) -> np.ndarray:
        """
        Создает модель транзита для заданных параметров
        
        Args:
            times: Временные метки
            period: Период
            duration: Длительность транзита
            depth: Глубина транзита
            
        Returns:
            np.ndarray: Модель транзита
        """
        model = np.ones_like(times)
        
        # Фазирование
        phases = (times % period) / period
        
        # Области транзита
        transit_start = 0.5 - duration / 2
        transit_end = 0.5 + duration / 2
        
        # Обработка пересечения фазы 0
        if transit_start < 0:
            mask1 = (phases >= 1 + transit_start) | (phases <= transit_end)
            mask2 = (phases >= transit_start) & (phases <= transit_end)
            mask = mask1 | mask2
        else:
            mask = (phases >= transit_start) & (phases <= transit_end)
        
        model[mask] -= depth
        
        return model
    
    def _find_transit_regions(self, fluxes: np.ndarray, model: np.ndarray,
                             min_length: int = 3) -> List[Dict]:
        """
        Находит регионы транзитов
        
        Args:
            fluxes: Наблюдаемые потоки
            model: Модель транзита
            min_length: Минимальная длина региона
            
        Returns:
            List[Dict]: Список регионов транзитов
        """
        # Вычисление остатков
        residuals = fluxes - model
        
        # Поиск областей с отрицательными остатками
        transit_mask = residuals < -np.std(residuals) * 0.5
        
        regions = []
        start_idx = None
        
        for i, is_transit in enumerate(transit_mask):
            if is_transit and start_idx is None:
                start_idx = i
            elif not is_transit and start_idx is not None:
                if i - start_idx >= min_length:
                    regions.append({
                        'start_idx': start_idx,
                        'end_idx': i - 1
                    })
                start_idx = None
        
        # Обработка случая, когда транзит доходит до конца
        if start_idx is not None and len(transit_mask) - start_idx >= min_length:
            regions.append({
                'start_idx': start_idx,
                'end_idx': len(transit_mask) - 1
            })
        
        return regions
    
    def _create_summary(self, candidates: List[Dict]) -> Dict:
        """
        Создает сводку результатов поиска
        
        Args:
            candidates: Список кандидатов
            
        Returns:
            Dict: Сводка результатов
        """
        if not candidates:
            return {
                'num_candidates': 0,
                'best_candidate': None,
                'avg_confidence': 0.0,
                'period_range': (0, 0),
                'depth_range': (0, 0)
            }
        
        confidences = [c['confidence'] for c in candidates]
        periods = [c['period'] for c in candidates]
        depths = [c['depth'] for c in candidates]
        
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        
        return {
            'num_candidates': len(candidates),
            'best_candidate': best_candidate,
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'period_range': (np.min(periods), np.max(periods)),
            'depth_range': (np.min(depths), np.max(depths)),
            'candidate_periods': periods,
            'candidate_confidences': confidences
        }


def train_neural_periodogram(model: NeuralPeriodogram, train_loader, 
                           val_loader, epochs: int = 50, device: str = 'cpu'):
    """
    Обучает Neural Periodogram на данных
    
    Args:
        model: Модель Neural Periodogram
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        epochs: Количество эпох
        device: Устройство для обучения
    """
    logger.info("Начинаем обучение Neural Periodogram")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Функции потерь
    classification_loss = nn.BCELoss()
    regression_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Предсказания
            transit_probs, transit_params = model(data)
            
            # Вычисление потерь
            # Предполагаем, что targets содержат метки транзитов и параметры
            if len(targets.shape) == 1:  # Только метки классов
                class_loss = classification_loss(transit_probs.squeeze(), targets.float())
                total_loss = class_loss
            else:  # Метки классов и параметры
                class_loss = classification_loss(transit_probs.squeeze(), targets[:, 0].float())
                param_loss = regression_loss(transit_params, targets[:, 1:4])
                total_loss = class_loss + 0.1 * param_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                transit_probs, transit_params = model(data)
                
                if len(targets.shape) == 1:
                    class_loss = classification_loss(transit_probs.squeeze(), targets.float())
                    total_loss = class_loss
                else:
                    class_loss = classification_loss(transit_probs.squeeze(), targets[:, 0].float())
                    param_loss = regression_loss(transit_params, targets[:, 1:4])
                    total_loss = class_loss + 0.1 * param_loss
                
                val_loss += total_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_neural_periodogram.pth')
    
    logger.info("Обучение Neural Periodogram завершено")


if __name__ == "__main__":
    # Пример использования
    logger.info("Тестирование гибридного поиска транзитов")
    
    # Создание тестовых данных
    np.random.seed(42)
    times = np.linspace(0, 100, 2000)
    
    # Создание синтетической кривой блеска с транзитом
    fluxes = np.ones_like(times) + 0.01 * np.random.randn(len(times))
    
    # Добавление транзита
    period = 10.0
    depth = 0.02
    duration = 0.1
    
    for i in range(len(times)):
        phase = (times[i] % period) / period
        if 0.45 <= phase <= 0.55:  # Транзит
            fluxes[i] -= depth
    
    # Инициализация гибридного поиска
    hybrid_search = HybridTransitSearch()
    
    # Поиск транзитов
    results = hybrid_search.search_transits(times, fluxes)
    
    # Вывод результатов
    print(f"Найдено кандидатов: {results['summary']['num_candidates']}")
    if results['summary']['best_candidate']:
        best = results['summary']['best_candidate']
        print(f"Лучший кандидат: период={best['period']:.3f}, глубина={best['depth']:.4f}, уверенность={best['confidence']:.3f}")
    
    logger.info("Тест завершен")
