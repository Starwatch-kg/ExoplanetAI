"""
Signal Processor для анализа астрономических данных
"""

import logging
from typing import Any, Dict

import numpy as np
import pywt
from scipy import signal
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Процессор для анализа астрономических сигналов (кривых блеска)
    """

    def __init__(self, flux_data: np.ndarray):
        """
        Инициализация процессора

        Args:
            flux_data: Массив данных потока (flux)
        """
        self.flux = flux_data.copy()
        self.original_flux = flux_data.copy()
        self.clean_curve = None
        self.transits = []
        self.features = {}
        self.classification_results = {}

    def remove_noise(self, method: str = "wavelet") -> "SignalProcessor":
        """
        Удаление шума из сигнала

        Args:
            method: Метод удаления шума ('wavelet', 'median', 'savgol')

        Returns:
            SignalProcessor: self для цепочки вызовов
        """
        logger.info(f"Removing noise using {method} method")

        if method == "wavelet":
            # Wavelet denoising
            try:
                # Используем Daubechies 4 wavelet
                coeffs = pywt.wavedec(self.flux, "db4", level=4)
                # Устанавливаем порог
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(self.flux)))

                # Применяем мягкий порог
                denoised_coeffs = []
                for coeff in coeffs:
                    denoised_coeffs.append(
                        pywt.threshold(coeff, threshold, mode="soft")
                    )

                # Восстанавливаем сигнал
                self.clean_curve = pywt.waverec(denoised_coeffs, "db4")

            except Exception as e:
                logger.warning(f"Wavelet denoising failed: {e}, using median filter")
                # Fallback to median filter
                self.clean_curve = signal.medfilt(self.flux, kernel_size=5)

        elif method == "median":
            # Медианная фильтрация
            kernel_size = min(5, len(self.flux) // 10)
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.clean_curve = signal.medfilt(self.flux, kernel_size=kernel_size)

        elif method == "savgol":
            # Savitzky-Golay filter
            window_length = min(51, len(self.flux) // 5)
            if window_length % 2 == 0:
                window_length += 1
            if window_length < 3:
                window_length = 3
            self.clean_curve = signal.savgol_filter(self.flux, window_length, 3)

        else:
            logger.warning(f"Unknown noise removal method: {method}")
            self.clean_curve = self.flux.copy()

        return self

    def detect_transits(
        self, threshold: float = 2.0, max_candidates: int = 10
    ) -> "SignalProcessor":
        """
        Обнаружение транзитов в сигнале

        Args:
            threshold: Порог обнаружения в сигмах
            max_candidates: Максимальное количество кандидатов

        Returns:
            SignalProcessor: self для цепочки вызовов
        """
        logger.info(f"Detecting transits with threshold {threshold}")

        if self.clean_curve is None:
            self.clean_curve = self.flux.copy()

        # Вычисляем статистику
        mean_flux = np.mean(self.clean_curve)
        std_flux = np.std(self.clean_curve)

        # Находим точки ниже порога
        transit_mask = self.clean_curve < (mean_flux - threshold * std_flux)

        # Находим группы транзитов
        from scipy.ndimage import label

        labeled_array, num_features = label(transit_mask)

        transit_indices = []
        for i in range(1, num_features + 1):
            transit_points = np.where(labeled_array == i)[0]
            if len(transit_points) >= 3:  # Минимум 3 точки для транзита
                # Находим центр транзита
                center_idx = int(np.mean(transit_points))
                transit_indices.append(center_idx)

        # Сортируем и ограничиваем количество
        transit_indices.sort()
        self.transits = transit_indices[:max_candidates]

        logger.info(f"Found {len(self.transits)} transit candidates")
        return self

    def analyze_periodicity(self) -> "SignalProcessor":
        """
        Анализ периодичности сигнала

        Returns:
            SignalProcessor: self для цепочки вызовов
        """
        logger.info("Analyzing periodicity")

        if self.clean_curve is None:
            self.clean_curve = self.flux.copy()

        try:
            # Простой анализ периодичности с помощью автокорреляции
            # Ограничиваем длину для производительности
            max_lag = min(len(self.clean_curve) // 3, 5000)

            if max_lag < 10:
                self.features["period"] = 1.0
                self.features["period_confidence"] = 0.1
                return self

            # Вычисляем автокорреляцию
            autocorr = np.correlate(self.clean_curve, self.clean_curve, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]
            autocorr = autocorr[:max_lag]

            # Находим пики
            peaks = signal.find_peaks(autocorr, prominence=0.1)[0]

            if len(peaks) > 0:
                # Берем первый значимый пик как период
                period = peaks[0] + 1  # +1 потому что индекс 0 соответствует лагу 0
                confidence = autocorr[peaks[0]] / np.max(autocorr)
            else:
                period = 1.0
                confidence = 0.1

            self.features["period"] = period
            self.features["period_confidence"] = confidence

        except Exception as e:
            logger.warning(f"Periodicity analysis failed: {e}")
            self.features["period"] = 1.0
            self.features["period_confidence"] = 0.1

        return self

    def extract_features(self) -> "SignalProcessor":
        """
        Извлечение статистических признаков

        Returns:
            SignalProcessor: self для цепочки вызовов
        """
        logger.info("Extracting features")

        if self.clean_curve is None:
            data = self.flux
        else:
            data = self.clean_curve

        # Базовые статистические признаки
        self.features.update(
            {
                "mean": np.mean(data),
                "std": np.std(data),
                "median": np.median(data),
                "min": np.min(data),
                "max": np.max(data),
                "skew": skew(data),
                "kurtosis": kurtosis(data),
                "snr": np.abs(np.mean(data)) / np.std(data) if np.std(data) > 0 else 0,
                "range": np.max(data) - np.min(data),
                "mad": np.median(
                    np.abs(data - np.median(data))
                ),  # Median Absolute Deviation
                "rms": np.sqrt(np.mean(data**2)),  # Root Mean Square
            }
        )

        # Признаки, связанные с транзитами
        if len(self.transits) > 0:
            self.features.update(
                {
                    "num_transits": len(self.transits),
                    "transit_depths": [
                        np.min(data[max(0, idx - 5) : min(len(data), idx + 5)])
                        for idx in self.transits
                    ],
                    "transit_strengths": [
                        abs(self.features["mean"] - depth)
                        for depth in self.features["transit_depths"]
                    ],
                }
            )

        return self

    def classify_signal(self) -> "SignalProcessor":
        """
        Классификация сигнала (планета/звезда/шум)

        Returns:
            SignalProcessor: self для цепочки вызовов
        """
        logger.info("Classifying signal")

        # Простая эвристическая классификация
        features = self.features

        # Критерии для классификации
        snr_score = min(features.get("snr", 0) / 10, 1.0)  # SNR > 10 = хорошая
        periodicity_score = features.get("period_confidence", 0)
        num_transits_score = min(
            features.get("num_transits", 0) / 5, 1.0
        )  # > 5 транзитов = хорошая
        depth_score = min(
            np.mean(features.get("transit_strengths", [0])) / 0.01, 1.0
        )  # Глубина > 1% = хорошая

        # Комбинированный score
        planet_score = (
            snr_score * 0.3
            + periodicity_score * 0.3
            + num_transits_score * 0.2
            + depth_score * 0.2
        )

        # Нормализуем вероятности
        star_score = 0.3  # Базовая вероятность звезды
        noise_score = 1.0 - planet_score - star_score

        if noise_score < 0:
            noise_score = 0.1
            star_score = 0.9 - planet_score

        self.features["probabilities"] = [planet_score, star_score, noise_score]
        self.features["classification"] = ["planet", "star", "noise"][
            np.argmax([planet_score, star_score, noise_score])
        ]

        logger.info(
            f"Signal classified as: {self.features['classification']} (planet: {planet_score:.2f}, "
            f"star: {star_score:.2f}, noise: {noise_score:.2f})"
        )

        return self

    def get_results(self) -> Dict[str, Any]:
        """
        Получение всех результатов обработки

        Returns:
            Dict с результатами анализа
        """
        return {
            "original_flux": self.original_flux,
            "clean_curve": self.clean_curve,
            "transits": self.transits,
            "features": self.features,
            "classification": self.features.get("classification"),
            "probabilities": self.features.get("probabilities"),
        }
