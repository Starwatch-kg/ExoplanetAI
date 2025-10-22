"""
Real ML Training Pipeline for ExoplanetAI
Реальный пайплайн обучения ML для ExoplanetAI

Использует только реальные данные NASA/TESS/Kepler без mock данных
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import joblib
from datetime import datetime

# ML библиотеки
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# Астрономические библиотеки
import lightkurve as lk
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)

class RealExoplanetTrainer:
    """
    Система реального обучения ML моделей на данных NASA
    """
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        
        # Конфигурация для реального обучения
        self.config = {
            "min_training_samples": 1000,
            "validation_split": 0.2,
            "min_snr": 5.0,
            "min_transit_depth": 0.001,
            "max_noise_level": 0.01,
            "real_data_only": True
        }

    def fetch_real_training_data(self) -> pd.DataFrame:
        """
        Загрузка реальных данных для обучения из NASA архивов
        """
        logger.info("🌌 Загрузка реальных данных NASA для обучения...")
        
        training_data = []
        
        # Загрузка подтвержденных экзопланет
        confirmed_planets = self._fetch_confirmed_planets()
        training_data.extend(confirmed_planets)
        
        # Загрузка кандидатов
        candidates = self._fetch_planet_candidates()
        training_data.extend(candidates)
        
        # Загрузка false positives
        false_positives = self._fetch_false_positives()
        training_data.extend(false_positives)
        
        df = pd.DataFrame(training_data)
        
        # Валидация качества данных
        df = self._validate_data_quality(df)
        
        logger.info(f"✅ Загружено {len(df)} реальных образцов для обучения")
        return df

    def _fetch_confirmed_planets(self) -> List[Dict]:
        """Загрузка подтвержденных экзопланет"""
        try:
            # Поиск подтвержденных планет в TESS
            confirmed = Catalogs.query_criteria(
                catalog="Tic",
                disposition="CONFIRMED",
                limit=500
            )
            
            planets = []
            for planet in confirmed:
                if self._is_valid_planet_data(planet):
                    planet_data = self._extract_planet_features(planet)
                    planet_data['label'] = 'CONFIRMED'
                    planets.append(planet_data)
            
            return planets
            
        except Exception as e:
            logger.error(f"Ошибка загрузки подтвержденных планет: {e}")
            return []

    def _fetch_planet_candidates(self) -> List[Dict]:
        """Загрузка кандидатов в планеты"""
        try:
            candidates = Catalogs.query_criteria(
                catalog="Tic", 
                disposition="CANDIDATE",
                limit=300
            )
            
            planets = []
            for candidate in candidates:
                if self._is_valid_planet_data(candidate):
                    planet_data = self._extract_planet_features(candidate)
                    planet_data['label'] = 'CANDIDATE'
                    planets.append(planet_data)
            
            return planets
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кандидатов: {e}")
            return []

    def _fetch_false_positives(self) -> List[Dict]:
        """Загрузка false positive объектов"""
        try:
            false_positives = Catalogs.query_criteria(
                catalog="Tic",
                disposition="FALSE POSITIVE", 
                limit=200
            )
            
            planets = []
            for fp in false_positives:
                if self._is_valid_planet_data(fp):
                    planet_data = self._extract_planet_features(fp)
                    planet_data['label'] = 'FALSE_POSITIVE'
                    planets.append(planet_data)
            
            return planets
            
        except Exception as e:
            logger.error(f"Ошибка загрузки false positives: {e}")
            return []

    def _is_valid_planet_data(self, planet_data) -> bool:
        """Проверка валидности данных планеты"""
        required_fields = ['ra', 'dec', 'Tmag']
        
        for field in required_fields:
            if field not in planet_data.colnames or np.isnan(planet_data[field]):
                return False
        
        # Проверка качества данных
        if planet_data['Tmag'] > 16.0:  # Слишком тусклая звезда
            return False
            
        return True

    def _extract_planet_features(self, planet_data) -> Dict:
        """Извлечение признаков планеты"""
        features = {
            'ra': float(planet_data['ra']),
            'dec': float(planet_data['dec']),
            'stellar_magnitude': float(planet_data['Tmag']),
            'stellar_radius': float(planet_data.get('rad', 1.0)),
            'stellar_mass': float(planet_data.get('mass', 1.0)),
            'stellar_temperature': float(planet_data.get('Teff', 5778)),
        }
        
        # Загрузка кривой блеска для дополнительных признаков
        try:
            lightcurve = self._fetch_lightcurve(planet_data['ra'], planet_data['dec'])
            if lightcurve is not None:
                lc_features = self._extract_lightcurve_features(lightcurve)
                features.update(lc_features)
        except Exception as e:
            logger.debug(f"Не удалось загрузить кривую блеска: {e}")
        
        return features

    def _fetch_lightcurve(self, ra: float, dec: float):
        """Загрузка кривой блеска для координат"""
        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            search_result = lk.search_lightcurve(coord, mission='TESS', radius=20*u.arcsec)
            
            if len(search_result) > 0:
                lc = search_result[0].download()
                return lc.normalize().remove_outliers()
            
        except Exception as e:
            logger.debug(f"Ошибка загрузки кривой блеска: {e}")
        
        return None

    def _extract_lightcurve_features(self, lightcurve) -> Dict:
        """Извлечение признаков из кривой блеска"""
        flux = lightcurve.flux.value
        
        features = {
            'flux_mean': np.mean(flux),
            'flux_std': np.std(flux),
            'flux_skewness': self._calculate_skewness(flux),
            'flux_kurtosis': self._calculate_kurtosis(flux),
            'flux_range': np.ptp(flux),
            'flux_median': np.median(flux),
            'flux_mad': np.median(np.abs(flux - np.median(flux))),
        }
        
        # BLS анализ для поиска транзитов
        try:
            from astropy.timeseries import BoxLeastSquares
            
            bls = BoxLeastSquares(lightcurve.time, lightcurve.flux)
            periodogram = bls.autopower(0.1)
            
            features.update({
                'bls_period': periodogram.period[np.argmax(periodogram.power)],
                'bls_power': np.max(periodogram.power),
                'bls_depth': periodogram.depth[np.argmax(periodogram.power)],
                'bls_duration': periodogram.duration[np.argmax(periodogram.power)]
            })
            
        except Exception as e:
            logger.debug(f"Ошибка BLS анализа: {e}")
        
        return features

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Расчет асимметрии"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Расчет эксцесса"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация качества данных"""
        logger.info("🔍 Валидация качества данных...")
        
        initial_count = len(df)
        
        # Удаление строк с NaN
        df = df.dropna()
        
        # Фильтрация по качеству
        if 'flux_std' in df.columns:
            df = df[df['flux_std'] < self.config['max_noise_level']]
        
        if 'bls_depth' in df.columns:
            df = df[df['bls_depth'] > self.config['min_transit_depth']]
        
        final_count = len(df)
        logger.info(f"Отфильтровано {initial_count - final_count} низкокачественных образцов")
        
        return df

    def train_models(self, df: pd.DataFrame) -> Dict:
        """Обучение ML моделей на реальных данных"""
        logger.info("🤖 Обучение ML моделей на реальных данных...")
        
        # Подготовка данных
        X, y = self._prepare_training_data(df)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['validation_split'], 
            stratify=y, random_state=42
        )
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение XGBoost
        xgb_model = self._train_xgboost(X_train_scaled, y_train)
        
        # Оценка модели
        metrics = self._evaluate_model(xgb_model, X_test_scaled, y_test)
        
        # Сохранение моделей
        self._save_models(xgb_model, scaler)
        
        logger.info("✅ Обучение завершено успешно!")
        return metrics

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        # Выбор признаков
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns].values
        
        # Кодирование меток
        y = self.label_encoder.fit_transform(df['label'])
        
        return X, y

    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
        """Обучение XGBoost модели"""
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train)
        return model

    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Оценка качества модели"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Метрики
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score'],
            'training_samples': len(X_test) * 5,  # Приблизительно
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"📊 Точность модели: {metrics['accuracy']:.3f}")
        logger.info(f"📊 F1-score: {metrics['f1_score']:.3f}")
        
        return metrics

    def _save_models(self, model, scaler):
        """Сохранение обученных моделей"""
        model_path = self.data_dir / "real_trained_model.joblib"
        scaler_path = self.data_dir / "feature_scaler.joblib"
        encoder_path = self.data_dir / "label_encoder.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        logger.info(f"💾 Модели сохранены в {self.data_dir}")

    def run_full_training_pipeline(self):
        """Запуск полного пайплайна обучения"""
        logger.info("🚀 Запуск полного пайплайна реального обучения...")
        
        # 1. Загрузка реальных данных
        df = self.fetch_real_training_data()
        
        if len(df) < self.config['min_training_samples']:
            logger.warning(f"Недостаточно данных для обучения: {len(df)} < {self.config['min_training_samples']}")
            return None
        
        # 2. Обучение моделей
        metrics = self.train_models(df)
        
        # 3. Сохранение отчета
        report_path = self.data_dir / "training_report.json"
        with open(report_path, 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        
        logger.info("🎉 Пайплайн реального обучения завершен!")
        return metrics

def main():
    """Главная функция для запуска обучения"""
    trainer = RealExoplanetTrainer()
    trainer.run_full_training_pipeline()

if __name__ == "__main__":
    main()
