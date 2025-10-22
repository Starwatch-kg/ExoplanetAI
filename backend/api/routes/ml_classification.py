"""
ML Classification API Routes
API маршруты для машинного обучения и классификации экзопланет

Эндпоинты:
- /classify - Классификация одного объекта
- /batch-classify - Пакетная классификация
- /train-model - Обучение модели
- /model-status - Статус модели
- /feature-importance - Важность признаков
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from pydantic import BaseModel, Field
import logging
import asyncio
from pathlib import Path
import json

from core.logging import get_logger
from ml.lightcurve_preprocessor import LightCurvePreprocessor
from ml.feature_extractor import ExoplanetFeatureExtractor
from ml.exoplanet_classifier import ExoplanetEnsembleClassifier, create_synthetic_training_data

logger = get_logger(__name__)

router = APIRouter()

# Глобальные объекты (ленивая инициализация для избежания блокировки при запуске)
preprocessor = None
feature_extractor = None
classifier = None

def get_preprocessor():
    global preprocessor
    if preprocessor is None:
        preprocessor = LightCurvePreprocessor()
    return preprocessor

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = ExoplanetFeatureExtractor()
    return feature_extractor

def get_classifier():
    global classifier
    if classifier is None:
        classifier = ExoplanetEnsembleClassifier()
    return classifier

# Путь к сохраненной модели
MODEL_PATH = Path("models/exoplanet_classifier.joblib")
MODEL_PATH.parent.mkdir(exist_ok=True)

# Статус модели
model_status = {
    "is_trained": False,
    "training_in_progress": False,
    "last_training_time": None,
    "model_metrics": None
}


# Pydantic модели для API
class LightCurveData(BaseModel):
    """Данные кривой блеска"""
    time: List[float] = Field(..., description="Временные метки")
    flux: List[float] = Field(..., description="Значения потока")
    flux_err: List[float] = Field(..., description="Ошибки потока")
    quality: Optional[List[int]] = Field(None, description="Quality flags")


class TransitParameters(BaseModel):
    """Параметры транзита"""
    period: float = Field(..., description="Орбитальный период (дни)")
    epoch: float = Field(..., description="Эпоха транзита (BJD)")
    duration: float = Field(..., description="Длительность транзита (часы)")


class ClassificationRequest(BaseModel):
    """Запрос на классификацию"""
    target_name: str = Field(..., description="Название объекта")
    lightcurve: LightCurveData = Field(..., description="Данные кривой блеска")
    transit_params: TransitParameters = Field(..., description="Параметры транзита")
    preprocessing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Опции предобработки"
    )


class ClassificationResult(BaseModel):
    """Результат классификации"""
    model_config = {"protected_namespaces": ()}
    
    target_name: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    model_predictions: Dict[str, Dict[str, Any]]
    processing_time: float
    data_quality_score: float
    extracted_features: Optional[Dict[str, float]] = None


class BatchClassificationRequest(BaseModel):
    """Запрос на пакетную классификацию"""
    requests: List[ClassificationRequest] = Field(..., max_items=100)


class TrainingRequest(BaseModel):
    """Запрос на обучение модели"""
    use_synthetic_data: bool = Field(True, description="Использовать синтетические данные для демо")
    n_synthetic_samples: int = Field(1000, description="Количество синтетических образцов")
    test_size: float = Field(0.2, description="Размер тестовой выборки")
    cv_folds: int = Field(5, description="Количество фолдов для кросс-валидации")


class ModelStatusResponse(BaseModel):
    """Статус модели"""
    model_config = {"protected_namespaces": ()}
    
    is_trained: bool
    training_in_progress: bool
    last_training_time: Optional[str]
    model_metrics: Optional[Dict[str, Any]]
    available_features: List[str]


@router.post("/classify", response_model=ClassificationResult)
async def classify_exoplanet(request: ClassificationRequest) -> ClassificationResult:
    """
    Классификация одного объекта
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Starting classification for {request.target_name}")
        
        # Проверяем, что модель обучена
        if not model_status["is_trained"]:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first using /train-model endpoint."
            )
        
        # Преобразуем данные в numpy arrays
        time_array = np.array(request.lightcurve.time)
        flux_array = np.array(request.lightcurve.flux)
        flux_err_array = np.array(request.lightcurve.flux_err)
        quality_array = np.array(request.lightcurve.quality) if request.lightcurve.quality else None
        
        # Предобработка данных
        preprocessing_options = request.preprocessing_options or {}
        preprocessed_data = get_preprocessor().full_preprocessing_pipeline(
            time_array, flux_array, flux_err_array, quality_array,
            smooth_method=preprocessing_options.get('smooth_method', 'savgol'),
            use_wavelet=preprocessing_options.get('use_wavelet', False),
            normalize_method=preprocessing_options.get('normalize_method', 'median')
        )
        
        # Извлечение признаков
        features = get_feature_extractor().extract_all_features(
            preprocessed_data['time'],
            preprocessed_data['flux_centered'],
            preprocessed_data['flux_err'],
            request.transit_params.period,
            request.transit_params.epoch,
            request.transit_params.duration / 24.0  # Конвертируем часы в дни
        )
        
        # Создание последовательности для CNN
        sequence = preprocessed_data['flux_centered']
        if len(sequence) > 64:
            # Берем сегмент вокруг транзита
            segments = get_preprocessor().segment_around_transits(
                preprocessed_data['time'],
                preprocessed_data['flux_centered'],
                request.transit_params.period,
                request.transit_params.epoch,
                request.transit_params.duration / 24.0,
                segment_length=64,
                n_transits=1
            )
            if segments:
                sequence = segments[0]
            else:
                sequence = sequence[:64]
        
        # Классификация
        prediction_result = get_classifier().predict_single(features, sequence)
        
        processing_time = time.time() - start_time
        
        result = ClassificationResult(
            target_name=request.target_name,
            predicted_class=prediction_result['predicted_class'],
            confidence=prediction_result['confidence'],
            class_probabilities=prediction_result['class_probabilities'],
            model_predictions=prediction_result['model_predictions'],
            processing_time=processing_time,
            data_quality_score=preprocessed_data['data_quality'],
            extracted_features=features
        )
        
        logger.info(f"Classification completed for {request.target_name}: {result.predicted_class} ({result.confidence:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Classification failed for {request.target_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/batch-classify")
async def batch_classify_exoplanets(request: BatchClassificationRequest) -> Dict[str, Any]:
    """
    Пакетная классификация объектов
    """
    if not model_status["is_trained"]:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first."
        )
    
    results = []
    failed_classifications = []
    
    for i, single_request in enumerate(request.requests):
        try:
            result = await classify_exoplanet(single_request)
            results.append(result.dict())
        except Exception as e:
            failed_classifications.append({
                "index": i,
                "target_name": single_request.target_name,
                "error": str(e)
            })
            logger.error(f"Batch classification failed for {single_request.target_name}: {str(e)}")
    
    return {
        "total_requests": len(request.requests),
        "successful_classifications": len(results),
        "failed_classifications": len(failed_classifications),
        "results": results,
        "failures": failed_classifications
    }


@router.post("/train-model")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Обучение модели (запускается в фоне)
    """
    if model_status["training_in_progress"]:
        raise HTTPException(
            status_code=400, 
            detail="Training already in progress"
        )
    
    # Запускаем обучение в фоне
    background_tasks.add_task(
        _train_model_background, 
        request.use_synthetic_data,
        request.n_synthetic_samples,
        request.test_size,
        request.cv_folds
    )
    
    model_status["training_in_progress"] = True
    
    return {
        "message": "Model training started in background",
        "status": "training_started"
    }


async def _train_model_background(use_synthetic_data: bool,
                                n_synthetic_samples: int,
                                test_size: float,
                                cv_folds: int):
    """
    Фоновое обучение модели
    """
    try:
        logger.info("Starting model training in background")
        
        if use_synthetic_data:
            # Создаем синтетические данные для демонстрации
            features, sequences, labels = create_synthetic_training_data(n_synthetic_samples)
            logger.info(f"Created {len(features)} synthetic training samples")
        else:
            # В реальности здесь загружались бы данные Kepler/TESS
            raise HTTPException(
                status_code=501, 
                detail="Real data training not implemented yet. Use synthetic data for demo."
            )
        
        # Обучение модели
        training_metrics = get_classifier().train(
            features=features,
            sequences=sequences,
            labels=labels,
            test_size=test_size,
            cv_folds=cv_folds
        )
        
        # Сохранение модели
        get_classifier().save_model(str(MODEL_PATH))
        
        # Обновление статуса
        model_status["is_trained"] = True
        model_status["training_in_progress"] = False
        model_status["last_training_time"] = pd.Timestamp.now().isoformat()
        model_status["model_metrics"] = training_metrics
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        model_status["training_in_progress"] = False
        model_status["model_metrics"] = {"error": str(e)}


@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """
    Получение статуса модели
    """
    return ModelStatusResponse(
        is_trained=model_status["is_trained"],
        training_in_progress=model_status["training_in_progress"],
        last_training_time=model_status["last_training_time"],
        model_metrics=model_status["model_metrics"],
        available_features=get_feature_extractor().get_feature_names()
    )


@router.get("/feature-importance")
async def get_feature_importance() -> Dict[str, Any]:
    """
    Получение важности признаков
    """
    if not model_status["is_trained"]:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained"
        )
    
    try:
        importance = get_classifier().get_feature_importance()
        feature_names = get_feature_extractor().get_feature_names()
        
        # Создаем читаемый формат
        formatted_importance = {}
        for model_name, importances in importance.items():
            formatted_importance[model_name] = [
                {
                    "feature_name": feature_names[i],
                    "importance": float(imp)
                }
                for i, imp in enumerate(importances)
            ]
            # Сортируем по важности
            formatted_importance[model_name].sort(
                key=lambda x: x["importance"], 
                reverse=True
            )
        
        return {
            "feature_importance": formatted_importance,
            "total_features": len(feature_names)
        }
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Загрузка данных для обучения (CSV файл)
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="Only CSV files are supported"
        )
    
    try:
        # Сохраняем файл
        upload_path = Path("data/training") / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Проверяем структуру файла
        df = pd.read_csv(upload_path)
        required_columns = ['time', 'flux', 'flux_err', 'label']
        
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_columns}"
            )
        
        logger.info(f"Training data uploaded: {file.filename} ({len(df)} rows)")
        
        return {
            "message": "Training data uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Failed to upload training data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-metrics")
async def get_detailed_model_metrics() -> Dict[str, Any]:
    """
    Получение детальных метрик модели
    """
    if not model_status["is_trained"]:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained"
        )
    
    metrics = model_status["model_metrics"]
    if not metrics:
        raise HTTPException(
            status_code=404, 
            detail="No metrics available"
        )
    
    # Форматируем метрики для лучшей читаемости
    formatted_metrics = {}
    
    for model_name, model_metrics in metrics.items():
        if isinstance(model_metrics, dict) and 'classification_report' in model_metrics:
            formatted_metrics[model_name] = {
                "accuracy": model_metrics.get("accuracy", 0),
                "auc_score": model_metrics.get("auc_score", 0),
                "per_class_performance": {}
            }
            
            # Извлекаем метрики по классам
            report = model_metrics["classification_report"]
            for class_name in ["CANDIDATE", "PC", "FP"]:
                if class_name in report:
                    formatted_metrics[model_name]["per_class_performance"][class_name] = {
                        "precision": report[class_name]["precision"],
                        "recall": report[class_name]["recall"],
                        "f1_score": report[class_name]["f1-score"],
                        "support": report[class_name]["support"]
                    }
        else:
            formatted_metrics[model_name] = model_metrics
    
    return {
        "training_completed": model_status["last_training_time"],
        "model_metrics": formatted_metrics,
        "summary": {
            "best_model": max(
                formatted_metrics.keys(), 
                key=lambda k: formatted_metrics[k].get("accuracy", 0)
                if isinstance(formatted_metrics[k], dict) else 0
            ) if formatted_metrics else None
        }
    }


# Инициализация при запуске
@router.on_event("startup")
async def load_existing_model():
    """
    Загрузка существующей модели при запуске
    """
    global model_status
    
    if MODEL_PATH.exists():
        try:
            get_classifier().load_model(str(MODEL_PATH))
            model_status["is_trained"] = True
            logger.info("Existing model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load existing model: {str(e)}")
            model_status["is_trained"] = False
    else:
        logger.info("No existing model found")


# Утилитарные эндпоинты
@router.get("/preprocessing-options")
async def get_preprocessing_options() -> Dict[str, Any]:
    """
    Получение доступных опций предобработки
    """
    return {
        "smooth_methods": ["savgol", "median", "none"],
        "normalize_methods": ["median", "mean", "robust"],
        "wavelet_types": ["db4", "db8", "haar", "coif2"],
        "default_options": {
            "smooth_method": "savgol",
            "use_wavelet": False,
            "normalize_method": "median"
        }
    }


@router.get("/class-descriptions")
async def get_class_descriptions() -> Dict[str, str]:
    """
    Описания классов объектов
    """
    return {
        "CANDIDATE": "Confirmed exoplanet - объект с высокой вероятностью быть экзопланетой",
        "PC": "Planetary candidate - планетарный кандидат, требующий дополнительного подтверждения", 
        "FP": "False positive - ложный позитив, не является экзопланетой"
    }
