#!/usr/bin/env python3
"""
Основной скрипт для поиска экзопланет по данным TESS/MAST
Использует комплексную систему машинного обучения для детекции транзитов
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Добавляем src в путь для импортов
sys.path.append(str(Path(__file__).parent / "src"))

from src.exoplanet_pipeline import ExoplanetSearchPipeline

def setup_logging(log_level: str = "INFO"):
    """Настройка логирования"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('exoplanet_search.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Поиск экзопланет по данным TESS/MAST',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Поиск экзопланет для конкретных звезд:
   python exoplanet_search.py --tic-ids 261136679 38846515 142802581

2. Поиск с указанием секторов TESS:
   python exoplanet_search.py --tic-ids 261136679 --sectors 1 2 3

3. Поиск с обучением моделей:
   python exoplanet_search.py --tic-ids 261136679 --train-models

4. Поиск с использованием предобученных моделей:
   python exoplanet_search.py --tic-ids 261136679 --no-train

5. Поиск с кастомной конфигурацией:
   python exoplanet_search.py --tic-ids 261136679 --config custom_config.yaml
        """
    )
    
    # Основные параметры
    parser.add_argument('--tic-ids', nargs='+', required=True,
                       help='Список TIC ID для анализа (обязательный параметр)')
    parser.add_argument('--sectors', nargs='+', type=int,
                       help='Список секторов TESS для анализа')
    parser.add_argument('--config', default='src/config.yaml',
                       help='Путь к файлу конфигурации (по умолчанию: src/config.yaml)')
    
    # Параметры обучения
    parser.add_argument('--train-models', action='store_true',
                       help='Обучать модели с нуля')
    parser.add_argument('--no-train', action='store_true',
                       help='Не обучать модели (использовать предобученные)')
    
    # Параметры вывода
    parser.add_argument('--top-n', type=int, default=50,
                       help='Количество топ кандидатов для экспорта (по умолчанию: 50)')
    parser.add_argument('--output-dir', default='results',
                       help='Директория для сохранения результатов (по умолчанию: results)')
    
    # Параметры логирования
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Уровень логирования (по умолчанию: INFO)')
    
    # Параметры производительности
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Устройство для вычислений (по умолчанию: auto)')
    parser.add_argument('--num-workers', type=int,
                       help='Количество рабочих процессов для загрузки данных')
    
    # Параметры данных
    parser.add_argument('--use-cache', action='store_true', default=True,
                       help='Использовать кэшированные данные')
    parser.add_argument('--no-cache', action='store_true',
                       help='Не использовать кэшированные данные')
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Проверка конфликтующих параметров
    if args.train_models and args.no_train:
        logger.error("Нельзя одновременно указать --train-models и --no-train")
        sys.exit(1)
    
    if args.no_cache:
        args.use_cache = False
    
    # Проверка существования файла конфигурации
    if not os.path.exists(args.config):
        logger.error(f"Файл конфигурации {args.config} не найден")
        sys.exit(1)
    
    # Создание директории для результатов
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("="*60)
        logger.info("ЗАПУСК ПОИСКА ЭКЗОПЛАНЕТ")
        logger.info("="*60)
        logger.info(f"TIC ID: {args.tic_ids}")
        logger.info(f"Секторы: {args.sectors or 'все доступные'}")
        logger.info(f"Конфигурация: {args.config}")
        logger.info(f"Обучение моделей: {not args.no_train}")
        logger.info(f"Топ кандидатов: {args.top_n}")
        logger.info(f"Выходная директория: {args.output_dir}")
        logger.info(f"Использование кэша: {args.use_cache}")
        
        # Инициализация pipeline
        logger.info("Инициализация pipeline...")
        pipeline = ExoplanetSearchPipeline(args.config)
        
        # Обновление конфигурации на основе аргументов
        if args.output_dir:
            pipeline.config['output_dir'] = args.output_dir
            pipeline.results_exporter.output_dir = Path(args.output_dir)
        
        if args.num_workers:
            pipeline.config['parallel_config']['num_workers'] = args.num_workers
        
        # Определение режима обучения
        train_models = args.train_models or (not args.no_train)
        
        # Запуск pipeline
        logger.info("Запуск поиска экзопланет...")
        results = pipeline.run_full_pipeline(
            tic_ids=args.tic_ids,
            sectors=args.sectors,
            train_models=train_models,
            top_n=args.top_n
        )
        
        # Вывод результатов
        logger.info("="*60)
        logger.info("РЕЗУЛЬТАТЫ ПОИСКА ЭКЗОПЛАНЕТ")
        logger.info("="*60)
        
        if 'error' in results:
            logger.error(f"ОШИБКА: {results['error']}")
            sys.exit(1)
        
        logger.info(f"Время выполнения: {results.get('duration', 'N/A')}")
        logger.info(f"Анализируемые звезды: {len(results['tic_ids'])}")
        logger.info(f"Загружено кривых блеска: {results.get('loaded_lightcurves', 0)}")
        logger.info(f"Найдено кандидатов: {results['total_candidates']}")
        logger.info(f"Топ кандидатов: {results['top_candidates']}")
        
        # Вывод информации о файлах
        if results.get('exported_files'):
            logger.info("\nЭкспортированные файлы:")
            for file_type, filepath in results['exported_files'].items():
                logger.info(f"  {file_type}: {filepath}")
        
        # Вывод итогового отчета
        if results.get('final_report'):
            logger.info(f"\nИтоговый отчет: {results['final_report']}")
        
        logger.info("="*60)
        logger.info("ПОИСК ЭКЗОПЛАНЕТ ЗАВЕРШЕН УСПЕШНО")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("Поиск прерван пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        logger.exception("Детали ошибки:")
        sys.exit(1)


if __name__ == "__main__":
    main()
