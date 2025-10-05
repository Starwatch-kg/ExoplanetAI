#!/usr/bin/env python3
"""
Упрощенная демонстрация системы управления данными ExoplanetAI
Simple Data Management System Demo for ExoplanetAI
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Импорты модулей системы управления данными
from ingest.data_manager import DataManager
from ingest.storage import StorageManager
from ingest.validator import DataValidator
from ingest.versioning import VersionManager
from preprocessing.lightcurve_processor import LightCurveProcessor
from core.config import get_settings

console = Console()
logger = logging.getLogger(__name__)

class SimpleDataDemo:
    """Упрощенная демонстрация системы управления данными"""
    
    def __init__(self):
        self.settings = get_settings()
        
    async def test_configuration(self) -> bool:
        """Тестирование конфигурации"""
        console.print("🔧 [bold blue]Тестирование конфигурации...[/bold blue]")
        
        try:
            # Проверяем настройки данных
            config_table = Table(title="Конфигурация системы данных")
            config_table.add_column("Параметр", style="cyan")
            config_table.add_column("Значение", style="yellow")
            
            config_table.add_row("Базовый путь данных", str(self.settings.data.data_path))
            config_table.add_row("Путь сырых данных", str(self.settings.data.raw_data_path))
            config_table.add_row("Путь обработанных данных", str(self.settings.data.processed_data_path))
            config_table.add_row("Путь кривых блеска", str(self.settings.data.lightcurves_path))
            config_table.add_row("Версионирование", str(self.settings.data.enable_versioning))
            config_table.add_row("Максимальное хранилище", f"{self.settings.data.max_storage_gb} GB")
            
            console.print(config_table)
            
            # Проверяем существование директорий
            paths_to_check = [
                self.settings.data.data_path,
                self.settings.data.raw_data_path,
                self.settings.data.processed_data_path,
                self.settings.data.lightcurves_path,
            ]
            
            paths_table = Table(title="Состояние директорий")
            paths_table.add_column("Путь", style="cyan")
            paths_table.add_column("Существует", style="green")
            paths_table.add_column("Размер", style="yellow")
            
            for path_str in paths_to_check:
                path = Path(path_str)
                exists = "✅" if path.exists() else "❌"
                
                if path.exists():
                    try:
                        # Подсчитываем размер директории
                        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        size_str = f"{total_size / 1024 / 1024:.2f} MB"
                    except:
                        size_str = "N/A"
                else:
                    size_str = "N/A"
                
                paths_table.add_row(str(path), exists, size_str)
            
            console.print(paths_table)
            
            return True
            
        except Exception as e:
            console.print(f"❌ [bold red]Ошибка конфигурации: {e}[/bold red]")
            return False
    
    async def test_storage_manager(self) -> bool:
        """Тестирование StorageManager"""
        console.print("\n💾 [bold blue]Тестирование StorageManager...[/bold blue]")
        
        try:
            storage = StorageManager()
            
            # Инициализация
            init_result = await storage.initialize()
            console.print(f"Инициализация: {'✅ Успешно' if init_result else '❌ Ошибка'}")
            
            # Тестовые данные
            test_data = {
                "test_key": "test_value",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": [1, 2, 3, 4, 5]
            }
            
            # Сохранение тестовых данных
            test_file = "test_storage.json"
            save_result = await storage.save_metadata(test_file, test_data)
            console.print(f"Сохранение метаданных: {'✅ Успешно' if save_result else '❌ Ошибка'}")
            
            # Загрузка тестовых данных
            loaded_data = await storage.load_metadata(test_file)
            if loaded_data and loaded_data.get("test_key") == "test_value":
                console.print("Загрузка метаданных: ✅ Успешно")
            else:
                console.print("Загрузка метаданных: ❌ Ошибка")
                return False
            
            # Получение статистики
            stats = await storage.get_storage_stats()
            if stats:
                stats_table = Table(title="Статистика хранилища")
                stats_table.add_column("Метрика", style="cyan")
                stats_table.add_column("Значение", style="yellow")
                
                stats_table.add_row("Общий размер", f"{stats.get('total_size_mb', 0):.2f} MB")
                stats_table.add_row("Всего файлов", str(stats.get('total_files', 0)))
                stats_table.add_row("Всего директорий", str(stats.get('total_directories', 0)))
                
                console.print(stats_table)
            
            await storage.cleanup()
            return True
            
        except Exception as e:
            console.print(f"❌ [bold red]Ошибка StorageManager: {e}[/bold red]")
            return False
    
    async def test_validator(self) -> bool:
        """Тестирование DataValidator"""
        console.print("\n🔍 [bold blue]Тестирование DataValidator...[/bold blue]")
        
        try:
            validator = DataValidator()
            
            # Тестовые данные с ошибками
            test_planet_data = {
                "pl_name": ["Valid Planet", "Invalid Planet"],
                "pl_orbper": [365.25, -10.0],  # Отрицательный период - ошибка
                "pl_rade": [1.0, 1000.0],      # Слишком большой радиус - предупреждение
                "ra": [180.0, 400.0],          # RA вне диапазона - ошибка
                "dec": [45.0, 95.0]            # DEC вне диапазона - ошибка
            }
            
            # Валидация планетных данных
            validation_result = await validator.validate_planet_data(test_planet_data)
            
            validation_table = Table(title="Результаты валидации")
            validation_table.add_column("Тип", style="cyan")
            validation_table.add_column("Количество", style="yellow")
            validation_table.add_column("Примеры", style="white")
            
            errors = validation_result.get("errors", [])
            warnings = validation_result.get("warnings", [])
            
            validation_table.add_row(
                "Ошибки", 
                str(len(errors)),
                "; ".join(errors[:2]) if errors else "Нет"
            )
            validation_table.add_row(
                "Предупреждения", 
                str(len(warnings)),
                "; ".join(warnings[:2]) if warnings else "Нет"
            )
            validation_table.add_row(
                "Валидные записи", 
                str(validation_result.get("valid_records", 0)),
                f"из {validation_result.get('total_records', 0)} общих"
            )
            
            console.print(validation_table)
            
            # Проверяем, что валидатор нашел ошибки
            if len(errors) > 0:
                console.print("✅ Валидатор корректно обнаружил ошибки")
                return True
            else:
                console.print("❌ Валидатор не обнаружил ошибки в некорректных данных")
                return False
            
        except Exception as e:
            console.print(f"❌ [bold red]Ошибка DataValidator: {e}[/bold red]")
            return False
    
    async def test_version_manager(self) -> bool:
        """Тестирование VersionManager"""
        console.print("\n📝 [bold blue]Тестирование VersionManager...[/bold blue]")
        
        try:
            version_manager = VersionManager()
            
            # Инициализация
            init_result = await version_manager.initialize()
            console.print(f"Инициализация: {'✅ Успешно' if init_result else '❌ Ошибка'}")
            
            # Создание тестовой версии
            version_name = "demo_test_v1.0"
            version_data = {
                "description": "Тестовая версия для демонстрации",
                "created_by": "simple_demo",
                "include_patterns": ["*.json", "*.csv"],
                "metadata": {
                    "purpose": "testing",
                    "components": ["storage", "validator"]
                }
            }
            
            create_result = await version_manager.create_version(version_name, version_data)
            if create_result.get("success"):
                console.print(f"✅ Версия создана: {version_name}")
                console.print(f"🔗 Commit: {create_result.get('commit_hash', 'N/A')[:8]}...")
            else:
                console.print(f"❌ Ошибка создания версии: {create_result.get('error', 'Unknown')}")
                return False
            
            # Получение списка версий
            versions = await version_manager.list_versions()
            if versions:
                versions_table = Table(title="Доступные версии")
                versions_table.add_column("Версия", style="cyan")
                versions_table.add_column("Дата", style="yellow")
                versions_table.add_column("Описание", style="white")
                
                for version in versions[:3]:  # Показываем первые 3
                    versions_table.add_row(
                        version.get("name", "N/A"),
                        version.get("created_at", "N/A")[:19] if version.get("created_at") else "N/A",
                        (version.get("description", "N/A")[:40] + "...") if len(version.get("description", "")) > 40 else version.get("description", "N/A")
                    )
                
                console.print(versions_table)
            
            return True
            
        except Exception as e:
            console.print(f"❌ [bold red]Ошибка VersionManager: {e}[/bold red]")
            return False
    
    async def test_lightcurve_processor(self) -> bool:
        """Тестирование LightCurveProcessor"""
        console.print("\n⚙️ [bold blue]Тестирование LightCurveProcessor...[/bold blue]")
        
        try:
            processor = LightCurveProcessor()
            
            # Создаем тестовую кривую блеска
            import numpy as np
            
            time_points = np.linspace(0, 10, 1000)
            # Добавляем периодический сигнал + шум + выбросы
            flux = 1.0 + 0.01 * np.sin(2 * np.pi * time_points) + 0.005 * np.random.normal(size=1000)
            # Добавляем несколько выбросов
            flux[100] = 1.1  # Выброс
            flux[500] = 0.8  # Выброс
            flux_err = np.full_like(flux, 0.005)
            
            test_lightcurve = {
                "time": time_points.tolist(),
                "flux": flux.tolist(),
                "flux_err": flux_err.tolist()
            }
            
            # Параметры предобработки
            processing_params = {
                "remove_outliers": True,
                "sigma_clip_sigma": 3.0,
                "baseline_window_length": 51,
                "normalize_method": "median",
                "wavelet_denoising": False
            }
            
            # Предобработка
            result = await processor.process_lightcurve(test_lightcurve, processing_params)
            
            if result.get("success"):
                stats = result.get("processing_stats", {})
                
                processing_table = Table(title="Результаты предобработки")
                processing_table.add_column("Параметр", style="cyan")
                processing_table.add_column("До", style="yellow")
                processing_table.add_column("После", style="green")
                
                processing_table.add_row(
                    "Точек данных",
                    str(stats.get("original_points", 0)),
                    str(stats.get("processed_points", 0))
                )
                processing_table.add_row(
                    "Выбросы удалены",
                    "-",
                    str(stats.get("outliers_removed", 0))
                )
                processing_table.add_row(
                    "RMS шума",
                    f"{stats.get('original_rms', 0):.6f}",
                    f"{stats.get('processed_rms', 0):.6f}"
                )
                
                console.print(processing_table)
                console.print(f"✅ Предобработка завершена за {result.get('processing_time_ms', 0):.1f} мс")
                
                return True
            else:
                console.print(f"❌ Ошибка предобработки: {result.get('error', 'Unknown')}")
                return False
            
        except Exception as e:
            console.print(f"❌ [bold red]Ошибка LightCurveProcessor: {e}[/bold red]")
            return False
    
    async def run_demo(self):
        """Запуск полной демонстрации"""
        console.print(Panel.fit(
            "[bold green]🌟 УПРОЩЕННАЯ ДЕМОНСТРАЦИЯ СИСТЕМЫ УПРАВЛЕНИЯ ДАННЫМИ 🌟[/bold green]\n"
            "[yellow]Simple Data Management System Demo[/yellow]",
            border_style="green"
        ))
        
        tests = [
            ("Конфигурация", self.test_configuration),
            ("StorageManager", self.test_storage_manager),
            ("DataValidator", self.test_validator),
            ("VersionManager", self.test_version_manager),
            ("LightCurveProcessor", self.test_lightcurve_processor)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                console.print(f"❌ [bold red]Критическая ошибка в тесте '{test_name}': {e}[/bold red]")
                results[test_name] = False
        
        # Итоговый отчет
        console.print("\n" + "="*60)
        console.print("[bold blue]📊 ИТОГОВЫЙ ОТЧЕТ[/bold blue]")
        console.print("="*60)
        
        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        result_table = Table(title="Результаты тестов")
        result_table.add_column("Компонент", style="cyan")
        result_table.add_column("Результат", style="white")
        
        for test_name, result in results.items():
            status = "✅ Успешно" if result else "❌ Ошибка"
            result_table.add_row(test_name, status)
        
        console.print(result_table)
        
        if success_count == total_count:
            console.print(f"\n🎉 [bold green]ВСЕ ТЕСТЫ ПРОЙДЕНЫ! ({success_count}/{total_count})[/bold green]")
            console.print("✅ Система управления данными полностью функциональна")
        else:
            console.print(f"\n⚠️ [bold yellow]ЧАСТИЧНЫЙ УСПЕХ: {success_count}/{total_count} тестов пройдено[/bold yellow]")
        
        return success_count == total_count


async def main():
    """Главная функция"""
    demo = SimpleDataDemo()
    success = await demo.run_demo()
    
    if success:
        console.print("\n🎯 [bold green]Демонстрация завершена успешно![/bold green]")
        return 0
    else:
        console.print("\n💥 [bold red]Демонстрация завершена с ошибками[/bold red]")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n🛑 Демонстрация прервана пользователем")
        exit(1)
    except Exception as e:
        console.print(f"\n💥 [bold red]Критическая ошибка: {e}[/bold red]")
        exit(1)
