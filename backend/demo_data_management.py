#!/usr/bin/env python3
"""
Демонстрационный скрипт для тестирования системы управления данными ExoplanetAI
Comprehensive Data Management System Demo for ExoplanetAI

Этот скрипт демонстрирует все возможности системы:
1. Автоматический ингест данных NASA/MAST/ExoFOP
2. Валидация и верификация данных
3. Версионирование с Git
4. Предобработка кривых блеска
5. Хранение и кэширование
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class DataManagementDemo:
    """Демонстрация системы управления данными"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_system_health(self) -> Dict:
        """Проверка состояния системы"""
        console.print("🔍 [bold blue]Проверка состояния системы...[/bold blue]")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                health_data = await response.json()
                
                # Создаем красивую таблицу статуса
                table = Table(title="Состояние системы ExoplanetAI")
                table.add_column("Компонент", style="cyan")
                table.add_column("Статус", style="green")
                table.add_column("Детали", style="yellow")
                
                overall_status = health_data.get("status", "unknown")
                table.add_row("Общий статус", overall_status, f"v{health_data.get('version', 'unknown')}")
                
                components = health_data.get("components", {})
                for comp_name, comp_data in components.items():
                    status = comp_data.get("status", "unknown")
                    details = []
                    
                    if comp_name == "data_sources":
                        details.append(f"{comp_data.get('initialized', 0)}/{comp_data.get('total', 0)} источников")
                    elif comp_name == "cache":
                        redis_status = "Redis подключен" if comp_data.get("redis_connected") else "Fallback кэш"
                        details.append(redis_status)
                    
                    table.add_row(comp_name.replace("_", " ").title(), status, ", ".join(details))
                
                console.print(table)
                return health_data
                
        except Exception as e:
            console.print(f"❌ [bold red]Ошибка проверки здоровья: {e}[/bold red]")
            return {}
    
    async def test_data_ingestion(self) -> bool:
        """Тестирование ингеста данных"""
        console.print("\n📥 [bold blue]Тестирование ингеста данных...[/bold blue]")
        
        # Тест 1: Ингест таблицы планет
        console.print("1️⃣ Тестирование ингеста таблицы планет...")
        
        table_request = {
            "source": "nasa",
            "table_name": "confirmed_planets",
            "filters": {
                "pl_name": "Kepler-452b"
            },
            "max_records": 10
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/api/v1/data/ingest/table",
                json=table_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    console.print(f"✅ Успешно загружено {result.get('records_ingested', 0)} записей")
                    console.print(f"📁 Сохранено в: {result.get('storage_path', 'N/A')}")
                else:
                    error_text = await response.text()
                    console.print(f"❌ Ошибка ингеста таблицы: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            console.print(f"❌ Исключение при ингесте таблицы: {e}")
            return False
        
        # Тест 2: Ингест кривой блеска
        console.print("2️⃣ Тестирование ингеста кривой блеска...")
        
        lightcurve_request = {
            "target": "TOI-715",
            "mission": "TESS",
            "sector": 1,
            "download_dir": "data/lightcurves/demo"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/api/v1/data/ingest/lightcurve",
                json=lightcurve_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    console.print(f"✅ Кривая блеска загружена: {result.get('file_path', 'N/A')}")
                    console.print(f"📊 Точек данных: {result.get('data_points', 0)}")
                else:
                    error_text = await response.text()
                    console.print(f"❌ Ошибка ингеста кривой блеска: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            console.print(f"❌ Исключение при ингесте кривой блеска: {e}")
            return False
            
        return True
    
    async def test_data_validation(self) -> bool:
        """Тестирование валидации данных"""
        console.print("\n🔍 [bold blue]Тестирование валидации данных...[/bold blue]")
        
        # Создаем тестовые данные для валидации
        test_data = {
            "data_type": "planet_table",
            "data": {
                "pl_name": ["Test Planet 1", "Test Planet 2"],
                "pl_orbper": [365.25, -10.5],  # Второй период отрицательный - ошибка
                "pl_rade": [1.0, 1000.0],      # Второй радиус слишком большой - предупреждение
                "ra": [180.0, 360.1],          # Второе RA вне диапазона - ошибка
                "dec": [45.0, 91.0]            # Второе DEC вне диапазона - ошибка
            },
            "validation_rules": {
                "check_physical_constraints": True,
                "check_coordinate_ranges": True,
                "check_duplicates": True
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/api/v1/data/validate",
                json=test_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Отображаем результаты валидации
                    validation_table = Table(title="Результаты валидации")
                    validation_table.add_column("Тип", style="cyan")
                    validation_table.add_column("Количество", style="yellow")
                    validation_table.add_column("Детали", style="white")
                    
                    validation_table.add_row(
                        "Ошибки", 
                        str(len(result.get("errors", []))),
                        "; ".join(result.get("errors", [])[:3])
                    )
                    validation_table.add_row(
                        "Предупреждения", 
                        str(len(result.get("warnings", []))),
                        "; ".join(result.get("warnings", [])[:3])
                    )
                    validation_table.add_row(
                        "Валидные записи", 
                        str(result.get("valid_records", 0)),
                        f"из {result.get('total_records', 0)} общих"
                    )
                    
                    console.print(validation_table)
                    
                    if result.get("is_valid", False):
                        console.print("✅ Данные прошли валидацию")
                    else:
                        console.print("⚠️ Данные содержат ошибки")
                    
                    return True
                else:
                    error_text = await response.text()
                    console.print(f"❌ Ошибка валидации: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            console.print(f"❌ Исключение при валидации: {e}")
            return False
    
    async def test_preprocessing(self) -> bool:
        """Тестирование предобработки кривых блеска"""
        console.print("\n⚙️ [bold blue]Тестирование предобработки кривых блеска...[/bold blue]")
        
        # Создаем тестовую кривую блеска
        preprocessing_request = {
            "lightcurve_data": {
                "time": list(range(100)),
                "flux": [1.0 + 0.01 * (i % 10) for i in range(100)],  # Простая периодическая кривая
                "flux_err": [0.001] * 100
            },
            "preprocessing_params": {
                "remove_outliers": True,
                "sigma_clip_sigma": 3.0,
                "baseline_window_length": 21,
                "wavelet_denoising": True,
                "wavelet_type": "db4",
                "normalize_method": "median"
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/api/v1/data/preprocess/lightcurve",
                json=preprocessing_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Отображаем результаты предобработки
                    preprocessing_table = Table(title="Результаты предобработки")
                    preprocessing_table.add_column("Параметр", style="cyan")
                    preprocessing_table.add_column("До", style="yellow")
                    preprocessing_table.add_column("После", style="green")
                    
                    stats = result.get("processing_stats", {})
                    preprocessing_table.add_row(
                        "Точек данных",
                        str(stats.get("original_points", 0)),
                        str(stats.get("processed_points", 0))
                    )
                    preprocessing_table.add_row(
                        "Выбросы удалены",
                        "-",
                        str(stats.get("outliers_removed", 0))
                    )
                    preprocessing_table.add_row(
                        "RMS шума",
                        f"{stats.get('original_rms', 0):.6f}",
                        f"{stats.get('processed_rms', 0):.6f}"
                    )
                    
                    console.print(preprocessing_table)
                    
                    console.print(f"✅ Предобработка завершена за {result.get('processing_time_ms', 0):.1f} мс")
                    console.print(f"📁 Результат сохранен: {result.get('output_path', 'N/A')}")
                    
                    return True
                else:
                    error_text = await response.text()
                    console.print(f"❌ Ошибка предобработки: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            console.print(f"❌ Исключение при предобработке: {e}")
            return False
    
    async def test_versioning(self) -> bool:
        """Тестирование системы версионирования"""
        console.print("\n📝 [bold blue]Тестирование системы версионирования...[/bold blue]")
        
        # Создаем новую версию
        version_request = {
            "version_name": f"demo_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Демонстрационная версия данных",
            "include_patterns": ["*.csv", "*.fits"],
            "metadata": {
                "created_by": "demo_script",
                "purpose": "testing",
                "data_sources": ["nasa", "tess"]
            }
        }
        
        try:
            # Создание версии
            async with self.session.post(
                f"{self.base_url}/api/v1/api/v1/data/version/create",
                json=version_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    console.print(f"✅ Версия создана: {result.get('version_name', 'N/A')}")
                    console.print(f"🔗 Commit hash: {result.get('commit_hash', 'N/A')}")
                else:
                    error_text = await response.text()
                    console.print(f"❌ Ошибка создания версии: {response.status} - {error_text}")
                    return False
            
            # Получение списка версий
            async with self.session.get(
                f"{self.base_url}/api/v1/api/v1/data/version/list"
            ) as response:
                if response.status == 200:
                    versions = await response.json()
                    
                    if versions.get("versions"):
                        version_table = Table(title="Доступные версии данных")
                        version_table.add_column("Версия", style="cyan")
                        version_table.add_column("Дата", style="yellow")
                        version_table.add_column("Описание", style="white")
                        
                        for version in versions["versions"][:5]:  # Показываем последние 5
                            version_table.add_row(
                                version.get("name", "N/A"),
                                version.get("created_at", "N/A"),
                                version.get("description", "N/A")[:50] + "..." if len(version.get("description", "")) > 50 else version.get("description", "N/A")
                            )
                        
                        console.print(version_table)
                    else:
                        console.print("📝 Версий пока нет")
                    
                    return True
                else:
                    error_text = await response.text()
                    console.print(f"❌ Ошибка получения версий: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            console.print(f"❌ Исключение при работе с версиями: {e}")
            return False
    
    async def test_storage_stats(self) -> bool:
        """Тестирование статистики хранилища"""
        console.print("\n💾 [bold blue]Получение статистики хранилища...[/bold blue]")
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/v1/api/v1/data/storage/stats"
            ) as response:
                if response.status == 200:
                    stats = await response.json()
                    
                    # Отображаем статистику хранилища
                    storage_table = Table(title="Статистика хранилища")
                    storage_table.add_column("Метрика", style="cyan")
                    storage_table.add_column("Значение", style="yellow")
                    
                    storage_stats = stats.get("storage_stats", {})
                    storage_table.add_row("Общий размер", f"{storage_stats.get('total_size_mb', 0):.2f} MB")
                    storage_table.add_row("Файлов", str(storage_stats.get("total_files", 0)))
                    storage_table.add_row("Директорий", str(storage_stats.get("total_directories", 0)))
                    
                    # Статистика по типам данных
                    data_types = storage_stats.get("by_data_type", {})
                    for data_type, type_stats in data_types.items():
                        storage_table.add_row(
                            f"  {data_type}",
                            f"{type_stats.get('files', 0)} файлов, {type_stats.get('size_mb', 0):.2f} MB"
                        )
                    
                    console.print(storage_table)
                    
                    return True
                else:
                    error_text = await response.text()
                    console.print(f"❌ Ошибка получения статистики: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            console.print(f"❌ Исключение при получении статистики: {e}")
            return False
    
    async def run_full_demo(self):
        """Запуск полной демонстрации"""
        console.print(Panel.fit(
            "[bold green]🌟 ДЕМОНСТРАЦИЯ СИСТЕМЫ УПРАВЛЕНИЯ ДАННЫМИ EXOPLANETAI 🌟[/bold green]\n"
            "[yellow]Comprehensive Data Management System Demo[/yellow]",
            border_style="green"
        ))
        
        # Проверяем здоровье системы
        health = await self.check_system_health()
        if not health or health.get("status") != "healthy":
            console.print("❌ [bold red]Система не готова для демонстрации[/bold red]")
            return False
        
        # Запускаем тесты
        tests = [
            ("Ингест данных", self.test_data_ingestion),
            ("Валидация данных", self.test_data_validation),
            ("Предобработка", self.test_preprocessing),
            ("Версионирование", self.test_versioning),
            ("Статистика хранилища", self.test_storage_stats)
        ]
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for test_name, test_func in tests:
                task = progress.add_task(f"Выполнение: {test_name}", total=None)
                
                try:
                    result = await test_func()
                    results[test_name] = result
                    
                    if result:
                        progress.update(task, description=f"✅ {test_name}")
                    else:
                        progress.update(task, description=f"❌ {test_name}")
                        
                except Exception as e:
                    console.print(f"❌ Ошибка в тесте '{test_name}': {e}")
                    results[test_name] = False
                    progress.update(task, description=f"❌ {test_name}")
                
                progress.remove_task(task)
        
        # Итоговый отчет
        console.print("\n" + "="*60)
        console.print("[bold blue]📊 ИТОГОВЫЙ ОТЧЕТ ДЕМОНСТРАЦИИ[/bold blue]")
        console.print("="*60)
        
        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        result_table = Table(title="Результаты тестов")
        result_table.add_column("Тест", style="cyan")
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
        
        console.print("\n📚 [bold blue]Документация доступна по адресу:[/bold blue]")
        console.print(f"   🌐 {self.base_url}/docs")
        
        return success_count == total_count


async def main():
    """Главная функция"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8001"
    
    console.print(f"🔗 Подключение к серверу: {base_url}")
    
    async with DataManagementDemo(base_url) as demo:
        success = await demo.run_full_demo()
        
        if success:
            console.print("\n🎯 [bold green]Демонстрация завершена успешно![/bold green]")
            return 0
        else:
            console.print("\n💥 [bold red]Демонстрация завершена с ошибками[/bold red]")
            return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n🛑 Демонстрация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n💥 [bold red]Критическая ошибка: {e}[/bold red]")
        sys.exit(1)
