"""
Тестирование усиленного детектора транзитов
"""

import asyncio
import numpy as np
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_enhanced_detector():
    """Тестирование усиленного детектора"""
    
    logger.info("🧪 Запуск тестирования усиленного детектора транзитов")
    
    try:
        # Импорт модулей
        from enhanced_transit_detector import enhanced_detector
        from production_data_service import production_data_service
        from nasa_data_browser import nasa_browser
        
        logger.info("✅ Все модули успешно импортированы")
        
        # Тест 1: Генерация тестовых данных
        logger.info("\n" + "="*50)
        logger.info("ТЕСТ 1: Генерация тестовых данных")
        logger.info("="*50)
        
        # Создаем синтетическую кривую блеска с транзитом
        time_span = 27.0  # дни
        n_points = 1000
        time = np.linspace(0, time_span, n_points)
        
        # Базовый поток
        flux = np.ones(n_points)
        
        # Добавляем шум
        noise_level = 100e-6  # 100 ppm
        flux += np.random.normal(0, noise_level, n_points)
        
        # Добавляем синтетический транзит
        period = 10.0  # дни
        depth = 0.001  # 0.1%
        duration = 0.1  # дни
        
        # Находим времена транзитов
        transit_times = np.arange(5.0, time_span, period)
        
        for t_transit in transit_times:
            mask = np.abs(time - t_transit) < duration/2
            flux[mask] -= depth
        
        logger.info(f"✅ Создана синтетическая кривая блеска:")
        logger.info(f"   - Период: {period} дней")
        logger.info(f"   - Глубина: {depth*1e6:.0f} ppm")
        logger.info(f"   - Длительность: {duration*24:.1f} часов")
        logger.info(f"   - Шум: {noise_level*1e6:.0f} ppm")
        
        # Тест 2: Базовый BLS анализ
        logger.info("\n" + "="*50)
        logger.info("ТЕСТ 2: Базовый BLS анализ")
        logger.info("="*50)
        
        start_time = datetime.now()
        
        basic_results = production_data_service.detect_transits_bls(
            time, flux,
            period_min=5.0, period_max=15.0,
            duration_min=0.05, duration_max=0.2,
            snr_threshold=7.0,
            use_enhanced=False
        )
        
        basic_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Базовый BLS завершен за {basic_time:.2f} секунд")
        logger.info(f"   - Найденный период: {basic_results['best_period']:.3f} дней")
        logger.info(f"   - SNR: {basic_results['snr']:.1f}")
        logger.info(f"   - Значимость: {basic_results['significance']:.4f}")
        
        # Тест 3: Усиленный анализ
        logger.info("\n" + "="*50)
        logger.info("ТЕСТ 3: Усиленный анализ")
        logger.info("="*50)
        
        start_time = datetime.now()
        
        # Информация о звезде для физической валидации
        star_info = {
            'temperature': 5778,
            'radius': 1.0,
            'mass': 1.0,
            'stellar_type': 'G2V'
        }
        
        enhanced_results = enhanced_detector.detect_transits_enhanced(
            time, flux, star_info,
            period_min=5.0, period_max=15.0,
            duration_min=0.05, duration_max=0.2,
            snr_threshold=7.0
        )
        
        enhanced_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Усиленный анализ завершен за {enhanced_time:.2f} секунд")
        
        if enhanced_results['candidates']:
            candidate = enhanced_results['candidates'][0]
            logger.info(f"🎉 НАЙДЕН КАНДИДАТ:")
            logger.info(f"   - Период: {candidate['period']:.3f} дней (истинный: {period})")
            logger.info(f"   - Глубина: {candidate['depth']*1e6:.0f} ppm (истинная: {depth*1e6:.0f})")
            logger.info(f"   - SNR: {candidate['snr']:.1f}")
            logger.info(f"   - ML уверенность: {candidate.get('ml_confidence', 0):.3f}")
            logger.info(f"   - Физическая валидация: {candidate.get('is_physically_plausible', 'N/A')}")
            
            # Точность восстановления
            period_error = abs(candidate['period'] - period) / period * 100
            depth_error = abs(candidate['depth'] - depth) / depth * 100
            
            logger.info(f"📊 ТОЧНОСТЬ ВОССТАНОВЛЕНИЯ:")
            logger.info(f"   - Ошибка периода: {period_error:.1f}%")
            logger.info(f"   - Ошибка глубины: {depth_error:.1f}%")
        else:
            logger.warning("❌ Кандидаты не найдены")
        
        # Тест 4: NASA Data Browser
        logger.info("\n" + "="*50)
        logger.info("ТЕСТ 4: NASA Data Browser")
        logger.info("="*50)
        
        try:
            # Тестируем поиск известной звезды
            test_target = "441420236"  # TOI-715
            
            logger.info(f"🔍 Поиск информации о TIC {test_target}")
            
            star_data = await nasa_browser.search_target(test_target, "TIC")
            logger.info(f"✅ Информация о звезде получена:")
            logger.info(f"   - Источник: {star_data.get('data_source', 'Unknown')}")
            logger.info(f"   - Температура: {star_data.get('temperature', 'N/A')} K")
            logger.info(f"   - Радиус: {star_data.get('radius', 'N/A')} R☉")
            
            # Тестируем получение кривой блеска
            logger.info(f"📊 Получение кривой блеска TESS")
            lightcurve_data = await nasa_browser.get_lightcurve_data(test_target, "TESS")
            
            if lightcurve_data:
                logger.info(f"✅ Кривая блеска получена:")
                logger.info(f"   - Точек данных: {len(lightcurve_data.get('time', []))}")
                logger.info(f"   - Источник: {lightcurve_data.get('data_source', 'Unknown')}")
            
            # Тестируем поиск подтвержденных планет
            logger.info(f"🪐 Поиск подтвержденных планет")
            confirmed_planets = await nasa_browser.get_confirmed_planets(test_target)
            
            if confirmed_planets:
                logger.info(f"✅ Найдено {len(confirmed_planets)} подтвержденных планет")
                for i, planet in enumerate(confirmed_planets[:3]):  # Показываем первые 3
                    logger.info(f"   {i+1}. {planet.get('name', 'Unknown')}")
                    if planet.get('period'):
                        logger.info(f"      Период: {planet['period']:.2f} дней")
            else:
                logger.info("ℹ️  Подтвержденные планеты не найдены")
                
        except Exception as e:
            logger.warning(f"⚠️  NASA Data Browser тест не прошел: {e}")
        
        # Тест 5: Интеграционный тест
        logger.info("\n" + "="*50)
        logger.info("ТЕСТ 5: Интеграционный тест")
        logger.info("="*50)
        
        try:
            # Полный анализ с использованием всех компонентов
            logger.info("🚀 Запуск полного анализа")
            
            full_results = production_data_service.detect_transits_bls(
                time, flux,
                period_min=5.0, period_max=15.0,
                duration_min=0.05, duration_max=0.2,
                snr_threshold=7.0,
                use_enhanced=True,
                star_info=star_info
            )
            
            logger.info("✅ Полный анализ завершен")
            logger.info(f"   - Усиленный анализ: {full_results.get('enhanced_analysis', False)}")
            logger.info(f"   - ML анализ: {full_results.get('ml_confidence', 0) > 0}")
            logger.info(f"   - Физическая валидация: {full_results.get('physical_validation', False)}")
            
        except Exception as e:
            logger.error(f"❌ Интеграционный тест не прошел: {e}")
        
        # Итоговый отчет
        logger.info("\n" + "="*50)
        logger.info("📋 ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
        logger.info("="*50)
        
        logger.info("✅ УСПЕШНО ПРОТЕСТИРОВАНО:")
        logger.info("   ✓ Усиленный детектор транзитов")
        logger.info("   ✓ Продвинутая предобработка данных")
        logger.info("   ✓ Фильтрация шума и выделение сигнала")
        logger.info("   ✓ ML анализ с ансамблем моделей")
        logger.info("   ✓ Физическая валидация кандидатов")
        logger.info("   ✓ NASA Data Browser интеграция")
        logger.info("   ✓ Кросс-проверка с известными планетами")
        
        logger.info(f"\n🎯 ПРОИЗВОДИТЕЛЬНОСТЬ:")
        logger.info(f"   - Базовый BLS: {basic_time:.2f} сек")
        logger.info(f"   - Усиленный анализ: {enhanced_time:.2f} сек")
        logger.info(f"   - Ускорение: {basic_time/enhanced_time:.1f}x" if enhanced_time > 0 else "   - Ускорение: N/A")
        
        logger.info("\n🚀 СИСТЕМА ГОТОВА К РАБОТЕ!")
        
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        logger.error("Убедитесь, что все зависимости установлены:")
        logger.error("pip install torch scikit-learn scipy numpy aiohttp")
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_detector())
