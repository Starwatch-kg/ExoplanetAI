import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import io
import base64
from datetime import datetime
import lightkurve as lk
from signal_processor import SignalProcessor
import logging

logger = logging.getLogger(__name__)

class LightcurveVisualizer:
    """
    Класс для визуализации кривых блеска с использованием matplotlib и lightkurve
    """

    def __init__(self):
        self.figsize = (12, 8)
        self.dpi = 100

    def create_lightcurve_plot(self, times: List[float], fluxes: List[float],
                             title: str = "Кривая блеска",
                             show_transits: bool = True,
                             transit_positions: Optional[List[int]] = None,
                             save_path: Optional[str] = None) -> str:
        """
        Создает график кривой блеска с возможностью показа транзитов

        Args:
            times: Временные метки
            fluxes: Значения потока
            title: Заголовок графика
            show_transits: Показывать позиции транзитов
            transit_positions: Позиции транзитов для отметки
            save_path: Путь для сохранения файла

        Returns:
            Base64 строка с изображением графика
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            times = np.array(times)
            fluxes = np.array(fluxes)

            # Основной график
            ax.plot(times, fluxes, 'b-', alpha=0.7, linewidth=1, label='Исходные данные')

            # Добавляем сглаженную кривую если есть
            if len(times) > 10:
                from scipy import signal
                window = min(21, len(times)//10)
                if window % 2 == 0:
                    window += 1
                smoothed = signal.savgol_filter(fluxes, window, 2)
                ax.plot(times, smoothed, 'r-', alpha=0.8, linewidth=2, label='Сглаженная')

            # Отмечаем транзиты
            if show_transits and transit_positions:
                for pos in transit_positions:
                    if 0 <= pos < len(times):
                        ax.axvline(x=times[pos], color='red', alpha=0.7, linestyle='--',
                                 label=f'Транзит {pos}')
                        # Добавляем область транзита
                        transit_start = max(0, pos - 5)
                        transit_end = min(len(times), pos + 5)
                        ax.axvspan(times[transit_start], times[transit_end],
                                 alpha=0.2, color='red', label='Область транзита')

            ax.set_xlabel('Время (дни)')
            ax.set_ylabel('Относительный поток')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Форматирование
            if len(times) > 1:
                ax.set_xlim(times[0], times[-1])

            plt.tight_layout()

            # Сохраняем в base64
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path

            # Конвертируем в base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Ошибка создания графика кривой блеска: {e}")
            return self._create_error_plot(f"Ошибка визуализации: {str(e)}")

    def create_phase_folded_plot(self, times: List[float], fluxes: List[float],
                               period: float, title: str = "Сложенная по фазе кривая блеска",
                               save_path: Optional[str] = None) -> str:
        """
        Создает фазово-сложенный график кривой блеска

        Args:
            times: Временные метки
            fluxes: Значения потока
            period: Период для фазового сложения
            title: Заголовок графика
            save_path: Путь для сохранения файла

        Returns:
            Base64 строка с изображением графика
        """
        try:
            if period <= 0:
                return self._create_error_plot("Некорректный период")

            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            times = np.array(times)
            fluxes = np.array(fluxes)

            # Вычисляем фазы
            phases = (times % period) / period

            # Сортируем по фазе
            sort_idx = np.argsort(phases)
            phases_sorted = phases[sort_idx]
            fluxes_sorted = fluxes[sort_idx]

            # Строим график
            ax.plot(phases_sorted, fluxes_sorted, 'b.', alpha=0.6, markersize=2)
            ax.plot(phases_sorted, fluxes_sorted, 'r-', alpha=0.3, linewidth=1)

            ax.set_xlabel('Фаза')
            ax.set_ylabel('Относительный поток')
            ax.set_title(f"{title}\nПериод: {period".3f"} дней", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Добавляем вертикальные линии для транзита
            ax.axvline(x=0.5, color='red', alpha=0.7, linestyle='--', label='Ожидаемый транзит')
            ax.axvspan(0.45, 0.55, alpha=0.2, color='red')

            ax.legend()
            ax.set_xlim(0, 1)
            plt.tight_layout()

            # Сохраняем
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Ошибка создания фазового графика: {e}")
            return self._create_error_plot(f"Ошибка фазовой визуализации: {str(e)}")

    def create_comparison_plot(self, times: List[float], fluxes: List[float],
                             nasa_data: Dict[str, Any],
                             title: str = "Сравнение с данными NASA",
                             save_path: Optional[str] = None) -> str:
        """
        Создает график сравнения пользовательских данных с данными NASA

        Args:
            times: Временные метки
            fluxes: Значения потока
            nasa_data: Данные NASA для сравнения
            title: Заголовок графика
            save_path: Путь для сохранения файла

        Returns:
            Base64 строка с изображением графика
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)

            times = np.array(times)
            fluxes = np.array(fluxes)

            # График 1: Пользовательские данные
            ax1.plot(times, fluxes, 'b-', alpha=0.7, label='Пользовательские данные')
            ax1.set_ylabel('Поток')
            ax1.set_title('Пользовательские данные', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # График 2: Данные NASA (если есть)
            if nasa_data and 'fluxes' in nasa_data:
                nasa_times = np.array(nasa_data['times'])
                nasa_fluxes = np.array(nasa_data['fluxes'])

                ax2.plot(nasa_times, nasa_fluxes, 'r-', alpha=0.7, label='Данные NASA')
                ax2.set_xlabel('Время (дни)')
                ax2.set_ylabel('Поток')
                ax2.set_title('Данные NASA', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                # Добавляем информацию о планетах
                planets = nasa_data.get('confirmed_planets', [])
                if planets:
                    planet_info = f"Планеты: {len(planets)}"
                    for planet in planets[:3]:  # Первые 3 планеты
                        planet_info += f"\n• {planet.get('planet_name', 'Unknown')}"
                    ax2.text(0.02, 0.98, planet_info,
                           transform=ax2.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
            else:
                ax2.text(0.5, 0.5, 'Данные NASA недоступны',
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
                ax2.set_title('Данные NASA (недоступны)')

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Сохраняем
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Ошибка создания графика сравнения: {e}")
            return self._create_error_plot(f"Ошибка сравнения: {str(e)}")

    def create_multi_panel_plot(self, times: List[float], fluxes: List[float],
                              analysis_results: Dict[str, Any],
                              title: str = "Полный анализ кривой блеска",
                              save_path: Optional[str] = None) -> str:
        """
        Создает многостраничный график с полным анализом

        Args:
            times: Временные метки
            fluxes: Значения потока
            analysis_results: Результаты анализа
            title: Заголовок графика
            save_path: Путь для сохранения файла

        Returns:
            Base64 строка с изображением графика
        """
        try:
            fig = plt.figure(figsize=(16, 12), dpi=self.dpi)

            # Создаем GridSpec для сложной компоновки
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            times = np.array(times)
            fluxes = np.array(fluxes)

            # 1. Основная кривая блеска
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(times, fluxes, 'b-', alpha=0.7, linewidth=1)
            ax1.set_ylabel('Поток')
            ax1.set_title('Кривая блеска', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # 2. Периодограмма
            ax2 = fig.add_subplot(gs[1, :2])
            if 'period' in analysis_results:
                frequencies = np.linspace(0.01, 1, 1000)
                power = np.sin(2 * np.pi * frequencies * analysis_results['period']) ** 2
                ax2.plot(frequencies, power, 'g-')
                ax2.axvline(x=1/analysis_results['period'], color='red', linestyle='--',
                           label=f'P = {analysis_results["period"]".3f"} дн.')
                ax2.set_xlabel('Частота (1/день)')
                ax2.set_ylabel('Мощность')
                ax2.set_title('Периодограмма', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. Фазово-сложенная кривая
            ax3 = fig.add_subplot(gs[2, :2])
            if 'period' in analysis_results and analysis_results['period'] > 0:
                phases = (times % analysis_results['period']) / analysis_results['period']
                sort_idx = np.argsort(phases)
                ax3.plot(phases[sort_idx], fluxes[sort_idx], 'b.', alpha=0.6, markersize=2)
                ax3.axvline(x=0.5, color='red', alpha=0.7, linestyle='--')
                ax3.set_xlabel('Фаза')
                ax3.set_ylabel('Поток')
                ax3.set_title(f'Фазово-сложенная (P={analysis_results["period"]".3f"} дн.)', fontweight='bold')
                ax3.grid(True, alpha=0.3)

            # 4. Статистика
            ax4 = fig.add_subplot(gs[0, 2])
            if analysis_results:
                stats_text = "Статистика:\n\n"
                stats_text += f"• Среднее: {analysis_results.get('mean', 0)".4f"}\n"
                stats_text += f"• СКО: {analysis_results.get('std', 0)".4f"}\n"
                stats_text += f"• Асимметрия: {analysis_results.get('skew', 0)".4f"}\n"
                stats_text += f"• Куртозис: {analysis_results.get('kurtosis', 0)".4f"}\n"
                stats_text += f"• Кандидаты: {analysis_results.get('total_candidates', 0)}\n"
                stats_text += f"• Период: {analysis_results.get('period', 'N/A')}\n"

                ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax4.set_title('Статистика', fontweight='bold')
            ax4.axis('off')

            # 5. Легенда кандидатов
            ax5 = fig.add_subplot(gs[1, 2])
            candidates = analysis_results.get('candidates', [])
            if candidates:
                legend_text = f"Кандидаты ({len(candidates)}):\n\n"
                for i, cand in enumerate(candidates[:5]):  # Первые 5
                    conf_level = "🔴" if cand.get('confidence', 0) > 0.7 else "🟡" if cand.get('confidence', 0) > 0.4 else "🟢"
                    legend_text += f"{conf_level} Кандидат {i+1}: {cand.get('confidence', 0)".3f"}\n"

                ax5.text(0.1, 0.5, legend_text, transform=ax5.transAxes,
                        fontsize=9, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax5.text(0.1, 0.5, "Кандидаты не найдены",
                        transform=ax5.transAxes, ha='center', va='center', fontsize=12)
            ax5.set_title('Кандидаты', fontweight='bold')
            ax5.axis('off')

            # 6. Статус анализа
            ax6 = fig.add_subplot(gs[2, 2])
            status_info = "Статус анализа:\n\n"
            status_info += f"• Время обработки: {analysis_results.get('processing_time', 0)".2f"} сек\n"
            status_info += f"• Качество данных: {'Хорошее' if len(times) > 100 else 'Ограниченное'}\n"
            status_info += f"• Метод: {analysis_results.get('method', 'Unknown')}\n"

            if analysis_results.get('nasa_verified', False):
                status_info += "• ✅ Проверено NASA\n"
            else:
                status_info += "• ⚠️ Требует проверки\n"

            ax6.text(0.1, 0.5, status_info, transform=ax6.transAxes,
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax6.set_title('Статус', fontweight='bold')
            ax6.axis('off')

            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Сохраняем
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Ошибка создания многостраничного графика: {e}")
            return self._create_error_plot(f"Ошибка анализа: {str(e)}")

    def _create_error_plot(self, error_message: str) -> str:
        """Создает простой график с сообщением об ошибке"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
            ax.text(0.5, 0.5, f"⚠️ {error_message}",
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcoral'))
            ax.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Ошибка создания графика ошибки: {e}")
            return f"Error creating plot: {error_message}"

# Глобальный экземпляр визуализатора
visualizer = LightcurveVisualizer()

def create_lightcurve_visualization(times: List[float], fluxes: List[float],
                                  analysis_results: Optional[Dict[str, Any]] = None,
                                  comparison_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Создает полную визуализацию кривой блеска

    Args:
        times: Временные метки
        fluxes: Значения потока
        analysis_results: Результаты анализа для статистики
        comparison_data: Данные NASA для сравнения

    Returns:
        Словарь с base64 строками графиков
    """
    try:
        plots = {}

        # Основная кривая блеска
        plots['lightcurve'] = visualizer.create_lightcurve_plot(
            times, fluxes,
            title=f"Кривая блеска - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        # Фазово-сложенная кривая (если есть период)
        if analysis_results and analysis_results.get('period') and analysis_results['period'] > 0:
            plots['phase_folded'] = visualizer.create_phase_folded_plot(
                times, fluxes, analysis_results['period'],
                title="Фазово-сложенная кривая блеска"
            )

        # Сравнение с NASA (если есть данные)
        if comparison_data:
            plots['comparison'] = visualizer.create_comparison_plot(
                times, fluxes, comparison_data,
                title="Сравнение с данными NASA"
            )

        # Многостраничный анализ (если есть результаты)
        if analysis_results:
            plots['full_analysis'] = visualizer.create_multi_panel_plot(
                times, fluxes, analysis_results,
                title="Полный анализ кривой блеска"
            )

        return plots

    except Exception as e:
        logger.error(f"Ошибка создания визуализации: {e}")
        return {
            'error': visualizer._create_error_plot(f"Ошибка визуализации: {str(e)}")
        }
