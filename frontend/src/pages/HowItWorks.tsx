import React from 'react';
import { Link } from 'react-router-dom';

const HowItWorks: React.FC = () => {
  return (
    <div className="min-h-screen bg-black text-slate-100">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <Link to="/" className="text-slate-400 hover:text-white">← Назад</Link>
        <h1 className="mt-6 text-4xl font-bold">Как это работает</h1>
        <ol className="mt-6 space-y-4 list-decimal list-inside text-slate-300">
          <li>Предобработка: очистка выбросов, нормализация, сглаживание.</li>
          <li>Поиск периодов: BLS/SSP + representation learning для подсказок.</li>
          <li>Валидация: ансамбль моделей фильтрует ложные срабатывания.</li>
          <li>Оценка достоверности: метрики глубины/длительности/формы транзита.</li>
          <li>Результаты: список кандидатов с визуализацией.</li>
        </ol>
        <div className="mt-8">
          <Link to="/demo" className="px-5 py-3 rounded-md bg-cyan-500 text-black font-medium">Открыть Demo</Link>
        </div>
      </div>
    </div>
  );
};

export default HowItWorks;


