import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { exoplanetApi } from '../api/exoplanetApi';

const Landing: React.FC = () => {
  const [stats, setStats] = useState<{ totalPlanets: number; totalHosts: number } | null>(null);

  useEffect(() => {
    exoplanetApi
      .healthCheck()
      .catch(() => null);
    fetch('/api/nasa/stats')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setStats(d))
      .catch(() => undefined);
  }, []);

  return (
    <div className="min-h-screen cosmic-gradient text-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <header className="flex items-center justify-between">
          <Link to="/" className="text-xl font-bold">Exoplanet AI</Link>
          <nav className="space-x-6 text-slate-300">
            <Link to="/how-it-works" className="hover:text-white">Как это работает</Link>
            <Link to="/demo" className="hover:text-white">Demo</Link>
            <a href="https://github.com/" className="hover:text-white" target="_blank" rel="noreferrer">GitHub</a>
          </nav>
        </header>

        <main className="mt-20 grid md:grid-cols-2 gap-10 items-center">
          <div>
            <h1 className="text-5xl md:text-6xl font-extrabold leading-tight">
              Поиск экзопланет с помощью ИИ
            </h1>
            <p className="mt-6 text-lg text-slate-300">
              Загрузите кривую блеска TESS и запустите гибридный алгоритм
              транзитного поиска. Высокая точность, скорость и удобный интерфейс.
            </p>
            <div className="mt-8 flex gap-4">
              <Link to="/demo" className="px-6 py-3 rounded-md bg-cyan-500 hover:bg-cyan-400 text-black font-medium">Попробовать Demo</Link>
              <Link to="/how-it-works" className="px-6 py-3 rounded-md border border-slate-600 hover:border-slate-400">Как это работает</Link>
            </div>
            <div className="mt-8 grid grid-cols-2 gap-6 text-slate-300">
              <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                <div className="text-2xl font-bold">{stats?.totalPlanets ?? '—'}</div>
                <div className="text-sm">Подтвержденных экзопланет (NASA)</div>
              </div>
              <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                <div className="text-2xl font-bold">{stats?.totalHosts ?? '—'}</div>
                <div className="text-sm">Звезд-хозяев</div>
              </div>
            </div>
          </div>
          <div>
            <img src="/vite.svg" alt="Illustration" className="w-full h-auto opacity-90" />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Landing;


