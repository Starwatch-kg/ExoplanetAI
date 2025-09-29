import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { exoplanetApi, type LightcurveData } from '../api/exoplanetApi';

interface DataLoaderProps {
  onDataLoaded: (data: LightcurveData) => void;
}

const DataLoader: React.FC<DataLoaderProps> = ({ onDataLoaded }) => {
  const [ticId, setTicId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleTicIdSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticId.trim()) return;

    setIsLoading(true);
    setError('');
    setSuccess(false);

    try {
      // Загружаем данные через API
      const response = await exoplanetApi.loadTICData(ticId);
      
      if (response.success) {
        onDataLoaded(response.data);
        setSuccess(true);
      } else {
        setError('Ошибка загрузки данных. Попробуйте другой TIC ID.');
      }
    } catch (err) {
      console.error('API Error:', err);
      setError('Ошибка загрузки данных. Проверьте подключение к серверу.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError('');
    setSuccess(false);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const lines = text.split('\n').filter(line => line.trim());
        
        // Parse CSV data (assuming time,flux format)
        const times: number[] = [];
        const fluxes: number[] = [];
        
        lines.forEach((line, index) => {
          if (index === 0) return; // Skip header
          const [time, flux] = line.split(',').map(Number);
          if (!isNaN(time) && !isNaN(flux)) {
            times.push(time);
            fluxes.push(flux);
          }
        });

        if (times.length > 0) {
          onDataLoaded({ tic_id: 'UPLOADED', times, fluxes });
          setSuccess(true);
        } else {
          setError('Неверный формат файла. Ожидается CSV с колонками time,flux');
        }
      } catch (err) {
        setError('Ошибка чтения файла');
      } finally {
        setIsLoading(false);
      }
    };

    reader.readAsText(file);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-card max-w-2xl mx-auto"
    >
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-4">
          Загрузка данных
        </h2>
        <p className="text-slate-400">
          Введите TIC ID звезды или загрузите файл с кривой блеска
        </p>
      </div>

      <div className="space-y-6">
        {/* TIC ID Input */}
        <form onSubmit={handleTicIdSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              TIC ID (TESS Input Catalog)
            </label>
            <div className="flex space-x-3">
              <input
                type="text"
                value={ticId}
                onChange={(e) => setTicId(e.target.value)}
                placeholder="Например: 261136679"
                className="cosmic-input flex-1"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !ticId.trim()}
                className="cosmic-button disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Upload className="w-5 h-5" />
                )}
                <span>Загрузить</span>
              </button>
            </div>
          </div>
        </form>

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-700" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-4 bg-slate-900 text-slate-400">или</span>
          </div>
        </div>

        {/* File Upload */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Загрузить файл CSV
          </label>
          <div className="relative">
            <input
              type="file"
              accept=".csv,.txt"
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
              disabled={isLoading}
            />
            <label
              htmlFor="file-upload"
              className="cosmic-input w-full cursor-pointer flex items-center justify-center space-x-3 py-8 border-dashed border-2 hover:border-cyan-400 transition-colors"
            >
              <FileText className="w-8 h-8 text-cyan-400" />
              <div className="text-center">
                <p className="text-white font-medium">
                  {isLoading ? 'Загрузка...' : 'Выберите файл или перетащите сюда'}
                </p>
                <p className="text-slate-400 text-sm mt-1">
                  Поддерживаются форматы: CSV, TXT
                </p>
              </div>
            </label>
          </div>
        </div>

        {/* Status Messages */}
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center space-x-3 p-4 bg-red-500/10 border border-red-500/30 rounded-lg"
          >
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <p className="text-red-400">{error}</p>
          </motion.div>
        )}

        {success && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center space-x-3 p-4 bg-green-500/10 border border-green-500/30 rounded-lg"
          >
            <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
            <p className="text-green-400">Данные успешно загружены!</p>
          </motion.div>
        )}

        {/* Example TIC IDs */}
        <div className="mt-8 p-4 bg-slate-800/50 rounded-lg">
          <h3 className="text-sm font-medium text-slate-300 mb-3">
            Примеры TIC ID для тестирования:
          </h3>
          <div className="flex flex-wrap gap-2">
            {['261136679', '38846515', '142802581', '123456789'].map((id) => (
              <button
                key={id}
                onClick={() => setTicId(id)}
                className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm hover:bg-cyan-500/30 transition-colors"
              >
                {id}
              </button>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default DataLoader;
