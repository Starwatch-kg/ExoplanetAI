import React, { useState } from 'react';
import { Database, Satellite, Telescope, Globe, Zap } from 'lucide-react';

interface DataSource {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  available: boolean;
}

interface DataSourceSelectorProps {
  selectedSource: string;
  onSourceChange: (source: string) => void;
}

const DataSourceSelector: React.FC<DataSourceSelectorProps> = ({
  selectedSource,
  onSourceChange
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const dataSources: DataSource[] = [
    {
      id: 'tess',
      name: 'TESS',
      description: 'Transiting Exoplanet Survey Satellite',
      icon: <Satellite className="w-5 h-5" />,
      color: 'from-blue-500 to-cyan-500',
      available: true
    },
    {
      id: 'kepler',
      name: 'Kepler',
      description: 'Kepler Space Telescope',
      icon: <Telescope className="w-5 h-5" />,
      color: 'from-purple-500 to-pink-500',
      available: true
    },
    {
      id: 'mast',
      name: 'MAST',
      description: 'Mikulski Archive for Space Telescopes',
      icon: <Database className="w-5 h-5" />,
      color: 'from-green-500 to-emerald-500',
      available: true
    },
    {
      id: 'nasa',
      name: 'NASA Exoplanet Archive',
      description: 'Official NASA exoplanet database',
      icon: <Globe className="w-5 h-5" />,
      color: 'from-orange-500 to-red-500',
      available: true
    },
    {
      id: 'simulated',
      name: 'Симуляция',
      description: 'Сгенерированные данные для тестирования',
      icon: <Zap className="w-5 h-5" />,
      color: 'from-gray-500 to-slate-500',
      available: true
    }
  ];

  const selectedDataSource = dataSources.find(source => source.id === selectedSource);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-3 px-4 py-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 hover:border-white/30 transition-all duration-300"
      >
        <div className={`p-2 rounded-lg bg-gradient-to-r ${selectedDataSource?.color || 'from-gray-500 to-gray-600'}`}>
          {selectedDataSource?.icon}
        </div>
        <div className="text-left">
          <p className="text-white font-medium">{selectedDataSource?.name || 'Выберите источник'}</p>
          <p className="text-gray-400 text-sm">{selectedDataSource?.description || 'Данные для анализа'}</p>
        </div>
        <div className="ml-auto">
          <div className="w-0 h-0 border-l-4 border-l-white/40 border-y-2 border-y-transparent ml-2" />
        </div>
      </button>

      {isOpen && (
        <>
          {/* Overlay */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown */}
          <div className="absolute top-full left-0 right-0 mt-2 bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 shadow-2xl z-20 max-h-80 overflow-y-auto">
            {dataSources.map((source) => (
              <button
                key={source.id}
                onClick={() => {
                  onSourceChange(source.id);
                  setIsOpen(false);
                }}
                disabled={!source.available}
                className={`w-full flex items-center space-x-3 p-4 text-left transition-all duration-200 ${
                  selectedSource === source.id
                    ? 'bg-white/20 border-l-4 border-blue-400'
                    : 'hover:bg-white/10'
                } ${
                  !source.available ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                }`}
              >
                <div className={`p-2 rounded-lg bg-gradient-to-r ${source.color}`}>
                  {source.icon}
                </div>
                <div className="flex-1">
                  <p className="text-white font-medium">{source.name}</p>
                  <p className="text-gray-400 text-sm">{source.description}</p>
                </div>
                {selectedSource === source.id && (
                  <div className="w-2 h-2 bg-blue-400 rounded-full" />
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default DataSourceSelector;
