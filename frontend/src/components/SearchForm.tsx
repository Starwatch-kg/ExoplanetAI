import { useState } from 'react'
import { Search, Loader, X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { SearchFormData } from '../../../front/frontend/src/types/api'

interface SearchFormProps {
  onSearch: (data: SearchFormData) => void
  loading: boolean
  onClear?: () => void
  hasResults?: boolean
}

export default function SearchForm({ onSearch, loading, onClear, hasResults }: SearchFormProps) {
  const { t } = useTranslation()
  const [formData, setFormData] = useState<SearchFormData>({
    target_name: '',
    catalog: 'TIC',
    mission: 'TESS',
    period_min: 1.0,
    period_max: 10.0,
    snr_threshold: 5.0
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (formData.target_name.trim()) {
      // Автоматически добавляем префикс каталога если его нет
      let targetName = formData.target_name.trim()
      
      // Проверяем, есть ли уже префикс
      const hasPrefix = targetName.toLowerCase().startsWith(formData.catalog.toLowerCase())
      
      if (!hasPrefix && /^\d+$/.test(targetName)) {
        // Если это просто число, добавляем префикс каталога
        targetName = `${formData.catalog} ${targetName}`
      }
      
      onSearch({
        ...formData,
        target_name: targetName
      })
    }
  }

  const handleChange = (field: keyof SearchFormData, value: string | number) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  return (
    <div className="bg-white/10 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-white/20 dark:border-gray-700/50">
      <h2 className="text-xl font-semibold text-white dark:text-gray-100 mb-6 text-center">
        {t('search.title')}
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Target Name */}
        <div>
          <label className="block text-sm font-medium text-gray-300 dark:text-gray-400 mb-2">
            {t('search.ticId.label')}
          </label>
          <input
            type="text"
            value={formData.target_name}
            onChange={(e) => handleChange('target_name', e.target.value)}
            placeholder={t('search.ticId.placeholder')}
            className="w-full px-4 py-2 rounded-lg bg-white/20 dark:bg-gray-700/50 text-white dark:text-gray-100 placeholder-gray-300 dark:placeholder-gray-500 border border-white/30 dark:border-gray-600 focus:border-blue-400 focus:outline-none transition-colors"
            required
          />
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
            {t('search.ticId.help')}
          </p>
        </div>

        {/* Catalog and Mission */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 dark:text-gray-400 mb-2">
              Catalog
            </label>
            <select
              value={formData.catalog}
              onChange={(e) => handleChange('catalog', e.target.value as 'TIC' | 'KIC' | 'EPIC')}
              className="w-full px-4 py-2 rounded-lg bg-white/20 dark:bg-gray-700/50 text-white dark:text-gray-100 border border-white/30 dark:border-gray-600 focus:border-blue-400 focus:outline-none transition-colors"
            >
              <option value="TIC">TIC</option>
              <option value="KIC">KIC</option>
              <option value="EPIC">EPIC</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 dark:text-gray-400 mb-2">
              Mission
            </label>
            <select
              value={formData.mission}
              onChange={(e) => handleChange('mission', e.target.value as 'TESS' | 'Kepler' | 'K2')}
              className="w-full px-4 py-2 rounded-lg bg-white/20 dark:bg-gray-700/50 text-white dark:text-gray-100 border border-white/30 dark:border-gray-600 focus:border-blue-400 focus:outline-none transition-colors"
            >
              <option value="TESS">TESS</option>
              <option value="Kepler">Kepler</option>
              <option value="K2">K2</option>
            </select>
          </div>
        </div>

        {/* Period Range */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 dark:text-gray-400 mb-2">
              Min Period ({t('units.days')})
            </label>
            <input
              type="number"
              value={formData.period_min}
              onChange={(e) => handleChange('period_min', parseFloat(e.target.value))}
              min="0.1"
              max="100"
              step="0.1"
              className="w-full px-4 py-2 rounded-lg bg-white/20 dark:bg-gray-700/50 text-white dark:text-gray-100 border border-white/30 dark:border-gray-600 focus:border-blue-400 focus:outline-none transition-colors"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 dark:text-gray-400 mb-2">
              Max Period ({t('units.days')})
            </label>
            <input
              type="number"
              value={formData.period_max}
              onChange={(e) => handleChange('period_max', parseFloat(e.target.value))}
              min="0.1"
              max="100"
              step="0.1"
              className="w-full px-4 py-2 rounded-lg bg-white/20 dark:bg-gray-700/50 text-white dark:text-gray-100 border border-white/30 dark:border-gray-600 focus:border-blue-400 focus:outline-none transition-colors"
            />
          </div>
        </div>

        {/* SNR Threshold */}
        <div>
          <label className="block text-sm font-medium text-gray-300 dark:text-gray-400 mb-2">
            {t('results.statistics.snr')}
          </label>
          <input
            type="number"
            value={formData.snr_threshold}
            onChange={(e) => handleChange('snr_threshold', parseFloat(e.target.value))}
            min="1"
            max="20"
            step="0.1"
            className="w-full px-4 py-2 rounded-lg bg-white/20 dark:bg-gray-700/50 text-white dark:text-gray-100 border border-white/30 dark:border-gray-600 focus:border-blue-400 focus:outline-none transition-colors"
          />
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            type="submit"
            disabled={loading || !formData.target_name.trim()}
            className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white rounded-lg flex items-center justify-center gap-2 transition-all duration-300 font-medium shadow-lg hover:shadow-xl disabled:shadow-none"
          >
            {loading ? (
              <>
                <Loader className="animate-spin" size={20} />
                {t('search.button.searching')}
              </>
            ) : (
              <>
                <Search size={20} />
                {t('search.button.search')}
              </>
            )}
          </button>

          {hasResults && onClear && (
            <button
              type="button"
              onClick={onClear}
              className="px-4 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
              title={t('search.button.clear')}
            >
              <X size={20} />
              <span className="hidden sm:inline">{t('search.button.clear')}</span>
            </button>
          )}
        </div>
      </form>
    </div>
  )
}
