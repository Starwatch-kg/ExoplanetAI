import React from 'react'
import { Languages } from 'lucide-react'
import { useTranslation } from 'react-i18next'

const LanguageToggle: React.FC = () => {
  const { i18n, t } = useTranslation()

  const toggleLanguage = () => {
    const newLang = i18n.language === 'en' ? 'ru' : 'en'
    i18n.changeLanguage(newLang)
    localStorage.setItem('exoplanet-ai-language', newLang)
  }

  return (
    <button
      onClick={toggleLanguage}
      className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
      aria-label={t('language.toggle')}
      title={t('language.toggle')}
    >
      <Languages size={16} />
      <span className="hidden sm:inline">
        {i18n.language === 'en' ? 'RU' : 'EN'}
      </span>
    </button>
  )
}

export default LanguageToggle
