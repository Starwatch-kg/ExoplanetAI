import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

// Import translation files
import enTranslations from './locales/en.json'
import ruTranslations from './locales/ru.json'

const resources = {
  en: {
    translation: enTranslations,
  },
  ru: {
    translation: ruTranslations,
  },
}

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: localStorage.getItem('exoplanet-ai-language') || 'en', // Default language
    fallbackLng: 'en',
    
    interpolation: {
      escapeValue: false, // React already does escaping
    },
    
    // Enable debug mode in development
    debug: process.env.NODE_ENV === 'development',
    
    // Namespace configuration
    defaultNS: 'translation',
    
    // React specific options
    react: {
      useSuspense: false,
    },
  })

export default i18n
