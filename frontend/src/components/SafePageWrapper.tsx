import { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

interface SafePageWrapperProps {
  children: ReactNode
  fallback?: ReactNode
}

const SafePageWrapper = ({ children, fallback }: SafePageWrapperProps) => {
  const { ready } = useTranslation()

  // Если переводы еще не загружены, показываем fallback
  if (!ready) {
    return (
      fallback || (
        <div className="flex items-center justify-center min-h-screen">
          <div className="flex flex-col items-center gap-4">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-amber-500"></div>
            <p className="text-gray-400">Загрузка переводов...</p>
          </div>
        </div>
      )
    )
  }

  return <>{children}</>
}

export default SafePageWrapper
