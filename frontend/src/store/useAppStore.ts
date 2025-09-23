import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { SearchResult, SearchRequest, HealthStatus, LoadingState } from '../types/api'
import { ToastMessage } from '../types/ui'

interface AppState {
  // Health status
  healthStatus: HealthStatus | null
  
  // Search state
  searchResults: SearchResult[]
  currentSearch: SearchRequest | null
  isSearching: boolean
  
  // Loading states
  loadingStates: Record<string, LoadingState>
  
  // Toast messages
  toasts: ToastMessage[]
  
  // UI state
  sidebarOpen: boolean
  theme: 'dark' | 'light'
  
  // Actions
  setHealthStatus: (status: HealthStatus) => void
  setSearchResults: (results: SearchResult[]) => void
  addSearchResult: (result: SearchResult) => void
  setCurrentSearch: (search: SearchRequest | null) => void
  setIsSearching: (isSearching: boolean) => void
  setLoadingState: (key: string, state: LoadingState) => void
  clearLoadingState: (key: string) => void
  addToast: (toast: Omit<ToastMessage, 'id'>) => void
  removeToast: (id: string) => void
  clearToasts: () => void
  setSidebarOpen: (open: boolean) => void
  setTheme: (theme: 'dark' | 'light') => void
  clearSearchResults: () => void
}

export const useAppStore = create<AppState>()(
  devtools(
    (set, get) => ({
      // Initial state
      healthStatus: null,
      searchResults: [],
      currentSearch: null,
      isSearching: false,
      loadingStates: {},
      toasts: [],
      sidebarOpen: false,
      theme: 'dark',

      // Actions
      setHealthStatus: (status) => set({ healthStatus: status }),
      
      setSearchResults: (results) => set({ searchResults: results }),
      
      addSearchResult: (result) => set((state) => ({
        searchResults: [result, ...state.searchResults]
      })),
      
      setCurrentSearch: (search) => set({ currentSearch: search }),
      
      setIsSearching: (isSearching) => set({ isSearching }),
      
      setLoadingState: (key, state) => set((appState) => ({
        loadingStates: {
          ...appState.loadingStates,
          [key]: state
        }
      })),
      
      clearLoadingState: (key) => set((state) => {
        const newLoadingStates = { ...state.loadingStates }
        delete newLoadingStates[key]
        return { loadingStates: newLoadingStates }
      }),
      
      addToast: (toast) => {
        const id = Math.random().toString(36).substr(2, 9)
        const newToast: ToastMessage = { ...toast, id }
        
        set((state) => ({
          toasts: [...state.toasts, newToast]
        }))
        
        // Auto-remove toast after duration
        if (toast.duration !== 0) {
          setTimeout(() => {
            get().removeToast(id)
          }, toast.duration || 5000)
        }
      },
      
      removeToast: (id) => set((state) => ({
        toasts: state.toasts.filter(toast => toast.id !== id)
      })),
      
      clearToasts: () => set({ toasts: [] }),
      
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      
      setTheme: (theme) => set({ theme }),
      
      clearSearchResults: () => set({ searchResults: [] }),
    }),
    {
      name: 'exoplanet-ai-store',
    }
  )
)
