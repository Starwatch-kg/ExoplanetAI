import ResultsPage from './results/ResultsPage'
import type { SearchResult } from '../types/api'

interface ResultsDisplayProps {
  result: SearchResult
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  return <ResultsPage result={result} />
}
