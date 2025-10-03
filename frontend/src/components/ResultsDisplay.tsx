import ResultsPage from './results/ResultsPage'
import type { SearchResult } from '../../../front/frontend/src/types/api'

interface ResultsDisplayProps {
  result: SearchResult
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  return <ResultsPage result={result} />
}
