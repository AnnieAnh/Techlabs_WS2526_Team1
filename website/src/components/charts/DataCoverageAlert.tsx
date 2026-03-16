import { AlertTriangle } from 'lucide-react'

type DataCoverageAlertProps = {
  known: number
  total: number
  label?: string
}

export const DataCoverageAlert = ({ known, total, label = 'records' }: DataCoverageAlertProps) => {
  const pct = total > 0 ? ((known / total) * 100).toFixed(1) : '0'
  return (
    <div className="flex items-center gap-2 rounded-md border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-700 dark:text-yellow-400">
      <AlertTriangle size={16} className="shrink-0" />
      <span>
        Only <strong>{known.toLocaleString()}</strong> of {total.toLocaleString()} {label} ({pct}%) have this data.
        Interpret with caution.
      </span>
    </div>
  )
}
