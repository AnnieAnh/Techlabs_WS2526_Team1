import { X } from 'lucide-react'

import { useFilters, useSetFilters } from '@/hooks/useFilters'
import type { DashboardFilters } from '@/types/filters'
import { parseFilterValue, serializeFilterValue } from '@/types/filters'

const FILTER_LABELS: Record<keyof DashboardFilters, string> = {
  family: 'Family',
  city: 'City',
  state: 'State',
  seniority: 'Seniority',
  modality: 'Modality',
  contract: 'Contract',
}

export const FilterChips = () => {
  const filters = useFilters()
  const { setFilters } = useSetFilters()

  const chips: { key: keyof DashboardFilters; value: string }[] = []
  for (const [key, raw] of Object.entries(filters) as [keyof DashboardFilters, string | undefined][]) {
    for (const value of parseFilterValue(raw)) {
      chips.push({ key, value })
    }
  }

  if (chips.length === 0) return null

  const removeChip = (key: keyof DashboardFilters, value: string) => {
    const current = parseFilterValue(filters[key])
    const next = current.filter((v) => v !== value)
    setFilters({ [key]: serializeFilterValue(next) })
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {chips.map(({ key, value }) => (
        <span
          key={`${key}-${value}`}
          className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/50 px-2.5 py-0.5 text-xs font-medium"
        >
          <span className="text-muted-foreground">{FILTER_LABELS[key]}:</span>
          {value}
          <button
            onClick={() => removeChip(key, value)}
            className="ml-0.5 rounded-full p-0.5 transition-colors hover:bg-destructive/10 hover:text-destructive"
            aria-label={`Remove ${value} from ${FILTER_LABELS[key]} filter`}
          >
            <X size={10} />
          </button>
        </span>
      ))}
    </div>
  )
}
