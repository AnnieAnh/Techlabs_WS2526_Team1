import { Filter, X } from 'lucide-react'

import { MultiSelect } from '@/components/shared/MultiSelect'
import { useFilters, useSetFilters } from '@/hooks/useFilters'
import { useOverview } from '@/hooks/useData'
import { activeFilterCount, parseFilterValue, serializeFilterValue } from '@/types/filters'
import { JobFamily, Seniority, WorkModality, ContractType } from '@/types'

const SENIORITY_OPTIONS = Object.values(Seniority)
const MODALITY_OPTIONS = Object.values(WorkModality)
const CONTRACT_OPTIONS = Object.values(ContractType)
const FAMILY_OPTIONS = Object.values(JobFamily).filter((f) => f !== JobFamily.Other)

const STATES = [
  'Baden-Württemberg', 'Bavaria', 'Berlin', 'Brandenburg', 'Bremen',
  'Hamburg', 'Hesse', 'Lower Saxony', 'Mecklenburg-West Pomerania',
  'North Rhine-Westphalia', 'Rhineland-Palatinate', 'Saarland',
  'Saxony', 'Saxony-Anhalt', 'Schleswig-Holstein', 'Thuringia',
]

export const FilterBar = () => {
  const filters = useFilters()
  const { setFilters, clearFilters } = useSetFilters()
  const { data: overview } = useOverview()
  const count = activeFilterCount(filters)

  const cities = overview?.top_cities ? Object.keys(overview.top_cities) : []

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Filter size={14} className="shrink-0 text-muted-foreground" />
      <MultiSelect
        label="Job Family"
        options={FAMILY_OPTIONS}
        selected={parseFilterValue(filters.family)}
        onChange={(v) => setFilters({ family: serializeFilterValue(v) })}
      />
      <MultiSelect
        label="City"
        options={cities}
        selected={parseFilterValue(filters.city)}
        onChange={(v) => setFilters({ city: serializeFilterValue(v) })}
      />
      <MultiSelect
        label="State"
        options={STATES}
        selected={parseFilterValue(filters.state)}
        onChange={(v) => setFilters({ state: serializeFilterValue(v) })}
      />
      <MultiSelect
        label="Seniority"
        options={SENIORITY_OPTIONS}
        selected={parseFilterValue(filters.seniority)}
        onChange={(v) => setFilters({ seniority: serializeFilterValue(v) })}
      />
      <MultiSelect
        label="Modality"
        options={MODALITY_OPTIONS}
        selected={parseFilterValue(filters.modality)}
        onChange={(v) => setFilters({ modality: serializeFilterValue(v) })}
      />
      <MultiSelect
        label="Contract"
        options={CONTRACT_OPTIONS}
        selected={parseFilterValue(filters.contract)}
        onChange={(v) => setFilters({ contract: serializeFilterValue(v) })}
      />
      {count > 0 && (
        <button
          onClick={clearFilters}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
        >
          <X size={12} />
          Clear ({count})
        </button>
      )}
    </div>
  )
}
