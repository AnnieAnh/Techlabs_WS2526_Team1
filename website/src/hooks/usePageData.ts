import { useMemo } from 'react'

import { useJobsSlim } from '@/hooks/useData'
import { useFilteredJobs } from '@/hooks/useFilteredJobs'
import { useFilters } from '@/hooks/useFilters'
import { activeFilterCount } from '@/types/filters'
import type { JobSlim } from '@/types'

/**
 * Returns filtered jobs when filters are active, or undefined when not.
 * Jobs-slim.json is only fetched when at least one filter is active.
 */
export const useFilterContext = () => {
  const filters = useFilters()
  const hasFilters = activeFilterCount(filters) > 0
  const { data: allJobs, isLoading: jobsLoading } = useJobsSlim(hasFilters)
  const filtered = useFilteredJobs(allJobs, filters)

  return {
    filters,
    hasFilters,
    filtered: hasFilters ? filtered : undefined,
    jobsLoading: hasFilters && jobsLoading,
  }
}

/**
 * Generic hybrid hook: returns precomputed static data when no filters,
 * or dynamically aggregated data when filters are active.
 */
export const useHybridData = <TStatic, TComputed>(
  staticData: TStatic | undefined,
  staticLoading: boolean,
  computeFn: (jobs: ReadonlyArray<JobSlim>) => TComputed,
) => {
  const { hasFilters, filtered, jobsLoading } = useFilterContext()

  const computed = useMemo(() => {
    if (!hasFilters || !filtered || filtered.length === 0) return undefined
    return computeFn(filtered)
  }, [hasFilters, filtered, computeFn])

  const data = hasFilters ? computed : staticData
  const isLoading = hasFilters ? jobsLoading : staticLoading
  const isEmpty = hasFilters && filtered !== undefined && filtered.length === 0

  return { data, isLoading, hasFilters, isEmpty, filteredCount: filtered?.length ?? 0 }
}
