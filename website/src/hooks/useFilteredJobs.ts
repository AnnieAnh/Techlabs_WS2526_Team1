import { useMemo } from 'react'

import type { JobSlim } from '@/types'
import type { DashboardFilters } from '@/types/filters'
import { parseFilterValue } from '@/types/filters'

const matchesFilter = (jobValue: string | null, filterValue: string | undefined): boolean => {
  if (!filterValue) return true
  if (!jobValue) return false
  const values = parseFilterValue(filterValue)
  return values.length === 0 || values.includes(jobValue)
}

export const useFilteredJobs = (
  jobs: ReadonlyArray<JobSlim> | undefined,
  filters: DashboardFilters,
): ReadonlyArray<JobSlim> => {
  return useMemo(() => {
    if (!jobs) return []
    return jobs.filter((job) => {
      if (!matchesFilter(job.job_family, filters.family)) return false
      if (!matchesFilter(job.city, filters.city)) return false
      if (!matchesFilter(job.state, filters.state)) return false
      if (!matchesFilter(job.seniority_from_title, filters.seniority)) return false
      if (!matchesFilter(job.work_modality, filters.modality)) return false
      if (!matchesFilter(job.contract_type, filters.contract)) return false
      return true
    })
  }, [jobs, filters])
}
