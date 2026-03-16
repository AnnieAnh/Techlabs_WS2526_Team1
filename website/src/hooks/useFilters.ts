import { useNavigate, useSearch } from '@tanstack/react-router'
import { useCallback } from 'react'

import type { DashboardFilters } from '@/types/filters'

export const useFilters = (): DashboardFilters => {
  const search = useSearch({ strict: false }) as Record<string, unknown>
  return {
    family: typeof search.family === 'string' ? search.family : undefined,
    city: typeof search.city === 'string' ? search.city : undefined,
    state: typeof search.state === 'string' ? search.state : undefined,
    seniority: typeof search.seniority === 'string' ? search.seniority : undefined,
    modality: typeof search.modality === 'string' ? search.modality : undefined,
    contract: typeof search.contract === 'string' ? search.contract : undefined,
  }
}

export const useSetFilters = () => {
  const navigate = useNavigate()

  const setFilters = useCallback(
    (updates: Partial<DashboardFilters>) => {
      void navigate({
        search: (prev: Record<string, unknown>) => {
          const next: Record<string, unknown> = { ...prev, ...updates }
          for (const k of Object.keys(next)) {
            if (!next[k]) delete next[k]
          }
          return next as DashboardFilters
        },
        replace: true,
      } as Parameters<typeof navigate>[0])
    },
    [navigate],
  )

  const clearFilters = useCallback(() => {
    void navigate({
      search: {} as DashboardFilters,
      replace: true,
    } as Parameters<typeof navigate>[0])
  }, [navigate])

  return { setFilters, clearFilters }
}
