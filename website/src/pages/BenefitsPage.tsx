import { useCallback } from 'react'

import { BarChartCard, HeatmapChart, PageSkeleton } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { useBenefits } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { countJsonArrayField } from '@/lib/aggregations'
import type { JobSlim } from '@/types'

const computeBenefits = (jobs: ReadonlyArray<JobSlim>) => ({
  category_counts: countJsonArrayField(jobs, 'benefit_categories', 20),
  by_family: {} as Record<string, Record<string, number>>,
})

export const BenefitsPage = () => {
  const { data: staticData, isLoading: staticLoading } = useBenefits()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeBenefits, [])
  const { data, isLoading, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const catData = Object.entries(data.category_counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const familyRows = Object.keys(data.by_family)
  const familyCols = familyRows.length > 0 ? Object.keys(data.by_family[familyRows[0]]) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Benefits Landscape</h1>
        <p className="text-sm text-muted-foreground">Benefit categories offered across German IT positions</p>
      </div>

      <BarChartCard title="Benefit Category Frequency" data={catData} height={450} />

      {familyRows.length > 0 && (
        <HeatmapChart
          title="Benefit Categories by Job Family (%)"
          data={data.by_family}
          rows={familyRows}
          cols={familyCols}
        />
      )}
    </div>
  )
}
