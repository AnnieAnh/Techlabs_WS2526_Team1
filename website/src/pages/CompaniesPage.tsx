import { useCallback } from 'react'

import { BarChartCard, PageSkeleton } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { useCompanies } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { countByField } from '@/lib/aggregations'
import { CHART_HEX } from '@/lib/colors'
import type { JobSlim } from '@/types'

const computeCompanies = (jobs: ReadonlyArray<JobSlim>) => ({
  top_20: countByField(jobs, 'company_name'),
  family_diversity: {} as Record<string, number>,
})

export const CompaniesPage = () => {
  const { data: staticData, isLoading: staticLoading } = useCompanies()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeCompanies, [])
  const { data, isLoading, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const topData = Object.entries(data.top_20)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([name, value]) => ({ name, value }))

  const diversityData = Object.entries(data.family_diversity)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Top Employers</h1>
        <p className="text-sm text-muted-foreground">Companies with the most IT job postings</p>
      </div>

      <BarChartCard title="Top 20 Companies by Job Postings" data={topData} height={550} />

      {diversityData.length > 0 && (
        <BarChartCard
          title="Companies by Job Family Diversity"
          data={diversityData}
          color={CHART_HEX[2]}
          height={450}
          valueFormatter={(v) => `${v} families`}
        />
      )}
    </div>
  )
}
