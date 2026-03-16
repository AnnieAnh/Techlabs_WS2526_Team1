import { useCallback } from 'react'

import { BarChartCard, HeatmapChart, KpiCard, PageSkeleton } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { useSeniority } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { countByField, crossTab } from '@/lib/aggregations'
import { formatNumber, formatPct } from '@/lib/format'
import type { JobSlim } from '@/types'

const computeSeniority = (jobs: ReadonlyArray<JobSlim>) => {
  const dist = countByField(jobs, 'seniority_from_title')
  const total = jobs.length
  const specified = Object.values(dist).reduce((s, v) => s + v, 0)
  return {
    distribution: dist,
    unspecified_count: total - specified,
    by_family: crossTab(jobs, 'job_family', 'seniority_from_title'),
  }
}

export const SeniorityPage = () => {
  const { data: staticData, isLoading: staticLoading } = useSeniority()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeSeniority, [])
  const { data, isLoading, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const total = Object.values(data.distribution).reduce((s, v) => s + v, 0) + data.unspecified_count
  const distData = Object.entries(data.distribution)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const familyRows = Object.keys(data.by_family).sort((a, b) => {
    const totalA = Object.values(data.by_family[a]).reduce((s, v) => s + v, 0)
    const totalB = Object.values(data.by_family[b]).reduce((s, v) => s + v, 0)
    return totalB - totalA
  }).slice(0, 15)

  const allSenLevels = new Set<string>()
  familyRows.forEach((f) => Object.keys(data.by_family[f]).forEach((k) => allSenLevels.add(k)))
  const senLevels = ['Junior', 'Senior', 'Lead', 'Director', 'C-Level', 'Unspecified'].filter((l) => allSenLevels.has(l))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Seniority & Experience</h1>
        <p className="text-sm text-muted-foreground">Seniority level distribution across the market</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <KpiCard title="Specified Seniority" value={formatNumber(total - data.unspecified_count)} subtitle={`${formatPct((total - data.unspecified_count) / total)} of postings`} />
        <KpiCard title="Unspecified" value={formatNumber(data.unspecified_count)} subtitle={`${formatPct(data.unspecified_count / total)} of postings`} />
        <KpiCard title="Most Common" value={distData[0]?.name ?? 'N/A'} subtitle={distData[0] ? `${formatNumber(distData[0].value)} postings` : ''} />
      </div>

      <BarChartCard title="Seniority Distribution" data={distData} layout="horizontal" />

      {familyRows.length > 0 && (
        <HeatmapChart
          title="Seniority by Job Family (count)"
          data={data.by_family}
          rows={familyRows}
          cols={senLevels}
          valueFormatter={(v) => String(Math.round(v))}
        />
      )}
    </div>
  )
}
