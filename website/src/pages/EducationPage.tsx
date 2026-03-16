import { useCallback } from 'react'

import { BarChartCard, HeatmapChart, KpiCard, PageSkeleton } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { useEducation } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { countByField, crossTab } from '@/lib/aggregations'
import { formatNumber, formatPct } from '@/lib/format'
import type { JobSlim } from '@/types'

const computeEducation = (jobs: ReadonlyArray<JobSlim>) => {
  const dist = countByField(jobs, 'education_level')
  const specified = Object.values(dist).reduce((s, v) => s + v, 0)
  return {
    distribution: dist,
    unspecified_count: jobs.length - specified,
    by_family: crossTab(jobs, 'job_family', 'education_level'),
    soft_skill_by_family: {} as Record<string, Record<string, number>>,
  }
}

export const EducationPage = () => {
  const { data: staticData, isLoading: staticLoading } = useEducation()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeEducation, [])
  const { data, isLoading, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const total = Object.values(data.distribution).reduce((s, v) => s + v, 0) + data.unspecified_count
  const distData = Object.entries(data.distribution)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const familyRows = Object.keys(data.by_family)
    .sort((a, b) => {
      const totalA = Object.values(data.by_family[a]).reduce((s, v) => s + v, 0)
      const totalB = Object.values(data.by_family[b]).reduce((s, v) => s + v, 0)
      return totalB - totalA
    })
    .slice(0, 15)
  const eduLevels = ['Bachelor', 'Master', 'PhD', 'Degree', 'Vocational', 'Unspecified'].filter((l) =>
    familyRows.some((f) => (data.by_family[f]?.[l] ?? 0) > 0),
  )

  const softRows = Object.keys(data.soft_skill_by_family)
  const softCols = softRows.length > 0 ? Object.keys(data.soft_skill_by_family[softRows[0]]) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Education & Soft Skills</h1>
        <p className="text-sm text-muted-foreground">Education requirements and soft skill expectations</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <KpiCard title="Education Specified" value={formatNumber(total - data.unspecified_count)} subtitle={formatPct((total - data.unspecified_count) / total)} />
        <KpiCard title="Most Common" value={distData[0]?.name ?? 'N/A'} subtitle={distData[0] ? `${formatNumber(distData[0].value)} postings` : ''} />
        <KpiCard title="Unspecified" value={formatNumber(data.unspecified_count)} subtitle={formatPct(data.unspecified_count / total)} />
      </div>

      <BarChartCard title="Education Level Distribution" data={distData} layout="horizontal" />

      {familyRows.length > 0 && (
        <HeatmapChart
          title="Education by Job Family (count)"
          data={data.by_family}
          rows={familyRows}
          cols={eduLevels}
          valueFormatter={(v) => String(Math.round(v))}
        />
      )}

      {softRows.length > 0 && (
        <HeatmapChart
          title="Soft Skill Categories by Job Family (%)"
          data={data.soft_skill_by_family}
          rows={softRows}
          cols={softCols}
        />
      )}
    </div>
  )
}
