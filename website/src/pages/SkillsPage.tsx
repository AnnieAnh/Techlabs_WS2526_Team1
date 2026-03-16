import { useCallback } from 'react'

import { BarChartCard, DataCoverageAlert, HeatmapChart, PageSkeleton } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useSkills } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { countJsonArrayField } from '@/lib/aggregations'
import { CHART_HEX } from '@/lib/colors'
import { formatSalary } from '@/lib/format'
import type { JobSlim } from '@/types'

const computeSkills = (jobs: ReadonlyArray<JobSlim>) => ({
  top_25: countJsonArrayField(jobs, 'technical_skills', 25),
  required_vs_optional: {} as Record<string, { required: number; nice_to_have: number }>,
  salary_premium: {} as Record<string, { median: number; count: number }>,
  by_family: {} as Record<string, Record<string, number>>,
  cooccurrence: {} as Record<string, Record<string, number>>,
})

export const SkillsPage = () => {
  const { data: staticData, isLoading: staticLoading } = useSkills()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeSkills, [])
  const { data, isLoading, hasFilters, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const top25Data = Object.entries(data.top_25)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const reqVsOptEntries = Object.entries(data.required_vs_optional)
  const reqVsOptData = reqVsOptEntries
    .slice(0, 15)
    .map(([name, d]) => ({ name, required: d.required, nice_to_have: d.nice_to_have }))

  const salaryPremData = Object.entries(data.salary_premium)
    .sort((a, b) => b[1].median - a[1].median)
    .slice(0, 15)
    .map(([name, d]) => ({ name, value: d.median }))

  const familyRows = Object.keys(data.by_family)
  const familyCols = familyRows.length > 0 ? Object.keys(data.by_family[familyRows[0]]) : []
  const coocRows = Object.keys(data.cooccurrence)

  // When filtered, some sections may be empty — only show what we have
  const showReqVsOpt = reqVsOptData.length > 0
  const showSalary = salaryPremData.length > 0
  const showHeatmaps = familyRows.length > 0

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Skills Demand</h1>
        <p className="text-sm text-muted-foreground">
          {hasFilters ? 'Technical skills for filtered postings' : 'Technical skills landscape across German IT job market'}
        </p>
      </div>

      <BarChartCard title="Top 25 Technical Skills" data={top25Data} height={600} />

      {(showReqVsOpt || showSalary) && (
        <div className="grid gap-4 lg:grid-cols-2">
          {showReqVsOpt && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Required vs Nice-to-Have</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {reqVsOptData.map((d) => {
                    const total = d.required + d.nice_to_have
                    const reqPct = total > 0 ? (d.required / total) * 100 : 0
                    return (
                      <div key={d.name} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="font-medium">{d.name}</span>
                          <span className="text-muted-foreground">{d.required} req / {d.nice_to_have} nth</span>
                        </div>
                        <div className="h-2 rounded-full bg-muted">
                          <div className="h-full rounded-full bg-chart-1" style={{ width: `${reqPct}%` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {showSalary && (
            <div className="space-y-4">
              <DataCoverageAlert known={Object.values(data.salary_premium).reduce((s, d) => s + d.count, 0)} total={19098} label="salary-known rows" />
              <BarChartCard title="Skills by Salary Premium" data={salaryPremData} color={CHART_HEX[2]} valueFormatter={formatSalary} />
            </div>
          )}
        </div>
      )}

      {showHeatmaps && (
        <>
          <HeatmapChart title="Skills by Job Family (%)" data={data.by_family} rows={familyRows} cols={familyCols} />
          {coocRows.length > 0 && (
            <HeatmapChart title="Skill Co-occurrence" data={data.cooccurrence} rows={coocRows} cols={coocRows} valueFormatter={(v) => String(Math.round(v))} />
          )}
        </>
      )}
    </div>
  )
}
