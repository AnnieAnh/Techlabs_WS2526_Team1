import { useCallback } from 'react'

import { PageSkeleton, PieChartCard } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useRemote } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { countByField, crossTab } from '@/lib/aggregations'
import type { JobSlim } from '@/types'

const MODALITY_COLORS: Record<string, string> = {
  'Remote': '#3d9b7f',
  'Hybrid': '#d4a843',
  'On-site': '#e07b54',
}

const computeRemote = (jobs: ReadonlyArray<JobSlim>) => ({
  overall: countByField(jobs, 'work_modality'),
  by_family: crossTab(jobs, 'job_family', 'work_modality'),
  by_seniority: crossTab(jobs, 'seniority_from_title', 'work_modality'),
})

export const RemotePage = () => {
  const { data: staticData, isLoading: staticLoading } = useRemote()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeRemote, [])
  const { data, isLoading, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const overallData = Object.entries(data.overall).map(([name, value]) => ({ name, value }))
  const families = Object.keys(data.by_family)
  const modalities = families.length > 0 ? Object.keys(data.by_family[families[0]]) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Remote Work</h1>
        <p className="text-sm text-muted-foreground">Work modality breakdown across the German IT market</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <PieChartCard title="Overall Work Modality" data={overallData} />

        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Work Modality by Job Family</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="p-2 text-left font-medium">Job Family</th>
                  {modalities.map((m) => (
                    <th key={m} className="p-2 text-right font-medium">{m}</th>
                  ))}
                  <th className="p-2 text-right font-medium">Total</th>
                </tr>
              </thead>
              <tbody>
                {families
                  .sort((a, b) => {
                    const totalA = Object.values(data.by_family[a]).reduce((s, v) => s + v, 0)
                    const totalB = Object.values(data.by_family[b]).reduce((s, v) => s + v, 0)
                    return totalB - totalA
                  })
                  .slice(0, 20)
                  .map((family) => {
                    const total = Object.values(data.by_family[family]).reduce((s, v) => s + v, 0)
                    return (
                      <tr key={family} className="border-b border-border/50">
                        <td className="p-2 font-medium">{family}</td>
                        {modalities.map((m) => (
                          <td key={m} className="p-2 text-right">{data.by_family[family][m] ?? 0}</td>
                        ))}
                        <td className="p-2 text-right font-medium">{total}</td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Work Modality by Seniority</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(data.by_seniority)
              .sort((a, b) => {
                const totalA = Object.values(a[1]).reduce((s, v) => s + v, 0)
                const totalB = Object.values(b[1]).reduce((s, v) => s + v, 0)
                return totalB - totalA
              })
              .map(([level, mods]) => {
                const total = Object.values(mods).reduce((s, v) => s + v, 0)
                return (
                  <div key={level} className="space-y-2 rounded-md border p-3">
                    <p className="text-sm font-medium">{level}</p>
                    {Object.entries(mods).map(([mod, count]) => (
                      <div key={mod} className="space-y-0.5">
                        <div className="flex justify-between text-xs">
                          <span>{mod}</span>
                          <span className="text-muted-foreground">{((count / total) * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-muted">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${(count / total) * 100}%`,
                              backgroundColor: MODALITY_COLORS[mod] ?? '#5bb89a',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )
              })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
