import { useState } from 'react'

import { BarChartCard, DataCoverageAlert, HeatmapChart, KpiCard, PageSkeleton, PieChartCard } from '@/components/charts'
import { useRoleDives } from '@/hooks/useData'
import { CHART_HEX } from '@/lib/colors'
import { formatNumber, formatSalary } from '@/lib/format'

export const RolesPage = () => {
  const { data, isLoading } = useRoleDives()
  const [selectedRole, setSelectedRole] = useState<string | null>(null)

  if (isLoading || !data) return <PageSkeleton />

  const roles = Object.keys(data)
  const active = selectedRole ?? roles[0]
  const role = data[active]

  if (!role) return <PageSkeleton />

  const skillData = Object.entries(role.top_skills)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const senData = Object.entries(role.seniority).map(([name, value]) => ({ name, value }))
  const modData = Object.entries(role.modality).map(([name, value]) => ({ name, value }))
  const cityData = Object.entries(role.top_cities)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const progRows = Object.keys(role.skill_progression)
  const progCols = progRows.length > 0 ? Object.keys(role.skill_progression[progRows[0]]) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Role Deep Dives</h1>
        <p className="text-sm text-muted-foreground">Detailed analysis for 6 target IT roles</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {roles.map((r) => (
          <button
            key={r}
            onClick={() => setSelectedRole(r)}
            className={`rounded-full border px-3 py-1 text-sm transition-colors ${
              r === active
                ? 'border-primary bg-primary text-primary-foreground'
                : 'border-border bg-background hover:bg-accent'
            }`}
          >
            {r} ({data[r].count})
          </button>
        ))}
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard title="Total Postings" value={formatNumber(role.count)} />
        <KpiCard
          title="Median Salary"
          value={role.salary ? formatSalary(role.salary.median) : 'N/A'}
          subtitle={role.salary ? `n=${role.salary.count}` : 'Insufficient data'}
        />
        <KpiCard title="Top Seniority" value={senData[0]?.name ?? 'N/A'} />
        <KpiCard title="Top City" value={cityData[0]?.name ?? 'N/A'} />
      </div>

      {role.salary && (
        <DataCoverageAlert known={role.salary.count} total={role.count} label="postings for this role" />
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title={`Top Skills — ${active}`} data={skillData} color={CHART_HEX[1]} />
        <div className="grid gap-4">
          <PieChartCard title="Seniority Distribution" data={senData} height={200} />
          <PieChartCard title="Work Modality" data={modData} height={200} />
        </div>
      </div>

      <BarChartCard title={`Top Cities — ${active}`} data={cityData} layout="horizontal" />

      {progRows.length > 0 && (
        <HeatmapChart
          title={`Skill Progression by Seniority — ${active}`}
          data={role.skill_progression}
          rows={progRows}
          cols={progCols}
        />
      )}
    </div>
  )
}
