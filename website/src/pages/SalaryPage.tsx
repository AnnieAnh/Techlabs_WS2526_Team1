import { useCallback } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { DataCoverageAlert, PageSkeleton } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useSalary } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { salaryByGroup } from '@/lib/aggregations'
import { formatSalary } from '@/lib/format'
import type { JobSlim } from '@/types'

type RangeChartProps = {
  title: string
  data: { name: string; p25: number; median: number; p75: number; count: number }[]
}

const SalaryRangeChart = ({ title, data }: RangeChartProps) => (
  <Card>
    <CardHeader className="pb-2">
      <CardTitle className="text-base">{title}</CardTitle>
    </CardHeader>
    <CardContent className="overflow-x-auto">
      <ResponsiveContainer width="100%" height={Math.max(300, data.length * 35)} minWidth={450}>
        <BarChart data={data} layout="vertical" margin={{ left: 120, right: 20, top: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={formatSalary} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={115} />
          <Tooltip
            formatter={(v, name) => [formatSalary(Number(v)), name === 'p25' ? 'P25' : name === 'median' ? 'Median' : 'P75']}
            labelFormatter={(l) => String(l)}
          />
          <Bar dataKey="p25" fill="#a3d4b8" name="P25" radius={[0, 0, 0, 0]} />
          <Bar dataKey="median" fill="#2d7d66" name="Median" radius={[0, 2, 2, 0]} />
          <Bar dataKey="p75" fill="#e07b54" name="P75" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
)

const toRangeData = (raw: Record<string, { median: number; p25: number; p75: number; count: number }>) =>
  Object.entries(raw)
    .sort((a, b) => b[1].median - a[1].median)
    .map(([name, d]) => ({ name, ...d }))

const computeSalary = (jobs: ReadonlyArray<JobSlim>) => ({
  n_with_salary: jobs.filter((j) => j.salary_min != null).length,
  n_total: jobs.length,
  by_family: salaryByGroup(jobs, 'job_family'),
  by_seniority: salaryByGroup(jobs, 'seniority_from_title'),
  by_city: salaryByGroup(jobs, 'city'),
})

export const SalaryPage = () => {
  const { data: staticData, isLoading: staticLoading } = useSalary()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeSalary, [])
  const { data, isLoading, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Salary Analysis</h1>
        <p className="text-sm text-muted-foreground">Salary ranges across roles, seniority levels, and cities</p>
      </div>

      <DataCoverageAlert known={data.n_with_salary} total={data.n_total} label="postings" />

      <SalaryRangeChart title="Salary by Job Family (Median, P25, P75)" data={toRangeData(data.by_family)} />

      <div className="grid gap-4 lg:grid-cols-2">
        <SalaryRangeChart title="Salary by Seniority" data={toRangeData(data.by_seniority)} />
        <SalaryRangeChart title="Salary by City" data={toRangeData(data.by_city)} />
      </div>
    </div>
  )
}
