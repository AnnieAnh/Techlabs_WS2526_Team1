import { useCallback } from 'react'
import { Briefcase, Building2, DollarSign, MapPin } from 'lucide-react'

import { BarChartCard, KpiCard, PageSkeleton, PieChartCard } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { useOverview } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { aggregateOverview } from '@/lib/aggregations'
import { formatNumber, formatPct } from '@/lib/format'

export const OverviewPage = () => {
  const { data: staticData, isLoading: staticLoading } = useOverview()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(aggregateOverview, [])
  const { data, isLoading, hasFilters, isEmpty } = useHybridData(staticData, staticLoading, compute)

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const familyData = Object.entries(data.families)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([name, value]) => ({ name, value }))

  const sourceData = Object.entries(data.source_split).map(([name, value]) => ({ name, value }))

  const seniorityData = Object.entries(data.seniority)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const cityData = Object.entries(data.top_cities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([name, value]) => ({ name, value }))

  // Static data has extra fields; computed data doesn't
  const isStatic = 'date_range' in data
  const subtitle = isStatic
    ? `${formatNumber(data.total_jobs)} German IT job postings from ${(data as typeof staticData & { date_range: { min: string; max: string } }).date_range.min} to ${(data as typeof staticData & { date_range: { min: string; max: string } }).date_range.max}`
    : `${formatNumber(data.total_jobs)} matching job postings`

  const hybridPct = isStatic ? (data as typeof staticData & { hybrid_pct: number }).hybrid_pct : 0
  const modalityKnownPct = isStatic ? (data as typeof staticData & { modality_known_pct: number }).modality_known_pct : 0
  const salaryKnownCount = isStatic
    ? (data as typeof staticData & { salary_known_count: number }).salary_known_count
    : Math.round((data.salary_known_pct / 100) * data.total_jobs)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Market Overview</h1>
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard title="Total Job Postings" value={formatNumber(data.total_jobs)} icon={<Briefcase size={20} />} />
        <KpiCard title="Companies Hiring" value={formatNumber(data.total_companies)} icon={<Building2 size={20} />} />
        <KpiCard
          title="Remote Work"
          value={formatPct(data.remote_pct)}
          subtitle={isStatic ? `${formatPct(hybridPct)} hybrid · ${formatPct(modalityKnownPct)} known` : undefined}
          icon={<MapPin size={20} />}
        />
        <KpiCard
          title="Salary Data"
          value={formatNumber(salaryKnownCount)}
          subtitle={`${formatPct(data.salary_known_pct)} of all postings`}
          icon={<DollarSign size={20} />}
        />
      </div>

      <BarChartCard title={hasFilters ? 'Job Families (filtered)' : 'Top 15 Job Families'} data={familyData} />

      <div className="grid gap-4 lg:grid-cols-3">
        <PieChartCard title="Source Distribution" data={sourceData} />
        <BarChartCard title="Seniority Distribution" data={seniorityData} />
        <BarChartCard title={hasFilters ? 'Top Cities (filtered)' : 'Top 15 Cities'} data={cityData} />
      </div>
    </div>
  )
}
