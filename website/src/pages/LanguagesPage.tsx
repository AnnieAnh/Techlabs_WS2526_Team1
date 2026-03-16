import { BarChartCard, KpiCard, PageSkeleton, PieChartCard } from '@/components/charts'
import { useLanguages } from '@/hooks/useData'
import { formatNumber, formatPct } from '@/lib/format'

export const LanguagesPage = () => {
  const { data, isLoading } = useLanguages()

  if (isLoading || !data) return <PageSkeleton />

  const germanPctData = Object.entries(data.german_pct_by_family)
    .sort((a, b) => b[1]['German %'] - a[1]['German %'])
    .map(([name, d]) => ({ name, value: Math.round(d['German %'] * 10) / 10 }))

  const englishPctData = Object.entries(data.english_pct_by_family)
    .sort((a, b) => b[1]['English %'] - a[1]['English %'])
    .map(([name, d]) => ({ name, value: Math.round(d['English %'] * 10) / 10 }))

  const cefrData = Object.entries(data.cefr_distribution)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Language Requirements</h1>
        <p className="text-sm text-muted-foreground">German and English language expectations in IT roles</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <KpiCard
          title="German Required"
          value={formatPct(data.german_mention_count / data.total_rows)}
          subtitle={`${formatNumber(data.german_mention_count)} postings`}
        />
        <KpiCard
          title="English Required"
          value={formatPct(data.english_mention_count / data.total_rows)}
          subtitle={`${formatNumber(data.english_mention_count)} postings`}
        />
        <KpiCard title="Total Postings" value={formatNumber(data.total_rows)} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title="German Requirement by Job Family (%)" data={germanPctData} />
        <BarChartCard title="English Requirement by Job Family (%)" data={englishPctData} />
      </div>

      {cefrData.length > 0 && (
        <PieChartCard title="CEFR Level Distribution (German)" data={cefrData} />
      )}
    </div>
  )
}
