import { useCallback, useMemo, useState } from 'react'
import { ComposableMap, Geographies, Geography } from 'react-simple-maps'
import { scaleLinear } from 'd3-scale'

import { BarChartCard, PageSkeleton } from '@/components/charts'
import { EmptyState } from '@/components/shared/EmptyState'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useLocation } from '@/hooks/useData'
import { useSetFilters } from '@/hooks/useFilters'
import { useHybridData } from '@/hooks/usePageData'
import { countByField, crossTab } from '@/lib/aggregations'
import { MAP_SCALE } from '@/lib/colors'
import { formatNumber } from '@/lib/format'
import { toEnglishState } from '@/lib/stateNames'
import type { JobSlim } from '@/types'

const GEO_URL = '/data/germany.topo.json'

const computeLocation = (jobs: ReadonlyArray<JobSlim>) => ({
  by_state: countByField(jobs, 'state'),
  by_city: countByField(jobs, 'city'),
  modality_by_state: crossTab(jobs, 'state', 'work_modality'),
})

export const LocationPage = () => {
  const { data: staticData, isLoading: staticLoading } = useLocation()
  const { clearFilters } = useSetFilters()
  const compute = useCallback(computeLocation, [])
  const { data, isLoading, isEmpty } = useHybridData(staticData, staticLoading, compute)
  const [hovered, setHovered] = useState<{ name: string; count: number } | null>(null)

  const maxVal = useMemo(() => {
    if (!data) return 1
    const vals = Object.values(data.by_state)
    return Math.max(...vals, 1)
  }, [data])

  const colorScale = useMemo(
    () =>
      scaleLinear<string>()
        .domain([0, maxVal])
        .range([MAP_SCALE[0], MAP_SCALE[MAP_SCALE.length - 1]]),
    [maxVal],
  )

  if (isLoading || !data) return <PageSkeleton />
  if (isEmpty) return <EmptyState onClear={clearFilters} />

  const cityData = Object.entries(data.by_city)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([name, value]) => ({ name, value }))

  const stateData = Object.entries(data.by_state)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))

  const modalityRows = Object.keys(data.modality_by_state)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Location Insights</h1>
        <p className="text-sm text-muted-foreground">Geographic distribution of IT jobs across Germany</p>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Jobs by Federal State</CardTitle>
        </CardHeader>
        <CardContent className="relative flex flex-col items-center justify-center">
          {hovered && (
            <div className="absolute top-2 left-1/2 z-10 -translate-x-1/2 rounded-md border bg-popover px-3 py-1.5 text-sm font-medium shadow-sm">
              {hovered.name}: {formatNumber(hovered.count)} jobs
            </div>
          )}
          <ComposableMap
            projection="geoMercator"
            projectionConfig={{ center: [10.4, 51.1], scale: 2600 }}
            width={800}
            height={700}
            style={{ maxWidth: '100%', height: 'auto' }}
          >
            <Geographies geography={GEO_URL}>
              {({ geographies }) =>
                geographies.map((geo) => {
                  const germanName = geo.properties.name as string
                  const englishName = toEnglishState(germanName)
                  const count = data.by_state[englishName] ?? 0
                  return (
                    <Geography
                      key={geo.rsmKey}
                      geography={geo}
                      fill={colorScale(count)}
                      stroke="#fff"
                      strokeWidth={0.5}
                      onMouseEnter={() => setHovered({ name: englishName, count })}
                      onMouseLeave={() => setHovered(null)}
                      style={{
                        default: { outline: 'none' },
                        hover: { outline: 'none', fill: '#e07b54', cursor: 'pointer' },
                        pressed: { outline: 'none' },
                      }}
                    />
                  )
                })
              }
            </Geographies>
          </ComposableMap>
        </CardContent>
      </Card>

      <BarChartCard title="Jobs by State" data={stateData} height={480} />

      <BarChartCard title="Top 20 Cities" data={cityData} />

      {modalityRows.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Work Modality by State</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="p-2 text-left font-medium">State</th>
                  {Object.keys(data.modality_by_state[modalityRows[0]]).map((mod) => (
                    <th key={mod} className="p-2 text-right font-medium">{mod}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modalityRows.map((state) => (
                  <tr key={state} className="border-b border-border/50">
                    <td className="p-2 font-medium">{state}</td>
                    {Object.values(data.modality_by_state[state]).map((val, i) => (
                      <td key={i} className="p-2 text-right">{typeof val === 'number' ? val.toFixed(1) : val}%</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
