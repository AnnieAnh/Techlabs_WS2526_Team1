import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

type HeatmapChartProps = {
  title: string
  data: Record<string, Record<string, number>>
  rows: string[]
  cols: string[]
  valueFormatter?: (v: number) => string
}

const getHeatColor = (value: number, max: number, isDark: boolean): string => {
  const t = max === 0 ? 0 : value / max
  if (isDark) {
    // Dark mode: range from very dark (low) to vivid green (high)
    const lightness = 20 + t * 40
    const chroma = 0.02 + t * 0.1
    return `oklch(${lightness}% ${chroma} 165)`
  }
  // Light mode: range from very light (low) to deep green (high)
  const lightness = 95 - t * 50
  return `oklch(${lightness}% 0.12 165)`
}

const getTextColor = (value: number, max: number, isDark: boolean): string => {
  const t = max === 0 ? 0 : value / max
  if (isDark) {
    // Dark mode: always light text, brighter for low values
    return t > 0.5 ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.7)'
  }
  // Light mode: dark text on light cells, white on dark cells
  return t > 0.45 ? 'white' : '#1a1a1a'
}

export const HeatmapChart = ({
  title,
  data,
  rows,
  cols,
  valueFormatter = (v) => `${v.toFixed(1)}%`,
}: HeatmapChartProps) => {
  const allValues = rows.flatMap((r) => cols.map((c) => data[r]?.[c] ?? 0))
  const max = Math.max(...allValues, 1)
  const isDark = typeof document !== 'undefined' && document.documentElement.classList.contains('dark')

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="p-1 text-left font-medium text-muted-foreground" />
              {cols.map((col) => (
                <th
                  key={col}
                  className="max-w-16 truncate p-1 text-center font-medium text-muted-foreground"
                  title={col}
                >
                  {col.length > 10 ? `${col.slice(0, 9)}…` : col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row}>
                <td className="whitespace-nowrap p-1 font-medium">{row}</td>
                {cols.map((col) => {
                  const val = data[row]?.[col] ?? 0
                  return (
                    <td key={col} className="p-0.5">
                      <Tooltip>
                        <TooltipTrigger>
                          <div
                            className="flex h-7 items-center justify-center rounded text-[10px] font-medium"
                            style={{
                              backgroundColor: getHeatColor(val, max, isDark),
                              color: getTextColor(val, max, isDark),
                            }}
                          >
                            {val > 0 ? valueFormatter(val) : ''}
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">
                            {row} × {col}: {valueFormatter(val)}
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}
