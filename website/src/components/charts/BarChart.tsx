import {
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { CHART_HEX } from '@/lib/colors'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

type BarChartProps = {
  title: string
  data: { name: string; value: number }[]
  color?: string
  layout?: 'horizontal' | 'vertical'
  height?: number
  valueFormatter?: (v: number) => string
}

export const BarChartCard = ({
  title,
  data,
  color = CHART_HEX[0],
  layout = 'vertical',
  height = 400,
  valueFormatter = String,
}: BarChartProps) => (
  <Card>
    <CardHeader className="pb-2">
      <CardTitle className="text-base">{title}</CardTitle>
    </CardHeader>
    <CardContent className="overflow-x-auto">
      <ResponsiveContainer width="100%" height={height} minWidth={350}>
        {layout === 'vertical' ? (
          <RechartsBarChart data={data} layout="vertical" margin={{ left: 100, right: 20, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 12 }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={95} />
            <Tooltip formatter={(v) => [valueFormatter(Number(v)), 'Count']} />
            <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]} />
          </RechartsBarChart>
        ) : (
          <RechartsBarChart data={data} margin={{ left: 10, right: 20, top: 5, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 11, angle: -45, textAnchor: 'end' }} height={60} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => [valueFormatter(Number(v)), 'Count']} />
            <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} />
          </RechartsBarChart>
        )}
      </ResponsiveContainer>
    </CardContent>
  </Card>
)
