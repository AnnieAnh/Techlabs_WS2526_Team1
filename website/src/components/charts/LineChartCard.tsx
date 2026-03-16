import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { CHART_HEX } from '@/lib/colors'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

type LineChartCardProps = {
  title: string
  data: { name: string; value: number }[]
  color?: string
  height?: number
}

export const LineChartCard = ({
  title,
  data,
  color = CHART_HEX[0],
  height = 300,
}: LineChartCardProps) => (
  <Card>
    <CardHeader className="pb-2">
      <CardTitle className="text-base">{title}</CardTitle>
    </CardHeader>
    <CardContent className="overflow-x-auto">
      <ResponsiveContainer width="100%" height={height} minWidth={350}>
        <LineChart data={data} margin={{ left: 10, right: 20, top: 5, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 11, angle: -45, textAnchor: 'end' }} height={60} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
)
