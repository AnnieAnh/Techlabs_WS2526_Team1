import type { ReactNode } from 'react'

import { Card, CardContent } from '@/components/ui/card'

type KpiCardProps = {
  title: string
  value: string | number
  subtitle?: string
  icon?: ReactNode
}

export const KpiCard = ({ title, value, subtitle, icon }: KpiCardProps) => (
  <Card>
    <CardContent className="flex items-start gap-4 p-4">
      {icon && (
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
          {icon}
        </div>
      )}
      <div className="min-w-0">
        <p className="text-sm text-muted-foreground">{title}</p>
        <p className="text-2xl font-bold tracking-tight">{value}</p>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </div>
    </CardContent>
  </Card>
)
