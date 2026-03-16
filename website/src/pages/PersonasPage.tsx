import { PageSkeleton } from '@/components/charts'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { usePersonas } from '@/hooks/useData'
import { formatNumber, formatPct, formatSalary } from '@/lib/format'
import type { PersonaEntry } from '@/types'

const PersonaCard = ({ persona }: { persona: PersonaEntry }) => {
  const skillData = Object.entries(persona.top_skills)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([name, value]) => ({ name, value }))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{persona.label}</CardTitle>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">{formatNumber(persona.count)} jobs</Badge>
          <Badge variant="secondary">{formatPct(persona.pct_of_market)} of market</Badge>
          <Badge variant="secondary">{formatPct(persona.remote_pct)} remote</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-2 sm:grid-cols-2">
          <div>
            <p className="text-xs text-muted-foreground">Top Job Family</p>
            <p className="text-sm font-medium">{persona.top_family ?? 'Various'}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Salary (Median)</p>
            <p className="text-sm font-medium">
              {persona.salary ? formatSalary(persona.salary.median) : 'Insufficient data'}
            </p>
          </div>
        </div>

        <div>
          <p className="mb-1 text-xs text-muted-foreground">Top Cities</p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(persona.top_cities).map(([city, count]) => (
              <Badge key={city} variant="outline" className="text-xs">
                {city} ({count})
              </Badge>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-1 text-xs text-muted-foreground">Top Skills</p>
          <div className="flex flex-wrap gap-1">
            {skillData.map(({ name }) => (
              <Badge key={name} className="text-xs">{name}</Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export const PersonasPage = () => {
  const { data, isLoading } = usePersonas()

  if (isLoading || !data) return <PageSkeleton />

  const personas = [
    data.junior_entry,
    data.senior_specialist,
    data.remote_first,
    data.career_changer,
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Job Seeker Guide</h1>
        <p className="text-sm text-muted-foreground">Four persona profiles with tailored market insights</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {personas.map((p) => (
          <PersonaCard key={p.label} persona={p} />
        ))}
      </div>
    </div>
  )
}
