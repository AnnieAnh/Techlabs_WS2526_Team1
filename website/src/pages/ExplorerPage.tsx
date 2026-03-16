import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Briefcase, DollarSign, Globe, Link2, Lock, MapPin, Sparkles, Search, ArrowLeftRight } from 'lucide-react'

import {
  BarChartCard,
  DataCoverageAlert,
  KpiCard,
  PageSkeleton,
  PieChartCard,
} from '@/components/charts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useJobsSlim } from '@/hooks/useData'
import { CHART_HEX, EXTENDED_PALETTE } from '@/lib/colors'
import { formatNumber, formatPct, formatSalary } from '@/lib/format'
import {
  compareRoles,
  estimateSalary,
  exploreByCity,
  exploreByRole,
  exploreBySkills,
  findNoGermanJobs,
  findSkillCombos,
  getAvailableCities,
  getAvailableSkills,
  type SkillCombo,
  type SkillMatch,
} from '@/lib/explore'
import type { JobSlim } from '@/types'

// ── Tab selector ────────────────────────────────────────

type Tab = 'role' | 'city' | 'skills' | 'salary' | 'no-german' | 'combos' | 'compare'

const TABS: { id: Tab; label: string; icon: React.ReactNode; highlight?: boolean }[] = [
  { id: 'role', label: 'By Role', icon: <Briefcase size={16} /> },
  { id: 'city', label: 'By City', icon: <MapPin size={16} /> },
  { id: 'skills', label: 'By Skills', icon: <Sparkles size={16} /> },
  { id: 'salary', label: 'Salary Estimator', icon: <DollarSign size={16} />, highlight: true },
  { id: 'no-german', label: 'No German', icon: <Globe size={16} /> },
  { id: 'combos', label: 'Skill Combos', icon: <Link2 size={16} /> },
  { id: 'compare', label: 'Role vs Role', icon: <ArrowLeftRight size={16} /> },
]

// ── Paywall overlay ─────────────────────────────────────

const Paywall = () => (
  <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm">
    <div className="mx-4 max-w-md rounded-2xl border border-border bg-background p-8 text-center shadow-2xl">
      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
        <Lock size={32} className="text-primary" />
      </div>
      <h2 className="mb-2 text-2xl font-bold tracking-tight">Premium Feature</h2>
      <p className="mb-6 text-sm text-muted-foreground">
        Advanced Information is available exclusively for premium subscribers.
        Unlock detailed role analysis, city insights, and personalized skill recommendations.
      </p>
      <div className="mb-4 rounded-lg border border-primary/20 bg-primary/5 p-4">
        <p className="text-lg font-bold text-primary">€3.00</p>
        <p className="text-xs text-muted-foreground">One-time payment only</p>
      </div>
      <button className="w-full rounded-lg bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90">
        Subscribe Now
      </button>
      <p className="mt-3 text-[10px] text-muted-foreground/60">
        Just kidding... or am I?
      </p>
    </div>
  </div>
)

// ── Main page ───────────────────────────────────────────

export const ExplorerPage = ({ bypassPaywall = false }: { bypassPaywall?: boolean }) => {
  const { data: jobs, isLoading } = useJobsSlim(true)
  const [tab, setTab] = useState<Tab>('role')
  const [showPaywall, setShowPaywall] = useState(false)

  useEffect(() => {
    if (bypassPaywall) return
    const timer = setTimeout(() => setShowPaywall(true), 3000)
    return () => clearTimeout(timer)
  }, [bypassPaywall])

  if (isLoading || !jobs) return <PageSkeleton />

  return (
    <>
      {showPaywall && <Paywall />}
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Advanced Information</h1>
          <p className="text-sm text-muted-foreground">
            Explore the job market by role, city, or your skills
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 rounded-full border px-4 py-1.5 text-sm transition-colors ${
                tab === t.id
                  ? 'border-[#3d9b7f] bg-[#3d9b7f] text-white'
                  : t.highlight
                    ? 'border-amber-400/60 bg-amber-50 text-amber-700 hover:bg-amber-100 dark:border-amber-500/40 dark:bg-amber-950/40 dark:text-amber-300 dark:hover:bg-amber-950/60'
                    : 'border-border bg-background hover:bg-accent'
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>

        {tab === 'role' && <ExploreByRoleTab jobs={jobs} />}
        {tab === 'city' && <ExploreByCityTab jobs={jobs} />}
        {tab === 'skills' && <ExploreBySkillsTab jobs={jobs} />}
        {tab === 'salary' && <SalaryEstimatorTab jobs={jobs} />}
        {tab === 'no-german' && <NoGermanTab jobs={jobs} />}
        {tab === 'combos' && <SkillCombosTab jobs={jobs} />}
        {tab === 'compare' && <RoleCompareTab jobs={jobs} />}
      </div>
    </>
  )
}

/** Easter egg entry — no paywall */
export const EggPage = () => <ExplorerPage bypassPaywall />

// ── By Role Tab ─────────────────────────────────────────

const ExploreByRoleTab = ({ jobs }: { jobs: ReadonlyArray<JobSlim> }) => {
  const roles = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const j of jobs) {
      if (j.job_family) counts[j.job_family] = (counts[j.job_family] ?? 0) + 1
    }
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name]) => name)
  }, [jobs])

  const [selected, setSelected] = useState('Data Scientist')
  const [searchTerm, setSearchTerm] = useState('')

  const filteredRoles = useMemo(
    () =>
      searchTerm
        ? roles.filter((r) => r.toLowerCase().includes(searchTerm.toLowerCase()))
        : roles,
    [roles, searchTerm],
  )

  const result = useMemo(
    () => (selected ? exploreByRole(jobs, selected) : null),
    [jobs, selected],
  )

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Explore by Role</h2>
        <p className="text-sm text-muted-foreground">
          Pick a job family to see its market footprint — top skills employers ask for, salary
          ranges, leading cities, hiring companies, and seniority distribution.
        </p>
      </div>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Select a Role</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative mb-3">
            <Search size={14} className="absolute left-2.5 top-2.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search roles..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring/50"
            />
          </div>
          <div className="flex max-h-48 flex-wrap gap-1.5 overflow-y-auto">
            {filteredRoles.map((r) => (
              <button
                key={r}
                onClick={() => setSelected(r)}
                className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                  r === selected
                    ? 'border-[#3d9b7f] bg-[#3d9b7f] text-white'
                    : 'border-border hover:bg-accent'
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {result && <RoleResults role={selected} data={result} />}
    </div>
  )
}

type RoleResultData = NonNullable<ReturnType<typeof exploreByRole>>

const RoleResults = ({ role, data }: { role: string; data: RoleResultData }) => {
  const skillData = toChartData(data.topSkills)
  const cityData = toChartData(data.topCities)
  const companyData = toChartData(data.topCompanies).slice(0, 15)
  const modalityData = toChartData(data.modality)
  const seniorityData = toChartData(data.seniority)
  const educationData = toChartData(data.education)
  const contractData = toChartData(data.contract)

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard title="Total Postings" value={formatNumber(data.count)} icon={<Briefcase size={20} />} />
        <KpiCard
          title="Median Salary"
          value={data.salary ? formatSalary(data.salary.median) : 'N/A'}
          subtitle={data.salary ? `n=${data.salary.count}` : 'Insufficient data'}
        />
        <KpiCard title="Top City" value={cityData[0]?.name ?? 'N/A'} icon={<MapPin size={20} />} />
        <KpiCard title="Top Seniority" value={seniorityData[0]?.name ?? 'N/A'} />
      </div>

      {data.salary && (
        <DataCoverageAlert known={data.salary.count} total={data.count} label={`${role} postings`} />
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title={`Top Skills — ${role}`} data={skillData} color={CHART_HEX[0]} />
        <BarChartCard title={`Top Cities — ${role}`} data={cityData} color={CHART_HEX[1]} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title="Top Companies" data={companyData} color={CHART_HEX[2]} />
        <div className="grid gap-4">
          <PieChartCard title="Work Modality" data={modalityData} height={200} />
          <PieChartCard title="Seniority Distribution" data={seniorityData} height={200} />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <PieChartCard title="Education Level" data={educationData} />
        <PieChartCard title="Contract Type" data={contractData} />
      </div>
    </div>
  )
}

// ── By City Tab ─────────────────────────────────────────

const ExploreByCityTab = ({ jobs }: { jobs: ReadonlyArray<JobSlim> }) => {
  const cities = useMemo(() => getAvailableCities(jobs, 10), [jobs])
  const [selected, setSelected] = useState('Düsseldorf')
  const [searchTerm, setSearchTerm] = useState('')

  const filteredCities = useMemo(
    () =>
      searchTerm
        ? cities.filter((c) => c.toLowerCase().includes(searchTerm.toLowerCase()))
        : cities,
    [cities, searchTerm],
  )

  const result = useMemo(
    () => (selected ? exploreByCity(jobs, selected) : null),
    [jobs, selected],
  )

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Explore by City</h2>
        <p className="text-sm text-muted-foreground">
          Select a German city to discover its local IT job market — which roles and companies
          dominate, what skills are in demand, and how remote-friendly the area is.
        </p>
      </div>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Select a City</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative mb-3">
            <Search size={14} className="absolute left-2.5 top-2.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search cities..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring/50"
            />
          </div>
          <div className="flex max-h-48 flex-wrap gap-1.5 overflow-y-auto">
            {filteredCities.map((c) => (
              <button
                key={c}
                onClick={() => setSelected(c)}
                className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                  c === selected
                    ? 'border-[#3d9b7f] bg-[#3d9b7f] text-white'
                    : 'border-border hover:bg-accent'
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {result && <CityResults city={selected} data={result} />}
    </div>
  )
}

type CityResultData = NonNullable<ReturnType<typeof exploreByCity>>

const CityResults = ({ city, data }: { city: string; data: CityResultData }) => {
  const roleData = toChartData(data.topRoles)
  const skillData = toChartData(data.topSkills)
  const companyData = toChartData(data.topCompanies).slice(0, 15)
  const modalityData = toChartData(data.modality)
  const seniorityData = toChartData(data.seniority)

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard title="Total Postings" value={formatNumber(data.count)} icon={<MapPin size={20} />} />
        <KpiCard
          title="Median Salary"
          value={data.salary ? formatSalary(data.salary.median) : 'N/A'}
          subtitle={data.salary ? `n=${data.salary.count}` : 'Insufficient data'}
        />
        <KpiCard title="Top Role" value={roleData[0]?.name ?? 'N/A'} icon={<Briefcase size={20} />} />
        <KpiCard title="Top Modality" value={modalityData[0]?.name ?? 'N/A'} />
      </div>

      {data.salary && (
        <DataCoverageAlert known={data.salary.count} total={data.count} label={`${city} postings`} />
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title={`Top Roles — ${city}`} data={roleData} color={CHART_HEX[0]} />
        <BarChartCard title={`Top Skills — ${city}`} data={skillData} color={CHART_HEX[1]} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title="Top Companies" data={companyData} color={CHART_HEX[2]} />
        <div className="grid gap-4">
          <PieChartCard title="Work Modality" data={modalityData} height={200} />
          <PieChartCard title="Seniority Distribution" data={seniorityData} height={200} />
        </div>
      </div>
    </div>
  )
}

// ── By Skills Tab ───────────────────────────────────────

const ExploreBySkillsTab = ({ jobs }: { jobs: ReadonlyArray<JobSlim> }) => {
  const availableSkills = useMemo(() => getAvailableSkills(jobs, 50), [jobs])
  const [selectedSkills, setSelectedSkills] = useState<string[]>(['Python', 'NumPy', 'pandas'])
  const [searchTerm, setSearchTerm] = useState('')

  const filteredSkills = useMemo(
    () =>
      searchTerm
        ? availableSkills.filter((s) => s.toLowerCase().includes(searchTerm.toLowerCase()))
        : availableSkills.slice(0, 80),
    [availableSkills, searchTerm],
  )

  const toggleSkill = useCallback(
    (skill: string) => {
      setSelectedSkills((prev) =>
        prev.includes(skill) ? prev.filter((s) => s !== skill) : [...prev, skill],
      )
    },
    [],
  )

  const result = useMemo(
    () => exploreBySkills(jobs, selectedSkills),
    [jobs, selectedSkills],
  )

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Explore by Skills</h2>
        <p className="text-sm text-muted-foreground">
          Select the skills you already have and get personalized role recommendations — see
          which job families match your profile, your skill overlap percentage, and what to learn next.
        </p>
      </div>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Select Your Skills</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative mb-3">
            <Search size={14} className="absolute left-2.5 top-2.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search skills... (type to find more)"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring/50"
            />
          </div>

          {selectedSkills.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-1.5">
              {selectedSkills.map((s) => (
                <button
                  key={s}
                  onClick={() => toggleSkill(s)}
                  className="flex items-center gap-1 rounded-full border border-primary bg-primary/10 px-2.5 py-1 text-xs text-primary"
                >
                  {s}
                  <span className="text-[10px]">×</span>
                </button>
              ))}
              <button
                onClick={() => setSelectedSkills([])}
                className="rounded-full border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-accent"
              >
                Clear all
              </button>
            </div>
          )}

          <div className="flex max-h-48 flex-wrap gap-1.5 overflow-y-auto">
            {filteredSkills.map((s) => (
              <button
                key={s}
                onClick={() => toggleSkill(s)}
                className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                  selectedSkills.includes(s)
                    ? 'border-[#3d9b7f] bg-[#3d9b7f] text-white'
                    : 'border-border hover:bg-accent'
                }`}
              >
                {s}
              </button>
            ))}
            {!searchTerm && availableSkills.length > 80 && (
              <span className="self-center px-2 text-xs text-muted-foreground">
                Type to search {availableSkills.length - 80} more skills...
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {selectedSkills.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center">
            <Sparkles size={32} className="text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground">
              Select your skills above to get personalized role and city recommendations
            </p>
          </CardContent>
        </Card>
      )}

      {selectedSkills.length > 0 && result.matches.length > 0 && (
        <SkillsResults matches={result.matches} bestCities={result.bestCities} />
      )}

      {selectedSkills.length > 0 && result.matches.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center">
            <Search size={32} className="text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground">
              No matching roles found for this skill combination. Try adding more skills.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

const SkillsResults = ({
  matches,
  bestCities,
}: {
  matches: SkillMatch[]
  bestCities: Record<string, number>
}) => {
  const cityData = toChartData(bestCities).slice(0, 15)

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Recommended Roles</h2>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {matches.slice(0, 9).map((m, i) => (
          <SkillMatchCard key={m.role} match={m} rank={i + 1} />
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard
          title="Best Cities for Your Skills"
          data={cityData}
          color={CHART_HEX[1]}
        />
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">All Matching Roles</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {matches.map((m) => (
                <div
                  key={m.role}
                  className="flex items-center justify-between rounded-md border border-border px-3 py-2 text-sm"
                >
                  <span className="font-medium">{m.role}</span>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span>{formatNumber(m.matchingJobs)} jobs</span>
                    <span>{formatPct(m.matchPct)} match</span>
                    <span>{formatPct(m.avgSkillOverlap)} overlap</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

const SkillMatchCard = ({ match, rank }: { match: SkillMatch; rank: number }) => {
  const accentColor = EXTENDED_PALETTE[(rank - 1) % EXTENDED_PALETTE.length]
  const topCity = Object.entries(match.topCities).sort((a, b) => b[1] - a[1])[0]

  return (
    <Card className="relative overflow-hidden">
      <div className="absolute left-0 top-0 h-full w-1" style={{ backgroundColor: accentColor }} />
      <CardContent className="space-y-3 p-4 pl-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-muted-foreground">#{rank} Recommended</p>
            <p className="text-sm font-semibold">{match.role}</p>
          </div>
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
            {formatPct(match.matchPct, 0)}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <p className="text-muted-foreground">Matching Jobs</p>
            <p className="font-semibold">{formatNumber(match.matchingJobs)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Skill Overlap</p>
            <p className="font-semibold">{formatPct(match.avgSkillOverlap, 0)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Median Salary</p>
            <p className="font-semibold">
              {match.salary ? formatSalary(match.salary.median) : 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Top City</p>
            <p className="font-semibold">{topCity ? topCity[0] : 'N/A'}</p>
          </div>
        </div>

        {match.missingSkills.length > 0 && (
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Skills to learn:</p>
            <div className="flex flex-wrap gap-1">
              {match.missingSkills.map((s) => (
                <span
                  key={s}
                  className="rounded-full border border-orange-300 bg-orange-50 px-2 py-0.5 text-[10px] text-orange-700 dark:border-orange-700 dark:bg-orange-950 dark:text-orange-300"
                >
                  {s}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ── Searchable Dropdown (autocomplete) ───────────────────

const SearchableDropdown = ({
  label,
  options,
  value,
  onChange,
  placeholder = 'Any',
}: {
  label: string
  options: string[]
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) => {
  const [inputValue, setInputValue] = useState(value)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Sync input text when value changes externally (e.g. reset)
  useEffect(() => { setInputValue(value) }, [value])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
        setInputValue(value) // revert to selected value on blur
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open, value])

  const filtered = inputValue && inputValue !== value
    ? options.filter((o) => o.toLowerCase().includes(inputValue.toLowerCase()))
    : options

  const handleInput = (text: string) => {
    setInputValue(text)
    if (!open) setOpen(true)
  }

  const handleSelect = (opt: string) => {
    onChange(opt)
    setInputValue(opt)
    setOpen(false)
  }

  const handleClear = () => {
    onChange('')
    setInputValue('')
    setOpen(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false)
      setInputValue(value)
      inputRef.current?.blur()
    }
  }

  return (
    <div ref={ref} className="relative">
      <label className="mb-1 block text-xs text-muted-foreground">{label}</label>
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => handleInput(e.target.value)}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="h-9 w-full rounded-md border border-input bg-background px-2 pr-7 text-sm outline-none focus:ring-2 focus:ring-ring/50 dark:bg-zinc-800"
        />
        <svg
          width="12" height="12" viewBox="0 0 12 12"
          className={`pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 shrink-0 text-muted-foreground transition-transform ${open ? 'rotate-180' : ''}`}
        >
          <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
        </svg>
      </div>
      {open && (
        <div className="absolute z-50 mt-1 w-full overflow-hidden rounded-md border border-border bg-popover shadow-lg">
          <div className="max-h-52 overflow-y-auto">
            <button
              onClick={handleClear}
              className={`w-full px-3 py-2 text-left text-xs transition-colors hover:bg-accent ${!value ? 'bg-accent/50 font-medium' : ''}`}
            >
              {placeholder}
            </button>
            {filtered.map((opt) => (
              <button
                key={opt}
                onClick={() => handleSelect(opt)}
                className={`w-full px-3 py-2 text-left text-xs transition-colors hover:bg-accent ${opt === value ? 'bg-accent/50 font-medium' : ''}`}
              >
                {opt}
              </button>
            ))}
            {filtered.length === 0 && (
              <p className="px-3 py-2 text-xs text-muted-foreground">No results</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Salary Estimator Tab ────────────────────────────────

const SalaryEstimatorTab = ({ jobs }: { jobs: ReadonlyArray<JobSlim> }) => {
  const roles = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const j of jobs) {
      if (j.job_family) counts[j.job_family] = (counts[j.job_family] ?? 0) + 1
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name]) => name)
  }, [jobs])

  const cities = useMemo(() => getAvailableCities(jobs, 5), [jobs])

  const seniorityLevels = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const j of jobs) {
      if (j.seniority_from_title) counts[j.seniority_from_title] = (counts[j.seniority_from_title] ?? 0) + 1
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name]) => name)
  }, [jobs])

  const [role, setRole] = useState('Data Scientist')
  const [city, setCity] = useState('')
  const [seniority, setSeniority] = useState('')

  const estimate = useMemo(
    () => estimateSalary(jobs, role || undefined, city || undefined, seniority || undefined),
    [jobs, role, city, seniority],
  )

  const hasSelection = role || city || seniority
  const reset = () => { setRole(''); setCity(''); setSeniority('') }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Salary Estimator</h2>
        <p className="text-sm text-muted-foreground">
          Filter by role, city, and seniority to see salary percentiles (P25 / Median / P75)
          based on real postings with published compensation. Only exact filter matches are shown — remove a filter to broaden results.
        </p>
      </div>
      <Card className="overflow-visible">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Configure Your Profile</CardTitle>
            {hasSelection && (
              <button
                onClick={reset}
                className="rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent"
              >
                Reset
              </button>
            )}
          </div>
        </CardHeader>
        <CardContent className="overflow-visible">
          <div className="grid gap-4 sm:grid-cols-3">
            <SearchableDropdown label="Role" options={roles} value={role} onChange={setRole} placeholder="Any role" />
            <SearchableDropdown label="City" options={cities} value={city} onChange={setCity} placeholder="Any city" />
            <SearchableDropdown label="Seniority" options={seniorityLevels} value={seniority} onChange={setSeniority} placeholder="Any level" />
          </div>
        </CardContent>
      </Card>

      <DataCoverageAlert known={402} total={18854} label="postings" />

      {estimate ? (
        <Card className="border-primary/50 ring-1 ring-primary/20">
          <CardContent className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-base font-semibold">{estimate.label}</p>
              <p className="text-xs text-muted-foreground">
                {formatNumber(estimate.totalFiltered)} matching postings,{' '}
                {estimate.stats.count} with salary data
              </p>
            </div>
            <div className="flex items-center gap-8">
              <div className="text-center">
                <p className="text-xs text-muted-foreground">P25</p>
                <p className="text-lg font-semibold">{formatSalary(estimate.stats.p25)}</p>
              </div>
              <div className="text-center">
                <p className="text-xs font-medium text-primary">Median</p>
                <p className="text-2xl font-bold text-primary">{formatSalary(estimate.stats.median)}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-muted-foreground">P75</p>
                <p className="text-lg font-semibold">{formatSalary(estimate.stats.p75)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="py-12 text-center">
            <DollarSign size={32} className="mx-auto mb-2 text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground">
              No salary data available for this combination. Try removing a filter.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

// ── No-German Finder Tab ────────────────────────────────

const NoGermanTab = ({ jobs }: { jobs: ReadonlyArray<JobSlim> }) => {
  const data = useMemo(() => findNoGermanJobs(jobs), [jobs])

  const roleData = toChartData(data.topRoles).slice(0, 15)
  const cityData = toChartData(data.topCities).slice(0, 15)
  const companyData = toChartData(data.topCompanies).slice(0, 15)
  const skillData = toChartData(data.topSkills)
  const modalityData = toChartData(data.modality)
  const remoteRoleData = toChartData(data.remoteTopRoles).slice(0, 10)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">No-German Finder</h2>
        <p className="text-sm text-muted-foreground">
          Jobs in Germany that don't list German as a language requirement — ideal for
          international professionals. Includes remote-friendly positions and the roles, cities,
          and companies most open to English-only candidates.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="No German Required"
          value={formatNumber(data.total)}
          subtitle={`${formatPct(data.pctOfMarket)} of all postings`}
          icon={<Globe size={20} />}
        />
        <KpiCard
          title="Remote + No German"
          value={formatNumber(data.remoteCount)}
          subtitle="Fully remote, no German"
        />
        <KpiCard
          title="Median Salary"
          value={data.salary ? formatSalary(data.salary.median) : 'N/A'}
          subtitle={data.salary ? `n=${data.salary.count}` : 'Insufficient data'}
        />
        <KpiCard
          title="Top Role"
          value={roleData[0]?.name ?? 'N/A'}
          icon={<Briefcase size={20} />}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title="Top Roles (No German)" data={roleData} color={CHART_HEX[0]} />
        <BarChartCard title="Top Cities (No German)" data={cityData} color={CHART_HEX[1]} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <BarChartCard title="Top Skills" data={skillData} color={CHART_HEX[2]} />
        <BarChartCard title="Top Companies" data={companyData} color={CHART_HEX[3]} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <PieChartCard title="Work Modality" data={modalityData} />
        <BarChartCard title="Remote Roles (No German)" data={remoteRoleData} color={CHART_HEX[0]} />
      </div>
    </div>
  )
}

// ── Skill Combos Tab ────────────────────────────────────

const SkillCombosTab = ({ jobs }: { jobs: ReadonlyArray<JobSlim> }) => {
  const [sortBy, setSortBy] = useState<'count' | 'salary'>('count')
  const combos = useMemo(() => findSkillCombos(jobs, 40), [jobs])

  const sorted = useMemo(() => {
    if (sortBy === 'salary') {
      return [...combos]
        .filter((c): c is SkillCombo & { salary: NonNullable<SkillCombo['salary']> } => c.salary !== null)
        .sort((a, b) => b.salary.median - a.salary.median)
    }
    return combos
  }, [combos, sortBy])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Skill Combos</h2>
        <p className="text-sm text-muted-foreground">
          The most frequently co-occurring skill pairs across all postings. See which
          technologies employers expect together and how each combo correlates with salary.
          Sort by popularity or highest-paying pair.
        </p>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Sort by:</span>
        <button
          onClick={() => setSortBy('count')}
          className={`rounded-full border px-3 py-1 text-xs transition-colors ${
            sortBy === 'count' ? 'border-[#3d9b7f] bg-[#3d9b7f] text-white' : 'border-border hover:bg-accent'
          }`}
        >
          Most Common
        </button>
        <button
          onClick={() => setSortBy('salary')}
          className={`rounded-full border px-3 py-1 text-xs transition-colors ${
            sortBy === 'salary' ? 'border-[#3d9b7f] bg-[#3d9b7f] text-white' : 'border-border hover:bg-accent'
          }`}
        >
          Highest Salary
        </button>
      </div>

      <DataCoverageAlert known={402} total={18854} label="postings (salary sort uses salary-known subset)" />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {sorted.slice(0, 21).map((combo, i) => (
          <Card key={combo.skills.join('+')} className="relative overflow-hidden">
            <div
              className="absolute left-0 top-0 h-full w-1"
              style={{ backgroundColor: EXTENDED_PALETTE[i % EXTENDED_PALETTE.length] }}
            />
            <CardContent className="p-4 pl-5">
              <div className="mb-2 flex items-start justify-between">
                <div className="flex flex-wrap gap-1">
                  {combo.skills.map((s) => (
                    <span
                      key={s}
                      className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary"
                    >
                      {s}
                    </span>
                  ))}
                </div>
                <span className="ml-2 text-xs text-muted-foreground">#{i + 1}</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <p className="text-muted-foreground">Job Postings</p>
                  <p className="font-semibold">{formatNumber(combo.count)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Median Salary</p>
                  <p className="font-semibold">
                    {combo.salary ? formatSalary(combo.salary.median) : 'N/A'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

// ── Role vs Role Tab ────────────────────────────────────

const ROLE_B_COLOR = '#e07b54' // terracotta — visually distinct from viridian Role A

const RoleCompareTab = ({ jobs }: { jobs: ReadonlyArray<JobSlim> }) => {
  const roles = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const j of jobs) {
      if (j.job_family) counts[j.job_family] = (counts[j.job_family] ?? 0) + 1
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name]) => name)
  }, [jobs])

  const [roleA, setRoleA] = useState('Data Scientist')
  const [roleB, setRoleB] = useState('Fullstack Developer')

  const result = useMemo(
    () => (roleA && roleB && roleA !== roleB ? compareRoles(jobs, roleA, roleB) : null),
    [jobs, roleA, roleB],
  )

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Role vs Role</h2>
        <p className="text-sm text-muted-foreground">
          Compare two job families side by side — salary, posting volume, skill overlap,
          top cities, work modality, and seniority distribution. Find out what skills transfer
          between roles and where each one stands in the market.
        </p>
      </div>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Select Two Roles to Compare</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">Role A</label>
              <select
                value={roleA}
                onChange={(e) => setRoleA(e.target.value)}
                className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm outline-none focus:ring-2 focus:ring-ring/50 dark:bg-zinc-800"
              >
                {roles.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">Role B</label>
              <select
                value={roleB}
                onChange={(e) => setRoleB(e.target.value)}
                className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm outline-none focus:ring-2 focus:ring-ring/50 dark:bg-zinc-800"
              >
                {roles.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {roleA === roleB && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            Select two different roles to compare.
          </CardContent>
        </Card>
      )}

      {result && (
        <div className="space-y-4">
          {/* KPI comparison */}
          <div className="grid gap-4 sm:grid-cols-2">
            <Card className="border-l-4" style={{ borderLeftColor: CHART_HEX[0] }}>
              <CardContent className="p-4">
                <p className="mb-2 text-sm font-semibold">{roleA}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div><p className="text-muted-foreground">Postings</p><p className="font-semibold">{formatNumber(result.a.count)}</p></div>
                  <div><p className="text-muted-foreground">Median Salary</p><p className="font-semibold">{result.a.salary ? formatSalary(result.a.salary.median) : 'N/A'}</p></div>
                </div>
              </CardContent>
            </Card>
            <Card className="border-l-4" style={{ borderLeftColor: ROLE_B_COLOR }}>
              <CardContent className="p-4">
                <p className="mb-2 text-sm font-semibold">{roleB}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div><p className="text-muted-foreground">Postings</p><p className="font-semibold">{formatNumber(result.b.count)}</p></div>
                  <div><p className="text-muted-foreground">Median Salary</p><p className="font-semibold">{result.b.salary ? formatSalary(result.b.salary.median) : 'N/A'}</p></div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Skill overlap */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Skill Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-3">
                <div>
                  <p className="mb-2 text-xs font-medium" style={{ color: CHART_HEX[0] }}>Only {roleA}</p>
                  <div className="flex flex-wrap gap-1">
                    {result.onlyASkills.map((s) => (
                      <span key={s} className="rounded-full border px-2 py-0.5 text-[11px]" style={{ borderColor: CHART_HEX[0], color: CHART_HEX[0] }}>{s}</span>
                    ))}
                    {result.onlyASkills.length === 0 && <span className="text-xs text-muted-foreground">None</span>}
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs font-medium text-primary">Shared Skills</p>
                  <div className="flex flex-wrap gap-1">
                    {result.sharedSkills.map((s) => (
                      <span key={s} className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] text-primary">{s}</span>
                    ))}
                    {result.sharedSkills.length === 0 && <span className="text-xs text-muted-foreground">None</span>}
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs font-medium" style={{ color: ROLE_B_COLOR }}>Only {roleB}</p>
                  <div className="flex flex-wrap gap-1">
                    {result.onlyBSkills.map((s) => (
                      <span key={s} className="rounded-full border px-2 py-0.5 text-[11px]" style={{ borderColor: ROLE_B_COLOR, color: ROLE_B_COLOR }}>{s}</span>
                    ))}
                    {result.onlyBSkills.length === 0 && <span className="text-xs text-muted-foreground">None</span>}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Side-by-side charts */}
          <div className="grid gap-4 lg:grid-cols-2">
            <BarChartCard title={`Top Skills — ${roleA}`} data={toChartData(result.a.topSkills)} color={CHART_HEX[0]} />
            <BarChartCard title={`Top Skills — ${roleB}`} data={toChartData(result.b.topSkills)} color={ROLE_B_COLOR} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <BarChartCard title={`Top Cities — ${roleA}`} data={toChartData(result.a.topCities).slice(0, 10)} color={CHART_HEX[0]} />
            <BarChartCard title={`Top Cities — ${roleB}`} data={toChartData(result.b.topCities).slice(0, 10)} color={ROLE_B_COLOR} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <PieChartCard title={`Modality — ${roleA}`} data={toChartData(result.a.modality)} />
            <PieChartCard title={`Modality — ${roleB}`} data={toChartData(result.b.modality)} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <PieChartCard title={`Seniority — ${roleA}`} data={toChartData(result.a.seniority)} />
            <PieChartCard title={`Seniority — ${roleB}`} data={toChartData(result.b.seniority)} />
          </div>
        </div>
      )}
    </div>
  )
}

// ── Helpers ─────────────────────────────────────────────

const toChartData = (record: Record<string, number>) =>
  Object.entries(record)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))
