import type { JobSlim } from '@/types'
import { countByField, countJsonArrayField, parseJsonArray, salaryStats } from './aggregations'

// ── By Role ──────────────────────────────────────────────

export const exploreByRole = (jobs: ReadonlyArray<JobSlim>, role: string) => {
  const filtered = jobs.filter((j) => j.job_family === role)
  if (filtered.length === 0) return null

  return {
    count: filtered.length,
    topSkills: countJsonArrayField(filtered, 'technical_skills', 20),
    topCities: countByField(filtered, 'city'),
    topCompanies: countByField(filtered, 'company_name'),
    modality: countByField(filtered, 'work_modality'),
    seniority: countByField(filtered, 'seniority_from_title'),
    education: countByField(filtered, 'education_level'),
    contract: countByField(filtered, 'contract_type'),
    salary: salaryStats(filtered),
  }
}

// ── By City ──────────────────────────────────────────────

export const exploreByCity = (jobs: ReadonlyArray<JobSlim>, city: string) => {
  const filtered = jobs.filter((j) => j.city === city)
  if (filtered.length === 0) return null

  return {
    count: filtered.length,
    topRoles: countByField(filtered, 'job_family'),
    topSkills: countJsonArrayField(filtered, 'technical_skills', 20),
    topCompanies: countByField(filtered, 'company_name'),
    modality: countByField(filtered, 'work_modality'),
    seniority: countByField(filtered, 'seniority_from_title'),
    salary: salaryStats(filtered),
  }
}

// ── By Skills (recommendation engine) ────────────────────

export type SkillMatch = {
  role: string
  totalJobs: number
  matchingJobs: number
  matchPct: number
  avgSkillOverlap: number
  topCities: Record<string, number>
  salary: ReturnType<typeof salaryStats>
  missingSkills: string[]
}

export const exploreBySkills = (
  jobs: ReadonlyArray<JobSlim>,
  userSkills: string[],
): { matches: SkillMatch[]; bestCities: Record<string, number> } => {
  if (userSkills.length === 0) return { matches: [], bestCities: {} }

  const userSet = new Set(userSkills.map((s) => s.toLowerCase()))

  // Score each job by skill overlap
  type ScoredJob = { job: JobSlim; overlap: number; jobSkills: string[] }
  const scored: ScoredJob[] = []

  for (const job of jobs) {
    const jobSkills = parseJsonArray(job.technical_skills)
    if (jobSkills.length === 0) continue
    const overlap = jobSkills.filter((s) => userSet.has(s.toLowerCase())).length
    if (overlap > 0) scored.push({ job, overlap, jobSkills })
  }

  // Group by role
  const byRole = new Map<string, ScoredJob[]>()
  for (const s of scored) {
    const role = s.job.job_family ?? 'Other'
    const arr = byRole.get(role) ?? []
    arr.push(s)
    byRole.set(role, arr)
  }

  // Build matches per role
  const matches: SkillMatch[] = []
  for (const [role, roleJobs] of byRole) {
    const totalRoleJobs = jobs.filter((j) => j.job_family === role).length
    const avgOverlap =
      roleJobs.reduce((sum, j) => sum + j.overlap / j.jobSkills.length, 0) / roleJobs.length

    // Find skills commonly required in this role but missing from user
    const roleSkillCounts: Record<string, number> = {}
    for (const j of roleJobs) {
      for (const s of j.jobSkills) {
        if (!userSet.has(s.toLowerCase())) {
          roleSkillCounts[s] = (roleSkillCounts[s] ?? 0) + 1
        }
      }
    }
    const missingSkills = Object.entries(roleSkillCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([s]) => s)

    // Top cities for matching jobs
    const cityCounts: Record<string, number> = {}
    for (const j of roleJobs) {
      if (j.job.city) cityCounts[j.job.city] = (cityCounts[j.job.city] ?? 0) + 1
    }

    matches.push({
      role,
      totalJobs: totalRoleJobs,
      matchingJobs: roleJobs.length,
      matchPct: totalRoleJobs > 0 ? (roleJobs.length / totalRoleJobs) * 100 : 0,
      avgSkillOverlap: avgOverlap * 100,
      topCities: Object.fromEntries(
        Object.entries(cityCounts).sort((a, b) => b[1] - a[1]).slice(0, 5),
      ),
      salary: salaryStats(roleJobs.map((j) => j.job)),
      missingSkills,
    })
  }

  // Sort by match percentage descending
  matches.sort((a, b) => b.matchPct - a.matchPct)

  // Best cities across all matching jobs
  const bestCities: Record<string, number> = {}
  for (const s of scored) {
    if (s.job.city) bestCities[s.job.city] = (bestCities[s.job.city] ?? 0) + 1
  }

  return {
    matches: matches.slice(0, 15),
    bestCities: Object.fromEntries(
      Object.entries(bestCities).sort((a, b) => b[1] - a[1]).slice(0, 15),
    ),
  }
}

// ── Salary Estimator ────────────────────────────────────

export type SalaryEstimate = {
  label: string
  stats: NonNullable<ReturnType<typeof salaryStats>>
  totalFiltered: number
}

export const estimateSalary = (
  jobs: ReadonlyArray<JobSlim>,
  role?: string,
  city?: string,
  seniority?: string,
): SalaryEstimate | null => {
  const filtered = jobs.filter((j) => {
    if (role && j.job_family !== role) return false
    if (city && j.city !== city) return false
    if (seniority && j.seniority_from_title !== seniority) return false
    return true
  })

  const parts: string[] = []
  if (role) parts.push(role)
  if (city) parts.push(`in ${city}`)
  if (seniority) parts.push(`(${seniority})`)
  const label = parts.length > 0 ? parts.join(' ') : 'Overall market'

  const stats = salaryStats(filtered)
  if (!stats) return null

  return { label, stats, totalFiltered: filtered.length }
}


// ── No-German Finder ────────────────────────────────────

/** Parse the languages JSON column and check if German is mentioned. */
const requiresGerman = (langStr: string | null): boolean => {
  if (!langStr) return false
  try {
    const parsed: unknown = JSON.parse(langStr)
    if (!Array.isArray(parsed)) return false
    return parsed.some(
      (entry: unknown) =>
        typeof entry === 'object' &&
        entry !== null &&
        'language' in entry &&
        typeof (entry as Record<string, unknown>).language === 'string' &&
        ((entry as Record<string, string>).language.toLowerCase() === 'german' ||
          (entry as Record<string, string>).language.toLowerCase() === 'deutsch'),
    )
  } catch {
    return false
  }
}

export const findNoGermanJobs = (jobs: ReadonlyArray<JobSlim>) => {
  const noGerman = jobs.filter((j) => !requiresGerman(j.languages))
  const remote = noGerman.filter((j) => j.work_modality === 'Remote')

  return {
    total: noGerman.length,
    pctOfMarket: jobs.length > 0 ? (noGerman.length / jobs.length) * 100 : 0,
    remoteCount: remote.length,
    topRoles: countByField(noGerman, 'job_family'),
    topCities: countByField(noGerman, 'city'),
    topCompanies: countByField(noGerman, 'company_name'),
    modality: countByField(noGerman, 'work_modality'),
    topSkills: countJsonArrayField(noGerman, 'technical_skills', 20),
    salary: salaryStats(noGerman),
    remoteTopRoles: countByField(remote, 'job_family'),
    remoteTopCities: countByField(remote, 'city'),
  }
}

// ── Skill Combos ────────────────────────────────────────

export type SkillCombo = {
  skills: [string, string]
  count: number
  salary: NonNullable<ReturnType<typeof salaryStats>> | null
}

export const findSkillCombos = (
  jobs: ReadonlyArray<JobSlim>,
  topN = 40,
): SkillCombo[] => {
  // Get top skills to limit combinations
  const skillCounts: Record<string, number> = {}
  for (const job of jobs) {
    for (const skill of parseJsonArray(job.technical_skills)) {
      skillCounts[skill] = (skillCounts[skill] ?? 0) + 1
    }
  }
  const topSkills = Object.entries(skillCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([s]) => s)
  const topSet = new Set(topSkills)

  // Count pairwise co-occurrences
  const pairCounts = new Map<string, { count: number; jobs: JobSlim[] }>()

  for (const job of jobs) {
    const skills = parseJsonArray(job.technical_skills).filter((s) => topSet.has(s))
    for (let i = 0; i < skills.length; i++) {
      for (let j = i + 1; j < skills.length; j++) {
        const key = [skills[i], skills[j]].sort().join('|||')
        const entry = pairCounts.get(key) ?? { count: 0, jobs: [] }
        entry.count++
        entry.jobs.push(job)
        pairCounts.set(key, entry)
      }
    }
  }

  // Build results sorted by count
  const combos: SkillCombo[] = []
  for (const [key, { count, jobs: comboJobs }] of pairCounts) {
    const [a, b] = key.split('|||')
    combos.push({
      skills: [a, b],
      count,
      salary: salaryStats(comboJobs),
    })
  }

  combos.sort((a, b) => b.count - a.count)
  return combos.slice(0, 30)
}

// ── Role vs Role ────────────────────────────────────────

export type RoleProfile = {
  count: number
  topSkills: Record<string, number>
  topCities: Record<string, number>
  modality: Record<string, number>
  seniority: Record<string, number>
  education: Record<string, number>
  contract: Record<string, number>
  salary: ReturnType<typeof salaryStats>
}

export const buildRoleProfile = (
  jobs: ReadonlyArray<JobSlim>,
  role: string,
): RoleProfile | null => {
  const filtered = jobs.filter((j) => j.job_family === role)
  if (filtered.length === 0) return null

  return {
    count: filtered.length,
    topSkills: countJsonArrayField(filtered, 'technical_skills', 15),
    topCities: countByField(filtered, 'city'),
    modality: countByField(filtered, 'work_modality'),
    seniority: countByField(filtered, 'seniority_from_title'),
    education: countByField(filtered, 'education_level'),
    contract: countByField(filtered, 'contract_type'),
    salary: salaryStats(filtered),
  }
}

export const compareRoles = (
  jobs: ReadonlyArray<JobSlim>,
  roleA: string,
  roleB: string,
) => {
  const a = buildRoleProfile(jobs, roleA)
  const b = buildRoleProfile(jobs, roleB)
  if (!a || !b) return null

  // Find shared and unique skills
  const skillsA = new Set(Object.keys(a.topSkills))
  const skillsB = new Set(Object.keys(b.topSkills))
  const shared = [...skillsA].filter((s) => skillsB.has(s))
  const onlyA = [...skillsA].filter((s) => !skillsB.has(s))
  const onlyB = [...skillsB].filter((s) => !skillsA.has(s))

  return { a, b, sharedSkills: shared, onlyASkills: onlyA, onlyBSkills: onlyB }
}

// ── Helpers ──────────────────────────────────────────────

/** Get unique skills from all jobs, sorted by frequency. Only skills with minCount occurrences. */
export const getAvailableSkills = (
  jobs: ReadonlyArray<JobSlim>,
  minCount = 50,
): string[] => {
  const counts: Record<string, number> = {}
  for (const job of jobs) {
    for (const skill of parseJsonArray(job.technical_skills)) {
      counts[skill] = (counts[skill] ?? 0) + 1
    }
  }
  return Object.entries(counts)
    .filter(([, c]) => c >= minCount)
    .sort((a, b) => b[1] - a[1])
    .map(([s]) => s)
}

/** Get unique cities sorted by frequency. Only cities with minCount postings. */
export const getAvailableCities = (
  jobs: ReadonlyArray<JobSlim>,
  minCount = 10,
): string[] => {
  const counts: Record<string, number> = {}
  for (const job of jobs) {
    if (job.city) counts[job.city] = (counts[job.city] ?? 0) + 1
  }
  return Object.entries(counts)
    .filter(([, c]) => c >= minCount)
    .sort((a, b) => b[1] - a[1])
    .map(([s]) => s)
}
