import type { JobSlim } from '@/types'

/** Safely parse a JSON array string, returning empty array on failure. */
export const parseJsonArray = (value: string | null | undefined): string[] => {
  if (!value) return []
  try {
    const parsed: unknown = JSON.parse(value)
    return Array.isArray(parsed) ? (parsed as string[]) : []
  } catch {
    return []
  }
}

/** Count occurrences of each value in a field. Returns sorted desc. */
export const countByField = (
  jobs: ReadonlyArray<JobSlim>,
  field: keyof JobSlim,
): Record<string, number> => {
  const counts: Record<string, number> = {}
  for (const job of jobs) {
    const val = job[field]
    if (val != null && val !== '') {
      const key = String(val)
      counts[key] = (counts[key] ?? 0) + 1
    }
  }
  return Object.fromEntries(Object.entries(counts).sort((a, b) => b[1] - a[1]))
}

/** Count items from a JSON array column (e.g. technical_skills). Top N. */
export const countJsonArrayField = (
  jobs: ReadonlyArray<JobSlim>,
  field: keyof JobSlim,
  topN = 25,
): Record<string, number> => {
  const counts: Record<string, number> = {}
  for (const job of jobs) {
    for (const item of parseJsonArray(job[field] as string | null)) {
      counts[item] = (counts[item] ?? 0) + 1
    }
  }
  return Object.fromEntries(
    Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN),
  )
}

/** Compute salary stats for jobs that have salary data. */
export const salaryStats = (
  jobs: ReadonlyArray<JobSlim>,
): { count: number; median: number; p25: number; p75: number } | null => {
  const mids = jobs
    .filter((j) => j.salary_min != null && j.salary_max != null)
    .map((j) => ((j.salary_min ?? 0) + (j.salary_max ?? 0)) / 2)
    .sort((a, b) => a - b)

  if (mids.length === 0) return null

  const q = (arr: number[], p: number) => {
    const idx = (arr.length - 1) * p
    const lo = Math.floor(idx)
    const hi = Math.ceil(idx)
    return lo === hi ? arr[lo] : arr[lo] * (hi - idx) + arr[hi] * (idx - lo)
  }

  return {
    count: mids.length,
    median: Math.round(q(mids, 0.5)),
    p25: Math.round(q(mids, 0.25)),
    p75: Math.round(q(mids, 0.75)),
  }
}

/** Group salary stats by a field. */
export const salaryByGroup = (
  jobs: ReadonlyArray<JobSlim>,
  field: keyof JobSlim,
): Record<string, { count: number; median: number; p25: number; p75: number }> => {
  const groups: Record<string, JobSlim[]> = {}
  for (const job of jobs) {
    const key = job[field] as string | null
    if (key) {
      ;(groups[key] ??= []).push(job)
    }
  }
  const result: Record<string, { count: number; median: number; p25: number; p75: number }> = {}
  for (const [key, group] of Object.entries(groups)) {
    const stats = salaryStats(group)
    if (stats && stats.count >= 3) result[key] = stats
  }
  return result
}

/** Cross-tabulate: count occurrences of colField values per rowField group. */
export const crossTab = (
  jobs: ReadonlyArray<JobSlim>,
  rowField: keyof JobSlim,
  colField: keyof JobSlim,
): Record<string, Record<string, number>> => {
  const result: Record<string, Record<string, number>> = {}
  for (const job of jobs) {
    const row = job[rowField] as string | null
    const col = job[colField] as string | null
    if (row && col) {
      ;(result[row] ??= {})[col] = ((result[row] ??= {})[col] ?? 0) + 1
    }
  }
  return result
}

/** Compute overview-like aggregation from filtered jobs. */
export const aggregateOverview = (jobs: ReadonlyArray<JobSlim>) => {
  const families = countByField(jobs, 'job_family')
  const seniority = countByField(jobs, 'seniority_from_title')
  const cities = countByField(jobs, 'city')
  const topCities = Object.fromEntries(Object.entries(cities).slice(0, 15))
  const sources = countByField(jobs, 'site')
  const modality = countByField(jobs, 'work_modality')
  const remoteCount = modality['Remote'] ?? 0
  const totalWithModality = Object.values(modality).reduce((s, v) => s + v, 0)
  const salaryKnown = jobs.filter((j) => j.salary_min != null).length

  return {
    total_jobs: jobs.length,
    total_companies: new Set(jobs.map((j) => j.company_name).filter(Boolean)).size,
    remote_pct: totalWithModality > 0 ? (remoteCount / totalWithModality) * 100 : 0,
    salary_known_pct: jobs.length > 0 ? (salaryKnown / jobs.length) * 100 : 0,
    families,
    seniority,
    top_cities: topCities,
    source_split: sources,
  }
}
