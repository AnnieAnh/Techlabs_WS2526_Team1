export type OverviewData = {
  total_jobs: number
  total_companies: number
  date_range: { min: string; max: string }
  salary_known_count: number
  salary_known_pct: number
  remote_pct: number
  hybrid_pct: number
  modality_known_pct: number
  source_split: Record<string, number>
  families: Record<string, number>
  seniority: Record<string, number>
  top_cities: Record<string, number>
  trends_monthly: Record<string, number>
}

export type SkillsData = {
  top_25: Record<string, number>
  by_family: Record<string, Record<string, number>>
  cooccurrence: Record<string, Record<string, number>>
  required_vs_optional: Record<string, { required: number; nice_to_have: number; ratio: number }>
  salary_premium: Record<string, { median: number; p25: number; p75: number; count: number }>
}

export type SalaryGroupStats = {
  median: number
  p25: number
  p75: number
  count: number
}

export type SalaryData = {
  n_with_salary: number
  n_total: number
  by_family: Record<string, SalaryGroupStats>
  by_seniority: Record<string, SalaryGroupStats>
  by_city: Record<string, SalaryGroupStats>
}

export type LocationData = {
  by_state: Record<string, number>
  by_city: Record<string, number>
  modality_by_state: Record<string, Record<string, number>>
}

export type RemoteData = {
  overall: Record<string, number>
  by_family: Record<string, Record<string, number>>
  by_seniority: Record<string, Record<string, number>>
}

export type SeniorityData = {
  distribution: Record<string, number>
  unspecified_count: number
  by_family: Record<string, Record<string, number>>
}

export type BenefitsData = {
  category_counts: Record<string, number>
  by_family: Record<string, Record<string, number>>
}

export type LanguagesData = {
  german_pct_by_family: Record<string, { 'German %': number }>
  english_pct_by_family: Record<string, { 'English %': number }>
  cefr_distribution: Record<string, number>
  german_mention_count: number
  english_mention_count: number
  total_rows: number
}

export type EducationData = {
  distribution: Record<string, number>
  unspecified_count: number
  by_family: Record<string, Record<string, number>>
  soft_skill_by_family: Record<string, Record<string, number>>
}

export type CompaniesData = {
  top_20: Record<string, number>
  family_diversity: Record<string, number>
}

export type RoleSalary = {
  median: number
  p25: number
  p75: number
  count: number
}

export type RoleDiveEntry = {
  count: number
  top_skills: Record<string, number>
  seniority: Record<string, number>
  modality: Record<string, number>
  top_cities: Record<string, number>
  skill_progression: Record<string, Record<string, number>>
  salary: RoleSalary | null
}

export type RoleDivesData = Record<string, RoleDiveEntry>

export type PersonaEntry = {
  label: string
  count: number
  pct_of_market: number
  top_family: string | null
  remote_pct: number
  top_skills: Record<string, number>
  top_cities: Record<string, number>
  salary: RoleSalary | null
}

export type PersonasData = {
  junior_entry: PersonaEntry
  senior_specialist: PersonaEntry
  remote_first: PersonaEntry
  career_changer: PersonaEntry
}

export type JobSlim = {
  job_family: string | null
  city: string | null
  state: string | null
  seniority_from_title: string | null
  contract_type: string | null
  work_modality: string | null
  salary_min: number | null
  salary_max: number | null
  education_level: string | null
  site: string | null
  company_name: string | null
  date_posted: string | null
  technical_skills: string | null
  nice_to_have_skills: string | null
  benefit_categories: string | null
  soft_skill_categories: string | null
  languages: string | null
}
