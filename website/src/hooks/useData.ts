import { useQuery } from '@tanstack/react-query'

import type {
  BenefitsData,
  CompaniesData,
  EducationData,
  JobSlim,
  LanguagesData,
  LocationData,
  OverviewData,
  PersonasData,
  RemoteData,
  RoleDivesData,
  SalaryData,
  SeniorityData,
  SkillsData,
} from '@/types'

const fetchJson = async <T>(path: string): Promise<T> => {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.statusText}`)
  return res.json() as Promise<T>
}

const STALE = Infinity

export const useOverview = () =>
  useQuery({ queryKey: ['overview'], queryFn: () => fetchJson<OverviewData>('/data/overview.json'), staleTime: STALE })

export const useSkills = () =>
  useQuery({ queryKey: ['skills'], queryFn: () => fetchJson<SkillsData>('/data/skills.json'), staleTime: STALE })

export const useSalary = () =>
  useQuery({ queryKey: ['salary'], queryFn: () => fetchJson<SalaryData>('/data/salary.json'), staleTime: STALE })

export const useLocation = () =>
  useQuery({ queryKey: ['location'], queryFn: () => fetchJson<LocationData>('/data/location.json'), staleTime: STALE })

export const useRemote = () =>
  useQuery({ queryKey: ['remote'], queryFn: () => fetchJson<RemoteData>('/data/remote.json'), staleTime: STALE })

export const useSeniority = () =>
  useQuery({ queryKey: ['seniority'], queryFn: () => fetchJson<SeniorityData>('/data/seniority.json'), staleTime: STALE })

export const useBenefits = () =>
  useQuery({ queryKey: ['benefits'], queryFn: () => fetchJson<BenefitsData>('/data/benefits.json'), staleTime: STALE })

export const useLanguages = () =>
  useQuery({ queryKey: ['languages'], queryFn: () => fetchJson<LanguagesData>('/data/languages.json'), staleTime: STALE })

export const useEducation = () =>
  useQuery({ queryKey: ['education'], queryFn: () => fetchJson<EducationData>('/data/education.json'), staleTime: STALE })

export const useCompanies = () =>
  useQuery({ queryKey: ['companies'], queryFn: () => fetchJson<CompaniesData>('/data/companies.json'), staleTime: STALE })

export const useRoleDives = () =>
  useQuery({ queryKey: ['role-dives'], queryFn: () => fetchJson<RoleDivesData>('/data/role-dives.json'), staleTime: STALE })

export const usePersonas = () =>
  useQuery({ queryKey: ['personas'], queryFn: () => fetchJson<PersonasData>('/data/personas.json'), staleTime: STALE })

export const useJobsSlim = (enabled = true) =>
  useQuery({ queryKey: ['jobs-slim'], queryFn: () => fetchJson<JobSlim[]>('/data/jobs-slim.json'), staleTime: STALE, enabled })
