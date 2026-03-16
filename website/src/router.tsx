import { createRootRoute, createRoute, createRouter } from '@tanstack/react-router'
import { lazy } from 'react'

import { DashboardLayout } from '@/components/layout/DashboardLayout'
import { NotFoundPage } from '@/pages/NotFoundPage'
import type { DashboardFilters } from '@/types/filters'

const rootRoute = createRootRoute({
  component: DashboardLayout,
  notFoundComponent: NotFoundPage,
  validateSearch: (search: Record<string, unknown>): DashboardFilters => ({
    family: typeof search.family === 'string' ? search.family : undefined,
    city: typeof search.city === 'string' ? search.city : undefined,
    state: typeof search.state === 'string' ? search.state : undefined,
    seniority: typeof search.seniority === 'string' ? search.seniority : undefined,
    modality: typeof search.modality === 'string' ? search.modality : undefined,
    contract: typeof search.contract === 'string' ? search.contract : undefined,
  }),
})

const overviewRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: lazy(() => import('@/pages/OverviewPage').then((m) => ({ default: m.OverviewPage }))),
})

const skillsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/skills',
  component: lazy(() => import('@/pages/SkillsPage').then((m) => ({ default: m.SkillsPage }))),
})

const salaryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/salary',
  component: lazy(() => import('@/pages/SalaryPage').then((m) => ({ default: m.SalaryPage }))),
})

const locationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/location',
  component: lazy(() => import('@/pages/LocationPage').then((m) => ({ default: m.LocationPage }))),
})

const remoteRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/remote',
  component: lazy(() => import('@/pages/RemotePage').then((m) => ({ default: m.RemotePage }))),
})

const seniorityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/seniority',
  component: lazy(() => import('@/pages/SeniorityPage').then((m) => ({ default: m.SeniorityPage }))),
})

const benefitsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/benefits',
  component: lazy(() => import('@/pages/BenefitsPage').then((m) => ({ default: m.BenefitsPage }))),
})

const languagesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/languages',
  component: lazy(() => import('@/pages/LanguagesPage').then((m) => ({ default: m.LanguagesPage }))),
})

const educationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/education',
  component: lazy(() => import('@/pages/EducationPage').then((m) => ({ default: m.EducationPage }))),
})

const companiesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/companies',
  component: lazy(() => import('@/pages/CompaniesPage').then((m) => ({ default: m.CompaniesPage }))),
})

const rolesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/roles',
  component: lazy(() => import('@/pages/RolesPage').then((m) => ({ default: m.RolesPage }))),
})

const personasRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/personas',
  component: lazy(() => import('@/pages/PersonasPage').then((m) => ({ default: m.PersonasPage }))),
})

const explorerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/explore',
  component: lazy(() => import('@/pages/ExplorerPage').then((m) => ({ default: m.ExplorerPage }))),
})

const eggRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/egg',
  component: lazy(() => import('@/pages/ExplorerPage').then((m) => ({ default: m.EggPage }))),
})

const routeTree = rootRoute.addChildren([
  overviewRoute,
  skillsRoute,
  salaryRoute,
  locationRoute,
  remoteRoute,
  seniorityRoute,
  benefitsRoute,
  languagesRoute,
  educationRoute,
  companiesRoute,
  rolesRoute,
  personasRoute,
  explorerRoute,
  eggRoute,
])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  // eslint-disable-next-line @typescript-eslint/consistent-type-definitions
  interface Register {
    router: typeof router
  }
}
