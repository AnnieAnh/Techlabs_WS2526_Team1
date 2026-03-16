import { Link, useRouterState, useSearch } from '@tanstack/react-router'
import {
  BarChart3,
  BrainCircuit,
  Briefcase,
  Building2,
  Compass,
  DollarSign,
  GraduationCap,
  Languages,
  LayoutDashboard,
  MapPin,
  Sparkles,
  Users,
  Wifi,
  X,
} from 'lucide-react'

import { cn } from '@/lib/utils'

type NavItem = {
  label: string
  to: string
  icon: React.ReactNode
}

const NAV_ITEMS: NavItem[] = [
  { label: 'Overview', to: '/', icon: <LayoutDashboard size={18} /> },
  { label: 'Skills', to: '/skills', icon: <BrainCircuit size={18} /> },
  { label: 'Salary', to: '/salary', icon: <DollarSign size={18} /> },
  { label: 'Location', to: '/location', icon: <MapPin size={18} /> },
  { label: 'Remote Work', to: '/remote', icon: <Wifi size={18} /> },
  { label: 'Seniority', to: '/seniority', icon: <BarChart3 size={18} /> },
  { label: 'Benefits', to: '/benefits', icon: <Sparkles size={18} /> },
  { label: 'Languages', to: '/languages', icon: <Languages size={18} /> },
  { label: 'Education', to: '/education', icon: <GraduationCap size={18} /> },
  { label: 'Companies', to: '/companies', icon: <Building2 size={18} /> },
  { label: 'Role Deep Dives', to: '/roles', icon: <Briefcase size={18} /> },
  { label: 'Job Seeker Guide', to: '/personas', icon: <Users size={18} /> },
  { label: 'Advanced Info', to: '/explore', icon: <Compass size={18} /> },
]

type SidebarProps = {
  open: boolean
  onClose: () => void
}

// Routes where global filters don't apply — navigate without search params
const NO_FILTER_ROUTES = new Set(['/roles', '/personas', '/explore', '/egg'])

export const Sidebar = ({ open, onClose }: SidebarProps) => {
  const router = useRouterState()
  const currentPath = router.location.pathname
  const search = useSearch({ strict: false }) as Record<string, unknown>

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={onClose}
          onKeyDown={(e) => e.key === 'Escape' && onClose()}
          role="button"
          tabIndex={0}
          aria-label="Close sidebar"
        />
      )}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-border bg-sidebar',
          'transition-transform duration-200 lg:translate-x-0 lg:static lg:z-auto',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex h-14 items-center justify-between border-b border-border px-4">
          <Link to="/" search={search} className="text-sm font-semibold text-sidebar-foreground no-underline">
            DE IT Jobs
          </Link>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-sidebar-foreground hover:bg-sidebar-accent lg:hidden"
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>
        <nav className="flex-1 overflow-y-auto p-2">
          <ul className="space-y-0.5">
            {NAV_ITEMS.map((item) => (
              <li key={item.to}>
                <Link
                  to={item.to}
                  search={NO_FILTER_ROUTES.has(item.to) ? {} : search}
                  onClick={onClose}
                  className={cn(
                    'flex items-center gap-3 rounded-md px-3 py-2 text-sm no-underline transition-colors',
                    currentPath === item.to
                      ? 'bg-sidebar-accent text-sidebar-accent-foreground font-medium'
                      : 'text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground',
                  )}
                >
                  {item.icon}
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="border-t border-border p-4">
          <p className="text-xs text-muted-foreground">
            ~19K jobs from LinkedIn & Indeed
          </p>
        </div>
      </aside>
    </>
  )
}
