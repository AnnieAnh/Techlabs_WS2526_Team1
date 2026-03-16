import { useRouterState } from '@tanstack/react-router'
import { Menu, Moon, Sun } from 'lucide-react'

import { useTheme } from '@/hooks/useTheme'
import { Button } from '@/components/ui/button'
import { FilterBar } from './FilterBar'
import { FilterChips } from '@/components/shared/FilterChips'

const NO_FILTER_ROUTES = new Set(['/roles', '/personas', '/explore', '/egg'])

type HeaderProps = {
  onMenuClick: () => void
}

export const Header = ({ onMenuClick }: HeaderProps) => {
  const { theme, toggle } = useTheme()
  const path = useRouterState({ select: (s) => s.location.pathname })
  const showFilters = !NO_FILTER_ROUTES.has(path)

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur-sm">
      <div className="flex h-14 items-center justify-between px-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={onMenuClick}
          className="lg:hidden"
          aria-label="Open sidebar"
        >
          <Menu size={20} />
        </Button>
        {showFilters ? (
          <div className="hidden flex-1 lg:block">
            <FilterBar />
          </div>
        ) : (
          <div className="hidden lg:block" />
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggle}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </Button>
      </div>
      {showFilters && (
        <>
          <div className="px-4 pb-2 lg:hidden">
            <FilterBar />
          </div>
          <div className="px-4 pb-2">
            <FilterChips />
          </div>
        </>
      )}
    </header>
  )
}
