import { SearchX } from 'lucide-react'

type EmptyStateProps = {
  message?: string
  onClear?: () => void
}

export const EmptyState = ({
  message = 'No data matches your current filters.',
  onClear,
}: EmptyStateProps) => (
  <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
    <SearchX size={40} className="text-muted-foreground/50" />
    <p className="text-sm text-muted-foreground">{message}</p>
    {onClear && (
      <button
        onClick={onClear}
        className="rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-accent"
      >
        Clear all filters
      </button>
    )}
  </div>
)
