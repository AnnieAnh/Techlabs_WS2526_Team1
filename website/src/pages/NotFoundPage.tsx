import { Link } from '@tanstack/react-router'

export const NotFoundPage = () => (
  <div className="flex flex-col items-center justify-center gap-4 py-20">
    <h1 className="text-6xl font-bold text-muted-foreground">404</h1>
    <p className="text-lg text-muted-foreground">Page not found</p>
    <Link
      to="/"
      className="inline-flex items-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
    >
      Back to Overview
    </Link>
  </div>
)
