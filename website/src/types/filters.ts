export type DashboardFilters = {
  readonly family?: string
  readonly city?: string
  readonly state?: string
  readonly seniority?: string
  readonly modality?: string
  readonly contract?: string
}

/** Count active filters (each non-empty key counts as 1, even if multi-value). */
export const activeFilterCount = (filters: DashboardFilters): number =>
  Object.values(filters).filter(Boolean).length

/** Parse a comma-separated filter value into an array. */
export const parseFilterValue = (value: string | undefined): string[] => {
  if (!value) return []
  return value.split(',').map((v) => v.trim()).filter(Boolean)
}

/** Serialize an array of values into a comma-separated string, or undefined if empty. */
export const serializeFilterValue = (values: string[]): string | undefined =>
  values.length > 0 ? values.join(',') : undefined
