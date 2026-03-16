/** Format a number with thousands separator (e.g. 19098 → "19,098"). */
export const formatNumber = (n: number): string =>
  new Intl.NumberFormat('en-US').format(n)

/** Format a number as EUR currency (e.g. 55000 → "€55,000"). */
export const formatSalary = (n: number): string =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 0,
  }).format(n)

/** Format a number as percentage (e.g. 0.42 → "42%", 42 → "42%"). */
export const formatPct = (n: number, decimals = 0): string => {
  const value = n > 1 ? n : n * 100
  return `${value.toFixed(decimals)}%`
}

/** Parse a JSON array string into a string array. */
export const parseJsonList = (value: string | null): string[] => {
  if (!value) return []
  try {
    const parsed: unknown = JSON.parse(value)
    return Array.isArray(parsed) ? (parsed as string[]) : []
  } catch {
    return []
  }
}
