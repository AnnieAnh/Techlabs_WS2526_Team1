/** Dashboard chart color palette — viridian green theme. */

export const CHART_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
] as const

/** Recharts needs hex/hsl directly — viridian green ramp (light → dark). */
export const CHART_HEX = [
  '#5bb89a', // chart-1  bright viridian
  '#3d9b7f', // chart-2  mid viridian
  '#2d7d66', // chart-3  deep viridian
  '#1f5f4d', // chart-4  dark viridian
  '#134236', // chart-5  darkest viridian
] as const

/** Extended palette: viridian greens + complementary warm accents. */
export const EXTENDED_PALETTE = [
  '#5bb89a', // viridian 1
  '#3d9b7f', // viridian 2
  '#2d7d66', // viridian 3
  '#1f5f4d', // viridian 4
  '#134236', // viridian 5
  '#e07b54', // terracotta
  '#d4a843', // amber
  '#8b5fbf', // muted purple
  '#4ba3c3', // teal blue
  '#c75c8a', // dusty rose
  '#6b9e4f', // sage green
  '#d68c4f', // copper
  '#5c8dbd', // steel blue
  '#b3694e', // sienna
  '#7fb08a', // mint
] as const

/** Map choropleth color scale (light → dark viridian green). */
export const MAP_SCALE = [
  '#e8f5ef',
  '#c8e6d5',
  '#a3d4b8',
  '#7cc19e',
  '#5bb89a',
  '#3d9b7f',
  '#2d7d66',
  '#1f5f4d',
  '#134236',
] as const

export const getColor = (index: number): string =>
  EXTENDED_PALETTE[index % EXTENDED_PALETTE.length]
