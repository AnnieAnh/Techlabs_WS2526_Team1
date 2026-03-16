/**
 * Mapping between German state names (used in GeoJSON) and English names (used in pipeline data).
 * Source: src/extraction/config/german_states.yaml
 */
export const GERMAN_TO_ENGLISH: Record<string, string> = {
  'Baden-Württemberg': 'Baden-Württemberg',
  'Bayern': 'Bavaria',
  'Berlin': 'Berlin',
  'Brandenburg': 'Brandenburg',
  'Bremen': 'Bremen',
  'Hamburg': 'Hamburg',
  'Hessen': 'Hesse',
  'Mecklenburg-Vorpommern': 'Mecklenburg-West Pomerania',
  'Niedersachsen': 'Lower Saxony',
  'Nordrhein-Westfalen': 'North Rhine-Westphalia',
  'Rheinland-Pfalz': 'Rhineland-Palatinate',
  'Saarland': 'Saarland',
  'Sachsen': 'Saxony',
  'Sachsen-Anhalt': 'Saxony-Anhalt',
  'Schleswig-Holstein': 'Schleswig-Holstein',
  'Thüringen': 'Thuringia',
}

export const ENGLISH_TO_GERMAN: Record<string, string> = Object.fromEntries(
  Object.entries(GERMAN_TO_ENGLISH).map(([de, en]) => [en, de]),
)

export const toEnglishState = (germanName: string): string =>
  GERMAN_TO_ENGLISH[germanName] ?? germanName

export const toGermanState = (englishName: string): string =>
  ENGLISH_TO_GERMAN[englishName] ?? englishName
