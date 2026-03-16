# German IT Jobs Dashboard — Frontend

Interactive web dashboard for exploring **~19,000 German IT job postings** from LinkedIn and Indeed. Built with React 19, TypeScript, and Tailwind CSS — fully client-side with zero backend.

**Live:** [it-in-de.iebo-testt.workers.dev](https://it-in-de.iebo-testt.workers.dev/)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT-SIDE DASHBOARD                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │  Static JSON  │    │  TanStack      │    │  React Pages         │  │
│  │  (pre-agg)    │───>│  Query Cache   │───>│  (14 routes)         │  │
│  │  /public/data │    │  staleTime: ∞  │    │  Lazy-loaded         │  │
│  └──────────────┘    └───────────────┘    └──────────────────────┘  │
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │  jobs-slim    │    │  Client-side   │    │  Filtered Views      │  │
│  │  .json (12MB) │───>│  Aggregation   │───>│  (real-time compute) │  │
│  │  ~19K rows    │    │  Engine        │    │                      │  │
│  └──────────────┘    └───────────────┘    └──────────────────────┘  │
│                                                                     │
│  Hybrid Data Pattern:                                               │
│  • No filters → use pre-aggregated static JSON (instant)            │
│  • With filters → load jobs-slim.json, compute on client (dynamic)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pages & Routes

The dashboard has **14 pages** across two categories: analytics views (pre-aggregated) and interactive exploration (client-computed).

### Analytics Pages (10)

These pages display pre-computed statistics and support 6 filter dimensions (role, city, state, seniority, modality, contract type). When filters are active, they switch to real-time client-side aggregation.

| Route | Page | What It Shows |
|-------|------|---------------|
| `/` | **Overview** | Market snapshot — total jobs, companies, remote %, salary coverage, top families, posting trends |
| `/skills` | **Skills** | Top 25 skills, required vs nice-to-have, salary premium per skill, skill×family heatmap, co-occurrence matrix |
| `/salary` | **Salary** | Salary ranges (P25/median/P75) by role, seniority, and city with range charts |
| `/location` | **Location** | Interactive Germany choropleth map, top cities, work modality by state |
| `/remote` | **Remote Work** | Remote/hybrid/on-site breakdown by role and seniority level |
| `/seniority` | **Seniority** | Seniority distribution, by-family heatmap, unspecified counter |
| `/benefits` | **Benefits** | Benefit categories (11), by-family heatmap, top benefit types |
| `/languages` | **Languages** | German/English % by role, CEFR level distribution |
| `/education` | **Education** | Education levels by family/seniority, soft skills heatmap |
| `/companies` | **Companies** | Top 20 hiring companies, job family diversity per company |

### Deep Dive Pages (2)

No sidebar filters — self-contained with their own selectors.

| Route | Page | What It Shows |
|-------|------|---------------|
| `/roles` | **Role Deep Dives** | 6 target roles with skills, salary progression, seniority, modality, top cities |
| `/personas` | **Personas** | 4 job seeker profiles (Junior, Senior, Remote-First, Career Changer) with tailored recommendations |

### Advanced Information (2)

Interactive exploration powered by real-time computation on the full dataset.

| Route | Page | What It Shows |
|-------|------|---------------|
| `/explore` | **Advanced Info** | 7 interactive tabs (see below), gated behind a €3 paywall overlay |
| `/egg` | **Easter Egg** | Same as Advanced Info but bypasses the paywall |

---

## Advanced Information — 7 Tabs

The Advanced Information page (`/explore`) provides interactive tools for deep market analysis. All computations run client-side on ~19K job records.

| Tab | Description |
|-----|-------------|
| **By Role** | Select a job family → top skills, salary, cities, companies, seniority, modality, education, contract type |
| **By City** | Select a city → local market profile: dominant roles, skills, companies, remote-friendliness |
| **By Skills** | Multi-select your skills → personalized role recommendations with match %, skill overlap, and skills to learn |
| **Salary Estimator** | Filter by role + city + seniority → salary percentiles (P25 / Median / P75) from real posting data |
| **No-German Finder** | Jobs that don't require German — roles, cities, companies, remote options for international candidates |
| **Skill Combos** | Most frequent skill pairs across all postings, sortable by popularity or highest salary |
| **Role vs Role** | Side-by-side comparison of two job families — shared/unique skills, salary, cities, modality, seniority |

---

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| **Framework** | React | 19.2 |
| **Language** | TypeScript (strict) | 5.9 |
| **Build** | Vite | 6.4 |
| **Routing** | TanStack Router | 1.167 |
| **Data Fetching** | TanStack React Query | 5.90 |
| **Styling** | Tailwind CSS | 4.2 |
| **Components** | shadcn/ui | latest |
| **Charts** | Recharts | 3.8 |
| **Maps** | react-simple-maps | 1.0 |
| **Icons** | lucide-react | 0.577 |
| **Font** | Geist Variable | — |
| **Deployment** | Cloudflare Workers | — |

---

## Project Structure

```
website/
├── public/
│   ├── data/                          ← Pre-aggregated JSON files
│   │   ├── overview.json              ← Market overview stats
│   │   ├── skills.json                ← Skill rankings + heatmaps
│   │   ├── salary.json                ← Salary ranges by group
│   │   ├── location.json              ← Geographic distribution
│   │   ├── remote.json                ← Work modality stats
│   │   ├── seniority.json             ← Seniority distribution
│   │   ├── benefits.json              ← Benefit categories
│   │   ├── languages.json             ← Language requirements
│   │   ├── education.json             ← Education levels
│   │   ├── companies.json             ← Top companies
│   │   ├── role-dives.json            ← 6 role deep dives
│   │   ├── personas.json              ← 4 job seeker personas
│   │   ├── jobs-slim.json             ← ~19K jobs (on-demand, 12MB)
│   │   └── germany.topo.json          ← TopoJSON for choropleth map
│   ├── robots.txt                     ← Crawler rules
│   ├── sitemap.xml                    ← 13 public pages
│   └── llms.txt                       ← AI scraper description
│
├── src/
│   ├── App.tsx                        ← QueryClient + Router + Suspense
│   ├── router.tsx                     ← 14 routes, lazy-loaded pages
│   ├── index.css                      ← Tailwind config + viridian theme
│   │
│   ├── pages/                         ← Page components (1 per route)
│   │   ├── OverviewPage.tsx
│   │   ├── SkillsPage.tsx
│   │   ├── SalaryPage.tsx
│   │   ├── LocationPage.tsx
│   │   ├── RemotePage.tsx
│   │   ├── SeniorityPage.tsx
│   │   ├── BenefitsPage.tsx
│   │   ├── LanguagesPage.tsx
│   │   ├── EducationPage.tsx
│   │   ├── CompaniesPage.tsx
│   │   ├── RolesPage.tsx
│   │   ├── PersonasPage.tsx
│   │   └── ExplorerPage.tsx           ← Advanced Info (7 tabs + paywall)
│   │
│   ├── components/
│   │   ├── layout/                    ← App shell
│   │   │   ├── DashboardLayout.tsx    ← Sidebar + Header + content
│   │   │   ├── Sidebar.tsx            ← Navigation (13 items)
│   │   │   ├── Header.tsx             ← Filter bar + theme toggle
│   │   │   └── FilterBar.tsx          ← 6-dimension filter dropdowns
│   │   ├── charts/                    ← Visualization components
│   │   │   ├── BarChartCard.tsx       ← Recharts bar chart wrapper
│   │   │   ├── PieChartCard.tsx       ← Recharts pie chart wrapper
│   │   │   ├── LineChartCard.tsx      ← Recharts line chart wrapper
│   │   │   ├── HeatmapChart.tsx       ← Custom div-grid heatmap
│   │   │   ├── KpiCard.tsx            ← Metric cards (icon + value)
│   │   │   ├── DataCoverageAlert.tsx  ← Low-data warnings
│   │   │   └── PageSkeleton.tsx       ← Loading placeholder
│   │   ├── shared/                    ← Reusable UI
│   │   │   ├── EmptyState.tsx         ← "No results" + clear filters
│   │   │   ├── FilterChips.tsx        ← Active filter pills
│   │   │   └── MultiSelect.tsx        ← Multi-select dropdown
│   │   └── ui/                        ← shadcn/ui base components
│   │
│   ├── hooks/                         ← Custom React hooks
│   │   ├── useData.ts                 ← TanStack Query hooks (13 endpoints)
│   │   ├── useFilters.ts              ← URL search param sync
│   │   ├── usePageData.ts             ← Hybrid static/computed data
│   │   ├── useFilteredJobs.ts         ← Client-side job filtering
│   │   └── useTheme.ts               ← Dark/light mode toggle
│   │
│   ├── lib/                           ← Pure utility functions
│   │   ├── aggregations.ts            ← Count, salary stats, cross-tab
│   │   ├── explore.ts                 ← Advanced analysis engine
│   │   ├── colors.ts                  ← Viridian green palettes
│   │   ├── format.ts                  ← Number/salary/pct formatters
│   │   ├── stateNames.ts             ← German state name mapping
│   │   └── utils.ts                   ← cn() classname merger
│   │
│   └── types/                         ← TypeScript type definitions
│       ├── data.ts                    ← All data shape interfaces
│       ├── enums.ts                   ← JobFamily (44), Seniority, etc.
│       └── filters.ts                 ← Filter types + helpers
│
├── index.html                         ← SEO meta, OG tags, JSON-LD
├── wrangler.jsonc                     ← Cloudflare Workers config
├── vite.config.ts                     ← Vite + Tailwind plugin
├── tsconfig.json                      ← TypeScript strict config
└── package.json                       ← Dependencies + scripts
```

---

## Data Flow

```
Pipeline Output                    Dashboard Input
─────────────────                  ────────────────

cleaned_jobs.csv  ──┐
  (~19K rows,       │   generate_dashboard_data.py     public/data/
   29 columns)      ├──────────────────────────────>  ├── overview.json
                    │   (Python script aggregates      ├── skills.json
                    │    and exports static JSON)       ├── salary.json
                    │                                   ├── ...
                    │                                   └── jobs-slim.json
                    └──────────────────────────────>       (18 columns,
                        (slim export: drop description,    ~19K rows)
                         keep filterable fields)
```

The Python script `scripts/generate_dashboard_data.py` bridges the pipeline and the dashboard:
1. Reads `cleaned_jobs.csv` from the pipeline output
2. Pre-aggregates statistics for each analytics page (overview, skills, salary, etc.)
3. Exports a slim version of all jobs (`jobs-slim.json`) for client-side filtering
4. Writes all JSON files to `website/public/data/`

---

## Filter System

Six filter dimensions available across all analytics pages:

| Filter | Values | URL Param |
|--------|--------|-----------|
| **Job Family** | 44 IT roles (Data Scientist, Backend Developer, ...) | `?family=Data+Scientist` |
| **City** | German cities (Berlin, München, Hamburg, ...) | `?city=Berlin` |
| **State** | 16 federal states | `?state=Bavaria` |
| **Seniority** | Junior, Mid, Senior, Lead, C-Level, Director | `?seniority=Senior` |
| **Modality** | Remote, Hybrid, On-Site | `?modality=Remote` |
| **Contract** | Full-time, Part-time, Contract, Freelance, Internship | `?contract=Full-time` |

Filters are encoded in the URL for shareability. Multi-value filters use comma separation. Navigation between pages preserves active filters.

---

## Design System

### Color Palette

- **Primary:** Viridian green ramp (`#5bb89a` → `#134236`)
- **Accent:** Terracotta `#e07b54` for contrast elements (Role B in comparisons)
- **Extended:** 15 colors mixing greens with warm accents (amber, purple, teal, rose, sage)
- **Map:** 9-step light→dark viridian gradient for choropleth

### Theme

- Dark and light mode with localStorage persistence
- Toggle in header
- CSS custom properties for all semantic colors

### Typography

- Geist Variable (system-optimized sans-serif)
- Monospace font for code/data display

---

## SEO & Discoverability

- **Meta tags:** Title, description, keywords, canonical URL
- **Open Graph + Twitter Cards:** Social sharing previews
- **JSON-LD:** WebApplication + Dataset structured data schemas
- **sitemap.xml:** All 13 public pages with priority weights
- **robots.txt:** All crawlers allowed, `/egg` blocked
- **llms.txt:** Plain-text site description for AI scraper conventions
- **AI meta tags:** `ai-content-declaration`, `ai-site-description`

---

## Commands

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Production build (TypeScript check + Vite build)
npm run build

# Preview production build
npm run preview

# Lint
npm run lint

# Generate dashboard data (from repo root)
python scripts/generate_dashboard_data.py
```

---

## Deployment

Deployed to **Cloudflare Workers** via `wrangler`:

```bash
# Deploy to production
npx wrangler deploy

# Preview deployment
npx wrangler dev
```

Configuration in `wrangler.jsonc`.
