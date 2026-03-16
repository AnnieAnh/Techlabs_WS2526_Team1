"""Generate pre-aggregated JSON data files for the dashboard website.

Reads the full cleaned CSV produced by the pipeline and outputs static JSON
files into ``website/public/data/``. Reuses all existing analytical functions
from ``src/analysis/compute.py`` and ``src/analysis/filters.py``.

Usage:
    PYTHONPATH=src python scripts/generate_dashboard_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.compute import (
    benefit_category_by_family,
    company_job_family_diversity,
    job_density_by_state,
    language_requirement_pct,
    modality_by_state,
    postings_per_month,
    required_vs_optional_skills,
    salary_by_group,
    skill_by_job_family,
    skill_cooccurrence,
    skill_counts,
    skill_progression,
    skill_salary_premium,
    soft_skill_category_by_family,
    top_companies_by_postings,
)
from analysis.filters import (
    exclude_future_dates,
    exclude_other_family,
    filter_by_job_family,
    salary_df,
)
from shared.io import read_csv_safe
from shared.json_utils import parse_json_list

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "website" / "public" / "data"
CSV_PATH = REPO_ROOT / "data" / "cleaning" / "cleaned_jobs.csv"

# Columns kept in the slim jobs JSON for client-side filtering
SLIM_COLS = [
    "job_family", "city", "state", "seniority_from_title", "contract_type",
    "work_modality", "salary_min", "salary_max", "education_level", "site",
    "company_name", "date_posted", "technical_skills", "nice_to_have_skills",
    "benefit_categories", "soft_skill_categories", "languages",
]

TARGET_ROLES = [
    "Frontend Developer", "Backend Developer", "Fullstack Developer",
    "Data Scientist", "Data Engineer", "UI/UX Designer",
]


def _serializable(obj: object) -> object:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 2) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)} — {obj!r}")


def _write_json(data: object, name: str) -> None:
    """Write a JSON file to the output directory."""
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=_serializable, ensure_ascii=False)
    size_kb = path.stat().st_size / 1024
    log.info("  ✓ %s (%.1f KB)", name, size_kb)


def _series_to_dict(s: pd.Series) -> dict:
    """Convert a pandas Series to a plain dict with serializable keys."""
    return {str(k): v for k, v in s.items()}


def _df_to_dict(df: pd.DataFrame) -> dict:
    """Convert a DataFrame to nested dict {index: {col: val}}."""
    return {
        str(idx): {str(c): row[c] for c in df.columns}
        for idx, row in df.iterrows()
    }


def generate_overview(df: pd.DataFrame) -> None:
    """Generate overview.json — high-level market statistics."""
    sal_known = df[df["salary_min"].notna() & df["salary_max"].notna()]
    modality_counts = df["work_modality"].value_counts()
    modality_total = modality_counts.sum()

    monthly = postings_per_month(df)

    data = {
        "total_jobs": len(df),
        "total_companies": int(df["company_name"].nunique()),
        "date_range": {
            "min": str(df["date_posted"].min()),
            "max": str(df[df["date_posted"] != "2027-01-01"]["date_posted"].max()),
        },
        "salary_known_count": len(sal_known),
        "salary_known_pct": round(len(sal_known) / len(df) * 100, 1),
        "remote_pct": round(
            modality_counts.get("Remote", 0) / max(1, modality_total) * 100, 1
        ),
        "hybrid_pct": round(
            modality_counts.get("Hybrid", 0) / max(1, modality_total) * 100, 1
        ),
        "modality_known_pct": round(modality_total / len(df) * 100, 1),
        "source_split": _series_to_dict(df["site"].value_counts()),
        "families": _series_to_dict(df["job_family"].value_counts()),
        "seniority": _series_to_dict(df["seniority_from_title"].value_counts()),
        "top_cities": _series_to_dict(df["city"].value_counts().head(30)),
        "trends_monthly": {str(k): int(v) for k, v in monthly.items()},
    }
    _write_json(data, "overview.json")


def generate_skills(df: pd.DataFrame, df_sal: pd.DataFrame) -> None:
    """Generate skills.json — skill demand analytics."""
    df_no_other = exclude_other_family(df)

    data = {
        "top_25": _series_to_dict(skill_counts(df, top_n=25)),
        "by_family": _df_to_dict(
            skill_by_job_family(df_no_other, top_n_skills=15, top_n_families=12)
        ),
        "cooccurrence": _df_to_dict(skill_cooccurrence(df, top_n=15)),
        "required_vs_optional": _df_to_dict(
            required_vs_optional_skills(df, top_n=20)
        ),
        "salary_premium": _df_to_dict(
            skill_salary_premium(df_sal, top_n=20)
        ),
    }
    _write_json(data, "skills.json")


def generate_salary(df_sal: pd.DataFrame, total_rows: int) -> None:
    """Generate salary.json — salary distributions."""
    data = {
        "n_with_salary": len(df_sal),
        "n_total": total_rows,
        "by_family": _df_to_dict(salary_by_group(df_sal, "job_family", min_count=5)),
        "by_seniority": _df_to_dict(
            salary_by_group(df_sal, "seniority_from_title", min_count=5)
        ),
        "by_city": _df_to_dict(salary_by_group(df_sal, "city", min_count=5)),
    }
    _write_json(data, "salary.json")


def generate_location(df: pd.DataFrame) -> None:
    """Generate location.json — geographic distribution."""
    data = {
        "by_state": _series_to_dict(job_density_by_state(df)),
        "by_city": _series_to_dict(df["city"].value_counts().head(30)),
        "modality_by_state": _df_to_dict(modality_by_state(df, top_n_states=16)),
    }
    _write_json(data, "location.json")


def generate_remote(df: pd.DataFrame) -> None:
    """Generate remote.json — work modality analysis."""
    df_no_other = exclude_other_family(df)
    modality_known = df_no_other[df_no_other["work_modality"].notna()]

    by_family = (
        pd.crosstab(
            modality_known["job_family"],
            modality_known["work_modality"],
        )
    )
    by_seniority = pd.crosstab(
        modality_known["seniority_from_title"].fillna("Unspecified"),
        modality_known["work_modality"],
    )

    data = {
        "overall": _series_to_dict(df["work_modality"].value_counts()),
        "by_family": _df_to_dict(by_family),
        "by_seniority": _df_to_dict(by_seniority),
    }
    _write_json(data, "remote.json")


def generate_seniority(df: pd.DataFrame) -> None:
    """Generate seniority.json — seniority level distributions."""
    df_no_other = exclude_other_family(df)
    by_family = pd.crosstab(
        df_no_other["job_family"],
        df_no_other["seniority_from_title"].fillna("Unspecified"),
    )

    data = {
        "distribution": _series_to_dict(
            df["seniority_from_title"].value_counts()
        ),
        "unspecified_count": int(df["seniority_from_title"].isna().sum()),
        "by_family": _df_to_dict(by_family),
    }
    _write_json(data, "seniority.json")


def generate_benefits(df: pd.DataFrame) -> None:
    """Generate benefits.json — benefit category analysis."""
    df_no_other = exclude_other_family(df)

    data = {
        "category_counts": _series_to_dict(
            skill_counts(df, col="benefit_categories", top_n=15)
        ),
        "by_family": _df_to_dict(
            benefit_category_by_family(df_no_other, top_n_families=12)
        ),
    }
    _write_json(data, "benefits.json")


def generate_languages(df: pd.DataFrame) -> None:
    """Generate languages.json — language requirement analysis."""
    df_no_other = exclude_other_family(df)
    top_families = df_no_other["job_family"].value_counts().head(12).index.tolist()
    role_dfs = {f: filter_by_job_family(df_no_other, f) for f in top_families}

    lang_pct = language_requirement_pct(role_dfs)

    # Parse CEFR levels from the languages column
    cefr_counts: dict[str, int] = {}
    german_mention_count = 0
    english_mention_count = 0
    for langs_str in df["languages"]:
        langs = parse_json_list(langs_str)
        for entry in langs:
            if isinstance(entry, dict):
                lang_name = entry.get("language", "").lower()
                level = entry.get("level", "")
                if lang_name == "german":
                    german_mention_count += 1
                    if level:
                        cefr_counts[level] = cefr_counts.get(level, 0) + 1
                elif lang_name == "english":
                    english_mention_count += 1
            elif isinstance(entry, str):
                if "german" in entry.lower():
                    german_mention_count += 1
                elif "english" in entry.lower():
                    english_mention_count += 1

    data = {
        "german_pct_by_family": _df_to_dict(lang_pct[["German %"]]),
        "english_pct_by_family": _df_to_dict(lang_pct[["English %"]]),
        "cefr_distribution": cefr_counts,
        "german_mention_count": german_mention_count,
        "english_mention_count": english_mention_count,
        "total_rows": len(df),
    }
    _write_json(data, "languages.json")


def generate_education(df: pd.DataFrame) -> None:
    """Generate education.json — education and soft skill analysis."""
    df_no_other = exclude_other_family(df)

    edu_by_family = pd.crosstab(
        df_no_other["job_family"],
        df_no_other["education_level"].fillna("Unspecified"),
    )

    data = {
        "distribution": _series_to_dict(df["education_level"].value_counts()),
        "unspecified_count": int(df["education_level"].isna().sum()),
        "by_family": _df_to_dict(edu_by_family),
        "soft_skill_by_family": _df_to_dict(
            soft_skill_category_by_family(df_no_other, top_n_families=10)
        ),
    }
    _write_json(data, "education.json")


def generate_companies(df: pd.DataFrame) -> None:
    """Generate companies.json — top employers analysis."""
    data = {
        "top_20": _series_to_dict(top_companies_by_postings(df, top_n=20)),
        "family_diversity": _series_to_dict(
            company_job_family_diversity(df, min_postings=5, top_n=15)
        ),
    }
    _write_json(data, "companies.json")


def generate_role_dives(df: pd.DataFrame, df_sal: pd.DataFrame) -> None:
    """Generate role-dives.json — per-role deep dives for 6 target roles."""
    roles: dict[str, dict] = {}
    for role in TARGET_ROLES:
        role_df = filter_by_job_family(df, role)
        if role_df.empty:
            continue

        role_sal = filter_by_job_family(df_sal, role) if not df_sal.empty else pd.DataFrame()

        entry: dict = {
            "count": len(role_df),
            "top_skills": _series_to_dict(skill_counts(role_df, top_n=15)),
            "seniority": _series_to_dict(
                role_df["seniority_from_title"].value_counts()
            ),
            "modality": _series_to_dict(
                role_df["work_modality"].value_counts()
            ),
            "top_cities": _series_to_dict(
                role_df["city"].value_counts().head(10)
            ),
        }

        # Skill progression across seniority levels
        try:
            prog = skill_progression(df, role, top_n=10)
            entry["skill_progression"] = _df_to_dict(prog)
        except Exception:
            entry["skill_progression"] = {}

        # Salary stats for this role
        if not role_sal.empty and len(role_sal) >= 3:
            entry["salary"] = {
                "median": int(role_sal["salary_mid"].median()),
                "p25": int(role_sal["salary_mid"].quantile(0.25)),
                "p75": int(role_sal["salary_mid"].quantile(0.75)),
                "count": len(role_sal),
            }
        else:
            entry["salary"] = None

        roles[role] = entry

    _write_json(roles, "role-dives.json")


def generate_personas(df: pd.DataFrame, df_sal: pd.DataFrame) -> None:
    """Generate personas.json — 4 job seeker persona profiles."""

    def _persona_stats(
        subset: pd.DataFrame, label: str
    ) -> dict:
        sal_sub = subset[
            subset["salary_min"].notna() & subset["salary_max"].notna()
        ].copy()
        if not sal_sub.empty:
            sal_sub["salary_mid"] = (
                pd.to_numeric(sal_sub["salary_min"])
                + pd.to_numeric(sal_sub["salary_max"])
            ) // 2
            salary = {
                "median": int(sal_sub["salary_mid"].median()),
                "p25": int(sal_sub["salary_mid"].quantile(0.25)),
                "p75": int(sal_sub["salary_mid"].quantile(0.75)),
                "count": len(sal_sub),
            }
        else:
            salary = None

        modality = subset["work_modality"].value_counts()
        remote_n = modality.get("Remote", 0)
        modality_total = modality.sum()

        return {
            "label": label,
            "count": len(subset),
            "pct_of_market": round(len(subset) / len(df) * 100, 1),
            "top_family": (
                subset["job_family"].value_counts().index[0]
                if not subset["job_family"].value_counts().empty
                else None
            ),
            "remote_pct": round(
                remote_n / max(1, modality_total) * 100, 1
            ),
            "top_skills": _series_to_dict(
                skill_counts(subset, top_n=10)
            ),
            "top_cities": _series_to_dict(
                subset["city"].value_counts().head(5)
            ),
            "salary": salary,
        }

    # Junior Entry: seniority is Junior OR (NaN AND experience < 3)
    junior_mask = df["seniority_from_title"] == "Junior"
    exp_col = pd.to_numeric(df["experience_years"], errors="coerce")
    junior_mask = junior_mask | (
        df["seniority_from_title"].isna() & (exp_col < 3)
    )
    junior = df[junior_mask]

    # Senior Specialist: seniority is Senior, Lead, Director, or C-Level
    senior_mask = df["seniority_from_title"].isin(
        ["Senior", "Lead", "Director", "C-Level"]
    )
    senior = df[senior_mask]

    # Remote-First: work_modality is Remote
    remote = df[df["work_modality"] == "Remote"]

    # Career Changer: Junior postings in diverse families, no specific tech required
    career_changer_mask = (
        junior_mask
        & df["education_level"].isin(["Vocational", "Bachelor", "Degree"])
    )
    career_changer = df[career_changer_mask]

    personas = {
        "junior_entry": _persona_stats(junior, "Junior Entry"),
        "senior_specialist": _persona_stats(senior, "Senior Specialist"),
        "remote_first": _persona_stats(remote, "Remote-First"),
        "career_changer": _persona_stats(career_changer, "Career Changer"),
    }
    _write_json(personas, "personas.json")


def generate_jobs_slim(df: pd.DataFrame) -> None:
    """Generate jobs-slim.json — all rows with subset of columns for client-side filtering."""
    slim = df[SLIM_COLS].copy()
    # Ensure numeric columns stay numeric (not object dtype)
    for col in ["salary_min", "salary_max"]:
        slim[col] = pd.to_numeric(slim[col], errors="coerce")
    path = OUT_DIR / "jobs-slim.json"
    # orient="records" with pandas handles NaN → null natively for numeric cols
    slim.to_json(path, orient="records", force_ascii=False, double_precision=0)
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info("  ✓ jobs-slim.json (%.1f MB)", size_mb)


def main() -> None:
    """Run all data generation steps."""
    log.info("=" * 60)
    log.info("Generating dashboard data")
    log.info("=" * 60)

    if not CSV_PATH.exists():
        log.error("CSV not found: %s", CSV_PATH)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", CSV_PATH.name)
    df_raw = read_csv_safe(CSV_PATH)
    log.info("  Loaded %d rows × %d columns", len(df_raw), len(df_raw.columns))

    # Apply standard filters
    df = exclude_future_dates(df_raw)
    log.info("  After exclude_future_dates: %d rows", len(df))

    # Drop postings before October 1, 2025
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df = df[df["date_posted"] >= "2025-10-01"].copy()
    df["date_posted"] = df["date_posted"].dt.strftime("%Y-%m-%d")
    log.info("  After date cutoff (>= 2025-10-01): %d rows", len(df))

    # Prepare salary subset
    df_sal = salary_df(df)
    log.info("  Salary-known rows: %d (%.1f%%)", len(df_sal), len(df_sal) / len(df) * 100)

    log.info("\nGenerating JSON files:")
    generate_overview(df)
    generate_skills(df, df_sal)
    generate_salary(df_sal, total_rows=len(df))
    generate_location(df)
    generate_remote(df)
    generate_seniority(df)
    generate_benefits(df)
    generate_languages(df)
    generate_education(df)
    generate_companies(df)
    generate_role_dives(df, df_sal)
    generate_personas(df, df_sal)
    generate_jobs_slim(df)

    log.info("\n✓ All %d JSON files generated in %s", len(list(OUT_DIR.glob("*.json"))), OUT_DIR)


if __name__ == "__main__":
    main()
