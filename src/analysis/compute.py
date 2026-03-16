"""Reusable analytical computations for analysis notebooks.

All functions are pure: they accept a DataFrame and return a new DataFrame
or Series. No side effects, no chart rendering, no file I/O.

Design principle: any calculation used in more than one notebook, or complex
enough to warrant testing, belongs here rather than inline in a notebook cell.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd

from shared.json_utils import parse_json_list

# ---------------------------------------------------------------------------
# Skill computations
# ---------------------------------------------------------------------------


def skill_counts(
    df: pd.DataFrame,
    col: str = "technical_skills",
    top_n: int = 30,
) -> pd.Series:
    """Count occurrences of each skill across all rows.

    Args:
        df: Source DataFrame.
        col: JSON-list column to count from.
        top_n: Maximum number of skills to return (sorted descending).

    Returns:
        Series indexed by skill name, values are occurrence counts.
    """
    counter: Counter = Counter()
    for skills in df[col].apply(parse_json_list):
        counter.update(s.strip() for s in skills if s.strip())
    return pd.Series(dict(counter.most_common(top_n)), name="count")


def skill_cooccurrence(
    df: pd.DataFrame,
    col: str = "technical_skills",
    top_n: int = 20,
) -> pd.DataFrame:
    """Build a symmetric skill co-occurrence matrix.

    A cell (A, B) contains the number of job postings that list both skill A
    and skill B.  Only the top_n most frequent skills are included.

    Args:
        df: Source DataFrame.
        col: JSON-list column to build co-occurrence from.
        top_n: Number of most-frequent skills to include (controls matrix size).

    Returns:
        Square DataFrame with skills as both index and columns.
    """
    top_skills = set(skill_counts(df, col, top_n=top_n).index)
    cooc: Counter = Counter()
    for skills in df[col].apply(parse_json_list):
        present = [s.strip() for s in skills if s.strip() in top_skills]
        for a, b in combinations(sorted(set(present)), 2):
            cooc[(a, b)] += 1

    matrix = pd.DataFrame(0, index=sorted(top_skills), columns=sorted(top_skills))
    for (a, b), count in cooc.items():
        matrix.loc[a, b] = count
        matrix.loc[b, a] = count
    return matrix


def skill_by_job_family(
    df: pd.DataFrame,
    col: str = "technical_skills",
    top_n_skills: int = 15,
    top_n_families: int = 10,
) -> pd.DataFrame:
    """Cross-tabulation: skill frequency per job family (normalised to %).

    Args:
        df: Source DataFrame with 'job_family' and a JSON-list skill column.
        col: JSON-list column to use.
        top_n_skills: Number of most-frequent skills to include.
        top_n_families: Number of most-frequent job families to include.

    Returns:
        DataFrame with job families as rows, skills as columns, values are
        the percentage of rows in that family that mention the skill.
    """
    top_skills = list(skill_counts(df, col, top_n=top_n_skills).index)
    top_families = list(
        df["job_family"].value_counts().head(top_n_families).index
    )
    sub = df[df["job_family"].isin(top_families)].copy()
    sub["_skills"] = sub[col].apply(parse_json_list)

    rows = []
    for family in top_families:
        fam_df = sub[sub["job_family"] == family]
        total = max(1, len(fam_df))
        skill_pct = {
            skill: fam_df["_skills"].apply(lambda lst, s=skill: s in lst).sum() / total * 100
            for skill in top_skills
        }
        rows.append({"job_family": family, **skill_pct})

    result = pd.DataFrame(rows).set_index("job_family")
    return result[top_skills]


def required_vs_optional_skills(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compare how often each skill appears as required vs nice-to-have.

    Args:
        df: Source DataFrame with 'technical_skills' and 'nice_to_have_skills'.
        top_n: Number of top-required skills to include.

    Returns:
        DataFrame indexed by skill name, with columns ['required', 'nice_to_have', 'ratio'].
        ratio = required / (required + nice_to_have); higher means more often required.
    """
    req = skill_counts(df, "technical_skills", top_n=top_n)
    nth = skill_counts(df, "nice_to_have_skills", top_n=200)

    result = pd.DataFrame({"required": req}).fillna(0)
    result["nice_to_have"] = result.index.map(lambda s: nth.get(s, 0))
    total = result["required"] + result["nice_to_have"]
    result["ratio"] = result["required"] / total.clip(lower=1)
    return result.sort_values("required", ascending=False)


def skill_trends_by_month(
    df: pd.DataFrame,
    skills: list[str],
    col: str = "technical_skills",
) -> pd.DataFrame:
    """Monthly mention rate for a list of skills over the posting date range.

    Args:
        df: Source DataFrame with 'date_posted' (parseable as datetime) and
            a JSON-list skill column.
        skills: List of skill names to track.
        col: JSON-list column to search.

    Returns:
        DataFrame indexed by month (Period), one column per skill.
        Values are the fraction of that month's postings mentioning the skill.
    """
    tmp = df[df["date_posted"].notna()].copy()
    tmp["_month"] = pd.to_datetime(tmp["date_posted"], errors="coerce").dt.to_period("M")
    tmp = tmp[tmp["_month"].notna()].copy()
    tmp["_skills"] = tmp[col].apply(parse_json_list)

    rows = {}
    for period, group in tmp.groupby("_month"):
        total = max(1, len(group))
        rows[period] = {
            skill: group["_skills"].apply(lambda lst, s=skill: s in lst).sum() / total
            for skill in skills
        }
    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------
# Salary computations
# ---------------------------------------------------------------------------


def salary_by_group(
    df: pd.DataFrame,
    group_col: str,
    min_count: int = 10,
) -> pd.DataFrame:
    """Median, P25, and P75 salary midpoint per group.

    Args:
        df: Source DataFrame. Must have been processed by salary_df() from
            filters.py so salary_mid is present.
        group_col: Column to group by (e.g. 'job_family', 'city', 'seniority_from_title').
        min_count: Minimum number of salary-known rows per group to be included.

    Returns:
        DataFrame with columns ['median', 'p25', 'p75', 'count'],
        indexed by group_col values, sorted by median descending.
    """
    if "salary_mid" not in df.columns:
        raise ValueError("salary_mid column missing — call salary_df(df) first")

    agg = (
        df[df[group_col].notna()]
        .groupby(group_col)["salary_mid"]
        .agg(
            count="count",
            median="median",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
        )
    )
    return agg[agg["count"] >= min_count].sort_values("median", ascending=False)


# ---------------------------------------------------------------------------
# Location computations
# ---------------------------------------------------------------------------


def job_density_by_state(df: pd.DataFrame) -> pd.Series:
    """Job posting count per German federal state.

    Args:
        df: Source DataFrame with a 'state' column.

    Returns:
        Series indexed by state name, sorted descending.
    """
    return df["state"].value_counts()


def modality_by_state(
    df: pd.DataFrame,
    top_n_states: int = 12,
) -> pd.DataFrame:
    """Work modality percentage breakdown per state.

    Args:
        df: Source DataFrame with 'state' and 'work_modality'.
        top_n_states: Limit to the N states with the most postings.

    Returns:
        DataFrame: states as rows, modalities as columns, values are percentages.
    """
    top_states = df["state"].value_counts().head(top_n_states).index
    sub = df[df["state"].isin(top_states) & df["work_modality"].notna()]
    ct = pd.crosstab(sub["state"], sub["work_modality"], normalize="index") * 100
    return ct.reindex(top_states)


# ---------------------------------------------------------------------------
# Company computations
# ---------------------------------------------------------------------------


def top_companies_by_postings(
    df: pd.DataFrame, top_n: int = 20
) -> pd.Series:
    """Companies ranked by total number of job postings.

    Args:
        df: Source DataFrame with a 'company_name' column.
        top_n: Number of top companies to return.

    Returns:
        Series indexed by company name, values are posting counts.
    """
    return df["company_name"].value_counts().head(top_n)


def company_job_family_diversity(
    df: pd.DataFrame,
    min_postings: int = 5,
    top_n: int = 20,
) -> pd.Series:
    """Companies ranked by the number of distinct job families they recruit for.

    A high diversity score means the company hires across many IT disciplines.

    Args:
        df: Source DataFrame with 'company_name' and 'job_family'.
        min_postings: Minimum total postings for a company to be included.
        top_n: Number of companies to return.

    Returns:
        Series indexed by company name, values are count of distinct job families.
    """
    valid = df[df["company_name"].notna() & df["job_family"].notna()]
    counts = valid.groupby("company_name").filter(
        lambda g: len(g) >= min_postings
    )
    diversity = (
        counts.groupby("company_name")["job_family"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return diversity


# ---------------------------------------------------------------------------
# Posting trends over time
# ---------------------------------------------------------------------------


def postings_per_month(df: pd.DataFrame) -> pd.Series:
    """Monthly job posting volume.

    Args:
        df: Source DataFrame with a 'date_posted' column.

    Returns:
        Series indexed by monthly Period, values are posting counts.
    """
    months = pd.to_datetime(df["date_posted"], errors="coerce").dt.to_period("M")
    return months.value_counts().sort_index()


def postings_per_week(df: pd.DataFrame) -> pd.Series:
    """Weekly job posting volume.

    Args:
        df: Source DataFrame with a 'date_posted' column.

    Returns:
        Series indexed by weekly Period, values are posting counts.
    """
    weeks = pd.to_datetime(df["date_posted"], errors="coerce").dt.to_period("W")
    return weeks.value_counts().sort_index()


# ---------------------------------------------------------------------------
# Soft skills
# ---------------------------------------------------------------------------


def soft_skill_category_by_family(
    df: pd.DataFrame,
    top_n_families: int = 10,
) -> pd.DataFrame:
    """Soft skill category prevalence (%) per job family.

    Args:
        df: Source DataFrame with 'job_family' and 'soft_skill_categories'
            (JSON-list column of canonical category labels).
        top_n_families: Number of most-frequent job families to include.

    Returns:
        DataFrame: job families as rows, soft skill categories as columns,
        values are % of rows in that family mentioning the category.
    """
    top_families = list(df["job_family"].value_counts().head(top_n_families).index)
    sub = df[df["job_family"].isin(top_families)].copy()
    sub["_cats"] = sub["soft_skill_categories"].apply(parse_json_list)

    all_cats: set[str] = set()
    for cats in sub["_cats"]:
        all_cats.update(cats)
    all_cats.discard("")

    rows = []
    for family in top_families:
        fam_df = sub[sub["job_family"] == family]
        total = max(1, len(fam_df))
        row = {"job_family": family}
        for cat in sorted(all_cats):
            row[cat] = fam_df["_cats"].apply(lambda lst, c=cat: c in lst).sum() / total * 100
        rows.append(row)

    return pd.DataFrame(rows).set_index("job_family")


# ---------------------------------------------------------------------------
# Benefit computations
# ---------------------------------------------------------------------------


def benefit_category_by_family(
    df: pd.DataFrame,
    top_n_families: int = 10,
) -> pd.DataFrame:
    """Benefit category prevalence (%) per job family.

    Args:
        df: Source DataFrame with 'job_family' and 'benefit_categories'.
        top_n_families: Number of most-frequent job families to include.

    Returns:
        DataFrame: families as rows, benefit categories as columns, values are %.
    """
    top_families = list(df["job_family"].value_counts().head(top_n_families).index)
    sub = df[df["job_family"].isin(top_families)].copy()
    sub["_cats"] = sub["benefit_categories"].apply(parse_json_list)

    all_cats: set[str] = set()
    for cats in sub["_cats"]:
        all_cats.update(cats)
    all_cats.discard("")

    rows = []
    for family in top_families:
        fam_df = sub[sub["job_family"] == family]
        total = max(1, len(fam_df))
        row = {"job_family": family}
        for cat in sorted(all_cats):
            row[cat] = fam_df["_cats"].apply(lambda lst, c=cat: c in lst).sum() / total * 100
        rows.append(row)

    return pd.DataFrame(rows).set_index("job_family")


# ---------------------------------------------------------------------------
# Skill-salary premium
# ---------------------------------------------------------------------------


def skill_salary_premium(
    df: pd.DataFrame,
    col: str = "technical_skills",
    min_count: int = 5,
    top_n: int = 25,
) -> pd.DataFrame:
    """Rank skills by median salary of postings that mention them.

    Args:
        df: Source DataFrame. Must have been processed by salary_df() so
            salary_mid is present.
        col: JSON-list column to explode skills from.
        min_count: Minimum postings with salary data for a skill to be included.
        top_n: Number of skills to return.

    Returns:
        DataFrame with columns ['median', 'p25', 'p75', 'count'],
        indexed by skill name, sorted by median descending.
    """
    if "salary_mid" not in df.columns:
        raise ValueError("salary_mid column missing — call salary_df(df) first")

    rows = []
    for _, r in df.iterrows():
        skills = parse_json_list(r[col])
        for s in skills:
            if s.strip():
                rows.append({"skill": s.strip(), "salary_mid": r["salary_mid"]})

    if not rows:
        return pd.DataFrame(columns=["median", "p25", "p75", "count"])

    skill_df = pd.DataFrame(rows)
    agg = skill_df.groupby("skill")["salary_mid"].agg(
        count="count",
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    )
    return agg[agg["count"] >= min_count].sort_values("median", ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Skill overlap / transferability
# ---------------------------------------------------------------------------


def skill_overlap_matrix(
    role_dfs: dict[str, pd.DataFrame],
    col: str = "technical_skills",
    top_n: int = 25,
) -> pd.DataFrame:
    """Jaccard similarity of skill sets between roles.

    For each role pair, computes |intersection| / |union| of their top-N
    most-frequent skills.  Higher values mean easier career transitions.

    Args:
        role_dfs: Mapping of role name → filtered DataFrame.
        col: JSON-list column of skills.
        top_n: Number of top skills per role to use for comparison.

    Returns:
        Square DataFrame with Jaccard similarity (0–100%) between roles.
    """
    role_skills: dict[str, set[str]] = {}
    for name, rdf in role_dfs.items():
        top = set(skill_counts(rdf, col, top_n=top_n).index)
        role_skills[name] = top

    names = list(role_dfs.keys())
    matrix = pd.DataFrame(0.0, index=names, columns=names)
    for a in names:
        for b in names:
            if a == b:
                matrix.loc[a, b] = 100.0
            else:
                inter = len(role_skills[a] & role_skills[b])
                union = len(role_skills[a] | role_skills[b])
                matrix.loc[a, b] = (inter / max(1, union)) * 100
    return matrix


# ---------------------------------------------------------------------------
# Categorical distributions (generic)
# ---------------------------------------------------------------------------


def categorical_distribution(
    role_dfs: dict[str, pd.DataFrame],
    col: str,
    categories: list[str],
    *,
    include_unspecified: bool = True,
) -> pd.DataFrame:
    """Percentage distribution of a categorical column across roles.

    Args:
        role_dfs: Mapping of role name → filtered DataFrame.
        col: Categorical column name (e.g. 'education_level', 'work_modality').
        categories: Ordered list of category values to count.
        include_unspecified: Whether to add an 'Unspecified' column for NaN values.

    Returns:
        DataFrame with roles as index, categories as columns, values as percentages.
    """
    rows: list[dict[str, str | float]] = []
    for name, rdf in role_dfs.items():
        total = max(1, len(rdf))
        row: dict[str, str | float] = {"Role": name}
        for cat in categories:
            row[cat] = (rdf[col] == cat).sum() / total * 100
        if include_unspecified:
            row["Unspecified"] = rdf[col].isna().sum() / total * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("Role")


def language_requirement_pct(
    role_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """German/English language requirement percentage per role.

    Args:
        role_dfs: Mapping of role name → filtered DataFrame.

    Returns:
        DataFrame with roles as index, 'German %' and 'English %' columns.
    """
    rows = []
    for name, rdf in role_dfs.items():
        total = max(1, len(rdf))
        has_german = rdf["languages"].apply(
            lambda x: "german" in str(x).lower() if pd.notna(x) else False
        ).sum()
        has_english = rdf["languages"].apply(
            lambda x: "english" in str(x).lower() if pd.notna(x) else False
        ).sum()
        rows.append({
            "Role": name,
            "German %": has_german / total * 100,
            "English %": has_english / total * 100,
        })
    return pd.DataFrame(rows).set_index("Role")


# ---------------------------------------------------------------------------
# Skill progression by seniority
# ---------------------------------------------------------------------------


def skill_progression(
    df: pd.DataFrame,
    job_family: str,
    col: str = "technical_skills",
    top_n: int = 10,
) -> pd.DataFrame:
    """Skill prevalence (%) at each seniority level within one job family.

    Rows with missing seniority_from_title are treated as "Mid" level.
    Useful for visualising skill progression from Junior to Mid to Senior to Lead.

    Args:
        df: Source DataFrame with 'job_family', 'seniority_from_title', and a
            JSON-list skill column.
        job_family: Job family to analyse.
        col: JSON-list column of skills.
        top_n: Number of top overall skills to include.

    Returns:
        DataFrame with seniority levels as columns, skills as rows,
        values are % of postings at that level mentioning the skill.
    """
    levels = ["Junior", "Mid", "Senior", "Lead"]
    fam = df[df["job_family"] == job_family].copy()

    # Treat NaN seniority as "Mid"
    fam["_level"] = fam["seniority_from_title"].fillna("Mid")
    fam = fam[fam["_level"].isin(levels)]
    fam["_skills"] = fam[col].apply(parse_json_list)

    # Find top skills overall for this family
    counter: dict[str, int] = {}
    for skills in fam["_skills"]:
        for s in skills:
            s = s.strip()
            if s:
                counter[s] = counter.get(s, 0) + 1
    top_skills = sorted(counter, key=lambda s: counter.get(s, 0), reverse=True)[:top_n]

    rows = []
    for skill in top_skills:
        row = {"skill": skill}
        for level in levels:
            level_df = fam[fam["_level"] == level]
            total = max(1, len(level_df))
            row[level] = level_df["_skills"].apply(
                lambda lst, s=skill: s in lst
            ).sum() / total * 100
        rows.append(row)

    return pd.DataFrame(rows).set_index("skill")[levels]
