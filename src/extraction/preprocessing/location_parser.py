"""Deterministic location parser: string → {city, state, country} with None for unknowns."""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from shared.constants import MISSING_SENTINELS as _MISSING_SENTINELS

logger = logging.getLogger("pipeline.location_parser")
_GERMANY_ALIASES = {"germany", "de", "deutschland"}

# Counters for the parse-strategy distribution log
_STRATEGY_STANDARD = "standard_3part"
_STRATEGY_CODED = "coded_3part"
_STRATEGY_2PART_CITY_STATE = "2part_city_state"
_STRATEGY_2PART_STATE_ONLY = "2part_state_only"
_STRATEGY_2PART_CITY = "2part_city"
_STRATEGY_2PART_FOREIGN = "2part_foreign"
_STRATEGY_COUNTRY = "bare_country"
_STRATEGY_SUPER_REGION = "super_region"
_STRATEGY_REGION = "region"
_STRATEGY_STATE = "bare_state"
_STRATEGY_FALLBACK = "fallback"


US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR",
}

NON_GERMAN_COUNTRIES = {
    "united states", "usa", "us", "uk", "united kingdom", "france", "spain",
    "italy", "netherlands", "belgium", "austria", "switzerland", "czechia",
    "czech republic", "poland", "sweden", "norway", "denmark", "finland",
    "portugal", "ireland", "canada", "australia", "india", "china", "japan",
    "brazil", "singapore",
}

_NON_GERMAN_SUPER_REGION_KEYWORDS = {"emea", "apac", "worldwide", "global"}

_GERMAN_INDICATORS = {
    "germany", "german", "deutschland", "münchen", "berlin", "hamburg",
    "frankfurt", "cologne", "stuttgart", "rhein", "rhein-main", "dach",
}


def is_non_german(location_raw: str, state: str, country: str) -> bool:
    """Return True if the location is definitely NOT in Germany.

    Checks five conditions in order:
    1. Resolved state is a US state/territory code (e.g. CA, NY, DC).
    2. A known non-German country keyword appears in the raw location string.
    3. Resolved country is non-German and not an unknown/multi-country placeholder.
    4. A non-German super-region keyword (EMEA, APAC, Worldwide, Global) appears in raw.
    5. Raw location contains an "area" pattern but resolved country is NA and has no
       German indicators (likely a foreign metro area that didn't match any region rule).
    """
    def _is_blank(v) -> bool:
        """True if v is None or NaN."""
        if v is None:
            return True
        if isinstance(v, float):
            import math
            return math.isnan(v)
        return False

    # Condition 1: US state/territory code in resolved state field
    if not _is_blank(state) and str(state).upper() in US_STATE_CODES:
        return True

    loc_lower = str(location_raw).lower()

    # Condition 2: Known non-German country keyword in raw location string
    for keyword in NON_GERMAN_COUNTRIES:
        if keyword in loc_lower:
            return True

    # Condition 3: Resolved country is neither Germany nor unknown (None/NaN)
    if not _is_blank(country) and country != "Germany":
        return True

    # Condition 4: Non-German super-region keyword in raw (EMEA, APAC, etc.)
    for keyword in _NON_GERMAN_SUPER_REGION_KEYWORDS:
        if keyword in loc_lower:
            return True

    # Condition 5: "Area" pattern with unresolved country and no German indicators
    if _is_blank(country) and "area" in loc_lower:
        if not any(ind in loc_lower for ind in _GERMAN_INDICATORS):
            return True

    return False


def _na() -> dict[str, str | None]:
    return {"city": None, "state": None, "country": None}


def load_geo_config(config_path: Path | None = None) -> dict:
    """Load the german_states.yaml geography config.

    Args:
        config_path: Path to german_states.yaml. Defaults to config/german_states.yaml.

    Returns:
        Dict with keys: states, state_codes, city_is_state, regions, super_regions,
        known_countries, german_to_english_states.
    """
    path = config_path or Path(__file__).parent.parent / "config" / "german_states.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_lookups(geo_config: dict) -> dict:
    """Pre-build lowercase lookup sets and dicts for fast matching."""
    states_lower = {s.lower(): s for s in geo_config["states"]}
    state_codes = {k.upper(): v for k, v in geo_config["state_codes"].items()}
    city_is_state_lower = {c.lower() for c in geo_config["city_is_state"]}
    regions_lower = {k.lower(): v for k, v in geo_config["regions"].items()}
    super_regions_lower = {s.lower() for s in geo_config["super_regions"]}
    known_countries_lower = {c.lower(): c for c in geo_config["known_countries"]}
    german_to_english = {
        k.lower(): v for k, v in geo_config.get("german_to_english_states", {}).items()
    }
    return {
        "states_lower": states_lower,
        "state_codes": state_codes,
        "city_is_state_lower": city_is_state_lower,
        "regions_lower": regions_lower,
        "super_regions_lower": super_regions_lower,
        "known_countries_lower": known_countries_lower,
        "german_to_english": german_to_english,
    }


def _resolve_state(raw: str, lookups: dict) -> str:
    """Resolve a raw state string to its canonical English name.

    Handles 2-letter codes (BY→Bavaria), German names (Bayern→Bavaria),
    and returns the raw string if already in canonical form.
    """
    # Try 2-letter state code first
    upper = raw.strip().upper()
    if upper in lookups["state_codes"]:
        return lookups["state_codes"][upper]

    lower = raw.strip().lower()
    # Try German→English mapping
    if lower in lookups["german_to_english"]:
        return lookups["german_to_english"][lower]
    # Try direct match against known English state names
    if lower in lookups["states_lower"]:
        return lookups["states_lower"][lower]
    # Return as-is (may be a valid foreign state/province)
    return raw.strip()


def parse_location(location: object, geo_config: dict) -> dict[str, str | None]:
    """Parse a location string into city, state, and country components.

    Args:
        location: Raw location value (string, None, NaN, etc.).
        geo_config: Loaded geography config (from load_geo_config()).

    Returns:
        Dict with keys 'city', 'state', 'country'. Unknown/absent fields use
        None. Country is always a string when resolved.
    """
    lookups = _build_lookups(geo_config)
    result, _ = _parse_with_strategy(location, lookups)
    return result


def _parse_with_strategy(location: object, lookups: dict) -> tuple[dict[str, str | None], str]:
    """Internal: parse location and return (result, strategy_name) for reporting."""

    if location is None:
        return _na(), _STRATEGY_FALLBACK
    loc_str = str(location).strip()
    if loc_str.lower() in _MISSING_SENTINELS:
        return _na(), _STRATEGY_FALLBACK

    if loc_str.lower() in lookups["super_regions_lower"]:
        return {"city": None, "state": None, "country": loc_str}, _STRATEGY_SUPER_REGION

    parts = [p.strip() for p in loc_str.split(",")]

    if len(parts) >= 3:
        return _parse_multi(parts, loc_str, lookups)
    if len(parts) == 2:
        return _parse_two_part(parts[0], parts[1], loc_str, lookups)
    # Single-part
    return _parse_single(loc_str, lookups)


def _parse_multi(
    parts: list[str], original: str, lookups: dict
) -> tuple[dict[str, str | None], str]:
    """Handle locations with 3 or more comma-separated parts."""
    city_raw, state_raw, country_raw = parts[0], parts[1], ", ".join(parts[2:])
    country_lower = country_raw.strip().lower()

    # Germany / DE formats
    if country_lower in _GERMANY_ALIASES:
        state = _resolve_state(state_raw, lookups)
        return {"city": city_raw, "state": state, "country": "Germany"}, _STRATEGY_STANDARD

    # Check if it ends with a known non-German country
    if country_lower in lookups["known_countries_lower"]:
        canonical_country = lookups["known_countries_lower"][country_lower]
        return {"city": city_raw, "state": state_raw, "country": canonical_country}, _STRATEGY_CODED

    # Unknown country — still parse best-effort
    return {"city": city_raw, "state": state_raw, "country": country_raw}, _STRATEGY_CODED


def _parse_two_part(
    first: str, second: str, original: str, lookups: dict
) -> tuple[dict[str, str | None], str]:
    """Handle 'First, Second' patterns."""
    second_lower = second.strip().lower()
    first_lower = first.strip().lower()

    # --- Ends with Germany or DE ---
    if second_lower in _GERMANY_ALIASES:
        # City-state (Berlin/Hamburg/Bremen)
        if first_lower in lookups["city_is_state_lower"]:
            city = lookups["states_lower"].get(first_lower, first)
            return {"city": city, "state": city, "country": "Germany"}, _STRATEGY_2PART_CITY_STATE

        # Known German state
        resolved = _resolve_state(first, lookups)
        if resolved.lower() in lookups["states_lower"]:
            return (
                {"city": None, "state": resolved, "country": "Germany"},
                _STRATEGY_2PART_STATE_ONLY,
            )

        # Treat as city in Germany
        return {"city": first, "state": None, "country": "Germany"}, _STRATEGY_2PART_CITY

    # --- Second part is a known non-German country ---
    if second_lower in lookups["known_countries_lower"]:
        canonical = lookups["known_countries_lower"][second_lower]
        return {"city": first, "state": None, "country": canonical}, _STRATEGY_2PART_FOREIGN

    # --- Non-German city, state ---
    return {"city": first, "state": second, "country": None}, _STRATEGY_2PART_FOREIGN


def _parse_single(
    loc: str, lookups: dict
) -> tuple[dict[str, str | None], str]:
    """Handle single-part location strings (no commas)."""
    loc_lower = loc.lower()

    # Bare country name
    if loc_lower in lookups["known_countries_lower"]:
        country = lookups["known_countries_lower"][loc_lower]
        if country == "Germany":
            return {"city": None, "state": None, "country": "Germany"}, _STRATEGY_COUNTRY
        return {"city": None, "state": None, "country": country}, _STRATEGY_COUNTRY

    # Known German state (bare, e.g. "Saarland" without ", Germany")
    if loc_lower in lookups["states_lower"]:
        return (
            {"city": None, "state": lookups["states_lower"][loc_lower], "country": "Germany"},
            _STRATEGY_STATE,
        )

    # Region lookup (exact match, case-insensitive)
    if loc_lower in lookups["regions_lower"]:
        r = lookups["regions_lower"][loc_lower]
        return {
            "city": r.get("city"),
            "state": r.get("state"),
            "country": r.get("country", "Germany"),
        }, _STRATEGY_REGION

    # Partial region match (e.g. "Rhein-Neckar Metropolitan Region" vs slight variant)
    for region_key, region_data in lookups["regions_lower"].items():
        if region_key in loc_lower or loc_lower in region_key:
            return {
                "city": region_data.get("city"),
                "state": region_data.get("state"),
                "country": region_data.get("country", "Germany"),
            }, _STRATEGY_REGION

    # Fallback
    logger.warning("Location fallback (no rule matched): %r", loc)
    return _na(), _STRATEGY_FALLBACK


def parse_all_locations(
    df,  # pd.DataFrame
    geo_config: dict,
    checkpoint,
    reports_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Apply parse_location to every row and add city/state/country columns.

    Args:
        df: Validated DataFrame (output of validate_input).
        geo_config: Loaded geography config.
        checkpoint: Checkpoint instance.
        reports_dir: Directory to write location_report.json.

    Returns:
        Tuple of (annotated DataFrame, report dict).
    """
    logger.info("=== STAGE: Location Parsing ===")
    start = time.monotonic()

    lookups = _build_lookups(geo_config)
    strategies: list[str] = []
    fallback_examples: list[str] = []

    cities, states, countries = [], [], []

    for loc in df["location"]:
        result, strategy = _parse_with_strategy(loc, lookups)
        cities.append(result["city"])
        states.append(result["state"])
        countries.append(result["country"])
        strategies.append(strategy)
        if strategy == _STRATEGY_FALLBACK and str(loc).lower() not in _MISSING_SENTINELS:
            fallback_examples.append(str(loc))

    df = df.copy()
    df["city"] = cities
    df["state"] = states
    df["country"] = countries

    # Advance checkpoint
    for row_id in df["row_id"]:
        checkpoint.advance_stage(row_id, "located")

    # Strategy distribution
    total = len(strategies)
    strategy_counts: dict[str, int] = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    fallback_count = strategy_counts.get(_STRATEGY_FALLBACK, 0)
    fallback_pct = fallback_count / total * 100 if total > 0 else 0.0
    logger.info(
        "Location parsing complete: %d rows, %.1f%% fallback",
        total, fallback_pct
    )
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-25s : %5d  (%.1f%%)", strategy, count, count / total * 100)

    if fallback_examples:
        logger.warning("%d unmatched location strings (showing up to 20):", len(fallback_examples))
        for ex in fallback_examples[:20]:
            logger.warning("  Fallback: %r", ex)

    top_cities = df["city"].dropna().value_counts().head(20).to_dict()
    top_states = df["state"].dropna().value_counts().head(20).to_dict()
    country_dist = df["country"].value_counts(dropna=False).to_dict()

    report = {
        "total_rows": total,
        "strategy_distribution": strategy_counts,
        "fallback_count": fallback_count,
        "fallback_percent": round(fallback_pct, 2),
        "top_cities": top_cities,
        "top_states": top_states,
        "country_distribution": {str(k): v for k, v in country_dist.items()},
        "na_breakdown": {
            "city_na": int(df["city"].isna().sum()),
            "state_na": int(df["state"].isna().sum()),
            "country_na": int(df["country"].isna().sum()),
        },
    }

    report_path = reports_dir / "location_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Location report saved to %s", report_path)

    elapsed = time.monotonic() - start
    logger.info("Stage Location Parsing complete: %d rows, %.1fs", total, elapsed)
    return df, report
