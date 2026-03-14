"""Smoke test: run 5 pre-LLM pipeline stages on a synthetic fixture DataFrame.

Stages tested (no LLM calls, no CSV committed to git):
    build fixture → validate_input → parse_all_locations →
    deduplicate_rows → normalize_all_titles → regex_extract (per-row)

Asserts:
    - No rows unexpectedly dropped (validate_input keeps all clean rows)
    - All required columns are present after each stage
    - No NaN in key fields after location/title stages
    - Regex fields have correct Python types (int or None, str or None)
    - At least one duplicate is removed by dedup (fixture has a known duplicate)
"""

import hashlib

import pandas as pd
import pytest

from extraction.checkpoint import Checkpoint
from extraction.dedup.row_dedup import deduplicate_rows
from extraction.preprocessing.location_parser import load_geo_config, parse_all_locations
from extraction.preprocessing.regex_extractor import extract_regex_fields
from extraction.preprocessing.title_normalizer import load_title_translations, normalize_all_titles
from extraction.preprocessing.validate_input import validate_input

# ---------------------------------------------------------------------------
# Synthetic fixture data (50 rows, inline — no CSV committed to git)
# Each row covers a different job family / location / salary format.
# Row index 20 is an intentional duplicate of row 0 (same title+company+location,
# different URL) so the dedup stage can be verified.
# ---------------------------------------------------------------------------

_ROWS = [
    # (job_url, title, company_name, location, description, date_posted)
    ("https://example.com/1", "Senior Python Developer", "TechCorp GmbH", "Berlin Germany",
     "We are looking for a Senior Python Developer with 5+ Jahre Berufserfahrung. Vollzeit. "
     "Remote moeglich. Gehalt: 80.000 - 100.000 EUR. Required: Python, FastAPI, PostgreSQL. "
     "Nice to have: Docker, Kubernetes. Benefits: 30 Urlaubstage, Homeoffice.", "2026-01-15"),
    ("https://example.com/2", "Junior Frontend Entwickler", "Startup AG", "Muenchen Deutschland",
     "Junior Frontend Entwickler gesucht. Teilzeit oder Vollzeit. React und TypeScript erforderlich. "
     "Werkstudenten herzlich willkommen. Gehalt: ab 40.000 EUR. vor Ort in Muenchen.", "2026-01-16"),
    ("https://example.com/3", "DevOps Engineer", "Cloud Solutions GmbH", "Hamburg Germany",
     "AWS und Kubernetes Kenntnisse erforderlich. CI/CD Pipeline-Erfahrung. 3+ Jahre Berufserfahrung. "
     "Vollzeit. Hybrid. Gehalt: 70.000 bis 90.000 EUR. Englischkenntnisse B2 erforderlich.", "2026-01-17"),
    ("https://example.com/4", "Data Scientist (m/w/d)", "Analytics Corp", "Frankfurt Germany",
     "Machine Learning Erfahrung. Python (scikit-learn, TensorFlow) erforderlich. SQL-Kenntnisse. "
     "Vollzeit. 100% Remote. Gehalt: 75.000 - 95.000 EUR. Deutsch fliessend erforderlich.", "2026-01-18"),
    ("https://example.com/5", "Fullstack Developer", "Digital GmbH", "Stuttgart Deutschland",
     "React Frontend, Node.js Backend. 4 Jahre Erfahrung. Vollzeit. Hybrides Arbeiten moeglich. "
     "Gehalt: 65.000 - 85.000 EUR. TypeScript, MongoDB.", "2026-01-19"),
    ("https://example.com/6", "Backend Developer Java", "Enterprise AG", "Duesseldorf Germany",
     "Java Spring Boot Entwickler mit 5 Jahren Erfahrung gesucht. Vollzeit. Praesenzpflicht. "
     "PostgreSQL Kenntnisse. Gehalt: 80.000 EUR. Bachelor Abschluss bevorzugt.", "2026-01-20"),
    ("https://example.com/7", "ML Engineer", "AI Startup", "Berlin Germany",
     "Machine Learning Engineer. TensorFlow PyTorch erforderlich. Promotion von Vorteil. Remote. "
     "Gehalt: 90.000 - 110.000 EUR. Vollzeit. 5+ Jahre Berufserfahrung.", "2026-01-21"),
    ("https://example.com/8", "IT Security Engineer", "SecureIT GmbH", "Koeln Deutschland",
     "IT Security Spezialist. Penetrationstesting. Vollzeit. Vor Ort. CISSP Zertifizierung erwuenscht. "
     "Gehalt: 75.000 - 95.000 EUR. Deutsch verhandlungssicher.", "2026-01-22"),
    ("https://example.com/9", "Product Manager", "ProductCo AG", "Berlin Germany",
     "Produktmanager fuer SaaS Produkt. 3+ Jahre Erfahrung. Agile Methoden. Scrum Master Erfahrung. "
     "Vollzeit. Hybrid. Englisch C1. Gehalt: 70.000 - 85.000 EUR.", "2026-01-23"),
    ("https://example.com/10", "QA Engineer", "Quality First GmbH", "Muenchen Germany",
     "Qualitaetssicherung. Selenium WebDriver. API Testing. Vollzeit. Homeoffice moeglich. "
     "Gehalt: 55.000 - 70.000 EUR. 2+ Jahre Berufserfahrung. Python oder Java.", "2026-01-24"),
    ("https://example.com/11", "Cloud Architect", "CloudBuilder AG", "Hamburg Germany",
     "Cloud Architect fuer AWS und Azure. Senior Level. 8+ Jahre Berufserfahrung. Vollzeit. Hybrid. "
     "Gehalt: 100.000 - 130.000 EUR. Terraform Kenntnisse.", "2026-01-25"),
    ("https://example.com/12", "Scrum Master", "AgileWorks GmbH", "Frankfurt Germany",
     "Scrum Master fuer Entwicklungsteams. Agile Coach Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 65.000 - 80.000 EUR. CSM Zertifizierung bevorzugt.", "2026-01-26"),
    ("https://example.com/13", "Data Engineer", "DataPipe Corp", "Stuttgart Deutschland",
     "Apache Spark und Kafka Kenntnisse. Python. SQL. 4 Jahre Erfahrung. Vollzeit. Remote. "
     "Gehalt: 70.000 - 90.000 EUR. Master Abschluss bevorzugt.", "2026-01-27"),
    ("https://example.com/14", "Mobile Developer Android", "MobileFirst AG", "Berlin Germany",
     "Android Entwickler. Kotlin erforderlich. 3 Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 65.000 - 80.000 EUR. Material Design Kenntnisse.", "2026-01-28"),
    ("https://example.com/15", "UX Designer", "DesignHub GmbH", "Muenchen Germany",
     "UI/UX Designer mit Figma Kenntnissen. 3+ Jahre Erfahrung. Vollzeit. Teilweise Remote. "
     "Gehalt: 55.000 - 70.000 EUR. Adobe XD Kenntnisse von Vorteil.", "2026-01-29"),
    ("https://example.com/16", "Software Architect", "ArchCo AG", "Koeln Germany",
     "Software Architekt fuer Microservices. Java Spring Boot. 10+ Jahre Berufserfahrung. "
     "Vollzeit. Vor Ort. Gehalt: 110.000 - 130.000 EUR.", "2026-01-30"),
    ("https://example.com/17", "Business Analyst", "BizAnalytics GmbH", "Duesseldorf Germany",
     "Business Analyst fuer Digitalisierungsprojekte. SQL Kenntnisse. 3 Jahre Erfahrung. "
     "Vollzeit. Hybrid. Gehalt: 60.000 - 75.000 EUR.", "2026-01-31"),
    ("https://example.com/18", "Site Reliability Engineer", "Reliability AG", "Berlin Germany",
     "SRE fuer Kubernetes Cluster. Python oder Go. Monitoring mit Prometheus Grafana. "
     "5 Jahre Erfahrung. Vollzeit. Remote first. Gehalt: 90.000 - 110.000 EUR.", "2026-02-01"),
    ("https://example.com/19", "Embedded Developer", "HardwareTech GmbH", "Stuttgart Germany",
     "Embedded Software Entwickler. C und C++ erforderlich. RTOS Kenntnisse. 4+ Jahre Erfahrung. "
     "Vollzeit. Vor Ort. Gehalt: 70.000 - 85.000 EUR.", "2026-02-02"),
    ("https://example.com/20", "IT Consultant SAP", "SAP Consulting AG", "Frankfurt Germany",
     "SAP Berater fuer S/4HANA. ABAP Kenntnisse von Vorteil. 5 Jahre Erfahrung. Vollzeit. "
     "Reisebereitschaft erforderlich. Gehalt: 80.000 - 100.000 EUR.", "2026-02-03"),
    # Row index 20: intentional duplicate of row 0 (same title+company+location, different URL)
    ("https://example.com/21", "Senior Python Developer", "TechCorp GmbH", "Berlin Germany",
     "We are looking for a Senior Python Developer with 5+ Jahre Berufserfahrung. Vollzeit. "
     "Remote moeglich. Gehalt: 80.000 - 100.000 EUR. Required: Python, FastAPI, PostgreSQL. "
     "Nice to have: Docker, Kubernetes. Benefits: 30 Urlaubstage, Homeoffice.", "2026-01-15"),
    ("https://example.com/22", "Backend Developer Go", "GoLang Startup", "Hamburg Germany",
     "Go Entwickler. Microservices. gRPC. 3 Jahre Erfahrung. Vollzeit. 100% Remote. "
     "Gehalt: 75.000 - 90.000 EUR. Docker Kubernetes.", "2026-02-04"),
    ("https://example.com/23", "Frontend Developer React", "ReactCo GmbH", "Berlin Germany",
     "React Entwickler. TypeScript. Redux. 3+ Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 65.000 - 80.000 EUR. Jest Testing.", "2026-02-05"),
    ("https://example.com/24", "Platform Engineer", "PlatformTech AG", "Muenchen Germany",
     "Platform Engineer fuer interne Entwicklerplattform. Kubernetes. Terraform. ArgoCD. "
     "5 Jahre Erfahrung. Vollzeit. Remote. Gehalt: 85.000 - 105.000 EUR.", "2026-02-06"),
    ("https://example.com/25", "Business Intelligence Developer", "BIWorks GmbH", "Koeln Germany",
     "BI Entwickler. Power BI. Tableau. SQL. 3+ Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 60.000 - 75.000 EUR. Bachelor Studium.", "2026-02-07"),
    ("https://example.com/26", "Network Engineer", "NetConnect AG", "Duesseldorf Germany",
     "Netzwerktechniker. Cisco CCNA. Firewalls. VPN. 4 Jahre Erfahrung. Vollzeit. Vor Ort. "
     "Gehalt: 55.000 - 70.000 EUR.", "2026-02-08"),
    ("https://example.com/27", "Technical Lead", "LeadDev GmbH", "Frankfurt Germany",
     "Technical Lead fuer Backend Team. Java oder Python. Team Leadership. 8 Jahre Berufserfahrung. "
     "Vollzeit. Hybrid. Gehalt: 95.000 - 115.000 EUR.", "2026-02-09"),
    ("https://example.com/28", "Database Administrator", "DBMS Corp", "Stuttgart Germany",
     "DBA fuer PostgreSQL und Oracle. Performance Tuning. Backup Recovery. 5+ Jahre Erfahrung. "
     "Vollzeit. Vor Ort. Gehalt: 65.000 - 80.000 EUR.", "2026-02-10"),
    ("https://example.com/29", "Engineering Manager", "ManageTech AG", "Berlin Germany",
     "Engineering Manager fuer 3 Teams. Agile. OKRs. 10+ Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 120.000 - 150.000 EUR.", "2026-02-11"),
    ("https://example.com/30", "IT Support Helpdesk", "HelpDesk GmbH", "Muenchen Germany",
     "IT Support Spezialist. Windows Linux. Ticketsystem. 1 Jahr Erfahrung. Vollzeit. Vor Ort. "
     "Gehalt: 35.000 - 45.000 EUR. Ausbildung im IT-Bereich.", "2026-02-12"),
    ("https://example.com/31", "MLOps Engineer", "MLPlatform AG", "Hamburg Germany",
     "MLOps fuer Machine Learning Plattform. Kubeflow. MLflow. Python. 4 Jahre Erfahrung. "
     "Vollzeit. Remote. Gehalt: 85.000 - 105.000 EUR.", "2026-02-13"),
    ("https://example.com/32", "AI Engineer", "AILab GmbH", "Berlin Germany",
     "KI Ingenieur. LLM Fine-tuning. RAG Systeme. Python. PyTorch. 3+ Jahre Erfahrung. "
     "Vollzeit. Hybrid. Gehalt: 85.000 - 110.000 EUR.", "2026-02-14"),
    ("https://example.com/33", "Working Student Software", "StudentJob AG", "Muenchen Germany",
     "Werkstudent Softwareentwicklung. Python oder JavaScript. Flexible Arbeitszeiten. "
     "20 Stunden pro Woche. Teilzeit. Homeoffice moeglich. Gehalt: 15 EUR pro Stunde.", "2026-02-01"),
    ("https://example.com/34", "Game Developer", "GameStudio GmbH", "Koeln Germany",
     "Game Developer. Unity oder Unreal Engine. C# oder C++. 3 Jahre Erfahrung. "
     "Vollzeit. Vor Ort. Gehalt: 55.000 - 70.000 EUR.", "2026-02-02"),
    ("https://example.com/35", "Solution Architect", "SolArch AG", "Frankfurt Germany",
     "Solution Architect fuer Enterprise Kunden. AWS Azure GCP. 8+ Jahre Berufserfahrung. "
     "Vollzeit. Reisebereitschaft. Gehalt: 105.000 - 125.000 EUR.", "2026-02-03"),
    ("https://example.com/36", "ERP Consultant", "ERPSolutions GmbH", "Stuttgart Germany",
     "ERP Consultant fuer SAP und Oracle. Prozessberatung. 5 Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 75.000 - 95.000 EUR.", "2026-02-04"),
    ("https://example.com/37", "System Administrator Linux", "SysAdmin Corp", "Berlin Germany",
     "Linux Systemadministrator. Ansible. Puppet. 4 Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 55.000 - 70.000 EUR. Englisch B1.", "2026-02-05"),
    ("https://example.com/38", "Technical Writer", "DocTeam GmbH", "Hamburg Germany",
     "Technischer Redakteur fuer API Dokumentation. Confluence. Swagger. 2+ Jahre Erfahrung. "
     "Vollzeit. Remote. Gehalt: 50.000 - 65.000 EUR.", "2026-02-06"),
    ("https://example.com/39", "Blockchain Developer", "CryptoTech AG", "Muenchen Germany",
     "Blockchain Entwickler. Solidity. Web3.js. Smart Contracts. 3 Jahre Erfahrung. "
     "Vollzeit. Remote. Gehalt: 80.000 - 100.000 EUR.", "2026-02-07"),
    ("https://example.com/40", "Software Developer C++", "SystemSoft GmbH", "Duesseldorf Germany",
     "C++ Entwickler fuer Echtzeitsysteme. RTOS. 5 Jahre Berufserfahrung. Vollzeit. Vor Ort. "
     "Gehalt: 75.000 - 90.000 EUR.", "2026-02-08"),
    ("https://example.com/41", "Data Analyst", "DataViz Corp", "Frankfurt Germany",
     "Datenanalyst. SQL. Power BI. Python pandas. 2+ Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 50.000 - 65.000 EUR. Hochschulabschluss.", "2026-02-09"),
    ("https://example.com/42", "Project Manager IT", "ProjectPro AG", "Stuttgart Germany",
     "IT Projektleiter. Agile Waterfall. PMP Zertifizierung bevorzugt. 6 Jahre Erfahrung. "
     "Vollzeit. Hybrid. Gehalt: 75.000 - 95.000 EUR.", "2026-02-10"),
    ("https://example.com/43", "Senior React Developer", "WebDev GmbH", "Berlin Germany",
     "Senior React Entwickler. Next.js. TypeScript. GraphQL. 5+ Jahre Erfahrung. "
     "Vollzeit. 100% Remote. Gehalt: 80.000 - 100.000 EUR.", "2026-02-11"),
    ("https://example.com/44", "iOS Developer Swift", "MobileApp AG", "Muenchen Germany",
     "iOS Entwickler. Swift SwiftUI. Xcode. 3 Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 65.000 - 80.000 EUR.", "2026-02-12"),
    ("https://example.com/45", "Intern Software Development", "InternCo GmbH", "Hamburg Germany",
     "Praktikant Softwareentwicklung. Python oder Java. 6 Monate. Vollzeit. Vor Ort. "
     "Verguetung: 1.200 EUR pro Monat. Student oder Absolvent.", "2026-02-13"),
    ("https://example.com/46", "Senior Data Scientist", "BigData AG", "Berlin Germany",
     "Senior Data Scientist. Python R. Machine Learning. NLP. 7+ Jahre Berufserfahrung. "
     "Vollzeit. Remote first. Gehalt: 95.000 - 115.000 EUR. Promotion bevorzugt.", "2026-02-14"),
    ("https://example.com/47", "Backend PHP Developer", "PHPShop GmbH", "Koeln Germany",
     "PHP Laravel Entwickler. MySQL. Redis. 3 Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 55.000 - 70.000 EUR. Symfony von Vorteil.", "2026-02-01"),
    ("https://example.com/48", "Kubernetes Engineer", "K8sOps AG", "Frankfurt Germany",
     "Kubernetes Administrator. Helm. Istio. Service Mesh. 4 Jahre Erfahrung. Vollzeit. Remote. "
     "Gehalt: 80.000 - 100.000 EUR.", "2026-02-02"),
    ("https://example.com/49", "Fullstack Vue Developer", "VueDev GmbH", "Stuttgart Germany",
     "Fullstack Entwickler. Vue.js. Node.js. PostgreSQL. 3+ Jahre Erfahrung. Vollzeit. Hybrid. "
     "Gehalt: 65.000 - 80.000 EUR.", "2026-02-03"),
    ("https://example.com/50", "Python Data Engineer", "StreamData AG", "Muenchen Germany",
     "Python Data Engineer. Apache Spark. Airflow. 4 Jahre Erfahrung. Vollzeit. Remote. "
     "Gehalt: 75.000 - 90.000 EUR. Deutsch fliessend.", "2026-02-04"),
]

_COLUMNS = ["job_url", "title", "company_name", "location", "description", "date_posted"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_id(job_url: str) -> str:
    return hashlib.sha256(job_url.encode()).hexdigest()[:12]


def _make_df() -> pd.DataFrame:
    """Build the fixture DataFrame with all pipeline-required columns."""
    df = pd.DataFrame(_ROWS, columns=_COLUMNS)
    df["row_id"] = df["job_url"].apply(_row_id)
    df["source_file"] = "smoke_fixture"
    df["site"] = "test"
    return df


def _make_cfg(tmp_path):
    reports_dir = tmp_path / "reports"
    deduped_dir = tmp_path / "deduped"
    reports_dir.mkdir()
    deduped_dir.mkdir()
    return {
        "paths": {
            "reports_dir": reports_dir,
            "deduped_dir": deduped_dir,
        },
        "validation": {
            "min_description_length": 50,
            "date_anomaly_cutoff": "2020-01-01",
        },
    }


def _register_rows(cp, df):
    for row_id in df["row_id"]:
        cp._conn.execute(
            "INSERT OR IGNORE INTO rows (row_id, file_path) VALUES (?, ?)",
            (row_id, "smoke_fixture"),
        )
    cp._conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cp(tmp_path):
    checkpoint = Checkpoint(tmp_path / "smoke.db")
    yield checkpoint
    checkpoint.close()


@pytest.fixture
def cfg(tmp_path):
    return _make_cfg(tmp_path)


# ---------------------------------------------------------------------------
# Stage 0: fixture shape
# ---------------------------------------------------------------------------

def test_fixture_has_correct_shape():
    df = _make_df()
    assert len(df) == 50, f"Expected 50 rows, got {len(df)}"
    required = {"job_url", "title", "company_name", "location", "description", "date_posted"}
    assert required.issubset(df.columns)
    assert df["row_id"].nunique() == 50, "row_ids must be unique (one per URL)"


# ---------------------------------------------------------------------------
# Stage 1: validate_input
# ---------------------------------------------------------------------------

def test_validate_input_no_unexpected_drops(cp, cfg):
    """validate_input must not drop rows; all fixture rows are clean."""
    df = _make_df()
    _register_rows(cp, df)

    validated, _ = validate_input(df, cfg, cp)

    assert len(validated) == len(df)
    assert "input_flags" in validated.columns
    privacy_wall = validated["input_flags"].apply(lambda f: "privacy_wall" in f).sum()
    assert privacy_wall == 0, f"Fixture should have no privacy-wall rows, got {privacy_wall}"


def test_validate_input_flags_column_is_list(cp, cfg):
    df = _make_df()
    _register_rows(cp, df)
    validated, _ = validate_input(df, cfg, cp)
    assert validated["input_flags"].apply(lambda f: isinstance(f, list)).all()


# ---------------------------------------------------------------------------
# Stage 2: parse_all_locations
# ---------------------------------------------------------------------------

def test_parse_all_locations_adds_columns(cp, cfg):
    df = _make_df()
    df["input_flags"] = [[] for _ in range(len(df))]

    geo = load_geo_config()
    located, _ = parse_all_locations(df, geo, cp, cfg["paths"]["reports_dir"])

    for col in ("city", "state", "country"):
        assert col in located.columns, f"Missing column: {col}"


def test_parse_all_locations_no_non_german_countries(cp, cfg):
    """All fixture locations are German cities — no non-German country should be assigned."""
    df = _make_df()
    df["input_flags"] = [[] for _ in range(len(df))]

    geo = load_geo_config()
    located, _ = parse_all_locations(df, geo, cp, cfg["paths"]["reports_dir"])

    # country column may contain None (fallback) or 'Germany'; nothing else
    non_null_countries = located["country"].dropna()
    unexpected = non_null_countries[~non_null_countries.isin(["Germany"])]
    assert len(unexpected) == 0, (
        f"Expected only Germany or None, found: {unexpected.value_counts().to_dict()}"
    )


# ---------------------------------------------------------------------------
# Stage 3: deduplicate_rows
# ---------------------------------------------------------------------------

def test_dedup_removes_known_duplicate(cp, cfg):
    """Row index 20 duplicates row 0 by title+company+location; dedup must remove it."""
    df = _make_df()
    df["input_flags"] = [[] for _ in range(len(df))]
    _register_rows(cp, df)

    deduped, _ = deduplicate_rows(df, cp, cfg)

    assert len(deduped) < len(df), "Dedup must remove at least one duplicate"
    assert len(deduped) == 49, f"Expected 49 rows after dedup (1 duplicate), got {len(deduped)}"


def test_dedup_no_duplicate_row_ids(cp, cfg):
    df = _make_df()
    df["input_flags"] = [[] for _ in range(len(df))]
    _register_rows(cp, df)

    deduped, _ = deduplicate_rows(df, cp, cfg)
    assert deduped["row_id"].nunique() == len(deduped)


# ---------------------------------------------------------------------------
# Stage 4: normalize_all_titles
# ---------------------------------------------------------------------------

def test_normalize_all_titles_adds_title_cleaned(cp, cfg):
    df = _make_df()
    df["input_flags"] = [[] for _ in range(len(df))]

    translations = load_title_translations()
    normalized, _ = normalize_all_titles(df, translations, cp, cfg["paths"]["reports_dir"])

    assert "title_cleaned" in normalized.columns
    assert "title_original" in normalized.columns


def test_normalize_all_titles_strips_gender_suffix(cp, cfg):
    """'Data Scientist (m/w/d)' is in the fixture — cleaned title must not contain (m/w/d)."""
    df = _make_df()
    df["input_flags"] = [[] for _ in range(len(df))]

    translations = load_title_translations()
    normalized, _ = normalize_all_titles(df, translations, cp, cfg["paths"]["reports_dir"])

    still_mwd = normalized["title_cleaned"].str.contains(r"\(m/w/d\)", case=False, na=False).sum()
    assert still_mwd == 0, f"{still_mwd} cleaned titles still contain (m/w/d)"


def test_normalize_all_titles_no_null_cleaned(cp, cfg):
    df = _make_df()
    df["input_flags"] = [[] for _ in range(len(df))]

    translations = load_title_translations()
    normalized, _ = normalize_all_titles(df, translations, cp, cfg["paths"]["reports_dir"])

    assert normalized["title_cleaned"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Stage 5: regex_extract (per-row, no LLM)
# ---------------------------------------------------------------------------

def test_regex_extract_returns_correct_types():
    """extract_regex_fields must return correct Python types for every fixture row."""
    df = _make_df()
    df["title_cleaned"] = df["title"]

    for _, row in df.iterrows():
        result = extract_regex_fields(
            str(row.get("description", "")),
            str(row.get("title_cleaned", "")),
        )

        for field in ("salary_min", "salary_max"):
            val = result.get(field)
            assert val is None or isinstance(val, int), (
                f"row={row['row_id']}: {field}={val!r} must be int|None"
            )

        exp = result.get("experience_years")
        assert exp is None or isinstance(exp, int), (
            f"row={row['row_id']}: experience_years={exp!r} must be int|None"
        )

        for field in ("contract_type", "work_modality", "seniority_from_title", "education_level"):
            val = result.get(field)
            assert val is None or isinstance(val, str), (
                f"row={row['row_id']}: {field}={val!r} must be str|None"
            )

        langs = result.get("languages")
        assert isinstance(langs, list), (
            f"row={row['row_id']}: languages={langs!r} must be a list"
        )


def test_regex_extract_salary_extracted_for_most_rows():
    """Nearly all fixture rows include a salary — salary_min should be extracted for most."""
    df = _make_df()
    df["title_cleaned"] = df["title"]

    extracted = sum(
        1 for _, row in df.iterrows()
        if extract_regex_fields(
            str(row.get("description", "")),
            str(row.get("title_cleaned", "")),
        ).get("salary_min") is not None
    )
    assert extracted >= 25, f"Expected >=25 rows with salary_min extracted, got {extracted}"


def test_regex_extract_contract_type_extracted_for_most_rows():
    """All fixture descriptions include Vollzeit/Teilzeit — contract_type should be extracted."""
    df = _make_df()
    df["title_cleaned"] = df["title"]

    extracted = sum(
        1 for _, row in df.iterrows()
        if extract_regex_fields(
            str(row.get("description", "")),
            str(row.get("title_cleaned", "")),
        ).get("contract_type") is not None
    )
    assert extracted >= 40, f"Expected >=40 rows with contract_type, got {extracted}"


# ---------------------------------------------------------------------------
# End-to-end: run all 5 stages in sequence
# ---------------------------------------------------------------------------

def test_full_smoke_pipeline(tmp_path, cp, cfg):
    """Run all 5 pre-LLM stages sequentially and assert invariants on the final result."""
    df = _make_df()
    assert len(df) == 50

    _register_rows(cp, df)

    # Stage 1: validate_input
    df, _ = validate_input(df, cfg, cp)
    assert "input_flags" in df.columns
    assert len(df) == 50

    # Stage 2: parse_all_locations
    geo = load_geo_config()
    df, _ = parse_all_locations(df, geo, cp, cfg["paths"]["reports_dir"])
    assert all(col in df.columns for col in ("city", "state", "country"))

    # Stage 3: deduplicate_rows
    df, _ = deduplicate_rows(df, cp, cfg)
    assert len(df) == 49
    assert df["row_id"].nunique() == len(df)

    # Stage 4: normalize_all_titles
    translations = load_title_translations()
    df, _ = normalize_all_titles(df, translations, cp, cfg["paths"]["reports_dir"])
    assert "title_cleaned" in df.columns
    assert df["title_cleaned"].isna().sum() == 0

    # Stage 5: regex_extract (iterate over raw dicts to avoid pandas type coercion)
    regex_rows = [
        extract_regex_fields(
            str(row.get("description", "")),
            str(row.get("title_cleaned", "")),
        )
        for _, row in df.iterrows()
    ]
    assert len(regex_rows) == len(df)

    # All required keys present in every result dict
    expected_keys = {
        "contract_type", "work_modality", "salary_min", "salary_max",
        "experience_years", "seniority_from_title", "languages", "education_level",
    }
    assert all(expected_keys.issubset(r.keys()) for r in regex_rows)

    # salary values are int or None (checked on raw dicts before any pandas conversion)
    for i, r in enumerate(regex_rows):
        for field in ("salary_min", "salary_max"):
            val = r.get(field)
            assert val is None or isinstance(val, int), (
                f"Row {i}: {field}={val!r} must be int|None"
            )

    # languages is always a list
    assert all(isinstance(r.get("languages"), list) for r in regex_rows)
