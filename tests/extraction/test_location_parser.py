"""Tests for extraction/preprocessing/location_parser.py — minimum 35 parametrized cases."""

import pytest

from extraction.preprocessing.location_parser import is_non_german, load_geo_config, parse_location

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def geo():
    """Load the real german_states.yaml once for all tests."""
    return load_geo_config()


# ---------------------------------------------------------------------------
# Parametrized: all rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("location, exp_city, exp_state, exp_country", [
    # -----------------------------------------------------------------------
    # Standard 3-part German  (City, State, Germany)
    # -----------------------------------------------------------------------
    ("Munich, Bavaria, Germany",                  "Munich",      "Bavaria",                   "Germany"),
    ("Berlin, Berlin, Germany",                   "Berlin",      "Berlin",                    "Germany"),
    ("Cologne, North Rhine-Westphalia, Germany",  "Cologne",     "North Rhine-Westphalia",    "Germany"),
    ("Stuttgart, Baden-Württemberg, Germany",     "Stuttgart",   "Baden-Württemberg",         "Germany"),
    ("Frankfurt, Hesse, Germany",                 "Frankfurt",   "Hesse",                     "Germany"),

    # -----------------------------------------------------------------------
    # German format with state codes + "DE"  (City, CODE, DE)
    # -----------------------------------------------------------------------
    ("Berlin, BE, DE",                            "Berlin",      "Berlin",                    "Germany"),
    ("München, BY, DE",                           "München",     "Bavaria",                   "Germany"),
    ("Frankfurt am Main, HE, DE",                 "Frankfurt am Main", "Hesse",               "Germany"),
    ("Stuttgart, BW, DE",                         "Stuttgart",   "Baden-Württemberg",         "Germany"),
    ("Köln, NW, DE",                              "Köln",        "North Rhine-Westphalia",    "Germany"),
    ("Hamburg, HH, DE",                           "Hamburg",     "Hamburg",                   "Germany"),
    ("Dresden, SN, DE",                           "Dresden",     "Saxony",                    "Germany"),
    ("Hannover, NI, DE",                          "Hannover",    "Lower Saxony",              "Germany"),

    # -----------------------------------------------------------------------
    # 2-part city-state  (Berlin/Hamburg/Bremen, Germany)
    # -----------------------------------------------------------------------
    ("Berlin, Germany",                           "Berlin",      "Berlin",                    "Germany"),
    ("Hamburg, Germany",                          "Hamburg",     "Hamburg",                   "Germany"),
    ("Bremen, Germany",                           "Bremen",      "Bremen",                    "Germany"),

    # -----------------------------------------------------------------------
    # "Deutschland" alias for Germany
    # -----------------------------------------------------------------------
    ("Berlin, Deutschland",                       "Berlin",      "Berlin",                    "Germany"),
    ("München, Bayern, Deutschland",              "München",     "Bavaria",                   "Germany"),
    ("Stuttgart, Deutschland",                    "Stuttgart",   None,                        "Germany"),

    # -----------------------------------------------------------------------
    # 2-part state-only  (State, Germany)
    # -----------------------------------------------------------------------
    ("Bavaria, Germany",                          None,          "Bavaria",                   "Germany"),
    ("North Rhine-Westphalia, Germany",           None,          "North Rhine-Westphalia",    "Germany"),
    ("Saxony, Germany",                           None,          "Saxony",                    "Germany"),
    ("Baden-Württemberg, Germany",                None,          "Baden-Württemberg",         "Germany"),
    ("Schleswig-Holstein, Germany",               None,          "Schleswig-Holstein",        "Germany"),

    # -----------------------------------------------------------------------
    # Bare "Germany"
    # -----------------------------------------------------------------------
    ("Germany",                                   None,          None,                        "Germany"),

    # -----------------------------------------------------------------------
    # Single-part region patterns
    # -----------------------------------------------------------------------
    ("Stuttgart Region",                          "Stuttgart",   "Baden-Württemberg",         "Germany"),
    ("Greater Munich Metropolitan Area",          "Munich",      "Bavaria",                   "Germany"),
    ("Berlin Metropolitan Area",                  "Berlin",      "Berlin",                    "Germany"),
    ("Ruhr Region",                               None,          "North Rhine-Westphalia",    "Germany"),
    ("Frankfurt Rhine-Main Metropolitan Area",    "Frankfurt",   "Hesse",                     "Germany"),
    ("Greater Hamburg Area",                      "Hamburg",     "Hamburg",                   "Germany"),
    ("Cologne Bonn Region",                       None,          "North Rhine-Westphalia",    "Germany"),
    ("Greater Nuremberg Metropolitan Area",       "Nuremberg",   "Bavaria",                   "Germany"),
    ("Greater Leipzig Area",                      "Leipzig",     "Saxony",                    "Germany"),

    # -----------------------------------------------------------------------
    # Non-German 3-part
    # -----------------------------------------------------------------------
    ("Liège, Walloon Region, Belgium",            "Liège",       "Walloon Region",            "Belgium"),
    ("Brussels, Brussels Region, Belgium",        "Brussels",    "Brussels Region",           "Belgium"),
    ("Heerlen, Limburg, Netherlands",             "Heerlen",     "Limburg",                   "Netherlands"),
    ("Vienna, Vienna, Austria",                   "Vienna",      "Vienna",                    "Austria"),

    # -----------------------------------------------------------------------
    # Non-German 2-part
    # -----------------------------------------------------------------------
    ("Washington, DC",                            "Washington",  "DC",                        None),
    ("Zurich, Switzerland",                       "Zurich",      None,                        "Switzerland"),

    # -----------------------------------------------------------------------
    # Super-regions
    # -----------------------------------------------------------------------
    ("DACH",                                      None,          None,                        "DACH"),
    ("European Union",                            None,          None,                        "European Union"),
    ("European Economic Area",                    None,          None,                        "European Economic Area"),

    # -----------------------------------------------------------------------
    # Single-part known state (no ", Germany")
    # -----------------------------------------------------------------------
    ("Saarland",                                  None,          "Saarland",                  "Germany"),

    # -----------------------------------------------------------------------
    # Bare non-German country
    # -----------------------------------------------------------------------
    ("Netherlands",                               None,          None,                        "Netherlands"),
    ("Switzerland",                               None,          None,                        "Switzerland"),

    # -----------------------------------------------------------------------
    # Edge cases: empty / None / garbage
    # -----------------------------------------------------------------------
    ("",                                          None,          None,                        None),
    ("nan",                                       None,          None,                        None),
    ("none",                                      None,          None,                        None),
])
def test_parse_location(location, exp_city, exp_state, exp_country, geo):
    result = parse_location(location, geo)
    assert result["city"] == exp_city,    f"city mismatch for {location!r}: {result['city']!r} != {exp_city!r}"
    assert result["state"] == exp_state,  f"state mismatch for {location!r}: {result['state']!r} != {exp_state!r}"
    assert result["country"] == exp_country, f"country mismatch for {location!r}: {result['country']!r} != {exp_country!r}"


# ---------------------------------------------------------------------------
# None input (not parametrized because pytest can't serialize None easily)
# ---------------------------------------------------------------------------

def test_none_input(geo):
    result = parse_location(None, geo)
    assert result == {"city": None, "state": None, "country": None}


# ---------------------------------------------------------------------------
# Return type guarantees
# ---------------------------------------------------------------------------

def test_result_has_three_keys(geo):
    result = parse_location("Munich, Bavaria, Germany", geo)
    assert set(result.keys()) == {"city", "state", "country"}


def test_unknown_fields_are_none(geo):
    """Unknown/absent fields use None (not 'NA', not NaN)."""
    test_inputs = [
        ("", {"city", "state", "country"}),            # all unknown
        ("Germany", {"city", "state"}),                # city + state unknown
        ("Bavaria, Germany", {"city"}),                # city only unknown
        ("Washington, DC", {"country"}),               # country unknown
    ]
    for inp, expected_none_keys in test_inputs:
        result = parse_location(inp, geo)
        for key in expected_none_keys:
            assert result[key] is None, (
                f"Expected {key}=None for {inp!r}, got {result[key]!r}"
            )


def test_known_fields_are_strings(geo):
    """Resolved fields are always strings, never None."""
    result = parse_location("Munich, Bavaria, Germany", geo)
    for key, val in result.items():
        assert isinstance(val, str), f"Field {key} should be str, got {type(val).__name__}: {val!r}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic(geo):
    """Same input always produces the same output."""
    loc = "Munich, Bavaria, Germany"
    assert parse_location(loc, geo) == parse_location(loc, geo)


# ---------------------------------------------------------------------------
# German state code coverage (all 16 codes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code, expected_state", [
    ("BE", "Berlin"),
    ("BY", "Bavaria"),
    ("BW", "Baden-Württemberg"),
    ("HH", "Hamburg"),
    ("HE", "Hesse"),
    ("NW", "North Rhine-Westphalia"),
    ("SN", "Saxony"),
    ("HB", "Bremen"),
    ("NI", "Lower Saxony"),
    ("RP", "Rhineland-Palatinate"),
    ("SL", "Saarland"),
    ("ST", "Saxony-Anhalt"),
    ("SH", "Schleswig-Holstein"),
    ("MV", "Mecklenburg-West Pomerania"),
    ("BB", "Brandenburg"),
    ("TH", "Thuringia"),
])
def test_state_code_mapping(code, expected_state, geo):
    result = parse_location(f"TestCity, {code}, DE", geo)
    assert result["state"] == expected_state
    assert result["country"] == "Germany"


# ---------------------------------------------------------------------------
# is_non_german() tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("location_raw, state, country, expected", [
    # Condition 1: US state codes
    ("Washington, DC",          "DC",       None,              True),
    ("Philadelphia, PA",        "PA",       None,              True),
    ("Arlington, VA",           "VA",       None,              True),
    # Condition 2: Non-German country keyword in raw
    ("Prague, Czechia",         None,       "Czech Republic",  True),
    ("Paris, France",           None,       "France",          True),
    # Condition 3: Resolved country is non-German non-None
    ("Amsterdam, Netherlands",  None,       "Netherlands",     True),
    ("Zurich, Switzerland",     None,       "Switzerland",     True),
    # Condition 4: EMEA/global keyword in raw
    ("EMEA",                    None,       None,              True),
    # Condition 5: Area pattern with None country, no German indicators
    ("Brussels Metropolitan Area", None,   None,              True),
    # Negative cases: German locations must return False
    ("Munich, Bavaria, Germany", "Bavaria", "Germany",         False),
    ("Berlin, Germany",         "Berlin",  "Germany",         False),
    ("Stuttgart Region",        "Baden-Württemberg", "Germany", False),
    ("Germany",                 None,      "Germany",         False),
])
def test_is_non_german(location_raw, state, country, expected):
    result = is_non_german(location_raw, state, country)
    assert result == expected, (
        f"is_non_german({location_raw!r}, {state!r}, {country!r}) "
        f"returned {result}, expected {expected}"
    )
