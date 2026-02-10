# NLP & Machine Learning for Information Extraction
## A Project-Based Curriculum: From Raw Job Descriptions to Structured Data

---

**Duration:** 8–10 weeks (self-paced, ~10–15 hours/week)
**Prerequisites:** Python fundamentals, pandas basics, comfort with the command line
**Project Context:** Extract structured fields (skills, seniority, salary, benefits, etc.) from 20,000 German IT job postings scraped from Indeed
**Final Deliverable:** A fully functional extraction pipeline that combines regex, NLP, NER, and ML classification

---

## Curriculum Overview

| Chapter | Title | Duration | Core Skill |
|---------|-------|----------|------------|
| 1 | Foundations: Text as Data | Week 1 | Text preprocessing, encoding, tokenization |
| 2 | Pattern Matching with Regex | Week 2 | Regular expressions for structured extraction |
| 3 | NLP Pipelines with spaCy | Week 3 | Tokenization, POS tagging, dependency parsing |
| 4 | Named Entity Recognition — Theory & Pre-trained Models | Week 4 | Understanding NER, using existing models |
| 5 | Data Annotation for Machine Learning | Week 5 | Labeling data, annotation tools, data quality |
| 6 | Training a Custom NER Model | Weeks 6–7 | Transfer learning, training loops, evaluation |
| 7 | Text Classification with scikit-learn | Week 8 | TF-IDF, classifiers, feature engineering |
| 8 | Building the Extraction Pipeline | Week 9 | Software design, pipeline composition |
| 9 | Evaluation, Validation & Iteration | Week 10 | Metrics, cross-validation, error analysis |

---

# Chapter 1: Foundations — Text as Data

**Goal:** Understand how computers represent text, why text processing is hard, and how to prepare raw text for NLP tasks.

> Before you can extract anything from a job description, you need to understand what happens to text before it reaches a model. This chapter covers the invisible work that makes everything else possible.

---

## Lesson 1.1: Character Encoding and Unicode

**Goal:** Understand why `"Straße"` sometimes becomes `"StraÃŸe"` and how to prevent it.

### Why This Matters

German job descriptions contain characters like ä, ö, ü, ß, and €. If your encoding is wrong, these characters corrupt. This breaks every downstream step — regex patterns won't match, NER models will hallucinate, and your pipeline silently produces garbage.

### Key Concepts

**ASCII** — The original 128-character encoding (A–Z, a–z, 0–9, basic punctuation). Covers English perfectly. Covers nothing else.

**UTF-8** — The modern standard. Encodes every character in every language. A single character can be 1–4 bytes long. The `€` sign is 3 bytes. The letter `A` is 1 byte. Python 3 uses UTF-8 by default, but files you read may not.

**Latin-1 (ISO 8859-1)** — Common in older European systems. Covers German characters but not `€`. You will encounter this encoding in scraped data from German websites.

### Example: Encoding in Practice

```python
# The problem: reading a file with the wrong encoding
text_bytes = "Softwareentwickler (m/w/d) – 60.000€".encode("utf-8")

# If you decode as Latin-1, the € symbol breaks
broken = text_bytes.decode("latin-1")
print(broken)  # "Softwareentwickler (m/w/d) â\x80\x93 60.000â\x82¬"

# Always specify encoding explicitly
import pandas as pd
df = pd.read_csv("jobs.csv", encoding="utf-8")
# If that fails, try:
# df = pd.read_csv("jobs.csv", encoding="latin-1")
# df = pd.read_csv("jobs.csv", encoding="cp1252")  # Windows German
```

### Practical Exercise

1. Load your job postings CSV
2. Check for encoding issues: `df["description"].str.contains("Ã").sum()` — if this returns > 0, you have encoding corruption
3. Fix any encoding issues before proceeding

### Resources

- [Joel Spolsky — "The Absolute Minimum Every Developer Must Know About Unicode"](https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/)
- [Python docs — Unicode HOWTO](https://docs.python.org/3/howto/unicode.html)

---

## Lesson 1.2: Text Normalization

**Goal:** Clean raw text into a consistent format without losing information that matters.

### Why This Matters

Scraped job descriptions are messy. They contain HTML artifacts (`&amp;`, `<br>`, `&nbsp;`), inconsistent whitespace, smart quotes, and mixed line endings. If you don't normalize first, every extractor you build will need to handle 10 variations of the same thing.

### Key Operations

**HTML cleanup** — Remove tags and decode entities.

```python
import re
from html import unescape

def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = unescape(text)                          # &amp; → &, &nbsp; → space
    text = re.sub(r"<br\s*/?>", "\n", text)        # <br> → newline (preserve structure)
    text = re.sub(r"<[^>]+>", "", text)            # strip remaining tags
    return text

raw = "<p>Wir suchen einen <b>Python</b>&ndash;Entwickler</p>"
print(clean_html(raw))
# Output: "Wir suchen einen Python–Entwickler"
```

**Whitespace normalization** — Collapse multiple spaces, fix line endings.

```python
def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    text = text.replace("\r\n", "\n")              # Windows → Unix line endings
    text = re.sub(r"[ \t]+", " ", text)            # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)         # max 2 consecutive newlines
    return text.strip()
```

**Unicode normalization** — Handle the fact that `ü` can be stored as one character (ü) or two characters (u + combining ¨). This matters for string comparison and regex matching.

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form (composed characters)."""
    return unicodedata.normalize("NFC", text)

# These look identical but are different bytes:
a = "ü"                              # single codepoint U+00FC
b = "u\u0308"                        # u + combining diaeresis
print(a == b)                        # False!
print(normalize_unicode(a) == normalize_unicode(b))  # True
```

### Anti-Pattern: Over-Cleaning

Do NOT lowercase everything at this stage. Do NOT remove punctuation. Do NOT strip numbers. That information is critical for downstream extraction — salary patterns need numbers, skill names need capitalization, section headers need punctuation.

> **Rule:** Normalize formatting, preserve content. You can always strip more later; you can never recover what you deleted.

### Practical Exercise

1. Write a `clean_description(text)` function that chains the three operations above
2. Apply it to your entire DataFrame: `df["description_clean"] = df["description"].apply(clean_description)`
3. Spot-check 20 rows — do German characters look correct? Are section headers still intact? Are bullet points preserved?

### Resources

- [Python `re` module documentation](https://docs.python.org/3/library/re.html)
- [Unicode normalization forms explained](https://unicode.org/reports/tr15/)

---

## Lesson 1.3: Tokenization — Splitting Text into Units

**Goal:** Understand what tokens are, why tokenization is non-trivial, and how different tokenizers make different decisions.

### Why This Matters

Every NLP model processes text as a sequence of **tokens**, not characters. A token is the atomic unit the model sees. How you split text into tokens fundamentally changes what patterns the model can learn.

### Three Levels of Tokenization

**Word-level:** Split on whitespace and punctuation.

```python
text = "Python-Entwickler mit 3+ Jahren Erfahrung"

# Naive: split on whitespace
tokens_naive = text.split()
# ["Python-Entwickler", "mit", "3+", "Jahren", "Erfahrung"]

# Problem: "Python-Entwickler" — is that one token or two?
# Problem: "3+" — is the "+" part of the number or punctuation?
```

**Subword-level:** Split words into smaller pieces. This is what transformer models (BERT, GPT) use. It handles unknown words by breaking them into known subwords.

```
"Kubernetes" → ["Kub", "erne", "tes"]
"Python-Entwickler" → ["Python", "-", "Ent", "wick", "ler"]
```

**Character-level:** Every character is a token. Handles any text but loses word-level meaning.

### spaCy Tokenization (What You'll Actually Use)

```python
import spacy
nlp = spacy.load("de_core_news_sm")

doc = nlp("Python-Entwickler mit 3+ Jahren C++-Erfahrung")
for token in doc:
    print(f"  '{token.text}':12s  is_alpha={token.is_alpha}  is_punct={token.is_punct}  like_num={token.like_num}")

# Output:
#   'Python'       is_alpha=True   is_punct=False  like_num=False
#   '-'            is_alpha=False  is_punct=True   like_num=False
#   'Entwickler'   is_alpha=True   is_punct=False  like_num=False
#   'mit'          is_alpha=True   is_punct=False  like_num=False
#   '3'            is_alpha=False  is_punct=False  like_num=True
#   '+'            is_alpha=False  is_punct=True   like_num=False
#   'Jahren'       is_alpha=True   is_punct=False  like_num=False
#   'C++'          is_alpha=False  is_punct=False  like_num=False  ← spaCy knows C++ is one token!
#   '-'            is_alpha=False  is_punct=True   like_num=False
#   'Erfahrung'    is_alpha=True   is_punct=False  like_num=False
```

Notice that spaCy correctly keeps "C++" as one token. It has special rules for known patterns. This is why you use a real tokenizer instead of `.split()`.

### Practical Exercise

1. Install spaCy and download the German model: `python -m spacy download de_core_news_sm`
2. Tokenize 10 job descriptions from your dataset
3. Find 3 cases where tokenization does something unexpected with German compound words or technical terms
4. Write down what you observe — this understanding will matter when you build your NER system

### Resources

- [spaCy 101 — Tokenization](https://spacy.io/usage/spacy-101#annotations-token)
- [Hugging Face — Tokenizers summary](https://huggingface.co/docs/transformers/tokenizer_summary)

---

## Lesson 1.4: Language Detection

**Goal:** Automatically identify whether a job description is in German, English, or a mix of both.

### Why This Matters

Your dataset has German and English job descriptions (and some that mix both). German and English have different sentence structures, different regex patterns for the same concept ("Vollzeit" vs "Full-time"), and different NLP models. You need to know the language before routing text through the right extractor.

### Approach: Use `lingua-py`

```bash
pip install lingua-language-detector
```

```python
from lingua import Language, LanguageDetectorBuilder

# Build detector for the languages you expect
detector = LanguageDetectorBuilder.from_languages(
    Language.GERMAN, Language.ENGLISH
).with_minimum_relative_distance(0.1).build()

def detect_language(text: str) -> str:
    """Detect language of text. Returns 'de', 'en', or 'mixed'."""
    if not text or len(text.strip()) < 20:
        return "unknown"
    
    # Check overall detection
    lang = detector.detect_language_of(text)
    if lang is None:
        return "unknown"
    
    # Check for mixed language by looking at confidence
    confidences = detector.compute_language_confidence_values(text)
    conf_dict = {c.language.name: c.value for c in confidences}
    
    de_conf = conf_dict.get("GERMAN", 0)
    en_conf = conf_dict.get("ENGLISH", 0)
    
    # If both languages have significant confidence, it's mixed
    if de_conf > 0.3 and en_conf > 0.3:
        return "mixed"
    
    return "de" if lang == Language.GERMAN else "en"

# Test
print(detect_language("Wir suchen einen erfahrenen Python-Entwickler"))  # "de"
print(detect_language("We are looking for a senior Python developer"))   # "en"
print(detect_language("Wir suchen a Senior Python Developer mit Erfahrung"))  # "mixed"
```

### Practical Exercise

1. Run language detection on your entire dataset
2. Create a distribution: how many DE, EN, and mixed descriptions do you have?
3. Save the `language` column — you'll use it in every subsequent chapter

### Resources

- [lingua-py GitHub](https://github.com/pemistahl/lingua-py)
- [Alternative: langdetect](https://github.com/Mimino666/langdetect)

---

## Chapter 1 Checkpoint

By the end of this chapter, you should have:

- [ ] A cleaned DataFrame with `description_clean` and `language` columns
- [ ] An understanding of UTF-8 encoding and why it matters
- [ ] The ability to tokenize German text with spaCy
- [ ] A language distribution of your dataset (% German, % English, % mixed)

---

# Chapter 2: Pattern Matching with Regular Expressions

**Goal:** Build regex-based extractors for fields that have predictable patterns, and learn when regex is the right tool versus when it isn't.

> Regex is the workhorse of text extraction. For fields with finite vocabularies (contract type, work modality) or numeric patterns (salary, years of experience), regex is faster, cheaper, and more reliable than any ML model. The trick is knowing where to stop.

---

## Lesson 2.1: Regex Fundamentals for German Text

**Goal:** Master the regex patterns you'll need for bilingual (German/English) extraction.

### Key Regex Features You'll Use

**Word boundaries `\b`** — Match the edge of a word. Critical for avoiding partial matches.

```python
import re

text = "Teilzeit und Vollzeit"

# Without word boundary: "Zeit" matches inside both words
re.findall(r"Zeit", text)               # ["Zeit", "Zeit"] — wrong

# With word boundary: only matches standalone words
re.findall(r"\bVollzeit\b", text)        # ["Vollzeit"] — correct
```

**Named groups `(?P<name>...)`** — Extract specific parts of a match by name.

```python
pattern = r"(?P<min>\d{2,3})[.,](?P<thousands>\d{3})\s*[€EUR]"
match = re.search(pattern, "Gehalt: 65.000€ brutto")
if match:
    salary = int(match.group("min") + match.group("thousands"))
    print(f"Salary: €{salary:,}")  # Salary: €65,000
```

**`re.IGNORECASE`** — Match regardless of case. Essential because job postings are inconsistent with capitalization.

```python
# "remote", "Remote", "REMOTE" all match
re.findall(r"\bremote\b", "Remote work möglich", re.IGNORECASE)
```

**Non-capturing groups `(?:...)`** — Group patterns without capturing them. Useful for alternations.

```python
# Match "Full-time" or "Full time" or "Fulltime"
pattern = r"\bFull[\s-]?time\b"

# Match German OR English variants
pattern = r"\b(?:Vollzeit|Full[\s-]?time)\b"
```

### Anti-Pattern: Over-Complex Regex

If your regex pattern is longer than ~100 characters, you're probably trying to do too much with regex. Split it into multiple simpler patterns or switch to an NLP approach.

```python
# BAD: one massive regex that tries to handle everything
pattern = r"(?:(?:(?:mind(?:estens)?|wenigstens|ab|über|mehr\s+als)\s+)?(\d{1,2})(?:\s*(?:\+|oder\s+mehr))?\s*(?:Jahre?|years?|J\.)\s*(?:Berufs)?(?:erfahrung|experience|Praxis))"

# GOOD: multiple simple patterns, tried in order
EXPERIENCE_PATTERNS = [
    r"(\d{1,2})\+?\s*(?:Jahre|years)\s*(?:Berufserfahrung|experience)",
    r"(?:mindestens|mind\.)\s*(\d{1,2})\s*Jahre",
    r"(\d{1,2})\s*(?:years?|Jahre)\s*(?:of\s+)?experience",
]

def extract_experience(text):
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None
```

### Practical Exercise

1. Open 10 job descriptions from your dataset
2. For each one, manually identify: contract type, work modality, salary, and experience years
3. Write down the exact text that contains each piece of information
4. Note the patterns — this becomes the basis for your regex patterns in the next lesson

### Resources

- [regex101.com](https://regex101.com/) — interactive regex tester (select Python flavor)
- [Python `re` documentation](https://docs.python.org/3/library/re.html)
- [Regular Expressions Cheat Sheet (DataCamp)](https://www.datacamp.com/cheat-sheet/regular-expresso)

---

## Lesson 2.2: Building Extractors Field by Field

**Goal:** Build and test regex extractors for the 5 fields where regex works well.

### Structure: One Function Per Field

Each extractor follows the same pattern:

```
1. Define patterns (German + English variants)
2. Try each pattern in order (most specific first)
3. Return the first match, or None
4. Include the raw matched text for verification
```

### Example: Contract Type Extractor

```python
from typing import Optional
import re

CONTRACT_PATTERNS = {
    "Full-time": [
        r"\bVollzeit\b",
        r"\bFull[\s-]?time\b",
        r"\bGanztags\b",
    ],
    "Part-time": [
        r"\bTeilzeit\b",
        r"\bPart[\s-]?time\b",
    ],
    "Mini-Job": [
        r"\bMini[\s-]?[Jj]ob\b",
        r"\b(?:450|520)[\s-]?Euro[\s-]?(?:Job|Basis)\b",
        r"\b[Gg]eringfügig\b",
    ],
    "Freelance": [
        r"\b(?:Freelance|Freiberuflich)\b",
    ],
}

def extract_contract_type(text: str) -> dict:
    """
    Extract contract type from job description.
    
    Returns:
        dict with keys 'value' (normalized label or None) and 'raw' (matched text or None)
    """
    if not text:
        return {"value": None, "raw": None}
    
    for label, patterns in CONTRACT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {"value": label, "raw": match.group(0)}
    
    return {"value": None, "raw": None}

# Test
assert extract_contract_type("Diese Stelle ist in Vollzeit zu besetzen")["value"] == "Full-time"
assert extract_contract_type("Part-time position available")["value"] == "Part-time"
assert extract_contract_type("No info here")["value"] is None
```

### Example: Salary Extractor (German Number Formatting)

This is harder because German uses `.` as thousands separator and `,` as decimal separator — opposite to English.

```python
def extract_salary(text: str) -> dict:
    """
    Extract salary range from job description.
    
    Handles:
        - German: "60.000 - 80.000 €"
        - English: "€60,000 - €80,000"  
        - Shorthand: "60k - 80k"
        - Monthly: "5.000€/Monat" (converts to annual × 12)
    
    Returns:
        dict with keys: min, max, raw, period
    """
    if not text:
        return {"min": None, "max": None, "raw": None, "period": None}
    
    # Pattern 1: German format "60.000 - 80.000 €"
    match = re.search(
        r"(\d{2,3})[.](\d{3})\s*(?:[€EUR]?\s*[-–bis]+\s*[€EUR]?\s*(\d{2,3})[.](\d{3}))?\s*[€EUR]",
        text, re.IGNORECASE
    )
    if match:
        salary_min = int(match.group(1) + match.group(2))
        salary_max = int(match.group(3) + match.group(4)) if match.group(3) else salary_min
        raw = match.group(0)
        
        # Detect if monthly
        is_monthly = bool(re.search(r"(?:pro\s*Monat|/\s*Monat|monthly|mtl)", text[match.end():match.end()+30], re.IGNORECASE))
        period = "monthly" if is_monthly else "annual"
        
        if is_monthly:
            salary_min *= 12
            salary_max *= 12
        
        return {"min": salary_min, "max": salary_max, "raw": raw, "period": period}
    
    # Pattern 2: Shorthand "60k - 80k"
    match = re.search(
        r"(\d{2,3})\s*[kK]\s*(?:[€EUR]?\s*[-–bis]+\s*[€EUR]?\s*(\d{2,3})\s*[kK])?\s*[€EUR]?",
        text
    )
    if match:
        salary_min = int(match.group(1)) * 1000
        salary_max = int(match.group(2)) * 1000 if match.group(2) else salary_min
        return {"min": salary_min, "max": salary_max, "raw": match.group(0), "period": "annual"}
    
    return {"min": None, "max": None, "raw": None, "period": None}

# Test
result = extract_salary("Gehalt: 60.000 - 80.000 € brutto/Jahr")
assert result["min"] == 60000
assert result["max"] == 80000

result = extract_salary("Salary: 65k€")
assert result["min"] == 65000
```

### Your Task: Build the Remaining Extractors

Using the same pattern, build extractors for:

| Field | Key Patterns to Handle |
|-------|----------------------|
| `work_modality` | "Remote", "Hybrid", "Homeoffice", "Vor Ort", "Onsite", "Büro", "100% remote" |
| `experience_years` | "3+ Jahre", "mind. 5 Jahre Berufserfahrung", "3-5 years experience", "mehrjährige Erfahrung" |
| `languages_required` | "Deutsch C1", "fließend Deutsch", "English fluent", "Englisch B2", "Deutschkenntnisse" |

---

## Lesson 2.3: Testing Your Extractors

**Goal:** Write automated tests so that when you change a pattern, you immediately know if you broke something.

### Why Testing Matters

You will iterate on your regex patterns dozens of times. Without tests, you'll fix one pattern and unknowingly break another. Automated tests catch this instantly.

```python
# tests/test_extractors.py
import pytest
from src.extractors import extract_contract_type, extract_salary, extract_experience

# ── Contract Type Tests ─────────────────────────────────
@pytest.mark.parametrize("text, expected", [
    # German
    ("Vollzeit-Stelle in München", "Full-time"),
    ("Teilzeit möglich (20h/Woche)", "Part-time"),
    ("450-Euro-Job an der Kasse", "Mini-Job"),
    # English
    ("Full-time position", "Full-time"),
    ("Part time available", "Part-time"),
    # Edge cases
    ("", None),                                     # empty string
    ("Competitive compensation package", None),     # no contract info
    ("Vollzeit oder Teilzeit", "Full-time"),         # takes first match
])
def test_contract_type(text, expected):
    result = extract_contract_type(text)
    assert result["value"] == expected, f"For '{text}': expected {expected}, got {result['value']}"


# ── Salary Tests ────────────────────────────────────────
@pytest.mark.parametrize("text, expected_min, expected_max", [
    ("60.000 - 80.000 €", 60000, 80000),
    ("Gehalt: 65.000€", 65000, 65000),
    ("65k - 85k EUR", 65000, 85000),
    ("No salary mentioned", None, None),
])
def test_salary(text, expected_min, expected_max):
    result = extract_salary(text)
    assert result["min"] == expected_min
    assert result["max"] == expected_max
```

Run with: `pytest tests/ -v`

### Practical Exercise

1. Write at least 10 test cases per extractor (5 positive matches, 3 edge cases, 2 expected `None`)
2. Include real examples from your dataset — copy-paste actual text snippets
3. Run `pytest` after every pattern change

### Resources

- [pytest documentation](https://docs.pytest.org/en/stable/)
- [pytest parametrize guide](https://docs.pytest.org/en/stable/how-to/parametrize.html)

---

## Lesson 2.4: Running the Regex Layer and Measuring Your Baseline

**Goal:** Apply all extractors to your full dataset and measure coverage per field.

```python
import pandas as pd
from tqdm import tqdm

def run_regex_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all regex extractors to the dataset."""
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Regex extraction"):
        text = row["description_clean"]
        title = row.get("title", "")
        
        contract = extract_contract_type(text)
        salary = extract_salary(text)
        experience = extract_experience(text)
        modality = extract_work_modality(text)
        languages = extract_languages(text)
        
        results.append({
            "contract_type": contract["value"],
            "contract_raw": contract["raw"],
            "salary_min": salary["min"],
            "salary_max": salary["max"],
            "salary_raw": salary["raw"],
            "experience_years": experience,
            "work_modality": modality["value"],
            "languages_required": languages,
        })
    
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

# Run it
df_enriched = run_regex_layer(df)

# ── Measure coverage ────────────────────────────────────
print("=== Regex Layer Coverage ===")
for col in ["contract_type", "work_modality", "salary_min", "experience_years"]:
    non_null = df_enriched[col].notna().sum()
    pct = non_null / len(df_enriched) * 100
    print(f"  {col:25s}: {non_null:6d} / {len(df_enriched)} ({pct:.1f}%)")
```

### What to Expect

| Field | Typical Coverage | Notes |
|-------|-----------------|-------|
| `contract_type` | 50–70% | Many postings don't explicitly state this |
| `work_modality` | 40–60% | Growing trend to include, but not universal |
| `salary_min` | 10–25% | German postings rarely include salary |
| `experience_years` | 30–50% | Often vague ("mehrjährige" without a number) |
| `languages_required` | 40–60% | Usually mentioned if German is required |

**This is your baseline. Write these numbers down. Every improvement from NLP/NER gets measured against them.**

---

## Chapter 2 Checkpoint

By the end of this chapter, you should have:

- [ ] Working extractors for 5 fields (contract, modality, salary, experience, languages)
- [ ] pytest test suite with 10+ tests per extractor
- [ ] Coverage percentages for each field across your full dataset
- [ ] An understanding of where regex works well and where it hits its limits

---

# Chapter 3: NLP Pipelines with spaCy

**Goal:** Understand how NLP pipelines process text, what linguistic annotations they produce, and how to use those annotations for information extraction.

> spaCy is the production NLP library. Not the most cutting-edge, but the most practical. Learning spaCy teaches you the fundamental NLP concepts (tokenization, POS tagging, dependency parsing, NER) in a framework you can actually ship.

---

## Lesson 3.1: The spaCy Processing Pipeline

**Goal:** Understand what happens when you call `nlp(text)` — every component in the pipeline and what it adds.

### Setup

```bash
pip install spacy
python -m spacy download de_core_news_lg    # German, large (best accuracy)
python -m spacy download en_core_web_lg     # English, large
```

### The Pipeline

When you call `nlp(text)`, your text flows through a series of components:

```
Raw Text → Tokenizer → Tagger → Parser → NER → Doc
              ↓            ↓         ↓        ↓
           Token      POS tags   Dep tree  Entities
           objects    (NOUN,     (subject,  (ORG,
                      VERB...)   object...) PER...)
```

```python
import spacy

nlp = spacy.load("de_core_news_lg")

# See what's in the pipeline
print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'morphologizer', 'parser', 'lemmatizer', 'attribute_ruler', 'ner']

text = "SAP sucht einen Senior Python-Entwickler in Berlin mit 5 Jahren Erfahrung."
doc = nlp(text)

# Each token has been enriched with linguistic information
for token in doc:
    print(f"  {token.text:20s}  POS={token.pos_:6s}  TAG={token.tag_:8s}  "
          f"DEP={token.dep_:12s}  HEAD={token.head.text}")
```

Output (simplified):
```
  SAP                   POS=PROPN   TAG=NE       DEP=sb           HEAD=sucht
  sucht                 POS=VERB    TAG=VVFIN    DEP=ROOT         HEAD=sucht
  einen                 POS=DET     TAG=ART      DEP=nk           HEAD=Entwickler
  Senior                POS=PROPN   TAG=NE       DEP=nk           HEAD=Entwickler
  Python-Entwickler     POS=NOUN    TAG=NN       DEP=oa           HEAD=sucht
  in                    POS=ADP     TAG=APPR     DEP=mnr          HEAD=Entwickler
  Berlin                POS=PROPN   TAG=NE       DEP=nk           HEAD=in
  mit                   POS=ADP     TAG=APPR     DEP=mnr          HEAD=Entwickler
  5                     POS=NUM     TAG=CARD     DEP=nk           HEAD=Jahren
  Jahren                POS=NOUN    TAG=NN       DEP=nk           HEAD=mit
  Erfahrung             POS=NOUN    TAG=NN       DEP=nk           HEAD=Jahren
  .                     POS=PUNCT   TAG=$.       DEP=punct        HEAD=sucht
```

### Key Annotations Explained

**POS (Part-of-Speech):** What grammatical role the word plays.
- `NOUN` — common noun ("Entwickler", "Erfahrung")
- `PROPN` — proper noun ("SAP", "Berlin") — your skill candidates often show up here
- `VERB` — verb ("sucht")
- `NUM` — number ("5") — useful for experience extraction
- `ADJ` — adjective ("Senior") — seniority indicators

**DEP (Dependency):** How words relate to each other in the sentence.
- `sb` — subject ("SAP" is the subject of "sucht")
- `oa` — accusative object ("Entwickler" is what SAP is looking for)
- `nk` — noun kernel (modifiers within a noun phrase)

**Lemma:** The base form of a word. "Jahren" → "Jahr", "sucht" → "suchen".

### Practical Exercise

1. Process 5 job descriptions through spaCy's German pipeline
2. For each one, print all tokens with their POS and DEP tags
3. Identify: which POS tags do skill names typically have? Which POS tags do benefit words have?
4. Run `spacy.explain("nk")` for any dependency labels you don't understand

### Resources

- [spaCy 101: Everything You Need to Know](https://spacy.io/usage/spacy-101)
- [spaCy Linguistic Features](https://spacy.io/usage/linguistic-features)
- [spaCy German model details](https://spacy.io/models/de)

---

## Lesson 3.2: Noun Chunks and Phrase Extraction

**Goal:** Use spaCy's noun chunk detection to extract candidate skill phrases.

### What Are Noun Chunks?

Noun chunks are "base noun phrases" — flat phrases that have a noun as their head. They're useful because skills and technologies are usually noun phrases.

```python
doc = nlp("Fundierte Kenntnisse in Python und Erfahrung mit Cloud-Infrastruktur wie AWS oder Azure")

print("=== Noun Chunks ===")
for chunk in doc.noun_chunks:
    print(f"  '{chunk.text}' (root: {chunk.root.text}, root_dep: {chunk.root.dep_})")

# Output:
#   'Fundierte Kenntnisse' (root: Kenntnisse, root_dep: sb)
#   'Python' (root: Python, root_dep: nk)
#   'Erfahrung' (root: Erfahrung, root_dep: cj)
#   'Cloud-Infrastruktur' (root: Cloud-Infrastruktur, root_dep: nk)
#   'AWS' (root: AWS, root_dep: nk)
#   'Azure' (root: Azure, root_dep: cj)
```

Notice: "Python", "AWS", and "Azure" are already isolated as noun chunks. This is a quick-and-dirty way to get skill candidates without any ML.

### Using Noun Chunks + POS Filtering for Skill Candidates

```python
def get_skill_candidates(doc) -> list[str]:
    """Extract potential skill terms using noun chunks and POS filtering."""
    candidates = set()
    
    # Strategy 1: Noun chunks that look like technologies
    for chunk in doc.noun_chunks:
        # Single proper nouns are strong candidates (Python, Docker, AWS)
        if chunk.root.pos_ == "PROPN" and len(chunk.text.split()) <= 3:
            candidates.add(chunk.text)
    
    # Strategy 2: Individual PROPN tokens not in chunks
    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop:
            candidates.add(token.text)
    
    return sorted(candidates)

doc = nlp("Erfahrung mit Python, Docker, Kubernetes und AWS. Gute SQL-Kenntnisse.")
print(get_skill_candidates(doc))
# ['AWS', 'Docker', 'Kubernetes', 'Python', 'SQL']
```

**Limitation:** This catches many skill names but also catches company names, cities, and people's names. It's a recall-optimized approach — it finds a lot, but needs filtering. That's where NER becomes valuable.

### Practical Exercise

1. Run noun chunk extraction on 20 job descriptions
2. Manually check: what percentage of noun chunks are actual skills vs. noise?
3. This accuracy percentage is your NLP-baseline for skills — compare it to your regex keyword-list accuracy from Chapter 2

---

## Lesson 3.3: Pre-trained NER — What spaCy Already Knows

**Goal:** Understand what spaCy's built-in NER detects and — critically — what it misses.

```python
nlp = spacy.load("de_core_news_lg")
doc = nlp("SAP in Walldorf sucht einen Python-Entwickler. Gehalt ab 65.000€.")

print("=== Named Entities ===")
for ent in doc.ents:
    print(f"  {ent.text:25s} → {ent.label_:6s} ({spacy.explain(ent.label_)})")

# Output:
#   SAP                       → ORG    (Companies, agencies, institutions)
#   Walldorf                  → LOC    (Non-GPE locations)
#   65.000€                   → MONEY  (Monetary values)
```

### What spaCy's German Model Recognizes

| Label | Meaning | Example |
|-------|---------|---------|
| `PER` | Person names | "Max Mustermann" |
| `ORG` | Organizations | "SAP", "Deutsche Bank" |
| `LOC` | Locations | "Berlin", "Walldorf" |
| `MISC` | Miscellaneous | Nationalities, events |

### What spaCy Does NOT Recognize

**Skills, technologies, programming languages, frameworks, tools.** spaCy has no `SKILL` entity type. "Python" might get tagged as `MISC` or not tagged at all. "Kubernetes" will likely be missed entirely.

This is exactly why you need to train a custom NER model (Chapter 6). The pre-trained model gives you companies and locations for free, but you have to teach it what a "skill" is.

### Practical Exercise

1. Run spaCy NER on 20 of your job descriptions
2. List every entity it finds and its type
3. List every skill/technology that it MISSES
4. Calculate: what percentage of skills does the pre-trained model catch?

This gap — between what spaCy's NER catches and what you need — is what you'll close in Chapters 5 and 6.

---

## Lesson 3.4: Section Splitting with NLP

**Goal:** Split job descriptions into labeled sections (requirements, tasks, benefits, about) using a combination of regex headers and NLP-based classification.

This lesson was covered in detail in our earlier conversation (Phase 3). Revisit and implement the `split_sections()` and `classify_section_by_content()` functions.

### Key Learning

The section splitter teaches you a fundamental pattern in production NLP: **rules first, ML as fallback.** When a clear header exists ("Anforderungen"), use regex. When no header exists, use linguistic features (keyword density, POS distribution) to classify. This hybrid approach is how most real systems work.

### Practical Exercise

1. Implement the section splitter
2. Run it on your full dataset
3. Measure: what percentage of descriptions have detectable section headers?
4. For headerless descriptions, does the keyword-based classifier correctly identify sections?

---

## Chapter 3 Checkpoint

By the end of this chapter, you should have:

- [ ] A working understanding of the spaCy pipeline components
- [ ] The ability to use POS tags, dependency parsing, and noun chunks
- [ ] An understanding of what pre-trained NER catches and what it misses
- [ ] A working section splitter for your job descriptions
- [ ] A clear understanding of WHY you need a custom NER model for skill extraction

### Resources

- [spaCy official course (free)](https://course.spacy.io/en/) — **Highly recommended. Do this entire course.**
- [spaCy API documentation](https://spacy.io/api)
- [spaCy Universe — community projects](https://spacy.io/universe)
- [Introduction to Named Entity Recognition (Python Humanities)](https://ner.pythonhumanities.com/) — excellent free textbook

---

# Chapter 4: Named Entity Recognition — Theory & Pre-trained Models

**Goal:** Understand how NER works conceptually, the difference between rule-based and ML-based NER, and how to use pre-trained transformer models for skill extraction.

> This is the heart of the curriculum. NER is the technique that will transform your job descriptions from unstructured text into structured skill lists. Understanding how it works — not just how to call it — is what separates someone who uses tools from someone who can build systems.

---

## Lesson 4.1: How NER Works — From Rules to Neural Networks

**Goal:** Understand the three generations of NER systems and what makes each one tick.

### Generation 1: Rule-Based NER (Gazetteers)

The simplest NER: maintain a dictionary of known entities and look them up.

```
Dictionary: {Python, Java, Docker, Kubernetes, AWS, ...}
Text: "Experience with Python and Docker"
Output: [Python=SKILL, Docker=SKILL]
```

**Pros:** Fast, predictable, no training needed, 100% precision on known terms.
**Cons:** Zero recall on unknown terms. Misses "Golang" if it's not in your dictionary. Doesn't handle context — tags "Java" in "Went to Java for vacation."

This is what spaCy's `EntityRuler` does:

```python
import spacy

nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")

patterns = [
    {"label": "SKILL", "pattern": "Python"},
    {"label": "SKILL", "pattern": "Docker"},
    {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
]
ruler.add_patterns(patterns)

doc = nlp("Experience with Python and Machine Learning")
for ent in doc.ents:
    print(f"  {ent.text} → {ent.label_}")
# Python → SKILL
# Machine Learning → SKILL
```

### Generation 2: Statistical NER (CRF, HMM)

Instead of looking up words in a dictionary, look at **features** of each word and its neighbors:

- Is the word capitalized? (Capital letters suggest a proper noun)
- What are the words before and after it? (Context matters: "experience with X" suggests X is a skill)
- What is the POS tag? (PROPN is more likely to be an entity than VERB)
- What shape does it have? (CamelCase like "JavaScript" suggests a technology)

A **Conditional Random Field (CRF)** or **Hidden Markov Model (HMM)** learns these patterns from labeled training data.

**Pros:** Handles unseen entities if they have similar features. Considers context.
**Cons:** Requires feature engineering. Limited by the features you define.

### Generation 3: Neural NER (Transformers)

Modern NER uses transformer models (BERT, RoBERTa, etc.) that learn contextual representations of words. Instead of hand-crafted features, the model learns what features matter directly from data.

```
Text: "Experience with Python and Docker"
                       ↓
            Transformer Encoder
            (BERT / RoBERTa)
                       ↓
      Context-aware embeddings for each token
                       ↓
         Token Classification Layer
                       ↓
Output: [O, O, O, B-SKILL, O, B-SKILL]
```

The key innovation: the model sees "Python" differently in "Python programming" vs "Monty Python" because the transformer encodes the entire surrounding context.

**Pros:** State-of-the-art accuracy. Handles context. Generalizes to unseen entities.
**Cons:** Requires GPU for reasonable speed. Needs labeled training data. Black box.

### The IOB Tagging Scheme

All NER systems use **Inside-Outside-Beginning** tags:

```
Token:   Experience  with  Python  and  Machine  Learning
IOB Tag: O           O     B-SKILL O    B-SKILL  I-SKILL
```

- `B-SKILL` = **Beginning** of a SKILL entity
- `I-SKILL` = **Inside** a SKILL entity (continuation)
- `O` = **Outside** any entity

Multi-word entities like "Machine Learning" get `B-SKILL I-SKILL`. This lets the model handle entities of any length.

### Practical Exercise

1. Manually IOB-tag 5 sentences from your job descriptions
2. Notice the ambiguities: is "Python-Entwickler" one entity (B-SKILL I-SKILL) or just "Python" (B-SKILL)?
3. Decide on your annotation guidelines BEFORE you start labeling (Chapter 5)

### Resources

- [Introduction to NER — Python Humanities (free textbook)](https://ner.pythonhumanities.com/)
- [spaCy — Named Entity Recognition 101](https://spacy.io/usage/linguistic-features#named-entities)
- [Stanford NLP — NER explanation](https://nlp.stanford.edu/software/CRF-NER.html)

---

## Lesson 4.2: Using Pre-trained Transformer NER Models

**Goal:** Load and run a pre-trained skill extraction model (JobBERT) and critically evaluate its performance on your data.

### Setup

```bash
pip install transformers torch
```

### Loading JobBERT

```python
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load the model
ner = pipeline(
    "ner",
    model="jjzha/jobbert_skill_extraction",
    aggregation_strategy="simple",  # merges subword tokens
    device=device,
)
```

### Understanding the Output

```python
text = "We need experience with Python, Kubernetes, and CI/CD pipelines."
results = ner(text)

for entity in results:
    print(f"  Entity: {entity['word']:25s}  "
          f"Label: {entity['entity_group']:10s}  "
          f"Score: {entity['score']:.3f}  "
          f"Start: {entity['start']}  End: {entity['end']}")

# Output:
#   Entity: Python                     Label: Skill       Score: 0.943  Start: 28  End: 34
#   Entity: Kubernetes                 Label: Skill       Score: 0.912  Start: 36  End: 46
#   Entity: CI/CD pipelines            Label: Skill       Score: 0.867  Start: 52  End: 68
```

Each result gives you:
- `word`: the detected entity text
- `entity_group`: the label (always "Skill" for this model)
- `score`: **confidence** — how sure the model is (0.0 to 1.0)
- `start`/`end`: character positions in the original text

### The Critical Test: German Text

```python
# English text — model was trained on this
text_en = "Strong knowledge of Docker, AWS, and Terraform required."
results_en = ner(text_en)

# German text — model was NOT trained on this
text_de = "Fundierte Kenntnisse in Docker, AWS und Terraform erforderlich."
results_de = ner(text_de)

print("=== English Results ===")
for e in results_en:
    print(f"  {e['word']:20s} score={e['score']:.3f}")

print("\n=== German Results ===")
for e in results_de:
    print(f"  {e['word']:20s} score={e['score']:.3f}")

# Compare: does the model find the same skills in both languages?
# Are confidence scores lower for German text?
```

### Practical Exercise: The 50-Sample Evaluation

This is the most important exercise in the entire curriculum. It tells you whether this model is worth using.

```python
import pandas as pd
import json

# 1. Sample 50 diverse descriptions (mix of German and English)
sample = df.sample(50, random_state=42)

# 2. Run NER on each
all_results = []
for idx, row in sample.iterrows():
    text = row["description_clean"][:1000]  # first 1000 chars
    results = ner(text)
    all_results.append({
        "index": idx,
        "language": row["language"],
        "title": row["title"],
        "ner_skills": [r["word"] for r in results if r["score"] > 0.4],
        "ner_details": results,
    })

# 3. Save for manual review
with open("data/validation/ner_50_sample.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# 4. NOW: open this file and manually check each row
#    For each posting, write down:
#    - Skills the model found correctly (true positives)
#    - Skills the model found that aren't skills (false positives)
#    - Skills the model missed (false negatives)
```

**This manual evaluation is non-negotiable.** Don't skip it. Don't automate it. Read each description with your own eyes and judge whether the model is doing a good job.

### Resources

- [Hugging Face — Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- [JobBERT model card](https://huggingface.co/jjzha/jobbert_skill_extraction)
- [Hugging Face NER pipeline documentation](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline)

---

## Lesson 4.3: Confidence Thresholds — Precision vs. Recall

**Goal:** Understand the precision-recall tradeoff and how confidence thresholds control it.

### The Tradeoff

Every NER prediction comes with a confidence score. You choose a threshold: only accept predictions above it.

- **Low threshold (0.3):** You accept almost everything. High **recall** (you find most skills) but low **precision** (you also tag garbage as skills).
- **High threshold (0.7):** You only accept high-confidence predictions. High **precision** (what you tag is correct) but low **recall** (you miss many real skills).

```
Precision = True Positives / (True Positives + False Positives)
            "Of everything I tagged, how much was correct?"

Recall    = True Positives / (True Positives + False Negatives)
            "Of all real skills, how many did I find?"

F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
            "Harmonic mean — balances both"
```

### Hands-On: Sweeping Thresholds on Your 50 Samples

Use the results from Lesson 4.2's manual evaluation:

```python
def evaluate_at_threshold(ner_results, manual_labels, threshold):
    """
    ner_results: list of {word, score} dicts from model
    manual_labels: set of strings — the skills you manually identified
    """
    predicted = {r["word"].lower().strip() for r in ner_results if r["score"] >= threshold}
    actual = {s.lower().strip() for s in manual_labels}
    
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}

# Sweep
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    metrics = evaluate_at_threshold(ner_output, gold_skills, t)
    print(f"  t={t:.1f}  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  "
          f"F1={metrics['f1']:.2f}  (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})")
```

### Decision Point

After this exercise, you'll know:
- The best threshold for your data (probably 0.3–0.5)
- Whether JobBERT works well on German text (it might not)
- Whether the model adds meaningful value over a simple keyword list

If the pre-trained model's F1 on German text is below 0.50, it's not worth using. Fall back to keyword lists and invest your time in training a custom model (Chapter 6).

---

## Chapter 4 Checkpoint

By the end of this chapter, you should have:

- [ ] An understanding of IOB tagging and how NER models work internally
- [ ] A working EntityRuler (dictionary-based NER) with ~200 IT skills
- [ ] Experience running a pre-trained transformer NER model (JobBERT)
- [ ] A 50-sample evaluation with precision, recall, and F1 at different thresholds
- [ ] A decision: is the pre-trained model good enough, or do you need to train your own?

---

# Chapter 5: Data Annotation for Machine Learning

**Goal:** Create high-quality labeled training data for your custom NER model. Learn annotation guidelines, tools, and quality control.

> This chapter is where most beginners bail out. Annotation is tedious. But it's also the single most important factor in model quality. A mediocre model trained on great data will outperform a state-of-the-art model trained on sloppy data. Every time.

---

## Lesson 5.1: Annotation Guidelines — Defining What a "Skill" Is

**Goal:** Create precise, unambiguous guidelines that ensure consistent labeling.

### Why Guidelines Matter

If you label "Python" as a SKILL in one description but skip it in another, your model learns noise instead of patterns. Guidelines prevent this.

### Your Annotation Schema

Define exactly one custom entity type:

**`SKILL`** — A technical skill, programming language, framework, tool, platform, protocol, methodology, or specific technology mentioned as a requirement or qualification.

**Label as SKILL:**
- Programming languages: Python, Java, JavaScript, Go, Rust, C++
- Frameworks: React, Django, Spring Boot, Angular, FastAPI
- Tools: Docker, Kubernetes, Terraform, Git, Jenkins
- Platforms: AWS, Azure, GCP, Linux, Windows Server
- Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
- Concepts: CI/CD, REST API, Microservices, Agile, Scrum
- Data tools: Pandas, Spark, Kafka, Airflow

**Do NOT label as SKILL:**
- Soft skills: "Teamfähigkeit", "Kommunikationsstärke", "self-motivated"
- Degree requirements: "Bachelor", "Master", "Studium der Informatik"
- Languages: "Deutsch", "Englisch" (these are extracted by regex)
- Job titles: "Senior Developer", "Data Engineer" (these are not skills)
- Companies: "SAP", "Google" (even though you "know SAP" — these are ORG entities)
- Vague terms: "IT-Kenntnisse", "Softwareentwicklung", "programming" (too general)

### Boundary Rules

How far does an entity span?

```
✓ "Python"                     → [Python]=SKILL
✓ "Machine Learning"           → [Machine Learning]=SKILL
✓ "REST API"                   → [REST API]=SKILL
✓ "CI/CD"                      → [CI/CD]=SKILL

✗ "Python-Entwickler"          → [Python]=SKILL only, not the whole compound
✗ "AWS Cloud Services"         → [AWS]=SKILL only (Cloud Services is descriptive)
✗ "Erfahrung mit Python"       → [Python]=SKILL only, not the full phrase
✗ "gute Python-Kenntnisse"     → [Python]=SKILL only
```

### Practical Exercise

1. Write your annotation guidelines in a document (1–2 pages)
2. Label 10 descriptions using ONLY these guidelines
3. Have someone else (or yourself after a day) label the same 10 descriptions
4. Compare: do you agree on every entity? Where do you disagree? Refine the guidelines.

---

## Lesson 5.2: Annotation Tools — Label Studio

**Goal:** Set up Label Studio and annotate 200 job descriptions for NER training.

### Setup

```bash
pip install label-studio
label-studio start
# Opens at http://localhost:8080
# Create an account (local, no internet required)
```

### Configure a NER Project

1. Create a new project called "Job Skills NER"
2. In the labeling configuration, use:

```xml
<View>
  <Labels name="label" toName="text">
    <Label value="SKILL" background="#00ff00"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

3. Import your data: prepare a JSON file with your 200 sample descriptions:

```python
import json
import pandas as pd

# Select 200 diverse samples
sample = df.sample(200, random_state=42)

# Format for Label Studio
tasks = []
for _, row in sample.iterrows():
    tasks.append({
        "data": {
            "text": row["description_clean"][:2000],  # trim very long descriptions
        },
        "meta": {
            "original_index": int(row.name),
            "language": row.get("language", "unknown"),
            "title": row.get("title", ""),
        }
    })

with open("data/annotation/label_studio_import.json", "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)
```

4. Import this file into your Label Studio project

### Annotation Workflow

For each description:
1. Read the full text
2. Highlight every SKILL entity according to your guidelines
3. Click Submit and move to the next task
4. Aim for 20–30 descriptions per hour (this is normal speed)

**200 descriptions × ~2 minutes each = ~7 hours of annotation work.** Yes, it's a lot. Yes, it's necessary.

### Quality Tips

- Annotate in batches of 30–40 to avoid fatigue errors
- After every batch, review 5 random completed annotations — are you still consistent?
- Keep your annotation guidelines open in another tab at all times
- If you encounter an edge case not covered by your guidelines, update the guidelines first, then label

### Resources

- [Label Studio documentation](https://labelstud.io/guide/)
- [Label Studio NER template](https://labelstud.io/templates/named_entity.html)
- [Blog: Evaluating NER with spaCy and Label Studio](https://labelstud.io/blog/evaluating-named-entity-recognition-parsers-with-spacy-and-label-studio/)

---

## Lesson 5.3: Exporting Annotations and Creating Training Data

**Goal:** Convert Label Studio annotations to spaCy's training format.

### Export from Label Studio

1. Go to your project → Export
2. Select **JSON** format
3. Download the file

### Convert to spaCy Format

```python
import json
from spacy.tokens import DocBin
import spacy

nlp = spacy.blank("de")  # blank model, just tokenizer

def convert_label_studio_to_spacy(ls_export_path: str, output_path: str):
    """Convert Label Studio JSON export to spaCy binary training format."""
    
    with open(ls_export_path) as f:
        tasks = json.load(f)
    
    db = DocBin()
    skipped = 0
    
    for task in tasks:
        text = task["data"]["text"]
        doc = nlp.make_doc(text)
        
        # Get annotations (from the first annotator)
        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue
        
        # Extract entity spans
        ents = []
        for annotation in annotations[0].get("result", []):
            if annotation["type"] != "labels":
                continue
            
            start = annotation["value"]["start"]
            end = annotation["value"]["end"]
            label = annotation["value"]["labels"][0]
            
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
            else:
                # Character offsets don't align with token boundaries
                # This happens with subword tokenization mismatches
                print(f"  Warning: Could not create span for '{text[start:end]}' [{start}:{end}]")
        
        doc.ents = ents
        db.add(doc)
    
    db.to_disk(output_path)
    print(f"Saved {len(tasks) - skipped} docs to {output_path} (skipped {skipped})")

# Usage
convert_label_studio_to_spacy(
    "data/annotation/export.json",
    "data/training/train.spacy"
)
```

### Train/Validation Split

```python
# Split 200 annotations: 160 train, 40 validation
# NEVER evaluate on training data
import random

with open("data/annotation/export.json") as f:
    all_tasks = json.load(f)

random.seed(42)
random.shuffle(all_tasks)

train_tasks = all_tasks[:160]
val_tasks = all_tasks[160:]

# Save separately
with open("data/annotation/train_export.json", "w") as f:
    json.dump(train_tasks, f)
with open("data/annotation/val_export.json", "w") as f:
    json.dump(val_tasks, f)

# Convert both
convert_label_studio_to_spacy("data/annotation/train_export.json", "data/training/train.spacy")
convert_label_studio_to_spacy("data/annotation/val_export.json", "data/training/dev.spacy")
```

---

## Chapter 5 Checkpoint

By the end of this chapter, you should have:

- [ ] Written annotation guidelines (1–2 pages defining what is and isn't a SKILL)
- [ ] Set up Label Studio and annotated 200 job descriptions
- [ ] Exported annotations and converted to spaCy format
- [ ] Split data into 160 training + 40 validation examples

---

# Chapter 6: Training a Custom NER Model

**Goal:** Fine-tune spaCy's NER component on your annotated data to recognize SKILL entities in German IT job descriptions.

> This is the chapter where you go from "using models" to "training models." The concepts here — transfer learning, loss functions, overfitting, evaluation — are the core of applied ML.

---

## Lesson 6.1: Transfer Learning — Why You Don't Train from Scratch

**Goal:** Understand why starting from a pre-trained model is dramatically better than starting from zero.

### The Concept

Training an NER model from scratch requires millions of labeled examples. You have 200. That's not enough to learn language from nothing.

**Transfer learning** solves this: you start from a model that already understands language (spaCy's `de_core_news_lg` was trained on millions of German words). Then you **fine-tune** it — teach it one new thing (recognizing SKILL entities) while keeping all its existing knowledge about German grammar, word relationships, and other entity types.

```
Pre-trained German model (de_core_news_lg):
  ✓ Knows German grammar
  ✓ Knows word relationships (Python is related to programming)
  ✓ Recognizes PER, ORG, LOC entities
  ✗ Doesn't know what a SKILL entity is

After fine-tuning on your 200 labeled examples:
  ✓ Still knows German grammar
  ✓ Still knows word relationships
  ✓ Still recognizes PER, ORG, LOC
  ✓ NOW recognizes SKILL entities too
```

### The Risk: Catastrophic Forgetting

If you fine-tune too aggressively, the model "forgets" its pre-trained knowledge. It learns SKILL but loses the ability to recognize ORG and LOC. Mitigation strategies:

1. **Low learning rate** — make small updates so existing knowledge is preserved
2. **Short training** — 20–30 epochs, not 200
3. **Dropout** — randomly disable neurons during training (default: 0.3) to prevent overfitting

---

## Lesson 6.2: Training with spaCy v3

**Goal:** Configure and run model training using spaCy's config system.

### Step 1: Generate a Base Config

Go to [spaCy's training quickstart](https://spacy.io/usage/training#quickstart) and select:
- Language: **German**
- Components: **ner**
- Hardware: **CPU** (or GPU if available)
- Optimize for: **efficiency**

Download the generated `base_config.cfg`, then fill it:

```bash
python -m spacy init fill-config base_config.cfg config.cfg
```

### Step 2: Modify the Config

Edit `config.cfg` to use the pre-trained German model as starting point:

```ini
[paths]
train = "data/training/train.spacy"
dev = "data/training/dev.spacy"

[components.ner]
source = "de_core_news_lg"   # Transfer NER from pre-trained model

[training]
max_epochs = 30
patience = 5                  # Stop if no improvement for 5 epochs

[training.optimizer]
learn_rate = 0.001            # Low learning rate for fine-tuning

[training.batcher]
size = 32
```

### Step 3: Train

```bash
python -m spacy train config.cfg --output ./models/skill_ner --gpu-id -1
```

spaCy will output training progress:

```
E    #       LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE
---  ------  --------  ------  ------  ------  -----
  0       0     26.50   12.34   15.67    10.22   0.12
  5     200      8.12   58.45   62.33    55.01   0.58
 10     400      3.45   71.23   74.56    68.22   0.71
 15     600      1.89   75.67   78.90    72.78   0.76
 20     800      1.12   76.12   79.23    73.34   0.76  ← best model saved
 25    1000      0.78   75.89   79.01    73.11   0.76  ← patience: no improvement
```

### Understanding the Metrics

- **LOSS NER:** How wrong the model's predictions are (lower is better). Should decrease.
- **ENTS_F:** F1 score on validation set. This is your primary metric.
- **ENTS_P:** Precision — of predicted entities, how many are correct.
- **ENTS_R:** Recall — of actual entities, how many were found.
- **SCORE:** Combined score for all components.

### Step 4: Load and Test Your Model

```python
import spacy

nlp = spacy.load("./models/skill_ner/model-best")

# Test on text NOT in your training data
test_text = "Wir suchen jemanden mit Erfahrung in React, Node.js und PostgreSQL."
doc = nlp(test_text)

for ent in doc.ents:
    print(f"  {ent.text:20s} → {ent.label_}")

# Expected:
#   React                → SKILL
#   Node.js              → SKILL
#   PostgreSQL           → SKILL
```

### Practical Exercise

1. Train the model using the commands above
2. Note the best F1 score on validation — this is your custom model's performance
3. Test on 10 descriptions that were NOT in training or validation
4. Compare to the pre-trained JobBERT from Chapter 4 — which performs better on your data?

### Resources

- [spaCy Training Documentation](https://spacy.io/usage/training)
- [spaCy config system explained](https://spacy.io/usage/training#config)
- [NER Training Textbook (Python Humanities)](https://ner.pythonhumanities.com/03_02_train_spacy_ner_model.html)

---

## Lesson 6.3: Error Analysis — Why Your Model Fails

**Goal:** Systematically identify and fix the most common errors your model makes.

### Common Error Types

**False Positives (model tags non-skills as SKILL):**
- Company names: "SAP" tagged as SKILL
- Cities: "München" tagged as SKILL
- Generic nouns: "Erfahrung" tagged as SKILL

**False Negatives (model misses real skills):**
- Skills not in training data: if "Terraform" never appeared in your 200 samples, the model won't learn it
- Unusual formatting: "NodeJS" vs "Node.js" vs "node.js"
- Skills in German compound words: "Python-Kenntnisse"

### The Fix: Iterative Improvement

1. Run your trained model on 50 new descriptions
2. Manually mark every error (FP and FN)
3. For the most common error types, add more training examples that cover those cases
4. Re-annotate 30–50 more descriptions specifically targeting the gaps
5. Retrain and re-evaluate

This cycle is how real NER development works: train → evaluate → find errors → add data → retrain.

---

## Chapter 6 Checkpoint

By the end of this chapter, you should have:

- [ ] A trained custom NER model that recognizes SKILL entities
- [ ] Validation F1 score documented (target: >0.65 for a first model)
- [ ] Error analysis identifying the top 5 error types
- [ ] An understanding of transfer learning, overfitting, and early stopping

---

# Chapter 7: Text Classification with scikit-learn

**Goal:** Build a classifier that predicts seniority level from job description text, learning TF-IDF vectorization, model selection, and feature interpretation along the way.

> This chapter takes you beyond NER into traditional ML classification. The concepts — feature extraction, model training, cross-validation — apply to any ML problem, not just NLP.

---

## Lesson 7.1: TF-IDF — Turning Text into Numbers

**Goal:** Understand how TF-IDF converts text into a numerical feature vector that ML models can process.

### The Problem

ML models work with numbers, not text. You need to convert "Wir suchen einen Senior Python-Entwickler mit 5 Jahren Erfahrung" into a vector of numbers.

### TF-IDF: Term Frequency × Inverse Document Frequency

**TF (Term Frequency):** How often a word appears in THIS document. "Python" appears 3 times → TF is high.

**IDF (Inverse Document Frequency):** How rare a word is ACROSS ALL documents. "und" appears in every posting → IDF is low. "Kubernetes" appears in 5% of postings → IDF is high.

**TF-IDF = TF × IDF** — Words that are frequent in this document AND rare overall get the highest scores. These are the words that best characterize what this document is about.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Senior Python developer with 5 years experience leading teams",
    "Junior developer position, no experience required, training provided",
    "Experienced architect designing microservices at scale",
]

vectorizer = TfidfVectorizer(max_features=20, ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)

# See the feature names
print("Features:", vectorizer.get_feature_names_out())
# ['architect', 'developer', 'experience', 'experienced', 'junior', 'leading', ...]

# See the TF-IDF matrix
import pandas as pd
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df.round(2))
```

### Key Parameters

- `max_features`: Limit vocabulary size (5000 is a good start)
- `ngram_range=(1,2)`: Include both single words AND two-word phrases. "team lead" is a bigram that's very informative for seniority.
- `sublinear_tf=True`: Apply log scaling to term frequency. Prevents very long documents from dominating.
- `min_df=2`: Ignore terms that appear in fewer than 2 documents (eliminates typos and noise).

---

## Lesson 7.2: Training a Seniority Classifier

**Goal:** Train a model that predicts seniority (Junior/Mid/Senior/Lead) from job description text.

### Preparing Training Data

You need ~200 manually labeled examples. Use your existing annotated descriptions or label new ones.

```python
# Example labeled data (you'll have more)
texts = [
    "Senior Software Engineer with 8+ years of experience leading cross-functional teams...",
    "Looking for a Junior Developer to join our growing team. No prior experience needed...",
    "We seek an experienced Mid-level Backend Developer with 3-5 years...",
    # ... 200 more
]
labels = ["Senior", "Junior", "Mid", ...]  # corresponding labels
```

### Building the Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Split: 160 train, 40 test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handles class imbalance
        C=1.0,
    )),
])

# Cross-validation on training set
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
print(f"CV F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Final evaluation on held-out test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Interpreting the Model

The best part of Logistic Regression: you can see exactly what the model learned.

```python
feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
coefficients = pipeline.named_steps["clf"].coef_

for i, class_name in enumerate(pipeline.named_steps["clf"].classes_):
    top_indices = np.argsort(coefficients[i])[-15:]  # top 15 features
    top_features = [(feature_names[j], coefficients[i][j]) for j in top_indices]
    
    print(f"\n{'='*50}")
    print(f"Top features for '{class_name}':")
    print(f"{'='*50}")
    for feat, weight in sorted(top_features, key=lambda x: -x[1]):
        print(f"  {feat:30s} weight={weight:.3f}")
```

You'll see things like:
- Senior: "team lead", "architect", "8 years", "Führung", "strategisch"
- Junior: "Berufseinsteiger", "Ausbildung", "training", "entry level"
- Lead: "leading", "team", "Verantwortung", "manage"

This interpretability is extremely valuable for debugging and for understanding your data.

### Resources

- [scikit-learn Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [scikit-learn TfidfVectorizer docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Understanding TF-IDF (O'Reilly)](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/ch04.html)

---

## Chapter 7 Checkpoint

By the end of this chapter, you should have:

- [ ] An understanding of TF-IDF and how text becomes numerical features
- [ ] A working seniority classifier with cross-validated F1 score
- [ ] The ability to interpret model weights and understand what the model learned
- [ ] Knowledge of when to use classification vs. NER (fixed categories vs. variable spans)

---

# Chapter 8: Building the Extraction Pipeline

**Goal:** Compose all your models and extractors into a single, modular pipeline that processes raw job descriptions end-to-end.

> Everything you've built so far is a component. Now you wire them together into a system. This chapter is about software engineering as much as it is about ML.

---

## Lesson 8.1: Pipeline Architecture

**Goal:** Design a modular pipeline where each component has a single responsibility.

```
Raw Description
       │
       ▼
┌──────────────┐
│ Text Cleanup │  ← Chapter 1: encoding, HTML, whitespace
└──────┬───────┘
       ▼
┌──────────────┐
│ Language Det. │  ← Chapter 1: lingua-py
└──────┬───────┘
       ▼
┌──────────────┐
│ Section Split│  ← Chapter 3: headers + NLP classification
└──────┬───────┘
       │
       ├──────────────────┐
       ▼                  ▼
┌──────────────┐  ┌──────────────┐
│ Regex Layer  │  │  NER Layer   │
│ (contract,   │  │  (skills)    │  ← Chapter 4 & 6
│  salary,     │  │              │
│  experience, │  └──────┬───────┘
│  modality,   │         │
│  languages)  │         │
│ ← Chapter 2  │         │
└──────┬───────┘         │
       │                 │
       ├─────────────────┘
       ▼
┌──────────────┐
│ Classifier   │  ← Chapter 7: seniority
└──────┬───────┘
       ▼
┌──────────────┐
│    Merge     │  ← Combine all fields
└──────┬───────┘
       ▼
  Enriched DataFrame
```

### Implementation

Refer to the `ExtractionPipeline` class from our earlier conversation. The key principles:

1. **Load models once, reuse across rows** — don't reload spaCy models for every description
2. **Each extractor is a pure function** — input text, output extracted value. No side effects.
3. **Configuration in YAML, not hardcoded** — thresholds, model paths, pattern lists
4. **Checkpoint intermediate results** — save after each layer so you don't re-run everything

---

## Lesson 8.2: Data Storage — Parquet, Not CSV

**Goal:** Understand why CSV breaks for this project and what to use instead.

Your extracted data has list-type columns (`skills`, `benefits`, `languages_required`). CSV has no concept of lists. It will serialize `["Python", "Docker"]` as the string `"['Python', 'Docker']"` and you'll have to `eval()` it back — fragile and dangerous.

```python
# Save as Parquet — preserves types
df.to_parquet("data/extracted/jobs_enriched.parquet", index=False)

# Load back — lists are still lists
df_loaded = pd.read_parquet("data/extracted/jobs_enriched.parquet")
print(type(df_loaded["skills"].iloc[0]))  # <class 'list'> ✓
```

---

## Chapter 8 Checkpoint

- [ ] All components wired into a single pipeline
- [ ] Pipeline runs end-to-end on your full dataset
- [ ] Results saved as Parquet with all extracted fields
- [ ] Configuration externalized to YAML

---

# Chapter 9: Evaluation, Validation & Iteration

**Goal:** Rigorously measure the quality of your pipeline, identify weaknesses, and improve systematically.

---

## Lesson 9.1: Per-Field Accuracy Assessment

**Goal:** Measure the accuracy of every extracted field against manually verified ground truth.

### The Gold Standard

Sample 100 descriptions (stratified by language). For each one, manually fill in every field yourself. This is your ground truth.

```python
# Stratified sample: proportional DE/EN/mixed
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=42)
for _, test_idx in splitter.split(df, df["language"]):
    eval_sample = df.iloc[test_idx]
```

Then compare your pipeline's output against the ground truth per field and compute accuracy, precision, recall, and F1 for each.

### What to Do with the Results

| Field | Accuracy | Action |
|-------|----------|--------|
| > 85% | Ship it. Good enough for analysis. |
| 70–85% | Acceptable. Document limitations. Improve in next iteration. |
| 50–70% | Usable with caveats. Flag low-confidence rows. |
| < 50% | Don't use this field in analysis. Go back and improve the extractor. |

---

## Lesson 9.2: Error Taxonomy and Improvement

**Goal:** Categorize errors systematically to guide your next iteration.

For every error in your 100-sample evaluation, classify it:

- **Pattern miss:** Regex didn't cover this pattern → add the pattern
- **NER miss:** Model didn't detect a skill → add more training data covering this case
- **NER false positive:** Model tagged a non-skill → add negative examples to training data
- **Section split failure:** Section splitter couldn't find sections → improve header patterns
- **Language issue:** Extractor can't handle mixed DE/EN text → add bilingual patterns
- **Data quality:** The original description is too short/garbled to extract anything → exclude from analysis

Count each error type. Fix the most frequent one first. This is how you get the most improvement for the least effort.

---

## Chapter 9 Checkpoint

- [ ] 100-row gold standard with manual labels for all fields
- [ ] Per-field accuracy report with precision, recall, F1
- [ ] Error taxonomy with counts per error type
- [ ] Prioritized list of improvements for the next iteration

---

# Appendix A: Recommended Learning Path

For each chapter, do these in order:

1. **Read the lesson** in this curriculum
2. **Follow the code examples** in a Jupyter notebook
3. **Do the practical exercises** on YOUR data (not toy examples)
4. **Complete the chapter checkpoint** before moving on
5. **Read the linked resources** for deeper understanding

---

# Appendix B: Complete Resource Library

## Core Tools Documentation
- [spaCy documentation](https://spacy.io/usage) — your primary NLP library
- [spaCy free course](https://course.spacy.io/en/) — **Do this entire course in Week 3**
- [scikit-learn documentation](https://scikit-learn.org/stable/user_guide.html) — ML fundamentals
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — pre-trained models

## NER-Specific Resources
- [Introduction to NER (Python Humanities)](https://ner.pythonhumanities.com/) — free online textbook, covers everything from theory to spaCy training
- [spaCy NER training quickstart](https://spacy.io/usage/training#quickstart) — config generator
- [NewsCatcher — Train Custom NER with spaCy v3](https://www.newscatcherapi.com/blog-posts/train-custom-named-entity-recognition-ner-model-with-spacy-v3) — step-by-step tutorial

## Annotation Tools
- [Label Studio documentation](https://labelstud.io/guide/) — open-source annotation tool
- [Label Studio NER template](https://labelstud.io/templates/named_entity.html)
- [Label Studio + spaCy integration tutorial](https://labelstud.io/blog/evaluating-named-entity-recognition-parsers-with-spacy-and-label-studio/)

## Text Processing & Regex
- [regex101.com](https://regex101.com/) — interactive regex tester
- [Python `re` module documentation](https://docs.python.org/3/library/re.html)
- [Joel Spolsky on Unicode](https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/)

## ML Fundamentals
- [scikit-learn text classification tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [StatQuest (YouTube)](https://www.youtube.com/c/joshstarmer) — ML concepts explained visually

## Pre-trained Models for Job Data
- [jjzha/jobbert_skill_extraction](https://huggingface.co/jjzha/jobbert_skill_extraction) — English skill NER
- [Nucha/Nucha_ITSkillNER_BERT](https://huggingface.co/Nucha/Nucha_ITSkillNER_BERT) — IT skill NER
- [deepset/gelectra-base-germanquad](https://huggingface.co/deepset/gelectra-base-germanquad) — German QA model

---

# Appendix C: Project Folder Structure

```
job-extraction/
├── config/
│   ├── settings.yaml               # Paths, thresholds, model names
│   ├── skills_keywords.yaml        # Supplementary keyword list
│   └── section_headers.yaml        # Header → section mapping
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Chapter 1: text cleanup, encoding, language detection
│   ├── section_splitter.py         # Chapter 3: description → sections
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── contract_type.py        # Chapter 2: regex
│   │   ├── work_modality.py        # Chapter 2: regex
│   │   ├── salary.py               # Chapter 2: regex
│   │   ├── experience.py           # Chapter 2: regex
│   │   ├── languages.py            # Chapter 2: regex
│   │   ├── skills_ner.py           # Chapter 6: custom spaCy NER
│   │   └── seniority.py            # Chapter 7: title rules + ML classifier
│   ├── pipeline.py                 # Chapter 8: orchestration
│   └── evaluation.py               # Chapter 9: metrics and reporting
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_extractors.py          # Chapter 2: parametrized tests
│   ├── test_skills_ner.py
│   └── test_pipeline.py
│
├── models/
│   ├── skill_ner/                  # Chapter 6: trained spaCy NER
│   └── seniority_clf.pkl           # Chapter 7: trained classifier
│
├── data/
│   ├── raw/
│   ├── annotation/                 # Chapter 5: Label Studio exports
│   ├── training/                   # Chapter 6: .spacy binary files
│   ├── extracted/                  # Chapter 8: enriched Parquet
│   └── validation/                 # Chapter 9: gold standard
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Chapter 1
│   ├── 02_regex_development.ipynb  # Chapter 2
│   ├── 03_spacy_exploration.ipynb  # Chapter 3
│   ├── 04_ner_evaluation.ipynb     # Chapter 4
│   ├── 05_model_training.ipynb     # Chapter 6
│   ├── 06_classifier_dev.ipynb     # Chapter 7
│   └── 07_quality_analysis.ipynb   # Chapter 9
│
├── requirements.txt
└── README.md
```

---

# Final Note

This curriculum is designed around a single principle: **learn by building something real.** Every lesson exists because you need it for the next step of your extraction pipeline. There are no theoretical detours for their own sake.

The progression is intentional:
- Chapters 1–2 teach you the basics while producing immediately useful output (cleaned data, regex extractions)
- Chapters 3–4 deepen your understanding while you evaluate what ML can add
- Chapters 5–6 are the hard part — annotation and training. Don't rush them.
- Chapters 7–9 bring everything together and teach you how to measure and improve

When you finish, you'll have built a production-quality extraction pipeline AND learned the fundamentals of NLP, NER, and ML. Both the project and the knowledge are yours to keep.
