"""
Enhanced Multilingual Hybrid Skill Extraction with SQLite Checkpointing
- Auto-resume capability if process crashes
- Batch processing with periodic saves
- SQLite database for persistent storage
- Parallel processing with multiprocessing for speed
"""

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import Counter, defaultdict
from typing import List, Tuple
import json
import re
from tqdm import tqdm
from langdetect import detect, LangDetectException
import sqlite3
from datetime import datetime
import os

from multiprocessing import Pool, cpu_count
from functools import partial
import threading
import logging
import warnings

# Suppress verbose model loading messages
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*position_ids.*")

tqdm.pandas()

# ============================================================
# SQLite CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """Manages SQLite database for checkpoint/resume capability"""
    
    def __init__(self, db_path="skill_extraction_checkpoint.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.create_tables()
        
    def create_tables(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Table for processed jobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_jobs (
                job_index INTEGER PRIMARY KEY,
                title TEXT,
                company_name TEXT,
                location TEXT,
                lang_detected TEXT,
                languages TEXT,
                soft_skills_final TEXT,
                technical_skills_final TEXT,
                tech_keywords_regex TEXT,
                soft_skills_dict TEXT,
                soft_skills_categories TEXT,
                skill_spans TEXT,
                skill_spans_soft TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for tracking progress
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_processed_index INTEGER,
                total_jobs INTEGER,
                batch_size INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def get_last_processed_index(self):
        """Get the index of the last processed job"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(job_index) FROM processed_jobs")
        result = cursor.fetchone()[0]
        return result if result is not None else -1
    
    def get_processed_count(self):
        """Get count of processed jobs"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processed_jobs")
        return cursor.fetchone()[0]
    
    def save_batch(self, df_batch, start_index):
        """Save a batch of processed jobs (thread-safe)"""
        with self.lock:
            cursor = self.conn.cursor()
            
            for i, (idx, row) in enumerate(df_batch.iterrows()):
                actual_idx = start_index + i
                
                # Convert lists to JSON strings
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_jobs 
                    (job_index, title, company_name, location, lang_detected, languages,
                     soft_skills_final, technical_skills_final, tech_keywords_regex,
                     soft_skills_dict, soft_skills_categories, skill_spans, skill_spans_soft)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    actual_idx,
                    row.get('title', ''),
                    row.get('company_name', ''),
                    row.get('location', ''),
                    row.get('lang_detected', ''),
                    json.dumps(row.get('languages', [])),
                    json.dumps(row.get('soft_skills_final', [])),
                    json.dumps(row.get('technical_skills_final', [])),
                    json.dumps(row.get('tech_keywords_regex', [])),
                    json.dumps(row.get('soft_skills_dict', [])),
                    json.dumps(row.get('soft_skills_categories', [])),
                    json.dumps(row.get('skill_spans', [])),
                    json.dumps(row.get('skill_spans_soft', []))
                ))
            
            self.conn.commit()
            print(f"✓ Saved batch checkpoint at index {start_index + len(df_batch) - 1}")
            
            # Auto-backup: copy DB to backup folder after every save
            try:
                import shutil
                backup_dir = os.path.join(os.path.dirname(os.path.abspath(self.db_path)), "checkpoint_backups")
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f"backup_{start_index + len(df_batch) - 1}.db")
                shutil.copy2(self.db_path, backup_path)
                # Keep only the 5 most recent backups to save disk space
                backups = sorted(os.listdir(backup_dir))
                for old in backups[:-5]:
                    os.remove(os.path.join(backup_dir, old))
            except Exception:
                pass  # Never crash the main process due to backup failure
    
    def update_progress(self, last_index, total_jobs, batch_size):
        """Update progress tracking"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO progress (last_processed_index, total_jobs, batch_size)
            VALUES (?, ?, ?)
        """, (last_index, total_jobs, batch_size))
        self.conn.commit()
    
    def export_to_dataframe(self):
        """Export all processed jobs to DataFrame"""
        query = """
            SELECT 
                job_index, title, company_name, location, lang_detected,
                languages, soft_skills_final, technical_skills_final, 
                tech_keywords_regex, soft_skills_dict, soft_skills_categories,
                skill_spans, skill_spans_soft
            FROM processed_jobs 
            ORDER BY job_index
        """
        df = pd.read_sql_query(query, self.conn)
        
        # Convert JSON strings back to lists
        json_columns = ['languages', 'soft_skills_final', 'technical_skills_final', 
                       'tech_keywords_regex', 'soft_skills_dict', 'soft_skills_categories',
                       'skill_spans', 'skill_spans_soft']
        
        for col in json_columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else [])
        
        return df
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================
# SOFT SKILLS DICTIONARY (EN + DE)
# ============================================================
SOFT_SKILLS = [
    "communication", "communication skills", "clear communication", "written communication",
    "verbal communication", "presentation", "presentation skills", "active listening",
    "listening skills", "interpersonal skills", "collaboration", "collaboration skills",
    "cross-functional collaboration", "stakeholder management", "teamwork", "team player",
    "conflict resolution", "conflict management", "negotiation", "negotiation skills",
    "facilitation", "facilitation skills",
    "self-management", "self-starter", "initiative", "proactive", "self-motivation", "motivation",
    "independent working style", "autonomous working style", "ownership", "accountability",
    "responsibility", "sense of responsibility", "dependability", "reliability",
    "attention to detail", "detail-oriented", "structured working style", "organized",
   "goal-oriented working", "time management", "prioritization skills", "ability to meet deadlines",
    "work ethic", "professionalism", "integrity", "commitment",
    "problem solving", "problem-solving skills", "analytical thinking", "analytical skills",
    "critical thinking", "conceptual thinking", "strategic thinking", "decision making",
    "decision-making skills", "solution-oriented mindset", "systems thinking", "risk awareness",
    "ability to work under pressure", "stress tolerance",
    "leadership", "leadership skills", "people management", "mentoring", "coaching",
    "coaching skills", "ability to motivate others", "influencing skills",
    "adaptability", "flexibility", "learning ability", "willingness to learn", "learning mindset",
    "openness to feedback", "continuous improvement mindset", "change management", "resilience",
    "customer focus", "service orientation", "user-centric mindset", "consulting skills",
    "business acumen", "quality mindset",
    # German
    "kommunikation", "kommunikationsfähigkeit", "kommunikationsstärke", "klare kommunikation",
    "schriftliche kommunikationsfähigkeit", "mündliche kommunikationsfähigkeit", "präsentation",
    "präsentationsfähigkeit", "moderation", "moderationsfähigkeit", "aktives zuhören",
    "teamarbeit", "teamfähigkeit", "zusammenarbeit", "zusammenarbeitsfähigkeit",
    "bereichsübergreifende zusammenarbeit", "stakeholder-management", "konfliktlösung",
    "konfliktmanagement", "verhandlungsgeschick",
    "selbstmanagement", "eigeninitiative", "selbstständige arbeitsweise",
    "verantwortungsbewusstsein", "verantwortlichkeit", "verantwortungsübernahme", "zuverlässigkeit",
    "sorgfalt", "auge fürs detail", "strukturierte arbeitsweise", "organisatorisches geschick",
    "zielorientiertes arbeiten", "zeitmanagement", "priorisierungsfähigkeit", "termintreue",
    "professionalität", "integrität", "engagement", "einsatzbereitschaft",
    "problemlösungsfähigkeit", "analytische fähigkeiten", "analytisches denken", "kritisches denken",
    "konzeptionelles denken", "strategisches denken", "entscheidungsfähigkeit",
    "lösungsorientierte denkweise", "systemisches denken", "risikobewusstsein", "belastbarkeit",
    "stresstoleranz",
    "führungskompetenz", "mitarbeiterführung", "coaching-kompetenz", "mentoring",
    "motivationsfähigkeit", "überzeugungskraft",
    "anpassungsfähigkeit", "flexibilität", "lernfähigkeit", "lernbereitschaft", "feedbackfähigkeit",
    "kontinuierliche verbesserung", "change-management", "resilienz",
    "kundenorientierung", "serviceorientierung", "nutzerzentriertes denken", "beratungskompetenz",
    "unternehmerisches denken", "qualitätsbewusstsein",
    "teamorientierte arbeitsweise", "lösungsorientiertes arbeiten", "sicheres auftreten",
]


# ============================================================
# LANGUAGE EXTRACTOR
# ============================================================
class LanguageExtractor:
    LANGUAGE_PATTERNS = {
        "english": [
            r"\benglish\b", r"\benglisch\b",
            r"\b(fluent|proficient|excellent|good)\s+english\b",
            r"\benglish\s+(skills?|proficiency|kenntnisse)\b",
            r"\b(fließendes|sehr\s+gutes|gutes)\s+englisch\b",
            r"\benglisch\s+in\s+wort\s+und\s+schrift\b",
            r"\benglischkenntnisse\b",
            r"\b(verhandlungssicheres|business-?)\s*englisch\b",
        ],
        "german": [
            r"\bgerman\b", r"\bdeutsch\b",
            r"\b(fluent|proficient|excellent|good)\s+german\b",
            r"\bgerman\s+(skills?|proficiency|kenntnisse)\b",
            r"\b(fließendes|sehr\s+gutes|gutes)\s+deutsch\b",
            r"\bdeutsch\s+in\s+wort\s+und\s+schrift\b",
            r"\bdeutschkenntnisse\b",
            r"\b(verhandlungssicheres|business-?)\s*deutsch\b",
        ],
    }
    CEFR_PATTERNS = [
        r"\b(a1|a2|b1|b2|c1|c2)\b",
        r"\benglish\s+(b2|c1|c2)\b", r"\bgerman\s+(b2|c1|c2)\b",
        r"\benglisch\s+(b2|c1|c2)\b", r"\bdeutsch\s+(b2|c1|c2)\b",
    ]

    def __init__(self):
        self.compiled_patterns = {
            lang: [re.compile(p, re.IGNORECASE) for p in patterns]
            for lang, patterns in self.LANGUAGE_PATTERNS.items()
        }
        self.cefr_patterns = [re.compile(p, re.IGNORECASE) for p in self.CEFR_PATTERNS]

    def extract_languages(self, text: str) -> List[str]:
        if pd.isna(text):
            return []
        text = str(text)
        found = set()
        for lang, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    found.add(lang)
                    break
        for pattern in self.cefr_patterns:
            match = pattern.search(text)
            if match:
                level = match.group(0).lower()
                if "englisch" in text.lower() or "english" in text.lower() or level in ("b2", "c1"):
                    found.add("english")
                if "deutsch" in text.lower() or "german" in text.lower() or level in ("b2", "c1"):
                    found.add("german")
        return list(found)


# ============================================================
# SOFT SKILL EXTRACTOR
# ============================================================
class FastSoftSkillExtractor:
    CATEGORY_TO_CANON = {
        "communication": "communication",
        "teamwork": "teamwork",
        "leadership": "leadership",
        "problem_solving": "problem solving",
        "adaptability": "adaptability",
        "time_management": "time management",
        "autonomy": "autonomy",
    }

    def __init__(self):
        self.soft_skills_dict = SOFT_SKILLS
        self.context_patterns = {
            "communication": [
                r"\b(strong|excellent|good|effective|clear)\s+(verbal|written)?\s*communication\b",
                r"\b(communication|presentation)\s+skills\b",
                r"\bactive\s+listening\b",
                r"\b(ability|capability)\s+to\s+communicat(e|ing)\b",
                r"\bcommunicat(e|es|ed|ing)\s+.*\b(clearly|effectively|understandably)\b",
                r"\bstakeholder\s+management\b",
                r"\bkommunikation(sfähigkeit|sstärke)?\b",
                r"\bkommunikative\s+(fähigkeiten|kompetenz)\b",
                r"\bverständlich\s+kommunizier(en|t)\b",
            ],
            "teamwork": [
                r"\bteam\s+player\b", r"\bteamwork\b", r"\bteam-?orient(ed|ierte)\b",
                r"\bwork\s+(collaboratively|together)\b",
                r"\bwork\s+in\s+(a\s+)?team(s)?\b",
                r"\bin\s+(close|tight)\s+collaboration\b",
                r"\bcross-?functional\s+(team|collaboration)\b",
                r"\bmultinational\s+(engineering\s+)?team\b",
                r"\bglobally\s+distributed\s+(engineering\s+)?team\b",
                r"\bactively\s+collaborat(e|ing)\b",
                r"\bpart\s+of\s+(a\s+)?(multinational|global|cross-functional)\s+team\b",
                r"\bteamfähigkeit\b", r"\bteamarbeit\b", r"\bteamgeist\b",
                r"\bzusammenarbeit(en|et|e)?\b",
            ],
            "leadership": [
                r"\blead(ing|s|)\s+(a\s+)?team(s)?\b",
                r"\bmanage(s|d|ment)?\s+(a\s+)?team(s)?\b",
                r"\bpeople\s+management\b", r"\bmentoring\b", r"\bcoaching\b",
                r"\bführung\b", r"\bführungskompetenz\b", r"\bmitarbeiterführung\b",
            ],
            "problem_solving": [
                r"\bproblem[-\s]?solving\b",
                r"\b(troubleshoot(ing)?|debug(ging)?)\b",
                r"\bidentify\s+and\s+(fix|resolve)\s+(errors|issues|bugs)\b",
                r"\bsolve\s+(complex|challenging)\s+problems\b",
                r"\banalytical\s+(skills|thinking)\b", r"\bcritical\s+thinking\b",
                r"\bcomplex\s+(interrelationships|dependencies)\b",
                r"\bproblemlös(ung|ungsfähigkeit)\b",
                r"\banalytisch(es)?\s+denken\b",
                r"\bfehler\s+(analysieren|beheben)\b",
            ],
            "adaptability": [
                r"\badapt(ab(le|ility)|ing)\b", r"\bflexib(le|ility)\b", r"\bagile\b",
                r"\bopen\s+to\s+change\b", r"\bchange\s+management\b", r"\bresilien(t|ce)\b",
                r"\bfast\s+learner\b", r"\blearn\s+quickly\b", r"\bwilling(ness)?\s+to\s+learn\b",
                r"\bcontinuous\s+improvement\b",
                r"\banpassungsfähigkeit\b", r"\bler(n|n)-?bereitschaft\b",
                r"\bresilienz\b", r"\bflexibilität\b",
                r"\bability\s+to\s+learn\b",
                r"\blearn\s+(new\s+)?(tools|technologies|concepts|skills|frameworks|methoden|technologien)\b",
                r"\bbereitschaft.{0,20}(neue|new).{0,20}(technolog|tools|methoden)\b",
            ],
            "time_management": [
                r"\btime\s+management\b",
                r"\bprioriti[sz](e|ing|ation)\s+(of\s+)?(tasks|workload|deadlines)\b",
                r"\bmanage\s+(tasks|workload)\b",
                r"\bdeadline-?\s*oriented\b",
                r"\bability\s+to\s+meet\s+deadlines\b",
                r"\bwork\s+under\s+pressure\b",
                r"\bstress\s+tolerance\b",
                r"\bzeitmanagement\b",
                r"\bpriorisier(ung|ungsfähigkeit)\s*(von\s+)?(aufgaben|anforderungen)?\b",
                r"\btermintreue\b", r"\bbelastbarkeit\b", r"\bstress(toleranz)?\b",
            ],
            "autonomy": [
                r"\bself-?starter\b",
                r"\bproactive\b",
                r"\bautonomous(ly)?\b",
                r"\bself-?motivated\b",
                r"\bindependent(ly)?\s+work(ing)?\b",
                r"\bachievement-?based\s+drive\b",
                r"\bhighly[- ]motivated\b",
                r"\beigeninitiative\b",
                r"\bselbstständig(e)?\s+arbeitsweise\b",
                r"\beigenverantwortlich\b",
                r"\bselbständigkeit\b",
                r"\bselbstständig\b(?!\s+einzuarbeiten)",
                r"\bhohem?\s+maß\s+an\s+selbst(ständigkeit|ändigem\s+arbeiten)\b",
            ],
        }
        self.compiled_patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.context_patterns.items()
        }

    def extract_by_dictionary(self, text: str) -> set:
        if pd.isna(text):
            return set()
        text_lower = str(text).lower()
        found = set()
        for skill in self.soft_skills_dict:
            if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
                found.add(skill)
        return found

    def extract_by_patterns(self, text: str) -> dict:
        if pd.isna(text):
            return {}
        text = str(text)
        results = defaultdict(list)
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    results[category].append(match.group())
        return dict(results)

    def extract_all(self, text: str) -> dict:
        dict_skills = self.extract_by_dictionary(text)
        pattern_skills = self.extract_by_patterns(text)
        categories = set(pattern_skills.keys())
        canon_from_patterns = {
            self.CATEGORY_TO_CANON[c] for c in categories if c in self.CATEGORY_TO_CANON
        }
        soft_final = set(dict_skills) | canon_from_patterns
        return {
            "dictionary_match": list(dict_skills),
            "pattern_match": pattern_skills,
            "all_categories": list(categories),
            "soft_skills_final": list(soft_final),
        }


# ============================================================
# HYBRID MULTILINGUAL SKILL EXTRACTOR
# ============================================================
class MultilingualHybridSkillExtractor(FastSoftSkillExtractor):

    GERMAN_STRONG_MARKERS = [
        "deutschkenntnisse", "fließend",
        "erfahrung mit", "aufgaben", "anforderungen",
        "wir bieten", "das bringst du mit",
        "m/w/d",
    ]

    TECH_KEYWORD_REGEX = [
        r"\bjava\b", r"\b(java\s*ee|j2ee|jee)\b", r"\bpython\b",
        r"\bgolang\b",
        r"\bgo\b(?!\s+(to|ahead|back|through|over|on|away|home|out|further|live|beyond))",
        r"\bc\+\+\b", r"\bc#",
        r"\bkotlin\b", r"\bscala\b",
        r"\b(?<![a-z])c(?![+#\w])\b",
        r"\bjavascript\b", r"\btypescript\b",
        r"\bejb\b", r"\bjpa\b", r"\bjsf\b", r"\bsoap\b",
        r"\brest(ful)?\s*(api|service|endpoint)s?\b",
        r"\bgraphql\b",
        r"\bspring\b", r"\bspring\s+boot\b",
        r"\basp\.net\b", r"\basp\.net\s+core\b", r"\bblazor\b", r"\bentity\s+framework\b",
        r"\bunity3d\b", r"\bunity\b",
        r"\breact\b", r"\bvue\b", r"\bangular\b",
        r"\bsolidjs\b", r"\bsolid\.js\b", r"\bsolid\s+js\b",
        r"\bwebgl\b", r"\bweb\s*gl\b",
        r"\bplaywright\b", r"\bvite\b", r"\bnx\b", r"\bmonorepo\b", r"\bmono\s*repo\b",
        r"\bweb\s*components?\b",
        r"\bnode\.js\b",
        r"\bdocker\b", r"\bkubernetes\b", r"\bopenshift\b", r"\bargocd\b", r"\bci/?cd\b",
        r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bterraform\b",
        r"\bgithub\s*actions?\b", r"\bgitlab\s*ci\b", r"\bjenkins\b",
        r"\brelational\s+databases?\b", r"\brelationale\s+datenbanken?\b",
        r"\bpostgres(ql)?\b", r"\bmysql\b", r"\bmongodb\b", r"\bmongo\s+db\b",
        r"\boracle\b", r"\bsql\s+server\b", r"\bsql\b",
        r"\bredis\b", r"\belasticsearch\b", r"\bcassandra\b",
        r"\boauth2?\b", r"\bopenid\b", r"\bjwt\b",
    ]

    TECH_SECTION_MARKERS = [
        "what we expect", "requirements", "your profile", "qualifications",
        "skills", "what you bring", "what you bring to the table",
        "what skills & experience", "our tech stack", "tech stack",
        "some of the technologies", "technologies you will be working",
        "tools & technologies",
        "was wir erwarten", "anforderungen", "dein profil", "ihr profil",
        "qualifikation",
        "das bringst du mit", "du bringst mit",
        "voraussetzungen", "was du mitbringst", "womit du punktest",
        "fundierte kenntnisse",
        "kenntnisse in",
        "gute kenntnisse",
        "sehr gute kenntnisse",
        "deine kenntnisse",
        "ihre kenntnisse",
        "das beschreibt dich",
        "du hast",
    ]

    TECH_CANON_MAP = {
        "jee": "java ee", "j2ee": "java ee", "java ee": "java ee",
        "postgres": "postgresql", "postgresql": "postgresql",
        "cicd": "ci/cd", "ci cd": "ci/cd", "ci/cd": "ci/cd",
        "nodejs": "node.js", "node.js": "node.js",
        "mssql": "sql server", "ms sql": "sql server", "sql server": "sql server",
        "springboot": "spring boot", "spring boot": "spring boot",
        "solidjs": "solidjs", "solid.js": "solidjs", "solid js": "solidjs",
        "webgl": "webgl", "web gl": "webgl",
        "monorepo": "monorepo", "mono repo": "monorepo",
        "mongo db": "mongodb",
        "go": "go",
        "golang": "go",
    }

    SOFT_CANON_MAP = {
        "collaboration": "teamwork", "collaborative": "teamwork",
        "team player": "teamwork", "teamarbeit": "teamwork",
        "teamfähigkeit": "teamwork", "zusammenarbeit": "teamwork",
        "teamgeist": "teamwork",
        "analytical skills": "problem solving", "analytical thinking": "problem solving",
        "analytische fähigkeiten": "problem solving", "analytisches denken": "problem solving",
        "problem-solving": "problem solving", "problemlösungsfähigkeit": "problem solving",
        "willingness to learn": "adaptability", "lernbereitschaft": "adaptability",
        "flexibility": "adaptability", "flexibilität": "adaptability",
        "adaptable": "adaptability", "anpassungsfähigkeit": "adaptability",
        "communication skills": "communication", "kommunikationsfähigkeit": "communication",
        "verbal communication": "communication", "written communication": "communication",
        "prioritization": "time management", "priorisierungsfähigkeit": "time management",
        "zeitmanagement": "time management", "deadline management": "time management",
        "termintreue": "time management", "belastbarkeit": "time management",
        "self-starter": "autonomy",
        "proactive": "autonomy",
        "self-motivation": "autonomy",
        "initiative": "autonomy",
        "eigeninitiative": "autonomy",
        "autonomous working style": "autonomy",
        "independent working style": "autonomy",
        "selbstständige arbeitsweise": "autonomy",
        "attention to detail": "attention to detail",
        "detail-oriented": "attention to detail",
        "sorgfalt": "attention to detail",
        "reliability": "reliability",
        "dependability": "reliability",
        "zuverlässigkeit": "reliability",
    }

    def __init__(
        self,
        model_en: str = "jjzha/jobbert_skill_extraction",
        model_multi: str = "jjzha/escoxlmr_skill_extraction",
        device: int = -1,
        use_quantization: bool = True,
    ):
        super().__init__()

        if use_quantization:
            print("  ⚡ Using INT8 dynamic quantization (faster CPU inference)")
            tok_en = AutoTokenizer.from_pretrained(model_en)
            tok_en.model_max_length = 512
            mdl_en = AutoModelForTokenClassification.from_pretrained(model_en)
            mdl_en = torch.quantization.quantize_dynamic(
                mdl_en, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.ner_en = pipeline(
                "token-classification", model=mdl_en, tokenizer=tok_en,
                aggregation_strategy="first", device=device,
            )

            tok_multi = AutoTokenizer.from_pretrained(model_multi)
            tok_multi.model_max_length = 512
            mdl_multi = AutoModelForTokenClassification.from_pretrained(model_multi)
            mdl_multi = torch.quantization.quantize_dynamic(
                mdl_multi, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.ner_multi = pipeline(
                "token-classification", model=mdl_multi, tokenizer=tok_multi,
                aggregation_strategy="first", device=device,
            )
        else:
            self.ner_en = pipeline(
                "token-classification", model=model_en,
                aggregation_strategy="first", device=device,
            )
            self.ner_multi = pipeline(
                "token-classification", model=model_multi,
                aggregation_strategy="first", device=device,
            )

        self.tokenizer_en = self.ner_en.tokenizer
        self.tokenizer_multi = self.ner_multi.tokenizer
        self.soft_set = set(s.lower().strip() for s in SOFT_SKILLS)
        self.tech_keyword_compiled = [re.compile(p, re.IGNORECASE) for p in self.TECH_KEYWORD_REGEX]

        self.tech_tokens = {
            "java", "jee", "j2ee", "java ee", "ejb", "jpa", "jsf", "soap", "rest", "graphql",
            "spring", "spring boot", "python", "golang", "go", "c++", "c#", "kotlin", "scala",
            "javascript", "typescript", "node.js",
            "docker", "kubernetes", "openshift", "ci/cd", "aws", "azure", "gcp", "terraform",
            "sql", "postgres", "postgresql", "mysql", "mongodb", "oracle", "sql server",
            "oauth", "oauth2", "openid", "jwt",
            "api", "microservices", "backend", "database", "datenbank",
            "solidjs", "webgl", "playwright", "vite", "nx", "monorepo",
            "redis", "elasticsearch", "jenkins", "argocd", "openshift",
            "github actions", "gitlab ci",
            "asp.net", "asp.net core", "blazor", "entity framework", "unity", "unity3d",
        }

        self.stop_spans = {
            "benefits", "bonus", "salary", "pension", "vacation", "ticket",
            "equal opportunity", "privacy policy", "apply now",
            "job ticket", "deutschland-ticket", "work-life balance",
            "annual bonus", "company canteen", "application portal",
            "security check", "business trips",
        }

        self.generic_nontech = {
            "project", "projects", "product", "products", "solution", "solutions",
            "software", "software solutions", "modern software solutions",
            "system", "systems", "inventory system", "software landscape",
            "development", "further development", "training", "courses",
            "team", "teams", "collaboration", "customers", "user", "users",
            "requirements", "process", "processes", "experience",
            "knowledge", "skills", "ability", "abilities", "mindset",
            "office", "home office", "hybrid", "remote",
            "design", "exciting projects", "important projects",
            "your it skills", "it skills",
            "technical systems", "digital offerings", "new technologies",
            "geospatial data infrastructure", "geospatial cloud-based applications",
            "robust user interfaces", "high-performance geospatial",
            "important and exciting projects", "modern backend development",
            "denken", "ganzheitlich denken", "hands-on",
            "weiterentwicklung", "aufbau", "neue funktionen",
        }

        self.tech_hint_pattern = re.compile(
            r"((?:[a-z]+(?:\.[a-z]+)+)|(?:ci/cd)|(?:\b[a-z]*\d+[a-z]*\b)"
            r"|(?:\bapi\b|\betl\b|\bml\b|\bai\b|\bsql\b))",
            re.IGNORECASE,
        )

        self.section_markers_lower = [m.lower() for m in self.TECH_SECTION_MARKERS]
        self.language_extractor = LanguageExtractor()

    def detect_language(self, text: str) -> str:
        if not text:
            return "mixed"
        s = text[:3000].lower()
        if re.search(r"[äöüß]", s):
            return "de"
        if any(m in s for m in self.GERMAN_STRONG_MARKERS):
            return "de"
        if detect is not None:
            try:
                lang = detect(s[:1500])
                if lang.startswith("de"):
                    return "de"
                if lang.startswith("en"):
                    return "en"
            except Exception:
                pass
        return "mixed"

    @staticmethod
    def _clean_span(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"^[^\wäöüß+#./-]+|[^\wäöüß+#./-]+$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s)
        tech_2char_exceptions = {"nx", "go", "c", "r", "ai", "ml", "ci", "cd", "c#", "f#"}
        if len(s) < 3 and s not in tech_2char_exceptions:
            return ""
        tech_vowel_exceptions = {"ejb", "aws", "azure", "ajax", "api", "ios", "ide", "orm", "oauth"}
        if re.match(r"^[aeiou]", s) and len(s) < 6 and s not in tech_vowel_exceptions:
            return ""
        return s

    def _normalize_tech(self, skill: str) -> str:
        s = self._clean_span(skill)
        if not s:
            return ""
        s = s.replace("javaee", "java ee").replace("java- ee", "java ee")
        s = s.replace("springboot", "spring boot").replace("nodejs", "node.js")
        s = s.replace("ci cd", "ci/cd")
        s = s.replace("golang", "go")
        s = s.replace("mongo db", "mongodb")
        return self.TECH_CANON_MAP.get(s, s)

    def _slice_for_technical(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        lower = t.lower()
        hits = [lower.find(m) for m in self.section_markers_lower if lower.find(m) != -1]
        return t[min(hits):] if hits else t

    def _preprocess_for_regex(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'([A-Z]{2,})C#', r'\1 C#', text)
        text = re.sub(r'C#([A-Z][a-z])', r'C# \1', text)
        text = re.sub(r'([a-zA-Z0-9])/([a-zA-Z])', r'\1 / \2', text)
        text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r' \2', text)
        text = re.sub(r'(?<![CcFf])([#\+])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([A-BD-EG-Za-bd-eg-z])([#\+])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_tech_keywords(self, text: str) -> List[str]:
        if pd.isna(text) or not str(text).strip():
            return []
        t = self._preprocess_for_regex(str(text))
        found = []
        for pat in self.tech_keyword_compiled:
            for m in pat.finditer(t):
                norm = self._normalize_tech(m.group())
                if norm and norm not in found:
                    found.append(norm)
        return found

    def _run_ner_truncated(self, ner_pipe, tokenizer, text: str) -> List[str]:
        encoding = tokenizer(
            text, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=510,
            return_offsets_mapping=True
        )
        
        if len(encoding['offset_mapping']) > 0:
            last_offset = encoding['offset_mapping'][-1][1]
            clipped = text[:last_offset]
        else:
            clipped = text
        
        out = []
        for e in ner_pipe(clipped):
            raw_word = e.get("word", "")
            if raw_word.startswith("##"):
                continue

            start, end = e.get("start"), e.get("end")
            if start is not None and end is not None and end > start:
                span_text = clipped[start:end]
            else:
                span_text = raw_word

            w = self._clean_span(span_text)
            if not w or len(w) < 3:
                continue
            if w in self.stop_spans:
                continue
            if len(w.split()) > 5:
                continue
            out.append(w)
        return out

    def _run_ner_batch(self, ner_pipe, tokenizer, texts: List[str]) -> List[List[str]]:
        """Run NER on a batch of texts — far faster than one-by-one on CPU."""
        if not texts:
            return []

        clipped_texts = []
        for text in texts:
            if not text or not str(text).strip():
                clipped_texts.append("")
                continue
            encoding = tokenizer(
                text, add_special_tokens=False, truncation=True,
                max_length=510, return_offsets_mapping=True
            )
            if len(encoding['offset_mapping']) > 0:
                last_offset = encoding['offset_mapping'][-1][1]
                clipped_texts.append(text[:last_offset])
            else:
                clipped_texts.append(text)

        # Only pass non-empty texts to pipeline
        indexed = [(i, t) for i, t in enumerate(clipped_texts) if t and t.strip()]
        if not indexed:
            return [[] for _ in texts]

        indices, valid_texts = zip(*indexed)
        pipe_results = ner_pipe(list(valid_texts), batch_size=min(len(valid_texts), 8))

        result_map: dict = {}
        for idx, clipped, result_list in zip(indices, valid_texts, pipe_results):
            spans = []
            for e in result_list:
                raw_word = e.get("word", "")
                if raw_word.startswith("##"):
                    continue
                start, end = e.get("start"), e.get("end")
                if start is not None and end is not None and end > start:
                    span_text = clipped[start:end]
                else:
                    span_text = raw_word
                w = self._clean_span(span_text)
                if not w or len(w) < 3:
                    continue
                if w in self.stop_spans:
                    continue
                if len(w.split()) > 5:
                    continue
                spans.append(w)
            result_map[idx] = spans

        return [result_map.get(i, []) for i in range(len(texts))]

    def extract_spans_dual(self, text: str, lang: str) -> List[str]:
        if pd.isna(text) or not str(text).strip():
            return []
        text = str(text)
        spans = []
        if lang == "en":
            spans.extend(self._run_ner_truncated(self.ner_en, self.tokenizer_en, text))
        else:
            spans.extend(self._run_ner_truncated(self.ner_multi, self.tokenizer_multi, text))
            spans.extend(self._run_ner_truncated(self.ner_en, self.tokenizer_en, text))
        seen = set()
        return [w for w in spans if not (w in seen or seen.add(w))]

    def _matches_known_tech(self, s: str) -> bool:
        return any(pat.search(s) for pat in self.tech_keyword_compiled)

    def _contains_tech_token(self, s: str) -> bool:
        return any(tok in self.tech_tokens for tok in re.findall(r"[a-zA-Zäöüß0-9#+./-]+", s.lower()))

    def _looks_technical(self, span: str) -> bool:
        s = span.strip().lower()
        if not s or s in self.stop_spans or s in self.generic_nontech:
            return False
        if len(s) <= 2 and s not in {"c", "r"}:
            return False
        if len(s.split()) > 4 or len(s) > 50:
            return False
        reject_verbs = {
            "collaborate", "develop", "maintain", "build", "create", "design",
            "work", "contribute", "participate", "implement", "manage",
            "ensure", "provide", "support", "deliver", "drive", "lead",
            "analyze", "improve", "optimize", "conduct", "perform",
            "designst", "entwickelst", "berätst", "unterstützt",
            "gibst", "hilfst", "übernehmen", "mitgestalten",
        }
        if s.split()[0] in reject_verbs:
            return False
        if self._matches_known_tech(s):
            return True
        strong_signal = (
            any(ch in s for ch in ["#", ".", "/", "+"])
            or bool(re.search(r"\d", s))
            or bool(self.tech_hint_pattern.search(s))
        )
        return strong_signal and self._contains_tech_token(s)

    def classify_spans(self, spans: List[str]) -> Tuple[List[str], List[str]]:
        soft, technical_raw = [], []
        for s in spans:
            s_lower = s.lower().strip()
            is_soft = (
                s_lower in self.soft_set
                or any(s_lower in ss or ss in s_lower for ss in self.soft_set if len(ss) > 4)
            )
            (soft if is_soft else technical_raw).append(s)
        return soft, [s for s in technical_raw if self._looks_technical(s)]

    def _deduplicate_soft_skills(self, soft_skills: List[str]) -> List[str]:
        canonical = set()
        for skill in soft_skills:
            sl = skill.lower().strip()
            canonical.add(self.SOFT_CANON_MAP.get(sl, sl))
        return [s for s in canonical if len(s) > 2 and s not in {"skill", "skills"}]

    def extract_all(self, text: str) -> dict:
        base = super().extract_all(text)
        full_text = "" if pd.isna(text) else str(text)
        lang = self.detect_language(full_text)
        languages = self.language_extractor.extract_languages(full_text)
        tech_text = self._slice_for_technical(full_text)

        spans_all = self.extract_spans_dual(full_text, lang=lang)
        spans_req = self.extract_spans_dual(tech_text, lang=lang)
        span_soft, _ = self.classify_spans(spans_all)
        _, span_tech_raw = self.classify_spans(spans_req)

        span_tech = []
        for s in span_tech_raw:
            norm = self._normalize_tech(s)
            if norm and norm not in span_tech:
                span_tech.append(norm)

        tech_kw = self.extract_tech_keywords(tech_text if tech_text else full_text)

        final_tech, seen = [], set()
        for x in tech_kw + span_tech:
            x = self._normalize_tech(x)
            if x and x not in self.generic_nontech and x not in self.stop_spans and x not in seen:
                seen.add(x)
                final_tech.append(x)

        return {
            **base,
            "languages": languages,
            "lang_detected": lang,
            "soft_skills_final": self._deduplicate_soft_skills(base.get("soft_skills_final", [])),
            "skill_spans": spans_all,
            "skill_spans_soft": span_soft,
            "technical_skills_final": final_tech,
            "tech_keywords_regex": tech_kw,
        }

    def extract_all_batch(self, texts: List[str]) -> List[dict]:
        """Process a list of job descriptions with batched NER (~4-6x faster than one-by-one)."""
        if not texts:
            return []

        full_texts = ["" if pd.isna(t) else str(t) for t in texts]
        langs = [self.detect_language(t) for t in full_texts]
        tech_texts = [self._slice_for_technical(t) for t in full_texts]

        non_en_indices = [i for i, l in enumerate(langs) if l != "en"]

        # Batch NER: EN model on ALL texts (full + tech-sliced)
        full_spans_en = self._run_ner_batch(self.ner_en, self.tokenizer_en, full_texts)
        tech_spans_en = self._run_ner_batch(self.ner_en, self.tokenizer_en, tech_texts)

        # Batch NER: multilingual model on non-EN texts only
        full_spans_multi: dict = {i: [] for i in range(len(texts))}
        tech_spans_multi: dict = {i: [] for i in range(len(texts))}
        if non_en_indices:
            ne_full = self._run_ner_batch(self.ner_multi, self.tokenizer_multi,
                                          [full_texts[i] for i in non_en_indices])
            ne_tech = self._run_ner_batch(self.ner_multi, self.tokenizer_multi,
                                          [tech_texts[i] for i in non_en_indices])
            for k, (fs, ts) in enumerate(zip(ne_full, ne_tech)):
                full_spans_multi[non_en_indices[k]] = fs
                tech_spans_multi[non_en_indices[k]] = ts

        results = []
        for i in range(len(texts)):
            full_text = full_texts[i]
            lang = langs[i]
            tech_text = tech_texts[i]

            raw_all = full_spans_en[i] if lang == "en" else full_spans_multi[i] + full_spans_en[i]
            seen: set = set()
            spans_all = [w for w in raw_all if not (w in seen or seen.add(w))]

            raw_req = tech_spans_en[i] if lang == "en" else tech_spans_multi[i] + tech_spans_en[i]
            seen = set()
            spans_req = [w for w in raw_req if not (w in seen or seen.add(w))]

            span_soft, _ = self.classify_spans(spans_all)
            _, span_tech_raw = self.classify_spans(spans_req)

            span_tech = []
            for s in span_tech_raw:
                norm = self._normalize_tech(s)
                if norm and norm not in span_tech:
                    span_tech.append(norm)

            tech_kw = self.extract_tech_keywords(tech_text if tech_text else full_text)

            final_tech, seen_t = [], set()
            for x in tech_kw + span_tech:
                x = self._normalize_tech(x)
                if x and x not in self.generic_nontech and x not in self.stop_spans and x not in seen_t:
                    seen_t.add(x)
                    final_tech.append(x)

            base = FastSoftSkillExtractor.extract_all(self, full_text)
            languages = self.language_extractor.extract_languages(full_text)

            results.append({
                **base,
                "languages": languages,
                "lang_detected": lang,
                "soft_skills_final": self._deduplicate_soft_skills(base.get("soft_skills_final", [])),
                "skill_spans": spans_all,
                "skill_spans_soft": span_soft,
                "technical_skills_final": final_tech,
                "tech_keywords_regex": tech_kw,
            })

        return results


# ============================================================
# EXPORT HELPERS
# ============================================================
def export_json_records(df: pd.DataFrame, json_path: str) -> None:
    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ============================================================
# ENHANCED MAIN FUNCTION WITH CHECKPOINTING
# ============================================================

# ============================================================
# PARALLEL PROCESSING HELPER
# ============================================================

# Global extractor for worker processes (initialized once per worker)
_worker_extractor = None

def init_worker(device=-1):
    """Initialize worker process with extractor (called once per worker)"""
    global _worker_extractor
    import warnings
    import logging
    import os
    
    # Suppress all verbose output
    warnings.filterwarnings('ignore')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    _worker_extractor = MultilingualHybridSkillExtractor(device=device)

def process_job_batch(descriptions_list):
    """Process a BATCH of job descriptions using batched NER — much faster than one-by-one."""
    try:
        return _worker_extractor.extract_all_batch(descriptions_list)
    except Exception as e:
        error_result = {
            "lang_detected": "error",
            "languages": [],
            "soft_skills_final": [],
            "technical_skills_final": [],
            "tech_keywords_regex": [],
            "dictionary_match": [],
            "all_categories": [],
            "pattern_match": {},
            "skill_spans": [],
            "skill_spans_soft": [],
        }
        return [dict(error_result) for _ in descriptions_list]

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def analyze_dataset_with_checkpoint(
    csv_file: str,
    output_csv: str = "jobs_with_skills_extracted.csv",
    output_json: str = "jobs_with_skills_extracted.json",
    db_path: str = "skill_extraction_checkpoint.db",
    device: int = -1,
    batch_size: int = 100,
    sample_size: int = None,
    num_workers: int = 3,  # Number of parallel workers
):
    """
    Process dataset with SQLite checkpointing for auto-resume capability
    
    Parameters:
    - csv_file: Input CSV file path
    - output_csv: Final output CSV path
    - output_json: Final output JSON path
    - db_path: SQLite database path for checkpoints
    - device: -1 for CPU, 0 for GPU
    - batch_size: Number of jobs to process before saving checkpoint
    - sample_size: Number of jobs to process (None for all)
    """
    
    print("=" * 70)
    print("SKILL EXTRACTION WITH AUTO-CHECKPOINT/RESUME")
    print("=" * 70)
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager(db_path)
    
    # Read full dataset
    print("\n📂 Reading dataset...")
    df = pd.read_csv(csv_file)
    print(f"Total jobs in file: {len(df):,}")
    
    # Apply sample size if specified
    if sample_size is not None and sample_size < len(df):
        print(f"⚠️  TEST MODE: Will process only first {sample_size} rows")
        df = df.head(sample_size)
    
    total_jobs = len(df)
    
    # Check for existing progress
    last_processed = checkpoint.get_last_processed_index()
    processed_count = checkpoint.get_processed_count()
    
    if processed_count > 0:
        print(f"\n🔄 RESUMING from previous run")
        print(f"   Already processed: {processed_count:,} jobs")
        print(f"   Last processed index: {last_processed}")
        print(f"   Remaining: {total_jobs - processed_count:,} jobs")
    else:
        print(f"\n🆕 Starting fresh extraction")
    
    # Determine which jobs to process
    start_idx = last_processed + 1
    
    if start_idx >= total_jobs:
        print("\n✅ All jobs already processed!")
        print("Exporting results...")
        df_result = checkpoint.export_to_dataframe()
        
        # Merge with original data using job_index
        df_final = df.copy()
        df_final = df_final.reset_index(drop=False).rename(columns={'index': 'job_index'})
        
        # Merge on job_index
        df_result_merge = df_result.drop(columns=['title', 'company_name', 'location'], errors='ignore')
        df_final = df_final.merge(df_result_merge, on='job_index', how='left')
        df_final = df_final.drop(columns=['job_index'])
        
        checkpoint.close()
        return df_final
    
    # Initialize extractor (only needed for single-threaded mode)
    extractor = None
    if num_workers <= 1:
        print(f"\n🤖 Initializing extractor (EN + Multilingual)...")
        extractor = MultilingualHybridSkillExtractor(device=device)
    else:
        print(f"\n🤖 Parallel mode: Each worker will initialize its own extractor")
    
    # Process in batches
    print(f"\n⚙️  Processing in batches of {batch_size} with {num_workers} parallel workers")
    print(f"Starting from index {start_idx} to {total_jobs - 1}")
    print("-" * 70)
    
    import time

    def _run_batches(pool):
        """Inner function that runs all batches using the given pool (or None for single-threaded)."""
        global_start_time = time.time()

        for batch_start in range(start_idx, total_jobs, batch_size):
            batch_end = min(batch_start + batch_size, total_jobs)
            batch_df = df.iloc[batch_start:batch_end].copy()

            print(f"\n📦 Batch: [{batch_start:,} - {batch_end-1:,}] ({len(batch_df)} jobs)")

            if pool is not None:
                CHUNK_SIZE = 8  # jobs per worker call — batched NER within each chunk
                descriptions = list(batch_df["description"])
                chunks = [descriptions[j:j+CHUNK_SIZE] for j in range(0, len(descriptions), CHUNK_SIZE)]
                flat_results = []
                for chunk_results in pool.imap(process_job_batch, chunks):
                    flat_results.extend(chunk_results)
                    jobs_done_total = batch_start + len(flat_results)
                    total_elapsed = time.time() - global_start_time
                    jobs_since_start = jobs_done_total - start_idx
                    jobs_per_sec = jobs_since_start / total_elapsed if total_elapsed > 0 else 0.001
                    remaining_secs = (total_jobs - jobs_done_total) / jobs_per_sec

                    progress_pct = (jobs_done_total / total_jobs) * 100
                    bar_length = 30
                    filled = int(bar_length * jobs_done_total / total_jobs)
                    bar = '█' * filled + '░' * (bar_length - filled)

                    print(f"\r   [{bar}] {jobs_done_total:,}/{total_jobs:,} ({progress_pct:.1f}%) | "
                          f"{total_elapsed/3600:.1f}h elapsed | ~{remaining_secs/3600:.1f}h left",
                          end='', flush=True)
                print()
                batch_df["skills_extracted"] = flat_results
            else:
                # Single-threaded with batched NER (8 jobs per pipeline call)
                CHUNK_SIZE = 8
                descriptions = list(batch_df["description"])
                flat_results = []
                for j in range(0, len(descriptions), CHUNK_SIZE):
                    chunk = descriptions[j:j+CHUNK_SIZE]
                    flat_results.extend(extractor.extract_all_batch(chunk))

                    jobs_done_total = batch_start + len(flat_results)
                    total_elapsed = time.time() - global_start_time
                    jobs_since_start = jobs_done_total - start_idx
                    jobs_per_sec = jobs_since_start / total_elapsed if total_elapsed > 0 else 0.001
                    remaining_secs = (total_jobs - jobs_done_total) / jobs_per_sec

                    progress_pct = (jobs_done_total / total_jobs) * 100
                    bar_length = 30
                    filled = int(bar_length * jobs_done_total / total_jobs)
                    bar = '█' * filled + '░' * (bar_length - filled)

                    print(f"\r   [{bar}] {jobs_done_total:,}/{total_jobs:,} ({progress_pct:.1f}%) | "
                          f"{total_elapsed/3600:.1f}h elapsed | ~{remaining_secs/3600:.1f}h left",
                          end='', flush=True)
                print()
                batch_df["skills_extracted"] = flat_results

            # Create output columns
            batch_df["lang_detected"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("lang_detected", "mixed") if isinstance(x, dict) else "mixed"
            )
            batch_df["languages"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("languages", []) if isinstance(x, dict) else []
            )
            batch_df["soft_skills_final"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("soft_skills_final", []) if isinstance(x, dict) else []
            )
            batch_df["technical_skills_final"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("technical_skills_final", []) if isinstance(x, dict) else []
            )
            batch_df["tech_keywords_regex"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("tech_keywords_regex", []) if isinstance(x, dict) else []
            )
            batch_df["soft_skills_dict"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("dictionary_match", []) if isinstance(x, dict) else []
            )
            batch_df["soft_skills_categories"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("all_categories", []) if isinstance(x, dict) else []
            )
            batch_df["skill_spans"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("skill_spans", []) if isinstance(x, dict) else []
            )
            batch_df["skill_spans_soft"] = batch_df["skills_extracted"].apply(
                lambda x: x.get("skill_spans_soft", []) if isinstance(x, dict) else []
            )

            # Save batch to checkpoint
            checkpoint.save_batch(batch_df, batch_start)
            checkpoint.update_progress(batch_end - 1, total_jobs, batch_size)

            progress_pct = (batch_end / total_jobs) * 100
            print(f"   ✓ Checkpoint saved — overall: {batch_end:,}/{total_jobs:,} ({progress_pct:.1f}%)")

    try:
        if num_workers > 1:
            print(f"\n🤖 Initializing {num_workers} workers (loading models ONCE, ~30s wait)...")
            with Pool(processes=num_workers, initializer=init_worker, initargs=(device,)) as pool:
                print("✅ Workers ready — starting processing...\n")
                _run_batches(pool)
        else:
            _run_batches(None)

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("✓ Progress saved to checkpoint database")
        print(f"Run again to resume from index {checkpoint.get_last_processed_index() + 1}")
        checkpoint.close()
        return None

    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        print("✓ Progress saved to checkpoint database")
        print(f"Run again to resume from index {checkpoint.get_last_processed_index() + 1}")
        checkpoint.close()
        raise
    
    # Export final results
    print("\n" + "=" * 70)
    print("📊 EXPORTING RESULTS")
    print("=" * 70)
    
    df_result = checkpoint.export_to_dataframe()
    
    # Merge with original data to preserve all columns using job_index
    df_final = df.copy()
    df_final = df_final.reset_index(drop=False).rename(columns={'index': 'job_index'})
    
    # Merge on job_index
    df_result_merge = df_result.drop(columns=['title', 'company_name', 'location'], errors='ignore')
    df_final = df_final.merge(df_result_merge, on='job_index', how='left')
    df_final = df_final.drop(columns=['job_index'])
    
    # Add count columns
    df_final["soft_skills_count_dict"] = df_final["soft_skills_dict"].apply(len)
    df_final["soft_skills_count_final"] = df_final["soft_skills_final"].apply(len)
    df_final["span_soft_count"] = df_final["skill_spans_soft"].apply(len)
    df_final["span_tech_count"] = df_final["technical_skills_final"].apply(len)
    
    # Save outputs
    print(f"\n💾 Saving CSV to: {output_csv}")
    df_final.to_csv(output_csv, index=False)
    
    print(f"💾 Saving JSON to: {output_json}")
    df_final.to_json(output_json, orient='records', indent=2, force_ascii=False)
    
    # Display statistics
    print("\n" + "=" * 70)
    print("📈 TOP 15: Soft Skills")
    print("=" * 70)
    all_soft = [s for row in df_final["soft_skills_final"] for s in row]
    for i, (skill, count) in enumerate(Counter(all_soft).most_common(15), 1):
        print(f"{i:2d}. {skill:30s}: {count:5d} ({count / len(df_final) * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("📈 TOP 15: Technical Skills")
    print("=" * 70)
    all_tech = [s for row in df_final["technical_skills_final"] for s in row]
    for i, (skill, count) in enumerate(Counter(all_tech).most_common(15), 1):
        print(f"{i:2d}. {skill:30s}: {count:5d} ({count / len(df_final) * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("📈 TOP Languages Required")
    print("=" * 70)
    all_langs = [lang for row in df_final["languages"] for lang in row]
    for i, (lang, count) in enumerate(Counter(all_langs).most_common(), 1):
        print(f"{i:2d}. {lang:20s}: {count:5d} ({count / len(df_final) * 100:.1f}%)")
    
    print("\n✅ [OK] Done!")
    print(f"📁 Checkpoint database: {db_path}")
    print(f"💡 Tip: You can delete the .db file after successful completion")
    
    checkpoint.close()
    return df_final


if __name__ == "__main__":
    df_result = analyze_dataset_with_checkpoint(
        csv_file=r"E:\DS\Project-Group-1\Techlabs_WS2526_Team1_PC\combined_jobs.csv",
        output_csv="jobs_with_skills_extracted_full.csv",
        output_json="jobs_with_skills_extracted_full.json",
        db_path="skill_extraction_checkpoint.db",
        device=-1,         # -1 for CPU, 0 for CUDA GPU
        batch_size=100,    # Save checkpoint every 100 jobs
        sample_size=None,  # FULL DATASET: All 22,526 jobs
        num_workers=1,     # 1 worker — gets all CPU cores, uses batched NER for speed, no memory issues
    )
