
# ===========================
# IMPORT LIBRARIES
# ===========================
import math
import requests
import random
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import quote as encode
import pandas as pd
from datetime import datetime
from time import sleep
import os

# ===========================
# LOGGING SETUP
# ===========================

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "RESET": "\033[0m"
    }

    EMOJIS = {
        "INFO": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        emoji = self.EMOJIS.get(record.levelname, "")
        record.levelname = f"{emoji} {record.levelname}"
        return f"{color}{super().format(record)}{self.COLORS['RESET']}"

logger = logging.getLogger("LinkedInScraper")
logger.setLevel(logging.INFO)

formatter = ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler = logging.FileHandler("linkedin_scraper.log", encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ===========================
# BASE URLs
# ===========================
INIT_URL = "https://www.linkedin.com/jobs/search"
PAGE_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
POST_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/"

# ===========================
# SESSION & HEADERS
# ===========================
session = requests.Session()

HEADERS_POOL = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    }
]

# ===========================
# SEARCH FILTERS
# ===========================
KEYWORDS = [
    "software engineer", "software architect"
]

CITIES = ["Stuttgart", "Munich"]

TIME_RANGE = "r86400"   # last 24h
DISTANCE = "50"
JOB_TYPE = ["F"]
PLACE = ["2"]

# ===========================
# GLOBAL RETRY CONFIG
# ===========================
MAX_RETRIES = 3
BACKOFF_BASE = 5

# ===========================
# HELPER FUNCTIONS
# ===========================
def random_delay(min_s=2.5, max_s=6.5):
    sleep(random.uniform(min_s, max_s))

def params(start=0, keyword="", location=""):
    return (
        f"?keywords={encode(keyword)}"
        f"&f_TPR={TIME_RANGE}"
        f"&location={encode(location)}"
        f"&distance={DISTANCE}"
        f"&f_JT={encode(','.join(JOB_TYPE))}"
        f"&f_WT={encode(','.join(PLACE))}"
        f"&start={start}"
        f"&sortBy=DD"
    )

def request_with_retry(url, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            headers = random.choice(HEADERS_POOL)
            res = session.get(url, headers=headers, timeout=15)

            if res.status_code == 429:
                logger.warning(f"429 Rate limit hit (attempt {attempt}/{retries}) – sleeping...")
                sleep(60)
                continue

            if "captcha" in res.text.lower() or "sign in" in res.text.lower():
                logger.error("LinkedIn block detected (captcha/sign-in)")
                sleep(120)
                continue

            res.raise_for_status()
            return res

        except Exception as e:
            logger.warning(f"Request failed (attempt {attempt}/{retries}): {e}")
            sleep(BACKOFF_BASE * attempt)

    logger.error("Max retries exceeded")
    return None

# ===========================
# SCRAPER FUNCTIONS
# ===========================
def job_result(keyword, location):
    url = INIT_URL + params(keyword=keyword, location=location)
    res = request_with_retry(url)
    if not res:
        return 0

    soup = BeautifulSoup(res.text, "html.parser")
    span = soup.find("span", class_="results-context-header__job-count")
    if not span:
        return 0

    return int(span.text.replace(",", "").replace("+", "").strip())

def job_id_list_per_page(start, keyword, location):
    url = PAGE_URL + params(start, keyword, location)
    res = request_with_retry(url)
    if not res:
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    job_ids = []

    for li in soup.find_all("li"):
        try:
            urn = li.get("data-entity-urn") or li.find("a")["data-entity-urn"]
            job_ids.append(urn.split(":")[-1])
        except Exception:
            continue

    return job_ids

def job_detail(job_id):
    url = POST_URL + job_id
    res = request_with_retry(url)
    if not res:
        return None

    soup = BeautifulSoup(res.text, "html.parser")

    return {
        "id": job_id,
        "link": url,
        "title": soup.select_one("h1").text.strip() if soup.select_one("h1") else None,
        "company": soup.select_one(".topcard__org-name-link").text.strip() if soup.select_one(".topcard__org-name-link") else None,
        "location": soup.select_one(".topcard__flavor--bullet").text.strip() if soup.select_one(".topcard__flavor--bullet") else None,
        "time": soup.select_one(".posted-time-ago__text").text.strip() if soup.select_one(".posted-time-ago__text") else None,
        "description": soup.select_one(".show-more-less-html__markup").text.strip() if soup.select_one(".show-more-less-html__markup") else None,
    }

# ===========================
# MAIN
# ===========================
def main():
    all_jobs = []

    for city in CITIES:
        for keyword in KEYWORDS:
            total = job_result(keyword, city)
            if total == 0:
                logger.info(f"No jobs for '{keyword}' in {city}")
                continue

            pages = math.ceil(total / 25)
            logger.info(f"🔎 {keyword} | {city} → {total} jobs")

            job_ids = []
            for p in tqdm(range(pages), desc="Pages"):
                job_ids.extend(job_id_list_per_page(p * 25, keyword, city))
                random_delay()

            for jid in tqdm(job_ids, desc="Jobs"):
                detail = job_detail(jid)
                if detail:
                    all_jobs.append(detail)
                random_delay(2, 4)

            random_delay(10, 15)

    if all_jobs:
        df = pd.DataFrame(all_jobs)
        path = os.path.join(os.getcwd(), "linkedin_jobs.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Exported {len(df)} jobs → {path}")
    else:
        logger.warning("No jobs collected")

if __name__ == "__main__":
    main()
