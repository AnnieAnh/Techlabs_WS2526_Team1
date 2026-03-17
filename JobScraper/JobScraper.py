# =======================
# IMPORTS
# =======================
import math
import requests
import ratelimit
import random
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import quote as encode
import pandas as pd
from datetime import datetime
from time import sleep
import os
import json
import csv
from pathlib import Path

# =======================
# OUTPUT DIR
# =======================
OUTPUT_DIR = Path("job_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# =======================
# LOGGING
# =======================
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m'
    }
    RESET = '\033[0m'

    EMOJI = {
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌'
    }

    def format(self, record):
        emoji = self.EMOJI.get(record.levelname, '')
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{emoji} {record.levelname}"
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


def setup_logging():
    logger = logging.getLogger("linkedin_scraper")
    logger.setLevel(logging.DEBUG)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = OUTPUT_DIR / f"Log_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        ColoredFormatter("%(levelname)s %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()

# =======================
# CONSTANTS
# =======================
INIT_URL = 'https://www.linkedin.com/jobs/search'
PAGE_URL = 'https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search'
POST_URL = 'https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
}

# =======================
# SEARCH LISTS
# =======================
#TEST
#keywords_list = ["backend","software engineer"]
#locations_list = ["Berlin","Stuttgart"]

keywords_list = [
    "backend","software engineer","software architect","software developer","data engineer",
    "data analyst","data scientist","BI developer","Cloud Engineer","Cloud architect",
    "DevOps engineer","IT administrator","backend developer","frontend developer",
    "full stack developer","Apps developer","SAP developer","machine learning engineer",
    "AI engineer","cybersecurity engineer",
    "Softwareentwickler","Softwarearchitekt","Data Engineer","Datenanalyst","Data Scientist",
    "BI-Entwickler","Cloud Engineer","Cloud-Architekt","DevOps Engineer","IT-Administrator",
    "Backend Entwickler","Frontend Entwickler","FullStack Entwickler","Apps Entwickler",
    "SAP Entwickler","Machine Learning Engineer","KI-Ingenieur","IT-Security Engineer"
]

locations_list = [
    "Berlin","Stuttgart","Munich","Potsdam","Bremen","Hamburg","Frankfurt","Hanover",
    "Rostock","Cologne","Mainz","Saarbrücken","Dresden","Magdeburg","Kiel","Erfurt",
    "Düsseldorf","Dortmund","Essen","Leipzig","Nürnberg","Karlsruhe","Mannheim",
    "Augsburg","Wiesbaden","Münster","Bonn","Freiburg","Aachen","Heidelberg","Ulm",
    "Darmstadt","Regensburg","Bielefeld"
]

# =======================
# FILTERS
# =======================
time_range = 'r2592000'
distance = '100'
job_type = ['F']
place = ['2']
limit_jobs = 150

# =======================
# URL PARAMS
# =======================
def params(keyword, location, start=0):
    return (
        f"?keywords={encode(keyword)}"
        f"&f_TPR={time_range}"
        f"&location={encode(location)}"
        f"&distance={distance}"
        f"&f_JT={encode(','.join(job_type))}"
        f"&f_WT={encode(','.join(place))}"
        f"&position=1&pageNum=0"
        f"&start={start}&sortBy=DD"
    )

# =======================
# JOB COUNT
# =======================
def job_result(keyword, location):
    if limit_jobs > 0:
        return limit_jobs

    uri = INIT_URL + params(keyword, location)
    res = requests.get(uri, headers=HEADERS)
    soup = BeautifulSoup(res.text, 'html.parser')

    try:
        job_count = soup.find('span', {'class': 'results-context-header__job-count'}).text
        return int(job_count.strip().replace(",", "").replace("+", ""))
    except:
        return 0

# =======================
# JOB IDS
# =======================
def job_id_list_per_page(keyword, location, start):
    uri = PAGE_URL + params(keyword, location, start)
    res = requests.get(uri, headers=HEADERS)

    if not res.ok or len(res.history) > 0:
        sleep(1)
        return []

    soup = BeautifulSoup(res.text, 'html.parser')
    job_ids = []

    for li in soup.find_all('li'):
        try:
            job_id = li.find('div', {'class': 'base-card'}).get('data-entity-urn').split(':')[3]
            job_ids.append(job_id)
        except:
            pass

    return job_ids

# =======================
# JOB DETAILS
# =======================
def job_detail(job_id, keyword, location):
    try:
        uri = POST_URL + job_id
        res = requests.get(uri, headers=HEADERS)
        if not res.ok:
            return None

        soup = BeautifulSoup(res.text, 'html.parser')

        detail = {
            'search_keyword': keyword,
            'search_location': location,
            'id': job_id,
            'link': uri,
            'title': None,
            'company': None,
            'location_job': None,
            'time': None,
            'description': None,
            'level': None,
            'industry': None,
            'type': None,
            'function': None
        }

        try:
            anchor = soup.find("div", {"class": "top-card-layout__entity-info"}).find("a")
            detail['link'] = anchor.get('href')
            detail['title'] = anchor.text.strip()
        except:
            pass

        try:
            detail['company'] = soup.select_one('.topcard__org-name-link').text.strip()
        except:
            pass

        try:
            detail['location_job'] = soup.select_one('span.topcard__flavor--bullet').text.strip()
        except:
            pass

        try:
            detail['time'] = soup.select_one('.posted-time-ago__text').text.strip()
        except:
            pass

        try:
            detail['description'] = soup.select_one('.show-more-less-html__markup').text.strip()
        except:
            pass

        return detail
    except Exception as e:
        logger.error(e)
        return None

# =======================
# MAIN
# =======================
def main():
    all_jobs = []
    seen_job_ids = set()

    for keyword in keywords_list:
        for location in locations_list:
            logger.info(f"🔍 Searching: '{keyword}' in '{location}'")

            total = job_result(keyword, location)
            if total == 0:
                continue

            num_page = math.ceil(total / 25)

            job_ids = []
            for i in range(num_page):
                start = i * 25
                job_ids += job_id_list_per_page(keyword, location, start)

            for job_id in job_ids:
                if job_id in seen_job_ids:
                    continue

                seen_job_ids.add(job_id)

                detail = job_detail(job_id, keyword, location)
                if detail:
                    all_jobs.append(detail)

                sleep(0.5)

    if all_jobs:
        df = pd.DataFrame(all_jobs)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        csv_path = OUTPUT_DIR / f"dataset_{timestamp}.csv"
        json_path = OUTPUT_DIR / f"dataset_{timestamp}.json"

        df.to_csv(csv_path, index=False, encoding='utf-8',
                  quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)

        metadata = {
            "created_at": timestamp,
            "total_jobs": len(df),
            "unique_jobs": len(seen_job_ids)
        }

        with open(OUTPUT_DIR / f"Metadata_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"📦 Exported {len(all_jobs)} unique jobs")
    else:
        logger.warning("No jobs found")

# =======================
if __name__ == "__main__":
    main()
