"""
LinkedIn Scraper for JobMiner.

This scraper extracts job listings from LinkedIn.
"""

import sys
import os
import logging
import json
import re
from typing import List, Optional, Dict
from urllib.parse import urljoin, quote
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path to import base_scraper
# Path to JobMiner directory: e:\DS\Project-Group-1\JobMiner\
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels: Techlabs_WS2526_Team1\JobMiner -> Techlabs_WS2526_Team1 -> Project-Group-1
jobminer_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'JobMiner'))
sys.path.insert(0, jobminer_path)
from base_scraper import BaseScraper, JobListing


# ============================================================================
# EXTENDED JOB LISTING FOR LINKEDIN
# ============================================================================

@dataclass
class LinkedInJobListing(JobListing):
    """Extended job listing with additional LinkedIn-specific fields."""
    
    # Work arrangement
    work_mode: Optional[str] = None  # Remote, Hybrid, On-site
    
    # Company information
    company_industry: Optional[str] = None
    company_size: Optional[str] = None  # Number of employees
    company_description: Optional[str] = None
    company_address: Optional[str] = None
    company_rating: Optional[float] = None
    company_reviews_count: Optional[int] = None
    is_verified: Optional[bool] = None  # Company verification status
    
    # Job details
    seniority_level: Optional[str] = None  # Entry level, Mid-Senior, Executive, etc.
    required_skills: Optional[List[str]] = field(default_factory=list)
    experience_range: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert extended job listing to dictionary."""
        # Get base dictionary but handle scraped_at manually
        from datetime import datetime
        
        base_dict = {
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'description': self.description,
            'salary': self.salary,
            'job_type': self.job_type,
            'experience_level': self.experience_level,
            'posted_date': self.posted_date,
            'job_url': self.job_url,
            'scraped_at': self.scraped_at if isinstance(self.scraped_at, str) else (self.scraped_at.isoformat() if self.scraped_at else None),
        }
        base_dict.update({
            'work_mode': self.work_mode,
            'company_industry': self.company_industry,
            'company_size': self.company_size,
            'company_description': self.company_description,
            'company_address': self.company_address,
            'company_rating': self.company_rating,
            'company_reviews_count': self.company_reviews_count,
            'is_verified': self.is_verified,
            'seniority_level': self.seniority_level,
            'required_skills': self.required_skills,
            'experience_range': self.experience_range,
        })
        return base_dict


# ============================================================================
# COLORED LOGGING
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis"""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m',
    }

    EMOJI = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥',
    }

    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.COLORS['RESET'])
        emoji = self.EMOJI.get(levelname, '')
        record.levelname = f"{emoji} {levelname}"
        message = super().format(record)
        return f"{color}{message}{self.COLORS['RESET']}"


class LinkedinScraper(BaseScraper):
    """
    LinkedIn job scraper with time filtering.
    """
    
    def __init__(self, delay: float = 3.0, target_days: int = 120):
        """
        Initialize the LinkedIn scraper.
        
        Args:
            delay: Delay between requests in seconds
            target_days: Target number of days to scrape (for filtering, default 120)
        """
        super().__init__(delay=delay)
        self.base_url = "https://www.linkedin.com"
        self.search_url = f"{self.base_url}/jobs-guest/jobs/api/seeMoreJobPostings/search"
        self.target_days = target_days
    
    def parse_posted_date_to_days(self, posted_date_str: str) -> Optional[int]:
        """
        Parse LinkedIn posted date string to number of days ago.
        
        Args:
            posted_date_str: String like "2 days ago", "3 weeks ago", "1 month ago"
            
        Returns:
            Number of days ago, or None if cannot parse
        """
        if not posted_date_str:
            return None
        
        posted_date_str = posted_date_str.lower()
        
        # Match patterns like "2 days ago", "3 weeks ago", "1 month ago"
        # Hours
        if 'hour' in posted_date_str or 'hr' in posted_date_str:
            return 0  # Same day
        
        # Days
        match = re.search(r'(\d+)\s*day', posted_date_str)
        if match:
            return int(match.group(1))
        
        # Weeks
        match = re.search(r'(\d+)\s*week', posted_date_str)
        if match:
            return int(match.group(1)) * 7
        
        # Months
        match = re.search(r'(\d+)\s*month', posted_date_str)
        if match:
            return int(match.group(1)) * 30
        
        # Years (NEW: Handle "1 year ago", "2 years ago")
        match = re.search(r'(\d+)\s*year', posted_date_str)
        if match:
            return int(match.group(1)) * 365
        
        # If just says "posted" without number, assume recent
        if 'posted' in posted_date_str:
            return 1
        
        return None
    
    def scrape_jobs(self, search_term: str, location: str = "", max_pages: int = 1) -> List[LinkedInJobListing]:
        """
        Main method to scrape jobs for given criteria with age filtering.
        
        Args:
            search_term: Job title or keyword to search for
            location: Location to search in
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of LinkedInJobListing objects (filtered by age)
        """
        self.logger.info(f"Starting scrape for '{search_term}' in '{location}'")
        
        # Get job URLs
        job_urls = self.get_job_urls(search_term, location, max_pages)
        self.logger.info(f"Found {len(job_urls)} job URLs")
        
        # Parse each job
        jobs = []
        skipped_count = 0
        for i, url in enumerate(job_urls, 1):
            self.logger.debug(f"Parsing job {i}/{len(job_urls)}")
            job = self.parse_job(url)
            if job:
                jobs.append(job)
            else:
                # Job was either failed to parse or skipped due to age
                skipped_count += 1
        
        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} jobs (too old or failed to parse)")
        
        self.logger.info(f"Successfully scraped {len(jobs)} jobs within {self.target_days} days")
        return jobs
    
    def get_job_urls(self, search_term: str, location: str = "", max_pages: int = 1) -> List[str]:
        """
        Get job URLs from LinkedIn search results.
        """
        job_urls = []
        
        for page in range(max_pages):
            start = page * 25  # LinkedIn shows 25 jobs per page
            
            # Build search URL
            params = f"keywords={quote(search_term)}"
            if location:
                params += f"&location={quote(location)}"
            params += f"&start={start}"
            
            search_url = f"{self.search_url}?{params}"
            self.logger.info(f"Searching page {page + 1}: {search_url}")
            
            # Fetch and parse search results
            soup = self.get_page(search_url)
            if not soup:
                self.logger.warning(f"Failed to fetch page {page + 1}")
                break
            
            # Extract job URLs from search results
            job_cards = soup.find_all('div', {'class': 'base-card'})
            
            if not job_cards:
                self.logger.info("No more jobs found")
                break
            
            for card in job_cards:
                link = card.find('a', {'class': 'base-card__full-link'})
                if link and link.get('href'):
                    job_url = link.get('href').split('?')[0]  # Remove query params
                    if job_url not in job_urls:
                        job_urls.append(job_url)
            
            self.logger.info(f"Found {len(job_cards)} jobs on page {page + 1}")
        
        self.logger.info(f"Found {len(job_urls)} job URLs")
        return job_urls
    
    def parse_job(self, job_url: str) -> Optional[LinkedInJobListing]:
        """
        Parse a single job listing from LinkedIn with extended information.
        """
        soup = self.get_page(job_url)
        if not soup:
            return None
        
        try:
            # Extract job title
            title_elem = soup.find('h1', {'class': 'top-card-layout__title'})
            if not title_elem:
                title_elem = soup.find('h2', {'class': 'topcard__title'})
            title = self.clean_text(title_elem.text) if title_elem else "N/A"
            
            # Extract company name & verification status
            company_elem = soup.find('a', {'class': 'topcard__org-name-link'})
            if not company_elem:
                company_elem = soup.find('span', {'class': 'topcard__flavor'})
            company = self.clean_text(company_elem.text) if company_elem else "N/A"
            
            # Check if company is verified (has badge)
            is_verified = soup.find('span', {'class': 'verified__badge'}) is not None or \
                         soup.find('li-icon', {'type': 'official-badge'}) is not None
            
            # Extract location
            location_elem = soup.find('span', {'class': 'topcard__flavor--bullet'})
            if not location_elem:
                location_elem = soup.find('span', {'class': 'topcard__flavor topcard__flavor--bullet'})
            location = self.clean_text(location_elem.text) if location_elem else "N/A"
            
            # Extract work mode (Remote, Hybrid, On-site)
            work_mode = None
            work_mode_elem = soup.find('span', {'class': 'workplace-type'})
            if not work_mode_elem:
                # Check in job criteria
                for item in soup.find_all('li', {'class': 'description__job-criteria-item'}):
                    header = item.find('h3')
                    if header and 'workplace type' in self.clean_text(header.text).lower():
                        value = item.find('span', {'class': 'description__job-criteria-text'})
                        if value:
                            work_mode = self.clean_text(value.text)
                            break
                # Also check in description for keywords
                if not work_mode:
                    desc_text = soup.get_text().lower()
                    if 'remote' in desc_text or 'work from home' in desc_text:
                        work_mode = 'Remote'
                    elif 'hybrid' in desc_text:
                        work_mode = 'Hybrid'
                    elif 'on-site' in desc_text or 'onsite' in desc_text:
                        work_mode = 'On-site'
            else:
                work_mode = self.clean_text(work_mode_elem.text)
            
            # Extract job description
            desc_elem = soup.find('div', {'class': 'show-more-less-html__markup'})
            if not desc_elem:
                desc_elem = soup.find('div', {'class': 'description__text'})
            description = self.clean_text(desc_elem.text) if desc_elem else "N/A"
            
            # Extract posted date
            posted_date = None
            date_elem = soup.find('span', {'class': 'posted-time-ago__text'})
            if not date_elem:
                date_elem = soup.find('span', {'class': 'topcard__flavor--metadata'})
            if date_elem:
                posted_date = self.clean_text(date_elem.text)
            
            # Skip jobs older than target_days
            if posted_date:
                days_ago = self.parse_posted_date_to_days(posted_date)
                if days_ago is not None and days_ago > self.target_days:
                    self.logger.info(f"Skipping job (too old): {posted_date} ({days_ago} days > {self.target_days} days)")
                    return None
            
            # Extract salary (if available)
            salary_elem = soup.find('div', {'class': 'salary'})
            if not salary_elem:
                salary_elem = soup.find('span', {'class': 'salary-main-rail__salary-range'})
            salary = self.clean_text(salary_elem.text) if salary_elem else None
            
            # Extract job criteria (type, level, industry, etc.)
            criteria_list = soup.find_all('li', {'class': 'description__job-criteria-item'})
            job_type = None
            seniority_level = None
            company_industry = None
            
            for item in criteria_list:
                header = item.find('h3', {'class': 'description__job-criteria-subheader'})
                if header:
                    header_text = self.clean_text(header.text).lower()
                    value = item.find('span', {'class': 'description__job-criteria-text'})
                    if value:
                        value_text = self.clean_text(value.text)
                        if 'employment type' in header_text:
                            job_type = value_text
                        elif 'seniority level' in header_text:
                            seniority_level = value_text
                        elif 'industries' in header_text or 'industry' in header_text:
                            company_industry = value_text
            
            # Extract company information
            company_size = None
            company_description = None
            company_address = None
            
            # Company size
            size_elem = soup.find('span', {'class': 'company-size'})
            if not size_elem:
                for item in soup.find_all('li', {'class': 'description__job-criteria-item'}):
                    header = item.find('h3')
                    if header and 'company size' in self.clean_text(header.text).lower():
                        value = item.find('span', {'class': 'description__job-criteria-text'})
                        if value:
                            company_size = self.clean_text(value.text)
                            break
            else:
                company_size = self.clean_text(size_elem.text)
            
            # Company description (from about section if available)
            company_desc_elem = soup.find('div', {'class': 'company-description'})
            if not company_desc_elem:
                company_desc_elem = soup.find('section', {'class': 'company-summary'})
            if company_desc_elem:
                company_description = self.clean_text(company_desc_elem.text)
            
            # Company location/address
            company_loc_elem = soup.find('span', {'class': 'company-location'})
            if company_loc_elem:
                company_address = self.clean_text(company_loc_elem.text)
            
            # Extract skills
            required_skills = []
            skills_section = soup.find('div', {'class': 'skills-section'})
            if not skills_section:
                skills_section = soup.find_all('span', {'class': 'skill-badge'})
            
            if skills_section:
                if isinstance(skills_section, list):
                    for skill in skills_section:
                        skill_text = self.clean_text(skill.text)
                        if skill_text:
                            required_skills.append(skill_text)
                else:
                    skill_items = skills_section.find_all(['span', 'li'])
                    for skill in skill_items:
                        skill_text = self.clean_text(skill.text)
                        if skill_text and len(skill_text) < 50:  # Avoid long texts
                            required_skills.append(skill_text)
            
            # If no skills found in dedicated section, try to extract from description
            if not required_skills:
                # Common tech skills keywords
                skill_keywords = [
                    'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
                    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure',
                    'gcp', 'docker', 'kubernetes', 'git', 'agile', 'scrum', 'ci/cd',
                    'machine learning', 'data analysis', 'tensorflow', 'pytorch',
                    'spark', 'hadoop', 'tableau', 'power bi', 'excel', 'sap'
                ]
                desc_lower = description.lower()
                for skill in skill_keywords:
                    if skill in desc_lower:
                        required_skills.append(skill.title())
            
            # Extract experience range
            experience_range = None
            exp_patterns = [
                r'(\d+)\s*[-to]+\s*(\d+)\s*years?',
                r'(\d+)\+?\s*years?',
            ]
            for pattern in exp_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    experience_range = match.group(0)
                    break
            
            # Company rating and reviews (if available on page)
            company_rating = None
            company_reviews_count = None
            rating_elem = soup.find('span', {'class': 'rating'})
            if not rating_elem:
                rating_elem = soup.find('span', {'data-test': 'rating'})
            if rating_elem:
                try:
                    rating_text = self.clean_text(rating_elem.text)
                    company_rating = float(rating_text.split()[0])
                except:
                    pass
            
            reviews_elem = soup.find('span', {'class': 'review-count'})
            if reviews_elem:
                try:
                    reviews_text = self.clean_text(reviews_elem.text)
                    company_reviews_count = int(''.join(filter(str.isdigit, reviews_text)))
                except:
                    pass
            
            # Create extended JobListing object
            job = LinkedInJobListing(
                title=title,
                company=company,
                location=location,
                description=description,
                salary=salary,
                job_type=job_type,
                experience_level=seniority_level,  # Use seniority_level for experience_level
                posted_date=posted_date,
                job_url=job_url,
                work_mode=work_mode,
                company_industry=company_industry,
                company_size=company_size,
                company_description=company_description,
                company_address=company_address,
                company_rating=company_rating,
                company_reviews_count=company_reviews_count,
                is_verified=is_verified,
                seniority_level=seniority_level,
                required_skills=required_skills,
                experience_range=experience_range,
            )
            
            return job
            
        except Exception as e:
            self.logger.error(f"Error parsing job {job_url}: {e}")
            return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_file_logging(output_dir: str = None, log_file: str = None):
    """Setup file and console logging with colors."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"Log_LinkedIn_{timestamp}.log"
        if output_dir:
            log_file = os.path.join(output_dir, log_filename)
        else:
            log_file = log_filename
    
    # Create logger
    logger = logging.getLogger("LinkedInScraper")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler (detailed, no colors)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (colored with emojis)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter('%(levelname)s %(message)s'))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file, logger


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _is_german_keyword(keyword: str) -> bool:
    """Check if keyword is in German."""
    german_indicators = [
        'entwickler', 'ingenieur', 'wissenschaftler', 'administrator',
        'architekt', 'analyst', 'spezialist'
    ]
    return any(indicator in keyword.lower() for indicator in german_indicators)


def _save_metadata(output_dir: str, timestamp: str, job_titles: list, locations: list, 
                   total_jobs: int, stats: dict, duration: float, target_days: int, logger):
    """Save scraping metadata."""
    metadata_filename = f"Metadata_{timestamp}.json"
    metadata_file = os.path.join(output_dir, metadata_filename)
    
    metadata = {
        'scrape_date': datetime.now().isoformat(),
        'job_board': 'LinkedIn',
        'cities': locations,
        'job_titles': job_titles,
        'total_unique_jobs': total_jobs,
        'bilingual': True,
        'target_time_range_days': target_days,
        'target_time_range_description': f'Last {target_days} days (approximately {target_days//30} months)',
        'duration_seconds': duration,
        'duration_formatted': format_duration(duration),
        'statistics': stats,
        'note': f'Jobs older than {target_days} days were automatically skipped during scraping. All collected jobs are within the target time range.'
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"   📋 Metadata saved: {metadata_filename}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Scrape jobs for multiple job titles and locations - Save all to ONE file."""
    
    # Setup output directory (save all files in JobMiner folder)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup file logging
    log_file, logger = setup_file_logging(output_dir=output_dir)
    
    # Job titles (English and German)
    job_titles = [
        # English keywords
        "software engineer", "software architect", "software developer",
        "data engineer", "data analyst", "data scientist",
        "BI developer", "Cloud Engineer", "Cloud architect",
        "DevOps engineer", "IT administrator",
        "backend developer", "frontend developer", "full stack developer",
        "Apps developer", "SAP developer",
        "machine learning engineer", "AI engineer", "cybersecurity engineer",
        # German keywords
        "Softwareentwickler", "Softwarearchitekt",
        "Data Engineer", "Datenanalyst", "Data Scientist",
        "BI-Entwickler", "Cloud Engineer", "Cloud-Architekt",
        "DevOps Engineer", "IT-Administrator",
        "Backend Entwickler", "Frontend Entwickler", "FullStack Entwickler",
        "Apps Entwickler", "SAP Entwickler",
        "Machine Learning Engineer", "KI-Ingenieur", "IT-Security Engineer"
    ]
    
    # German cities
    locations = [
        "Stuttgart", "Munich", "Berlin", "Potsdam", "Bremen",
        "Hamburg", "Frankfurt", "Hanover", "Rostock", "Cologne",
        "Mainz", "Saarbrücken", "Dresden", "Magdeburg", "Kiel",
        "Erfurt", "Düsseldorf", "Dortmund", "Essen", "Leipzig",
        "Nürnberg", "Karlsruhe", "Mannheim", "Augsburg", "Wiesbaden",
        "Münster", "Bonn", "Freiburg", "Aachen", "Heidelberg",
        "Ulm", "Darmstadt", "Regensburg", "Bielefeld"
    ]
    
    # ========================================================================
    # CONFIGURATION - OPTIMIZED MODE (Balance: Speed + Safety + Max Data)
    # ========================================================================
    TARGET_DAYS = 120        # Scrape jobs from last 120 days (3 months)
    MAX_PAGES_PER_SEARCH = 2 # 2 pages = 50 jobs per search (balance)
    DELAY_SECONDS = 0.8      # ⚡ OPTIMIZED - Faster but still safe
                             # 0.8s base + random 0-0.5s variance
                             # NO multi-threading (sequential requests only)
    MAX_WORKERS = 1          # Sequential only - prevents pattern detection
    
    # ========================================================================
    # RESUME FROM CHECKPOINT (Set to True to continue from checkpoint)
    # ========================================================================
    RESUME_FROM_CHECKPOINT = False  # ❌ Run from scratch (fix applied)
    CHECKPOINT_FILE = "CHECKPOINT_1200_of_1258_2026-02-06_14-26-13.json"
    RESUME_FROM_SEARCH = 1201  # Start from search #1201 (skip first 1200)
    
    # Print header
    logger.info("="*80)
    if RESUME_FROM_CHECKPOINT:
        logger.info("🚀 LINKEDIN JOB SCRAPER - RESUMING FROM CHECKPOINT")
        logger.info(f"📂 Loading: {CHECKPOINT_FILE}")
    else:
        logger.info("🚀 LINKEDIN JOB SCRAPER - BILINGUAL (EN + DE)")
    logger.info("🐢 SAFE MODE - Human-like behavior (No multi-threading)")
    logger.info("="*80)
    logger.info(f"📍 Cities: {len(locations)} (ALL)")
    logger.info(f"🔑 Job Titles: {len(job_titles)} (ALL)")
    logger.info(f"📝 Log File: {log_file}")
    logger.info(f"📅 Time Range: Last {TARGET_DAYS} days")
    logger.info(f"⏱️  Pages per search: {MAX_PAGES_PER_SEARCH} (={MAX_PAGES_PER_SEARCH * 25} jobs max)")
    logger.info(f"🐢 Delay: {DELAY_SECONDS}s + random(0-0.5s) - HUMAN-LIKE")
    logger.info(f"🔄 Sequential scraping: 1 request at a time (safest)")
    
    total_combinations = len(job_titles) * len(locations)
    
    # Load checkpoint if resuming
    existing_jobs = []
    start_from = 1
    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                existing_jobs = [LinkedInJobListing(**job) for job in checkpoint_data]
            logger.info(f"✅ Loaded checkpoint: {len(existing_jobs)} existing jobs")
            logger.info(f"⏭️  Resuming from search #{RESUME_FROM_SEARCH}")
            start_from = RESUME_FROM_SEARCH
        else:
            logger.warning(f"⚠️  Checkpoint file not found: {CHECKPOINT_FILE}")
            logger.info("Starting from beginning...")
            RESUME_FROM_CHECKPOINT = False
    
    remaining_searches = total_combinations - (start_from - 1)
    
    # Estimate time
    estimated_time_seconds = remaining_searches * 35
    estimated_time_hours = estimated_time_seconds / 3600
    logger.info(f"🎯 Total searches: {total_combinations}")
    if RESUME_FROM_CHECKPOINT:
        logger.info(f"   Already completed: {start_from - 1}")
        logger.info(f"   Remaining: {remaining_searches} searches")
    logger.info(f"🕒 Est. time: {estimated_time_hours:.1f}h ({estimated_time_seconds/60:.0f} min)")
    logger.info(f"📊 Expected new jobs: ~{remaining_searches * 25:,} (before dedup)")
    logger.info(f"✅ Safest mode - mimics human browsing behavior")
    logger.info("="*80)
    
    # Start timing
    start_time = datetime.now()
    
    scraper = LinkedinScraper(delay=DELAY_SECONDS, target_days=TARGET_DAYS)
    all_jobs = existing_jobs.copy()  # Start with checkpoint jobs if any
    
    # Thread-safe locks
    jobs_lock = Lock()
    stats_lock = Lock()
    checkpoint_lock = Lock()
    
    # Statistics
    stats = {
        'successful_searches': 0,
        'failed_searches': 0,
        'no_jobs_searches': 0,
        'jobs_skipped_too_old': 0,
    }
    
    # Progress tracking
    completed_count = 0
    completed_lock = Lock()
    CHECKPOINT_INTERVAL = 100  # Auto-save every 100 searches
    
    def scrape_single_combination(job_title, location, combination_num):
        """Scrape a single job title/location combination (thread-safe)."""
        nonlocal all_jobs, stats, completed_count
        
        # Detect language for logging
        lang = "🇩🇪 DE" if _is_german_keyword(job_title) else "🇬🇧 EN"
        
        logger.info(f"🔎 [{combination_num}/{total_combinations}] {lang}: '{job_title}' in {location}")
        
        try:
            # Scrape jobs
            jobs = scraper.scrape_jobs(
                search_term=job_title,
                location=f"{location}, Germany",
                max_pages=MAX_PAGES_PER_SEARCH
            )
            
            # Thread-safe update of results
            with jobs_lock:
                if jobs:
                    logger.info(f"   ✓ Found {len(jobs)} jobs")
                    all_jobs.extend(jobs)
                    with stats_lock:
                        stats['successful_searches'] += 1
                else:
                    logger.warning(f"   ✗ No jobs found")
                    with stats_lock:
                        stats['no_jobs_searches'] += 1
                    
        except Exception as e:
            logger.error(f"   ✗ Error: {str(e)}")
            with stats_lock:
                stats['failed_searches'] += 1
        
        # Update completed count and check for checkpoint
        with completed_lock:
            completed_count += 1
            if completed_count % CHECKPOINT_INTERVAL == 0:
                with checkpoint_lock:
                    logger.info(f"\n💾 CHECKPOINT: Saving progress at {completed_count}/{total_combinations}...")
                    unique_checkpoint = []
                    seen_urls_checkpoint = set()
                    with jobs_lock:
                        for job in all_jobs:
                            if job.job_url not in seen_urls_checkpoint:
                                unique_checkpoint.append(job)
                                seen_urls_checkpoint.add(job.job_url)
                    
                    checkpoint_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    checkpoint_filename = f"CHECKPOINT_{completed_count}_of_{total_combinations}_{checkpoint_timestamp}.json"
                    checkpoint_file = os.path.join(output_dir, checkpoint_filename)
                    scraper.save_to_json(unique_checkpoint, checkpoint_file)
                    logger.info(f"   ✓ Checkpoint saved: {len(unique_checkpoint)} unique jobs so far\n")
    
    # Create list of all combinations with indices
    combinations = [
        (job_title, location, idx + 1)
        for idx, (job_title, location) in enumerate(
            (jt, loc) for jt in job_titles for loc in locations
        )
    ]
    
    # Filter to only remaining searches if resuming
    if RESUME_FROM_CHECKPOINT:
        combinations = [c for c in combinations if c[2] >= start_from]
        logger.info(f"\n⏭️  Skipping first {start_from - 1} searches (from checkpoint)")
        logger.info(f"🚀 Starting with {len(combinations)} remaining searches...\n")
    
    # Use ThreadPoolExecutor for parallel processing
    logger.info(f"\n🚀 Starting multi-threaded scraping with {MAX_WORKERS} workers...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [
            executor.submit(scrape_single_combination, job_title, location, idx)
            for job_title, location, idx in combinations
        ]
        
        # Wait for all to complete (no need to process results as they update shared state)
        for future in as_completed(futures):
            try:
                future.result()  # Raise any exceptions that occurred
            except Exception as exc:
                logger.error(f"❌ Thread exception: {exc}")
    
    # Remove duplicates based on job_url
    logger.info("\n🧹 Removing duplicate jobs...")
    unique_jobs = []
    seen_urls = set()
    for job in all_jobs:
        if job.job_url not in seen_urls:
            unique_jobs.append(job)
            seen_urls.add(job.job_url)
    
    # Calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("✅ SCRAPING COMPLETE!")
    logger.info("="*80)
    logger.info(f"⏱️  Duration: {format_duration(duration)}")
    logger.info(f"\n📊 STATISTICS:")
    logger.info(f"   ✓ Successful searches: {stats['successful_searches']}")
    logger.info(f"   ✗ Failed searches: {stats['failed_searches']}")
    logger.info(f"   - No jobs found: {stats['no_jobs_searches']}")
    logger.info(f"\n📦 JOBS COLLECTED (within {TARGET_DAYS} days):")
    logger.info(f"   Raw total: {len(all_jobs)}")
    logger.info(f"   Unique jobs: {len(unique_jobs)}")
    logger.info(f"   📅 All jobs are within last {TARGET_DAYS} days (older jobs were skipped)")
    
    if len(all_jobs) > len(unique_jobs):
        duplicates = len(all_jobs) - len(unique_jobs)
        logger.info(f"   🧹 Removed duplicates: {duplicates}")
    
    # Save ALL results to ONE set of files with timestamp
    if unique_jobs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_filename = f"Raw_Jobs_LINKEDIN_{timestamp}.json"
        csv_filename = f"Raw_Jobs_LINKEDIN_{timestamp}.csv"
        json_file = os.path.join(output_dir, json_filename)
        csv_file = os.path.join(output_dir, csv_filename)
        
        logger.info(f"\n💾 SAVING ALL DATA TO SINGLE FILES...")
        
        # Save to JSON
        scraper.save_to_json(unique_jobs, json_file)
        logger.info(f"   ✓ JSON saved: {json_file} ({len(unique_jobs)} jobs)")
        
        # Save to CSV
        scraper.save_to_csv(unique_jobs, csv_file)
        logger.info(f"   ✓ CSV saved: {csv_filename}")
        
        logger.info(f"\n📁 FINAL OUTPUT FILES (4 files total):")
        logger.info(f"   📄 {csv_filename}")
        logger.info(f"   📄 {json_filename}")
        logger.info(f"   📄 {os.path.basename(log_file)}")
        
        # Save metadata
        _save_metadata(output_dir, timestamp, job_titles, locations, len(unique_jobs), stats, duration, TARGET_DAYS, logger)
        logger.info(f"\n📂 Output Location: {output_dir}")
        
    else:
        logger.warning("No jobs found to save.")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
