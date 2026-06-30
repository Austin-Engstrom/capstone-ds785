"""
Shared configuration for the Pinkbike review scraper pipeline.

Purpose:
This file centralizes project paths, scraper settings, browser settings,
input/output file locations, and directory creation logic. Keeping these
values in one place makes the scraping and parsing scripts easier to update,
reuse, and troubleshoot.

AI Use:
AI tools were used to assist with code review and annotation.
"""

from pathlib import Path


"""
Project directory structure
PROJECT_ROOT resolves the project folder relative to this config file.
All data folders and output paths are built from this root path so that
the project can be run consistently across different computers.
"""

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
QA_DIR = DATA_DIR / "qa"

HTML_SAMPLES_DIR = RAW_DIR / "html_samples"
ARTICLE_HTML_DIR = RAW_DIR / "article_html"

"""
Reference input files
These files support repeatable scraping and parsing. The review links
file stores discovered Pinkbike review URLs, while the brand reference
file supports controlled brand extraction during article parsing.
"""

REVIEW_LINKS_FILE = REFERENCE_DIR / "review_links.csv"
BRAND_REFERENCE_FILE = REFERENCE_DIR / "brand_reference.csv"

"""
Processed output files
These files store the final structured datasets generated from the scraped
and parsed articles. The main dataset will be used in later project stages
for sentiment analysis and NLP modeling.
"""

REVIEWS_DATASET_FILE = PROCESSED_DIR / "pinkbike_reviews.csv"

"""
Quality assurance output files
These files help track failed URLs and summarize scraper results so
data collection issues can be reviewed and corrected.
"""

FAILED_URLS_FILE = QA_DIR / "failed_urls.csv"
SCRAPE_SUMMARY_FILE = QA_DIR / "scrape_summary.csv"

"""
Pinkbike source settings
These settings define the base URL for Pinkbike and the main reviews index page.
BASE_URL is used to convert relative article links into full URLs.
REVIEWS_INDEX_URL is the main Pinkbike reviews tag page used for discovering review article links.
"""
BASE_URL = "https://www.pinkbike.com"
REVIEWS_INDEX_URL = "https://www.pinkbike.com/news/tags/reviews/"

"""
Discovery metadata
This label identifies the source used to collect review links. It can
be stored with scraped URLs to support traceability.
"""
DISCOVERY_SOURCE = "pinkbike_reviews_tag"

"""
Scraper behavior settings
These values control how Playwright loads Pinkbike pages. The long
request delay and timeout settings are intentionally conservative to
reduce scraping failures and avoid making rapid requests to the site.
"""
HEADLESS = False
PAGE_TIMEOUT_MS = 60000
PAGE_WAIT_MS = 10000

"""
Batch processing settings
START_INDEX allows scraping to resume from a later point.
"""
START_INDEX = 0

"""
Browser identity settings
The user agent helps Playwright present as a standard browser session
when requesting Pinkbike pages.
"""
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


def create_project_directories() -> None:
    """
    Create all project data folders if they do not already exist.

    This function is called before scraping or parsing so the expected
    raw, processed, reference, and QA folders are available before files
    are wrote.
    """

    directories = [
        RAW_DIR,
        PROCESSED_DIR,
        REFERENCE_DIR,
        QA_DIR,
        HTML_SAMPLES_DIR,
        ARTICLE_HTML_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)