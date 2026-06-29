"""
Shared configuration for the Pinkbike review scraper pipeline.
"""

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
QA_DIR = DATA_DIR / "qa"

HTML_SAMPLES_DIR = RAW_DIR / "html_samples"
ARTICLE_HTML_DIR = RAW_DIR / "article_html"

# Reference datasets
REVIEW_LINKS_FILE = REFERENCE_DIR / "review_links.csv"
BRAND_REFERENCE_FILE = REFERENCE_DIR / "brand_reference.csv"

# Processed datasets
PARSED_ARTICLES_FILE = PROCESSED_DIR / "pinkbike_reviews.csv"

# QA outputs
FAILED_URLS_FILE = QA_DIR / "failed_urls.csv"
SCRAPE_SUMMARY_FILE = QA_DIR / "scrape_summary.csv"

# Pinkbike settings
BASE_URL = "https://www.pinkbike.com"
REVIEWS_INDEX_URL = "https://www.pinkbike.com/news/tags/reviews/"

# Discovery settings
DISCOVERY_SOURCE = "pinkbike_reviews_tag"

# Scraper settings
HEADLESS = False
REQUEST_DELAY_SECONDS = 60
PAGE_TIMEOUT_MS = 60000
PAGE_WAIT_MS = 10000

# Batch settings
ARTICLE_LIMIT = None
START_INDEX = 0
SCRAPE_ONE_ARTICLE_PER_RUN = True

# Browser settings
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


def create_project_directories() -> None:
    """
    Create the project data folders if they do not already exist.
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