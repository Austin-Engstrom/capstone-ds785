"""
Scrape Pinkbike review article pages using Playwright.

1. Reads review URLs from data/raw/review_links/review_links.csv
2. Visits either a limited batch or all remaining articles
3. Saves each raw article HTML file locally
4. Adds a delay between requests

"""

from review_scraper.config import (
    RAW_REVIEW_LINKS_FILE,
    ARTICLE_HTML_DIR,
    ARTICLE_LIMIT,
    START_INDEX,
    REQUEST_DELAY_SECONDS,
    PAGE_TIMEOUT_MS,
    PAGE_WAIT_MS,
    HEADLESS,
    USER_AGENT,
    create_project_directories,
)
import time
import pandas as pd
from playwright.sync_api import sync_playwright
from review_scraper.utils import make_safe_filename

INPUT_FILE = RAW_REVIEW_LINKS_FILE

def main() -> None:
    """
    Visit a small batch of review URLs and save raw HTML.
    """
    create_project_directories()

    links_df = pd.read_csv(INPUT_FILE)

    if ARTICLE_LIMIT is None:
        sample_links = links_df.iloc[START_INDEX:]
    else:
        sample_links = links_df.iloc[START_INDEX:START_INDEX + ARTICLE_LIMIT]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)

        try:
            page = browser.new_page(user_agent=USER_AGENT)

            for index, row in sample_links.iterrows():
                url = row["url"]
                output_file = ARTICLE_HTML_DIR / make_safe_filename(url, index + 1)

                if output_file.exists():
                    print(f"Skipping existing file: {output_file}")
                    continue

                print(f"Scraping article {index + 1}: {url}")

                try:
                    page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=PAGE_TIMEOUT_MS
                    )

                    page.wait_for_timeout(PAGE_WAIT_MS)

                    html = page.content()
                    output_file.write_text(html, encoding="utf-8")

                    print(f"Saved: {output_file}")
                    print(f"HTML size: {len(html):,} characters")

                except Exception as error:
                    print(f"Failed to scrape {url}")
                    print(f"Error: {error}")

                time.sleep(REQUEST_DELAY_SECONDS)

        finally:
            browser.close()
if __name__ == "__main__":
    main()