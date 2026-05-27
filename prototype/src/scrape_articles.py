"""
scrape_articles.py

Scrape a small batch of Pinkbike review article pages using Playwright.

This prototype:
1. Reads review URLs from data/processed/review_links.csv
2. Visits only a limited number of articles
3. Saves each raw article HTML file locally
4. Adds a short delay between requests
"""

from pathlib import Path
import time
import pandas as pd
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "review_links" / "review_links.csv"
ARTICLE_HTML_DIR = PROJECT_ROOT / "data" / "raw" / "article_html"

ARTICLE_LIMIT = 1
REQUEST_DELAY_SECONDS = 3
START_INDEX = 1

def make_safe_filename(url: str, index: int) -> str:
    """
    Create a safe local filename from an article URL.
    """

    slug = url.rstrip("/").split("/")[-1].replace(".html", "")
    return f"{index:03d}_{slug}.html"


def main() -> None:
    """
    Visit a small batch of review URLs and save raw HTML.
    """

    ARTICLE_HTML_DIR.mkdir(parents=True, exist_ok=True)

    links_df = pd.read_csv(INPUT_FILE)
    sample_links = links_df.iloc[START_INDEX:START_INDEX + ARTICLE_LIMIT]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        )

        for index, row in sample_links.iterrows():
            url = row["url"]
            output_file = ARTICLE_HTML_DIR / make_safe_filename(url, index + 1)

            print(f"Scraping article {index + 1}: {url}")

            try:
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=60000
                )

                page.wait_for_timeout(5000)

                html = page.content()
                output_file.write_text(html, encoding="utf-8")

                print(f"Saved: {output_file}")
                print(f"HTML size: {len(html):,} characters")

            except Exception as error:
                print(f"Failed to scrape {url}")
                print(f"Error: {error}")

            time.sleep(REQUEST_DELAY_SECONDS)

        browser.close()


if __name__ == "__main__":
    main()