"""
Scrape Pinkbike review article pages using Playwright.

1. Reads review URLs from data/reference/review_links.csv
2. Visits either a limited batch or all remaining articles
3. Saves each raw article HTML file locally
4. Adds a delay between requests
5. Stops if a Cloudflare block page is detected
"""

import time

import pandas as pd
from playwright.sync_api import sync_playwright

from review_scraper.config import (
    REVIEW_LINKS_FILE,
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
from review_scraper.utils import make_safe_filename


INPUT_FILE = REVIEW_LINKS_FILE


def is_cloudflare_block(html: str) -> bool:
    """
    Detect whether the downloaded HTML is a Cloudflare block page.
    """

    block_indicators = [
        "Attention Required! | Cloudflare",
        "Sorry, you have been blocked",
        "cf-error-details",
    ]

    return any(indicator in html for indicator in block_indicators)


def main() -> None:
    """
    Visit review URLs and save raw HTML.
    """

    create_project_directories()

    links_df = pd.read_csv(INPUT_FILE)

    if ARTICLE_LIMIT is None:
        sample_links = links_df.iloc[START_INDEX:]
    else:
        sample_links = links_df.iloc[START_INDEX:START_INDEX + ARTICLE_LIMIT]

    processed_count = 0
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)

        try:
            page = browser.new_page(user_agent=USER_AGENT)

            for index, row in sample_links.iterrows():
                processed_count += 1

                url = (
                    row["normalized_url"]
                    if "normalized_url" in row and pd.notna(row["normalized_url"])
                    else row["url"]
                )

                output_file = ARTICLE_HTML_DIR / make_safe_filename(url)

                if output_file.exists():
                    skipped_count += 1
                    print(f"[SKIP] {output_file.name}")
                    continue

                print(f"[SCRAPE] {index + 1}: {url}")

                try:
                    page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=PAGE_TIMEOUT_MS,
                    )

                    page.wait_for_timeout(PAGE_WAIT_MS)

                    html = page.content()

                    if is_cloudflare_block(html):
                        failed_count += 1
                        print(f"[BLOCKED] {url}")
                        print("Cloudflare block detected. Ending scrape.")
                        break

                    output_file.write_text(html, encoding="utf-8")

                    downloaded_count += 1
                    print(f"[SAVE] {output_file.name}")
                    print(f"HTML size: {len(html):,} characters")

                except Exception as error:
                    failed_count += 1
                    print(f"[FAIL] {url}")
                    print(f"Error: {error}")

                time.sleep(REQUEST_DELAY_SECONDS)

        finally:
            browser.close()

    print("\nScrape summary")
    print("--------------")
    print(f"Articles processed: {processed_count}")
    print(f"Downloaded:         {downloaded_count}")
    print(f"Skipped:            {skipped_count}")
    print(f"Failed:             {failed_count}")


if __name__ == "__main__":
    main()