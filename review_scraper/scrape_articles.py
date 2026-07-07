"""
Scrape one Pinkbike review article page using Playwright.

Purpose:
This script supports the data collection stage of the project. It reads
the discovered Pinkbike review URLs, identifies the next article that has
not already been saved, downloads that article's raw HTML, and stores it
locally for later parsing and preprocessing.

The scraper intentionally processes one article per run. This conservative
approach reduces scraping risk, makes failures easier to inspect, and helps
avoid repeatedly requesting many pages from Pinkbike in a short time.

AI Use:
AI tools were used to assist with code review, debugging, simplification,
and annotation.
"""

import pandas as pd
from playwright.sync_api import sync_playwright
from typing import Optional

from review_scraper.config import (
    REVIEW_LINKS_FILE,
    ARTICLE_HTML_DIR,
    START_INDEX,
    PAGE_TIMEOUT_MS,
    PAGE_WAIT_MS,
    HEADLESS,
    USER_AGENT,
    create_project_directories,
)
from review_scraper.utils import make_safe_filename


def is_cloudflare_block(html: str) -> bool:
    """
    Detect whether the downloaded HTML is a Cloudflare block page.

    This prevents the scraper from saving a block page as if it were a
    valid article.
    """

    block_indicators = [
        "Attention Required! | Cloudflare",
        "Sorry, you have been blocked",
        "cf-error-details",
    ]

    return any(indicator in html for indicator in block_indicators)


def get_next_unscraped_url(links_df: pd.DataFrame) -> Optional[str]:
    """
    Return the first review URL that does not already have a saved HTML file.

    START_INDEX can be used to resume checking from a later row in the link
    file if needed.
    """

    for _, row in links_df.iloc[START_INDEX:].iterrows():
        url = (
            row["normalized_url"]
            if "normalized_url" in row and pd.notna(row["normalized_url"])
            else row["url"]
        )

        output_file = ARTICLE_HTML_DIR / make_safe_filename(url)

        if not output_file.exists():
            return url

    return None


def main() -> None:
    """
    Scrape the next missing Pinkbike review article and save its raw HTML.
    """

    create_project_directories()

    links_df = pd.read_csv(REVIEW_LINKS_FILE)
    url = get_next_unscraped_url(links_df)

    if url is None:
        print("No missing article HTML files found.")
        return

    output_file = ARTICLE_HTML_DIR / make_safe_filename(url)

    print(f"[SCRAPE] {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)

        try:
            context = browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                timezone_id="America/Chicago",
                color_scheme="light",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )

            page = context.new_page()

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT_MS,
            )

            page.wait_for_timeout(PAGE_WAIT_MS)

            """
            Simulate a small amount of user interaction so lazy-loaded
            page content has a chance to appear before saving HTML.
            """
            page.mouse.move(400, 400)
            page.mouse.wheel(0, 500)
            page.wait_for_timeout(1500)

            html = page.content()

            if is_cloudflare_block(html):
                print(f"[BLOCKED] {url}")
                print("Cloudflare block detected. Article was not saved.")
                return

            output_file.write_text(html, encoding="utf-8")

            print(f"[SAVE] {output_file.name}")
            print(f"HTML size: {len(html):,} characters")

        except Exception as error:
            print(f"[FAIL] {url}")
            print(f"Error: {error}")

        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()