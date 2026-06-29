"""
Discover Pinkbike review article links and merge them into the existing link CSV.
"""

from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from review_scraper.config import (
    BASE_URL,
    REVIEWS_INDEX_URL,
    RAW_REVIEW_LINKS_FILE,
    HEADLESS,
    USER_AGENT,
    PAGE_TIMEOUT_MS,
    PAGE_WAIT_MS,
    create_project_directories,
)
from review_scraper.utils import normalize_url


SCROLL_ATTEMPTS = 10
DISCOVERY_SOURCE = "pinkbike_reviews_tag"


def extract_review_links(html: str) -> pd.DataFrame:
    """
    Extract candidate Pinkbike review article links from HTML.
    """

    soup = BeautifulSoup(html, "lxml")
    rows = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        title = link.get_text(" ", strip=True)

        if "/news/" in href and href.endswith(".html") and title:
            url = urljoin(BASE_URL, href)

            rows.append(
                {
                    "title": title,
                    "url": url,
                    "normalized_url": normalize_url(url),
                    "date_discovered": datetime.today().date().isoformat(),
                    "source": DISCOVERY_SOURCE,
                }
            )

    return pd.DataFrame(rows).drop_duplicates(subset=["normalized_url"])


def merge_with_existing_links(new_links: pd.DataFrame) -> pd.DataFrame:
    """
    Merge newly discovered links with the existing review links CSV.
    """

    if RAW_REVIEW_LINKS_FILE.exists():
        existing_links = pd.read_csv(RAW_REVIEW_LINKS_FILE)

        if "normalized_url" not in existing_links.columns:
            existing_links["normalized_url"] = existing_links["url"].apply(normalize_url)

        if "date_discovered" not in existing_links.columns:
            existing_links["date_discovered"] = None

        if "source" not in existing_links.columns:
            existing_links["source"] = "existing_review_links_csv"

        combined_links = pd.concat(
            [existing_links, new_links],
            ignore_index=True,
        )
    else:
        combined_links = new_links

    combined_links = combined_links.drop_duplicates(
        subset=["normalized_url"],
        keep="first",
    )

    combined_links = combined_links.sort_values(
        by=["date_discovered", "title"],
        na_position="last",
    )

    return combined_links


def main() -> None:
    """
    Scrape the Pinkbike reviews tag page, append new links to the existing
    CSV, and remove duplicates.
    """

    create_project_directories()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)

        try:
            page = browser.new_page(user_agent=USER_AGENT)

            print(f"Opening: {REVIEWS_INDEX_URL}")

            page.goto(
                REVIEWS_INDEX_URL,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT_MS,
            )

            page.wait_for_timeout(PAGE_WAIT_MS)

            previous_height = 0

            for attempt in range(1, SCROLL_ATTEMPTS + 1):
                print(f"Scroll attempt {attempt}")

                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(PAGE_WAIT_MS)

                current_height = page.evaluate("document.body.scrollHeight")

                if current_height == previous_height:
                    print("Page height did not change. Stopping scroll.")
                    break

                previous_height = current_height

            html = page.content()
            new_links = extract_review_links(html)

            existing_count = 0

            if RAW_REVIEW_LINKS_FILE.exists():
                existing_count = len(pd.read_csv(RAW_REVIEW_LINKS_FILE))

            combined_links = merge_with_existing_links(new_links)

            RAW_REVIEW_LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            combined_links.to_csv(RAW_REVIEW_LINKS_FILE, index=False)

            print(f"Existing links before run: {existing_count}")
            print(f"Links found this run: {len(new_links)}")
            print(f"Total links after dedupe: {len(combined_links)}")
            print(f"Net new links added: {len(combined_links) - existing_count}")
            print(f"Saved to: {RAW_REVIEW_LINKS_FILE}")

        finally:
            browser.close()


if __name__ == "__main__":
    main()