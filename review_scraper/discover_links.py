"""
Discover Pinkbike review article links and merge them into the master link CSV.

Purpose:
This script supports the data collection stage of the project. It visits the
Pinkbike reviews tag page, scrolls the page to load article links, extracts
candidate review article URLs, removes duplicates, and saves the updated
master list of review links. This allows for new articles to be discovered and added to the dataset over time.

The resulting review_links.csv file is used by scrape_articles.py to download
individual article HTML files.

AI Use:
AI tools were used to assist with code review, debugging, organization,
and annotation. All code and comments were reviewed by the student before
submission.
"""

from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from review_scraper.config import (
    BASE_URL,
    REVIEWS_INDEX_URL,
    REVIEW_LINKS_FILE,
    DISCOVERY_SOURCE,
    HEADLESS,
    USER_AGENT,
    PAGE_TIMEOUT_MS,
    PAGE_WAIT_MS,
    create_project_directories,
)
from review_scraper.utils import normalize_url


# Number of times to scroll the Pinkbike reviews page to load more links.
SCROLL_ATTEMPTS = 10


def extract_review_links(html: str) -> pd.DataFrame:
    """
    Extract candidate Pinkbike review article links from page HTML.

    The function searches all anchor tags, keeps links that appear to be
    Pinkbike news articles, converts relative URLs into full URLs, and adds
    metadata for traceability.
    """

    soup = BeautifulSoup(html, "lxml")
    rows = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        title = link.get_text(" ", strip=True)
        """
        Pinkbike article links are stored under /news/ and end in .html.
        A non-empty title is required so each URL has useful context.
        """
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

    if not rows:
        return pd.DataFrame(
            columns=[
                "title",
                "url",
                "normalized_url",
                "date_discovered",
                "source",
            ]
        )

    # Deduplicate links found during this discovery run.
    return pd.DataFrame(rows).drop_duplicates(subset=["normalized_url"])


def load_existing_links() -> pd.DataFrame:
    """
    Load the existing master review link CSV if it exists.

    If the file does not exist yet, an empty DataFrame with the expected
    schema is returned. The function also backfills expected columns so older
    link files remain compatible with the current pipeline.
    """

    if not REVIEW_LINKS_FILE.exists():
        return pd.DataFrame(
            columns=[
                "title",
                "url",
                "normalized_url",
                "date_discovered",
                "source",
            ]
        )

    existing_links = pd.read_csv(REVIEW_LINKS_FILE)

    if "normalized_url" not in existing_links.columns:
        existing_links["normalized_url"] = existing_links["url"].apply(normalize_url)

    if "date_discovered" not in existing_links.columns:
        existing_links["date_discovered"] = None

    if "source" not in existing_links.columns:
        existing_links["source"] = "existing_review_links_csv"

    return existing_links


def merge_links(existing_links: pd.DataFrame, new_links: pd.DataFrame) -> pd.DataFrame:
    """
    Combine existing and newly discovered links, then deduplicate.

    Deduplication is based on normalized_url so the same article is not stored
    multiple times if it appears with minor URL differences.
    """

    combined_links = pd.concat(
        [existing_links, new_links],
        ignore_index=True,
    )

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
    Discover Pinkbike review article links and update the master link file.
    """

    create_project_directories()

    existing_links = load_existing_links()
    existing_count = len(existing_links)

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
            """
            Scroll the reviews page so dynamically loaded article links
            have a chance to appear in the final page HTML.
            """
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

        finally:
            browser.close()

    combined_links = merge_links(existing_links, new_links)

    REVIEW_LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_links.to_csv(REVIEW_LINKS_FILE, index=False)

    new_link_count = len(new_links)
    total_count = len(combined_links)
    net_new_count = total_count - existing_count
    duplicate_count = existing_count + new_link_count - total_count

    print("\nDiscovery summary")
    print("-----------------")
    print(f"Existing links:      {existing_count}")
    print(f"Links discovered:    {new_link_count}")
    print(f"Duplicates removed:  {duplicate_count}")
    print(f"Net new links:       {net_new_count}")
    print(f"Total links:         {total_count}")
    print(f"Saved to:            {REVIEW_LINKS_FILE}")


if __name__ == "__main__":
    main()