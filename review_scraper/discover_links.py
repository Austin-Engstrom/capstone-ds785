"""
Discover Pinkbike review article links and merge them into the master link CSV.
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


SCROLL_ATTEMPTS = 10


def extract_review_links(html: str) -> pd.DataFrame:
    """
    Extract candidate Pinkbike article links from the loaded reviews page.
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

    return pd.DataFrame(rows).drop_duplicates(subset=["normalized_url"])


def load_existing_links() -> pd.DataFrame:
    """
    Load the existing master review link CSV if it exists.
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
    Scrape the Pinkbike reviews tag page, merge new links with existing links,
    deduplicate, and save the updated master review link file.
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