"""
test_playwright_fetch.py

Prototype test to determine whether Pinkbike review pages can be accessed
through a browser-rendered request instead of a basic requests call.
"""

from pathlib import Path
from playwright.sync_api import sync_playwright


REVIEWS_URL = "https://www.pinkbike.com/news/tags/reviews/"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_HTML_DIR = PROJECT_ROOT / "data" / "raw" / "html_samples"
OUTPUT_FILE = RAW_HTML_DIR / "pinkbike_reviews_index_playwright.html"


def main() -> None:
    """
    Open the Pinkbike reviews page in a browser context and save the HTML.
    """

    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        )

        print(f"Opening: {REVIEWS_URL}")

        page.goto(
            REVIEWS_URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        # Give scripts/images a few seconds to finish loading.
        page.wait_for_timeout(5000)

        print(f"Page title: {page.title()}")
        print(f"Current URL: {page.url}")

        html = page.content()
        OUTPUT_FILE.write_text(html, encoding="utf-8")

        print(f"Saved HTML to: {OUTPUT_FILE}")
        print(f"HTML Size: {len(html):,} characters")

        browser.close()


if __name__ == "__main__":
    main()