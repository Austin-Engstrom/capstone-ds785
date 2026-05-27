"""
parse_index.py

Parse saved Pinkbike reviews index HTML and extract candidate review article links.
"""

from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://www.pinkbike.com"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "html_samples" / "pinkbike_reviews_index_playwright.html"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "review_links" / "review_links.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def parse_review_links(html_path: Path) -> pd.DataFrame:
    """
    Extract Pinkbike news article links from saved HTML.
    """

    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    rows = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        title = link.get_text(" ", strip=True)

        if "/news/" in href and href.endswith(".html") and title:
            rows.append(
                {
                    "title": title,
                    "url": urljoin(BASE_URL, href),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["url"])

    return df


def main() -> None:
    """
    Parse index HTML and save review links to CSV.
    """

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = parse_review_links(INPUT_FILE)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved links to: {OUTPUT_FILE}")
    print(f"Rows extracted: {len(df)}")
    print(df.head(10))


if __name__ == "__main__":
    main()