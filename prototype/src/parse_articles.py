"""
parse_articles.py

Parse saved Pinkbike review article HTML files and extract structured article data.

This version is tuned using a real Pinkbike article HTML sample.
"""

from pathlib import Path
from typing import Optional
import json
import re

import pandas as pd
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTICLE_HTML_DIR = PROJECT_ROOT / "data" / "raw" / "article_html"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "parsed_articles.csv"


def get_json_ld_article(soup: BeautifulSoup) -> dict:
    """
    Extract the NewsArticle JSON-LD object when available.
    """

    for script in soup.find_all("script", type="application/ld+json"):
        script_text = script.get_text(strip=True)

        try:
            data = json.loads(script_text)
        except json.JSONDecodeError:
            continue

        if data.get("@type") == "NewsArticle":
            return data

    return {}


def extract_meta_content(soup: BeautifulSoup, property_name: str) -> Optional[str]:
    """
    Extract content from a meta tag by property name.
    """

    meta = soup.find("meta", {"property": property_name})

    if meta:
        return meta.get("content")

    return None


def extract_author(article_json: dict) -> Optional[str]:
    """
    Extract author name from JSON-LD.
    """

    authors = article_json.get("author")

    if isinstance(authors, list) and len(authors) > 0:
        return authors[0].get("name")

    if isinstance(authors, dict):
        return authors.get("name")

    return None


def extract_retail_price(text: str) -> Optional[str]:
    """
    Extract likely product retail price from article text.

    Prioritizes explicit labels like:
    - Price: $769 USD
    - MSRP: $5,999
    - Retail Price: $4,499
    """

    labeled_price_pattern = (
        r"(?:Price|MSRP|Retail Price|Retail|RRP)\s*[:\-]?\s*"
        r"(\$[\d,]+(?:\.\d{2})?\s*(?:USD|CAD|AUD|GBP|EUR)?)"
    )

    labeled_match = re.search(
        labeled_price_pattern,
        text,
        flags=re.IGNORECASE
    )

    if labeled_match:
        return labeled_match.group(1).strip()

    fallback_match = re.search(
        r"(\$[\d,]+(?:\.\d{2})?\s*(?:USD|CAD|AUD|GBP|EUR)?)",
        text,
        flags=re.IGNORECASE
    )

    if fallback_match:
        return fallback_match.group(1).strip()

    return None


def clean_article_text(text: str) -> str:
    """
    Clean article body text by removing common embedded player/navigation noise.
    """

    noise_patterns = [
        r"0 seconds of .*? Volume 0%",
        r"Press shift question mark .*? Keyboard Shortcuts",
        r"Keyboard Shortcuts Enabled Disabled .*? Email Link",
        r"facebook linkedin x tumblr reddit pinterest Email Link",
    ]

    cleaned_text = text

    for pattern in noise_patterns:
        cleaned_text = re.sub(
            pattern,
            " ",
            cleaned_text,
            flags=re.IGNORECASE
        )

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def extract_article_text(soup: BeautifulSoup) -> str:
    """
    Extract article body text from the Pinkbike blog body container.
    """

    body = soup.select_one(".blog-body")

    if body is None:
        body = soup.select_one("#blog-container")

    if body is None:
        return ""

    # Remove scripts, styles, videos, comments, and obvious non-article containers.
    for element in body.select(
        "script, style, iframe, video, .news-comments, .commentslist, #comment_wrap"
    ):
        element.decompose()

    article_text = body.get_text(" ", strip=True)

    return clean_article_text(article_text)


def parse_article(html_file: Path) -> dict:
    """
    Parse a single saved Pinkbike article HTML file.
    """

    html = html_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    article_json = get_json_ld_article(soup)

    title = article_json.get("headline")

    if not title:
        title_element = soup.find("h1")
        title = title_element.get_text(" ", strip=True) if title_element else None

    source_url = article_json.get("url")
    author = extract_author(article_json)

    publish_date = (
        article_json.get("datePublished")
        or extract_meta_content(soup, "article:published_time")
    )

    modified_date = article_json.get("dateModified")

    tags = article_json.get("keywords")

    if isinstance(tags, list):
        tags = ", ".join(tags)

    article_text = extract_article_text(soup)
    retail_price = extract_retail_price(article_text)

    return {
        "source_file": html_file.name,
        "source_url": source_url,
        "title": title,
        "author": author,
        "publish_date": publish_date,
        "modified_date": modified_date,
        "tags": tags,
        "retail_price": retail_price,
        "article_text": article_text,
        "article_text_length": len(article_text),
    }


def main() -> None:
    """
    Parse all saved article HTML files.
    """

    rows = []

    html_files = sorted(ARTICLE_HTML_DIR.glob("*.html"))

    print(f"Found {len(html_files)} article files")

    for html_file in html_files:
        print(f"Parsing: {html_file.name}")

        try:
            rows.append(parse_article(html_file))

        except Exception as error:
            print(f"Failed to parse {html_file.name}")
            print(f"Error: {error}")

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved parsed dataset to:")
    print(OUTPUT_FILE)

    print("\nPreview:")
    print(
        df[
            [
                "title",
                "author",
                "publish_date",
                "retail_price",
                "article_text_length",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()