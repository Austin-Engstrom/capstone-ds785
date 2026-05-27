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

def keyword_score(text: str, keywords: list[str]) -> int:
    """
    Count how many keywords appear in the text.
    """
    return sum(1 for keyword in keywords if keyword in text)


def classify_product_category(text: str) -> str:
    """
    Classify high-level product category using scored keyword groups.
    """

    text = text.lower()

    category_keywords = {
        "Bike": [
            "bike", "frame", "frameset", "hardtail", "trail bike",
            "enduro bike", "downhill bike", "dh bike", "xc bike",
            "cross-country bike", "gravel bike", "e-bike", "ebike",
            "electric mountain bike",
        ],
        "Component": [
            "fork", "shock", "wheel", "wheelset", "tire", "tyre",
            "brake", "rotor", "drivetrain", "derailleur", "cassette",
            "chain", "crank", "pedal", "dropper", "seatpost",
            "handlebar", "stem", "grip", "saddle",
        ],
        "Clothing": [
            "jersey", "pants", "shorts", "glove", "gloves",
            "shoe", "shoes", "jacket", "bib", "sock", "socks",
        ],
        "Protective Gear": [
            "helmet", "knee pad", "elbow pad", "body armor",
            "chest protector", "back protector", "goggles",
            "protection",
        ],
        "Accessory": [
            "pack", "hip pack", "backpack", "tool", "multi-tool",
            "pump", "light", "computer", "gps", "rack", "bag",
            "bottle", "cage",
        ],
    }

    scores = {
        category: keyword_score(text, keywords)
        for category, keywords in category_keywords.items()
    }

    best_category = max(scores, key=scores.get)

    if scores[best_category] == 0:
        return "Other"

    return best_category

def classify_product_subcategory(text: str) -> Optional[str]:
    """
    Classify more detailed product subcategory using ordered keyword matching.
    """

    text = text.lower()

    subcategory_map = {
        # Bikes
        "Downhill Bike": ["downhill bike", "dh bike"],
        "Enduro Bike": ["enduro bike", "enduro"],
        "Trail Bike": ["trail bike"],
        "XC Bike": ["xc bike", "cross-country bike", "cross country"],
        "Hardtail": ["hardtail"],
        "E-Bike": ["e-bike", "ebike", "electric mountain bike"],
        "Gravel Bike": ["gravel bike", "gravel"],

        # Components
        "Suspension Fork": ["suspension fork", "fork"],
        "Rear Shock": ["rear shock", "shock"],
        "Wheelset": ["wheelset", "wheels"],
        "Tire": ["tire", "tyre"],
        "Brake": ["brake", "rotor"],
        "Drivetrain": ["drivetrain", "derailleur", "cassette", "chainring", "chain"],
        "Crankset": ["crankset", "crank"],
        "Pedal": ["pedal"],
        "Dropper Post": ["dropper", "seatpost"],
        "Handlebar": ["handlebar", "bar"],
        "Stem": ["stem"],
        "Saddle": ["saddle"],
        "Grip": ["grip"],

        # Clothing
        "Jersey": ["jersey"],
        "Pants": ["pants"],
        "Shorts": ["shorts"],
        "Gloves": ["glove", "gloves"],
        "Shoes": ["shoe", "shoes"],
        "Jacket": ["jacket"],
        "Socks": ["sock", "socks"],

        # Protective Gear
        "Helmet": ["helmet"],
        "Knee Pads": ["knee pad", "knee pads"],
        "Elbow Pads": ["elbow pad", "elbow pads"],
        "Body Armor": ["body armor", "chest protector", "back protector"],
        "Goggles": ["goggle", "goggles"],

        # Accessories
        "Pack": ["hip pack", "backpack", "pack"],
        "Tool": ["multi-tool", "tool"],
        "Pump": ["pump"],
        "Light": ["light"],
        "Bike Computer": ["computer", "gps"],
        "Bag": ["bag"],
        "Bottle Cage": ["bottle cage", "cage"],
    }

    for subcategory, keywords in subcategory_map.items():
        if any(keyword in text for keyword in keywords):
            return subcategory

    return None

def classify_review_type(text: str) -> str:
    """
    Classify the review/article format.
    """

    text = text.lower()

    review_type_map = {
        "Long-Term Review": [
            "long-term review", "long term review", "long-term test",
            "long term test",
        ],
        "Field Test": [
            "field test",
        ],
        "First Ride": [
            "first ride",
        ],
        "Group Test": [
            "group test", "roundup", "round-up",
        ],
        "Value Comparison": [
            "value bike", "budget bike", "comparison", "vs.",
        ],
        "Product Review": [
            "review",
        ],
    }

    for review_type, keywords in review_type_map.items():
        if any(keyword in text for keyword in keywords):
            return review_type

    return "Unclassified"

def extract_brand(text: str) -> Optional[str]:
    """
    Extract likely product brand from article title/text using a known brand list.
    """

    brand_keywords = [
        # Bikes
        "Specialized", "Trek", "Santa Cruz", "Yeti", "Pivot", "Norco",
        "Commencal", "Canyon", "YT", "Giant", "Liv", "Cannondale",
        "Transition", "Rocky Mountain", "Ibis", "Evil", "Propain",
        "Orbea", "Scott", "Marin", "Kona", "Devinci", "GT",

        # Components
        "SRAM", "Shimano", "RockShox", "Fox", "Marzocchi", "Maxxis",
        "Schwalbe", "Continental", "DT Swiss", "Industry Nine",
        "Race Face", "OneUp", "Works Components", "PNW", "Hope",
        "TRP", "Hayes", "Magura", "Burgtec", "Renthal", "Deity",

        # Clothing / protective gear
        "Troy Lee Designs", "Fox Racing", "Giro", "Bell", "POC",
        "Smith", "100%", "Leatt", "7mesh", "Endura", "Five Ten",
        "Ride Concepts", "Rapha", "Pearl Izumi", "Dakine", "ION",
        "6D",
    ]

    # Check longer brand names first so "Fox Racing" beats "Fox".
    brand_keywords = sorted(
        brand_keywords,
        key=len,
        reverse=True
    )

    for brand in brand_keywords:
        pattern = rf"\b{re.escape(brand)}\b"

        if re.search(pattern, text, flags=re.IGNORECASE):
            return brand

    return None

def extract_product_name(
    title: str,
    brand: Optional[str]
) -> Optional[str]:
    """
    Attempt to extract product name from article title.

    Prototype approach:
    - remove review-type phrases
    - remove brand from front of title
    - clean separators
    """

    if not title:
        return None

    cleaned = title

    removal_patterns = [
        r"review",
        r"long-term review",
        r"long term review",
        r"field test",
        r"first ride",
        r"vs\.",
        r"comparison",
        r"tested",
    ]

    for pattern in removal_patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    # Remove brand from beginning if found.
    if brand:
        cleaned = re.sub(
            rf"^{re.escape(brand)}\s+",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    # Remove common separators.
    cleaned = re.sub(r"[-:|]+", " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned if cleaned else None

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
    combined_text = f"{title} {article_text}"
    product_category = classify_product_category(combined_text)
    product_subcategory = classify_product_subcategory(combined_text)
    review_type = classify_review_type(combined_text)
    brand = extract_brand(combined_text)
    product_name = extract_product_name(title, brand)

    return {
        "source_file": html_file.name,
        "source_url": source_url,
        "title": title,
        "author": author,
        "publish_date": publish_date,
        "modified_date": modified_date,
        "tags": tags,
        "retail_price": retail_price,
        "brand": brand,
        "product_name": product_name,
        "article_text": article_text,
        "article_text_length": len(article_text),
        "product_category": product_category,
        "product_subcategory": product_subcategory,
        "review_type": review_type
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
                "brand",
                "product_name",
                "retail_price",
                "product_category",
                "product_subcategory",
                "review_type",
                "article_text_length",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()