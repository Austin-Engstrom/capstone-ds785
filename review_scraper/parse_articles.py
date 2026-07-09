"""
Parse saved Pinkbike review article HTML files and extract structured article data.

Purpose:
This script supports the cleaning and preprocessing stage of the project.
It converts raw saved Pinkbike article HTML files into a structured CSV
dataset that can be used for later sentiment analysis and NLP modeling.

Main tasks:
1. Load saved article HTML files.
2. Extract article metadata such as title, author, publish date, URL, and tags.
3. Extract and clean article body text.
4. Extract product attributes such as brand, product name, retail price, product category, product subcategory, and review type.
5. Save the structured review-level dataset to data/processed.

AI Use:
AI tools were used to assist with code review, debugging, function design, extraction logic, and annotation.
"""
from pathlib import Path
from typing import Optional
import json
import re

import pandas as pd
from bs4 import BeautifulSoup

from review_scraper.config import (
    ARTICLE_HTML_DIR,
    REVIEWS_DATASET_FILE,
    BRAND_REFERENCE_FILE,
    create_project_directories,
)


OUTPUT_FILE = REVIEWS_DATASET_FILE

"""
Brand reference loading
The brand reference CSV allows brand extraction to use a controlled
reference table instead of relying only on hard-coded brand names.
This makes the parser easier to update as new brands or aliases are found.
"""

def load_brand_reference() -> pd.DataFrame:
    """
    Load the active brand reference table.
    """

    brand_reference = pd.read_csv(BRAND_REFERENCE_FILE)

    # Keep only active brands
    brand_reference = brand_reference[
        brand_reference["active"].astype(str).str.upper() == "TRUE"
    ].copy()

    # Longest aliases first, then highest priority
    brand_reference["alias_length"] = brand_reference["alias"].str.len()

    brand_reference = brand_reference.sort_values(
        by=["priority", "alias_length"],
        ascending=[False, False],
    )

    return brand_reference

BRAND_REFERENCE = load_brand_reference()

"""
Metadata extraction helpers
These functions extract structured article metadata from JSON-LD and
HTML meta tags when available. JSON-LD is preferred because it usually
contains cleaner article-level information than visible page text.
"""

def get_json_ld_article(soup: BeautifulSoup) -> dict:
    """
    Extract the NewsArticle JSON-LD object when available.
    Handles both dictionary and list JSON-LD structures.
    """

    for script in soup.find_all("script", type="application/ld+json"):
        script_text = script.get_text(strip=True)

        try:
            data = json.loads(script_text)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            if data.get("@type") == "NewsArticle":
                return data

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == "NewsArticle":
                    return item

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

"""
Price extraction and normalization
These functions identify raw price text from the article body and convert
it into a numeric price and currency field for later analysis.
"""

def extract_retail_price(text: str) -> Optional[str]:
    """
    Extract likely product retail price from article text.
    """

    labeled_price_pattern = (
        r"(?:Price|MSRP|Retail Price|Retail|RRP)\s*[:\-]?\s*"
        r"(\$[\d,]+(?:\.\d{2})?\s*(?:USD|CAD|AUD|GBP|EUR)?)"
    )

    labeled_match = re.search(
        labeled_price_pattern,
        text,
        flags=re.IGNORECASE,
    )

    if labeled_match:
        return labeled_match.group(1).strip()

    fallback_match = re.search(
        r"(\$[\d,]+(?:\.\d{2})?\s*(?:USD|CAD|AUD|GBP|EUR)?)",
        text,
        flags=re.IGNORECASE,
    )

    if fallback_match:
        return fallback_match.group(1).strip()

    return None


def normalize_price(price_text: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    """
    Normalize a raw retail price string into numeric value and currency.
    """

    if not price_text:
        return None, None

    currency_symbol_map = {
        "$": None,
        "€": "EUR",
        "£": "GBP",
    }

    currency_code_pattern = r"\b(USD|CAD|AUD|GBP|EUR)\b"

    currency_match = re.search(
        currency_code_pattern,
        price_text,
        flags=re.IGNORECASE,
    )

    currency = currency_match.group(1).upper() if currency_match else None

    if currency is None:
        for symbol, symbol_currency in currency_symbol_map.items():
            if symbol in price_text:
                currency = symbol_currency
                break

    numeric_match = re.search(
        r"[\$€£]?\s*([\d,]+(?:\.\d{2})?)",
        price_text,
    )

    if not numeric_match:
        return None, currency

    price_value = float(numeric_match.group(1).replace(",", ""))

    return price_value, currency

"""
Article text extraction and cleaning
These functions isolate the main Pinkbike article body and remove common
page noise such as embedded video controls, comments, scripts, and style
elements.
"""

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
            flags=re.IGNORECASE,
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

"""
Product classification helpers
These functions use rule-based keyword matching to classify each review
into a high-level product category, detailed subcategory, and review type.
This gives the project structured fields for exploratory analysis before the NLP modeling stage.
"""

def classify_product_category(text: str) -> str:
    """
    Classify high-level product category using keyword groups.
    """

    text = text.lower()

    protective_keywords = [
        "helmet", "full face", "full-face", "knee pad", "elbow pad",
        "body armor", "chest protector", "back protector", "goggles",
        "protection",
    ]

    clothing_keywords = [
        "jersey", "pants", "shorts", "glove", "gloves",
        "shoe", "shoes", "jacket", "bib", "sock", "socks",
    ]

    bike_keywords = [
        "bike", "frame", "frameset", "hardtail", "trail bike",
        "enduro bike", "downhill bike", "dh bike", "xc bike",
        "cross-country bike", "gravel bike", "e-bike", "ebike",
        "electric mountain bike",
    ]

    component_keywords = [
        "fork", "shock", "wheel", "wheelset", "tire", "tyre",
        "brake", "rotor", "drivetrain", "derailleur", "cassette",
        "chain", "crank", "pedal", "dropper", "seatpost",
        "handlebar", "stem", "grip", "saddle",
    ]

    accessory_keywords = [
        "pack", "hip pack", "backpack", "tool", "multi-tool",
        "pump", "light", "computer", "gps", "rack", "bag",
        "bottle", "cage",
    ]

    if any(keyword in text for keyword in protective_keywords):
        return "Protective Gear"

    if any(keyword in text for keyword in clothing_keywords):
        return "Clothing"

    if any(keyword in text for keyword in component_keywords):
        return "Component"

    if any(keyword in text for keyword in bike_keywords):
        return "Bike"

    if any(keyword in text for keyword in accessory_keywords):
        return "Accessory"

    return "Other"

def classify_product_subcategory(text: str) -> Optional[str]:
    """
    Classify detailed product subcategory using ordered keyword matching.
    """

    text = text.lower()

    subcategory_map = {
        "Helmet": ["helmet"],
        "Knee Pads": ["knee pad", "knee pads"],
        "Elbow Pads": ["elbow pad", "elbow pads"],
        "Body Armor": ["body armor", "chest protector", "back protector"],
        "Goggles": ["goggle", "goggles"],

        "Downhill Bike": ["downhill bike", "dh bike"],
        "Enduro Bike": ["enduro bike", "enduro"],
        "Trail Bike": ["trail bike"],
        "XC Bike": ["xc bike", "cross-country bike", "cross country"],
        "Hardtail": ["hardtail"],
        "E-Bike": ["e-bike", "ebike", "electric mountain bike"],
        "Gravel Bike": ["gravel bike", "gravel"],

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

        "Jersey": ["jersey"],
        "Pants": ["pants"],
        "Shorts": ["shorts"],
        "Gloves": ["glove", "gloves"],
        "Shoes": ["shoe", "shoes"],
        "Jacket": ["jacket"],
        "Socks": ["sock", "socks"],

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

"""
Brand and product name extraction
These functions extract the likely product brand and model name. Brand
matching uses the active brand reference table, while product name
extraction cleans the article title after removing review-related wording.
"""

def extract_brand(title: Optional[str], article_text: str) -> Optional[str]:
    """
    Extract the product brand using the brand reference table.
    """

    for source_text in [title, article_text]:
        if not source_text:
            continue

        for _, row in BRAND_REFERENCE.iterrows():
            alias = row["alias"]
            brand_name = row["brand_name"]

            pattern = rf"\b{re.escape(alias)}\b"

            if re.search(pattern, source_text, flags=re.IGNORECASE):
                return brand_name

    return None

def build_brand_slugs(brand: str) -> set[str]:
    """
    Build possible Pinkbike tag prefixes for a brand.
    """

    brand_slug = normalize_brand_slug(brand)
    brand_slugs = {brand_slug}

    suffixes = [
        "-bikes",
        "-bike",
        "-cycles",
        "-cycle-works",
        "-works",
        "-components",
        "-suspension",
        "-racing",
        "-clothing",
    ]

    for suffix in suffixes:
        if brand_slug.endswith(suffix):
            brand_slugs.add(brand_slug[:-len(suffix)])

    # Also keep the first word of longer manufacturer names.
    # Example: allied-cycle-works -> allied
    parts = brand_slug.split("-")
    if len(parts) > 1:
        brand_slugs.add(parts[0])

    return brand_slugs

def normalize_brand_slug(brand: str) -> str:
    """
    Normalize a brand name into the slug format commonly used by Pinkbike tags.

    Examples:
        Forge+Bond -> forge-and-bond
        Chris King -> chris-king
        Cane Creek -> cane-creek
        Santa Cruz -> santa-cruz
        YT -> yt
    """

    slug = brand.lower().strip()

    # Replace common symbols
    slug = slug.replace("&", "and")
    slug = slug.replace("+", "-and-")
    slug = slug.replace("'", "")

    # Replace any remaining non-alphanumeric characters with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)

    # Collapse duplicate hyphens
    slug = re.sub(r"-+", "-", slug).strip("-")

    return slug

def extract_product_from_tags(
    brand: Optional[str],
    tags: Optional[Union[list[str], str]]
) -> Optional[str]:
    """
    Extract product model name from article tags.
    """

    if not brand or not tags:
        return None

    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",")]

    brand_slugs = build_brand_slugs(brand)

    ignore_tags = {
        "reviews",
        "reviews-and-tech",
        "field-test",
        "first-ride",
        "big-brake-test",
        "trail-bikes",
        "enduro-bikes",
        "dh-bikes",
        "xc-bikes",
        "gravel-bikes",
        "emtb",
        "e-bikes",
        "helmets",
        "shoes",
        "tires",
        "wheels",
        "brakes",
        "suspension",
        "videos",
    }

    for tag in tags:
        tag_clean = tag.lower().strip()

        if tag_clean in ignore_tags:
            continue

        for slug in brand_slugs:
            if tag_clean == slug:
                continue

            if tag_clean.startswith(f"{slug}-"):
                product_slug = tag_clean[len(slug) + 1:]

                if not product_slug:
                    continue

                return product_slug.replace("-", " ").title()

    return None

def extract_product_name(
    title: Optional[str],
    brand: Optional[str],
    tags: Optional[list[str] | str] = None
) -> Optional[str]:
    """
    Extract product model name from article tags.
    Use title only to validate the tag-derived product.
    """

    tag_product = extract_product_from_tags(brand, tags)

    if not tag_product:
        return None

    if not title:
        return tag_product

    title_clean = title.lower()
    product_clean = tag_product.lower()

    if product_clean in title_clean:
        return tag_product

    return None

"""
Single-article parser
This function combines metadata extraction, text cleaning, price parsing,
brand extraction, and product classification into one structured record.
"""

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

    retail_price_raw = extract_retail_price(article_text)
    retail_price, retail_price_currency = normalize_price(retail_price_raw)

    classification_text = f"{title} {tags} {article_text}"

    brand = extract_brand(title, article_text)
    product_name = extract_product_name(title, brand, tags)

    product_category = classify_product_category(classification_text)
    product_subcategory = classify_product_subcategory(classification_text)
    review_type = classify_review_type(classification_text)

    return {
        "source_file": html_file.name,
        "source_url": source_url,
        "title": title,
        "author": author,
        "publish_date": publish_date,
        "modified_date": modified_date,
        "tags": tags,
        "retail_price_raw": retail_price_raw,
        "retail_price": retail_price,
        "currency": retail_price_currency,
        "brand": brand,
        "product_name": product_name,
        "product_category": product_category,
        "product_subcategory": product_subcategory,
        "review_type": review_type,
        "article_text_length": len(article_text),
        "article_text": article_text,
    }

"""
Batch parser entry point
Unlike the scraper, this parser intentionally processes all saved HTML
files. The scraper collects articles cautiously one at a time, but once
files are saved locally, parsing all available files is safe and repeatable.
"""

def main() -> None:
    """
    Parse all saved article HTML files.
    """

    create_project_directories()

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

    if not df.empty:
        print("\nPreview:")
        print(
            df[
                [
                    "title",
                    "author",
                    "publish_date",
                    "brand",
                    "product_name",
                    "retail_price_raw",
                    "retail_price",
                    "currency",
                    "product_category",
                    "product_subcategory",
                    "review_type",
                    "article_text_length",
                ]
            ].head()
        )


if __name__ == "__main__":
    main()