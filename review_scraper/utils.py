"""
Shared helper functions for the Pinkbike review scraper pipeline.

Purpose:
This file contains reusable utility functions used across the scraping
and parsing workflow. These functions support URL standardization and
safe local file naming, which help make the data collection process more
consistent and repeatable.

AI Use:
AI tools were used to assist with code review and annotation.
"""
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """
    Normalize article URLs for duplicate checking.

    This function standardizes URLs by:
    - forcing the scheme to https,
    - lowercasing the domain,
    - removing trailing slashes from the path,
    - removing UTM tracking query parameters, and
    - dropping URL fragments.

    Normalized URLs allow the scraper to identify duplicate article links
    even when the same page appears with different tracking parameters.
    """

    if not isinstance(url, str):
        return ""

    url = url.strip()

    parsed = urlsplit(url)

    scheme = "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    query_params = [
        (key, value)
        for key, value in parse_qsl(parsed.query)
        if not key.lower().startswith("utm_")
    ]

    query = urlencode(query_params)

    return urlunsplit((scheme, netloc, path, query, ""))


def make_safe_filename(url: str) -> str:
    """
    Create a safe HTML filename from an article URL.

    The scraper saves article HTML locally for QA and reproducibility.
    This function extracts the article slug from the URL, removes the
    .html suffix if present, replaces unsafe filename characters with
    underscores, and returns a standardized .html filename.
    """

    slug = url.rstrip("/").split("/")[-1]

    if slug.endswith(".html"):
        slug = slug[:-5]

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug)

    return f"{slug}.html"