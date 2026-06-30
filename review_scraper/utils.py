"""
Shared helper functions for the Pinkbike review scraper pipeline.
"""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """
    Normalize article URLs for duplicate checking.
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
    """

    slug = url.rstrip("/").split("/")[-1]

    if slug.endswith(".html"):
        slug = slug[:-5]

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug)

    return f"{slug}.html"