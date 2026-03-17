# -*- coding: utf-8 -*-
"""
utils.py
--------
Shared utility functions used across the pipeline:
  - HTTP helpers (safe_get, govinfo_get)
  - Date normalisation
  - Topic classification
  - Bioguide / URL validation
"""

import re
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

from src.config import (
    HTTP_HEADERS,
    HTTP_TIMEOUT,
    CONGRESS_API_KEY,
    GOVINFO_API_KEY,
    GOVINFO_BASE,
    LEGISLATORS_CSV,
    HEALTHCARE_KEYWORDS,
    AI_KEYWORDS,
    TOPIC_WHITELIST,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded set of valid bioguide IDs
# ---------------------------------------------------------------------------

_VALID_BIOGUIDES: Optional[set[str]] = None


def get_valid_bioguides() -> set[str]:
    """Return the set of valid bioguide IDs, loading from CSV on first call."""
    global _VALID_BIOGUIDES
    if _VALID_BIOGUIDES is None:
        try:
            df = pd.read_csv(LEGISLATORS_CSV)
            _VALID_BIOGUIDES = set(df["bioguide_id"].dropna().astype(str).tolist())
        except Exception as exc:
            logger.warning("Could not load legislators CSV: %s", exc)
            _VALID_BIOGUIDES = set()
    return _VALID_BIOGUIDES


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def safe_get(url: str, timeout: int = HTTP_TIMEOUT, json_expected: bool = False):
    """
    Perform a GET request, returning the response object (or parsed JSON).

    Returns None on any error or non-200 status so callers can handle
    missing data gracefully rather than crashing.
    """
    try:
        resp = requests.get(
            url,
            headers=HTTP_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            logger.debug("HTTP %s for %s", resp.status_code, url)
            return None
        if json_expected:
            try:
                return resp.json()
            except Exception as exc:
                logger.warning("JSON parse failed for %s: %s", url, exc)
                return None
        return resp
    except requests.RequestException as exc:
        logger.warning("Request error for %s: %s", url, exc)
        return None


def congress_get(path: str, **extra_params) -> Optional[dict]:
    """
    Convenience wrapper for the Congress.gov API.

    Args:
        path: The URL path *or* full URL (if it already starts with 'http').
        **extra_params: Additional query parameters merged with api_key.

    Returns:
        Parsed JSON dict, or None on failure.
    """
    from src.config import CONGRESS_API_BASE  # avoid circular at module level

    url = path if path.startswith("http") else f"{CONGRESS_API_BASE}{path}"
    params: dict = {"api_key": CONGRESS_API_KEY, "format": "json", **extra_params}
    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("Congress API HTTP %s for %s", resp.status_code, url)
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("Congress API error for %s: %s", url, exc)
        return None


def govinfo_get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """
    Convenience wrapper for the GovInfo API.

    Always injects the GovInfo API key and returns parsed JSON or None.
    """
    p = dict(params or {})
    p["api_key"] = GOVINFO_API_KEY
    try:
        resp = requests.get(url, params=p, timeout=20)
        if resp.status_code != 200:
            logger.debug("GovInfo HTTP %s for %s", resp.status_code, url)
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("GovInfo error for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%a, %d %b %Y %H:%M:%S %z")


def parse_valid_date(dt_str: Optional[str]) -> Optional[str]:
    """
    Parse a date string to ISO YYYY-MM-DD, or return None if invalid.

    Rejects dates before year 2000 or after today (avoids clearly wrong data).
    """
    if not dt_str:
        return None

    today = datetime.today().date()

    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(dt_str, fmt).date()
            if d.year < 2000 or d > today:
                return None
            return d.isoformat()
        except ValueError:
            continue

    # Regex fallback
    match = re.search(r"\d{4}-\d{2}-\d{2}", dt_str)
    if match:
        try:
            d = datetime.strptime(match.group(0), "%Y-%m-%d").date()
            if d.year < 2000 or d > today:
                return None
            return d.isoformat()
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_bioguide(bio: Optional[str]) -> bool:
    """Return True if `bio` is a known bioguide ID from the legislators CSV."""
    if not isinstance(bio, str):
        return False
    return bio in get_valid_bioguides()


def is_valid_url(u: Optional[str]) -> bool:
    """Return True if `u` looks like an http(s) URL."""
    return isinstance(u, str) and u.startswith("http")


# ---------------------------------------------------------------------------
# Topic classification
# ---------------------------------------------------------------------------

def classify_topics(text: str) -> list[str]:
    """
    Return a list of whitelisted topic tags for the given text.

    Currently detects 'healthcare' and 'ai' based on keyword matching.
    Only topics present in TOPIC_WHITELIST are returned.
    """
    text_lower = text.lower()
    topics: list[str] = []

    if any(keyword in text_lower for keyword in HEALTHCARE_KEYWORDS):
        topics.append("healthcare")
    if any(keyword in text_lower for keyword in AI_KEYWORDS):
        topics.append("ai")

    return [t for t in topics if t in TOPIC_WHITELIST]


def subjects_match(subjects: list[str]) -> bool:
    """Return True if any subject string contains a healthcare or AI keyword."""
    if not subjects:
        return False
    combined = " ".join(subjects).lower()
    return any(k in combined for k in AI_KEYWORDS + HEALTHCARE_KEYWORDS)
