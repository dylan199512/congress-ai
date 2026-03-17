# -*- coding: utf-8 -*-
"""
stances.py
----------
Scrapes and saves congressional member stances on healthcare and AI topics.

Data sources (tried in order per member):
  1. RSS feed (fast, structured)
  2. Inferred press-release pages on the member's official website

The module is built for *parallel* execution via a ThreadPoolExecutor so the
~500-member scrape completes in minutes rather than hours.

Output: data/member_stances.csv
Columns: bioguide_id | date | source_url | topic | text
"""

import csv
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import urljoin

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

import html
import re
from html.parser import HTMLParser


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode entities from text."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    class _S(HTMLParser):
        def __init__(self): super().__init__(); self.t = []
        def handle_data(self, d): self.t.append(d)
    s = _S(); s.feed(text)
    return re.sub(r'\s+', ' ', ' '.join(s.t)).strip()


from src.config import (
    LEGISLATORS_CSV,
    STANCES_CSV,
    SCRAPE_SLEEP,
    SCRAPE_MAX_WORKERS,
)
from src.utils import (
    classify_topics,
    is_valid_bioguide,
    is_valid_url,
    parse_valid_date,
    safe_get,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRESS_PAGE_SUFFIXES = [
    "/press-releases",
    "/newsroom",
    "/news",
    "/media",
    "/press",
]


def guess_press_urls(base_url: str) -> list[str]:
    """
    Generate candidate press-release URLs from a member's root website URL.

    Example: https://smith.house.gov → [
        https://smith.house.gov/press-releases,
        https://smith.house.gov/newsroom,
        ...
    ]
    """
    base = base_url.rstrip("/")
    return [base + suffix for suffix in _PRESS_PAGE_SUFFIXES]


def _make_stance_row(
    bioguide_id: str,
    date: str,
    source_url: str,
    topic: str,
    text: str,
) -> dict:
    return {
        "bioguide_id": bioguide_id,
        "date": date,
        "source_url": source_url,
        "topic": topic,
        "text": text,
    }


# ---------------------------------------------------------------------------
# RSS scraper
# ---------------------------------------------------------------------------

def parse_rss_for_stances(bioguide_id: str, rss_url: Optional[str]) -> list[dict]:
    """
    Parse an RSS/Atom feed and return stance rows matching healthcare/AI topics.

    Args:
        bioguide_id: The member's bioguide identifier.
        rss_url:     The RSS feed URL (may be None/empty).

    Returns:
        List of stance dicts (may be empty).
    """
    if not rss_url or not isinstance(rss_url, str):
        return []

    items: list[dict] = []

    try:
        feed = feedparser.parse(rss_url)
    except Exception as exc:
        logger.warning("[RSS] Parse error for %s: %s", rss_url, exc)
        return []

    for entry in feed.entries:
        title     = getattr(entry, "title",     "")
        summary   = getattr(entry, "summary",   "")
        link      = getattr(entry, "link",      "")
        published = (
            getattr(entry, "published", "")
            or getattr(entry, "updated",   "")
        )

        text   = _strip_html(f"{title}\n{summary}".strip())
        topics = classify_topics(text)

        if not topics:
            continue

        date_norm = parse_valid_date(published)
        if not date_norm:
            continue

        if not is_valid_url(link):
            continue

        if not is_valid_bioguide(bioguide_id):
            continue

        for topic in topics:
            items.append(_make_stance_row(bioguide_id, date_norm, link, topic, text))

    return items


# ---------------------------------------------------------------------------
# Press-page scraper
# ---------------------------------------------------------------------------

def scrape_press_page_for_stances(bioguide_id: str, url: str) -> list[dict]:
    """
    Scrape a member's press-release page for stance items.

    Looks for article-like containers, extracts title, date, link, and a
    short snippet, then classifies by topic.

    Args:
        bioguide_id: The member's bioguide identifier.
        url:         The press-release page URL.

    Returns:
        List of stance dicts (may be empty).
    """
    items: list[dict] = []

    resp = safe_get(url)
    if not resp:
        return []

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        logger.warning("[PRESS] BeautifulSoup error for %s: %s", url, exc)
        return []

    # Look for news/press article containers
    articles = soup.find_all(
        ["article", "div"],
        class_=re.compile(r"(press|news|release)", re.IGNORECASE),
    )

    for art in articles[:20]:
        # Title
        title_tag = art.find(["h1", "h2", "h3", "a"])
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        if not title:
            continue

        # Date
        date_tag  = art.find("time")
        date_text = ""
        if date_tag:
            date_text = date_tag.get("datetime") or date_tag.get_text(strip=True)
        date_norm = parse_valid_date(date_text)
        if not date_norm:
            continue

        # Link
        link_tag = art.find("a", href=True)
        link     = urljoin(url, link_tag["href"]) if link_tag else url
        if not is_valid_url(link):
            continue

        # Snippet
        p       = art.find("p")
        snippet = p.get_text(strip=True) if p else title
        text    = f"{title}\n{snippet}".strip()

        topics = classify_topics(text)
        if not topics:
            continue

        if not is_valid_bioguide(bioguide_id):
            continue

        for topic in topics:
            items.append(_make_stance_row(bioguide_id, date_norm, link, topic, text))

    return items


# ---------------------------------------------------------------------------
# Per-member worker (called from thread pool)
# ---------------------------------------------------------------------------

def _scrape_member(row: pd.Series) -> list[dict]:
    """
    Scrape one member: try RSS first, then press pages.

    Designed to be called from a thread pool — catches all exceptions
    so a single broken member doesn't kill the entire run.
    """
    bioguide_id = row.get("bioguide_id", "")
    url         = row.get("url",         "")
    rss_url     = row.get("rss_url",     "")

    logger.debug("[STANCES] Processing %s", bioguide_id)

    try:
        rss_items = parse_rss_for_stances(bioguide_id, rss_url)
    except Exception as exc:
        logger.warning("[STANCES] RSS exception for %s: %s", bioguide_id, exc)
        rss_items = []

    if rss_items:
        time.sleep(SCRAPE_SLEEP)
        return rss_items

    # Fall back to press pages only when RSS yields nothing
    if isinstance(url, str) and url.strip():
        for guess_url in guess_press_urls(url):
            logger.debug("[STANCES]   Trying press page: %s", guess_url)
            try:
                page_items = scrape_press_page_for_stances(bioguide_id, guess_url)
            except Exception as exc:
                logger.warning("[STANCES]   Press page exception for %s: %s", guess_url, exc)
                page_items = []

            if page_items:
                time.sleep(SCRAPE_SLEEP)
                return page_items

    time.sleep(SCRAPE_SLEEP)
    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_member_stances(max_workers: int = SCRAPE_MAX_WORKERS) -> None:
    """
    Build member_stances.csv by scraping all current legislators in parallel.

    Steps:
      1. Load legislators from CSV.
      2. Scrape each member (RSS → press pages) using a thread pool.
      3. Deduplicate rows by (bioguide_id, date, source_url, topic).
      4. Write output CSV (only if rows were found; avoids overwriting good data).

    Args:
        max_workers: Number of parallel scraping threads.
    """
    try:
        legislators = pd.read_csv(LEGISLATORS_CSV)
    except FileNotFoundError:
        logger.error("Legislators CSV not found: %s", LEGISLATORS_CSV)
        return

    # Ensure required columns exist
    for col in ("bioguide_id", "url", "rss_url"):
        if col not in legislators.columns:
            legislators[col] = None

    all_rows: list[dict] = []

    logger.info("[STANCES] Scraping %d legislators with %d workers…",
                len(legislators), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_scrape_member, row): idx
            for idx, row in legislators.iterrows()
        }
        for future in as_completed(futures):
            try:
                all_rows.extend(future.result())
            except Exception as exc:
                logger.warning("[STANCES] Worker exception: %s", exc)

    # Deduplicate
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in all_rows:
        key = (r["bioguide_id"], r["date"], r["source_url"], r["topic"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    logger.info("[STANCES] Total valid stance rows after dedup: %d", len(deduped))

    if not deduped:
        logger.warning("[STANCES] No valid stances found — not overwriting existing file.")
        return

    with open(STANCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bioguide_id", "date", "source_url", "topic", "text"],
        )
        writer.writeheader()
        writer.writerows(deduped)

    logger.info("[STANCES] Saved to %s", STANCES_CSV)
