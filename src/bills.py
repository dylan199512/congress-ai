# -*- coding: utf-8 -*-
"""
bills.py
--------
Fetches congressional bill metadata and summaries from the Congress.gov API
and saves them to data/bills.csv.

Pre-filters bills by keyword in the title at the list stage so we only
fetch detail for relevant bills — reducing API calls from ~40,000 to ~500.

Output: data/bills.csv
Columns: bill_id | title | summary | sponsor_bioguide_id | topics
"""

import csv
import logging
import time
from typing import Optional

from src.config import (
    BILLS_CSV,
    CONGRESSES,
    HEALTHCARE_KEYWORDS,
    AI_KEYWORDS,
)
from src.utils import classify_topics, congress_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_is_relevant(title: str) -> bool:
    """Return True if the bill title contains a healthcare or AI keyword."""
    title_lower = title.lower()
    return any(k in title_lower for k in HEALTHCARE_KEYWORDS + AI_KEYWORDS)


def _extract_sponsor_bioguide(sponsors) -> str:
    """Return the bioguide ID of the first sponsor, or empty string."""
    if not isinstance(sponsors, list) or not sponsors:
        return ""
    return str(sponsors[0].get("bioguideId", "") or "")


def _extract_subjects_text(subjects) -> str:
    """Join a list of subject dicts into a string."""
    if not isinstance(subjects, list):
        return ""
    return ", ".join(
        str(s.get("name", "")) for s in subjects if isinstance(s, dict)
    )


# ---------------------------------------------------------------------------
# API accessors
# ---------------------------------------------------------------------------

def _fetch_bill_list(congress: int) -> list[dict]:
    """
    Retrieve bills for a given congress, pre-filtering by title keyword.
    Only fetches detail for bills matching healthcare/AI keywords in title.
    """
    bills: list[dict] = []
    url = f"/bill/{congress}"
    page = 0

    while url:
        page += 1
        print(f"[BILLS] Congress {congress} — page {page} …", flush=True)
        data = congress_get(url)

        if not data:
            print("[BILLS] No data, stopping.")
            break

        page_bills = data.get("bills", [])
        if not isinstance(page_bills, list):
            break

        # Pre-filter by title — skips ~95% of irrelevant bills
        relevant = [b for b in page_bills if _title_is_relevant(b.get("title", ""))]
        bills.extend(relevant)
        print(f"         {len(relevant)}/{len(page_bills)} relevant | total: {len(bills)}", flush=True)

        pagination = data.get("pagination") or {}
        next_url   = pagination.get("next")
        url        = next_url if next_url else None
        time.sleep(0.15)

    print(f"[BILLS] Congress {congress}: {len(bills)} relevant bills to fetch detail for")
    return bills


def _fetch_bill_detail(congress: int, bill_type: str, number: str) -> Optional[dict]:
    """Fetch full detail for a single bill."""
    path = f"/bill/{congress}/{bill_type}/{number}"
    data = congress_get(path)
    if not data:
        return None
    bill = data.get("bill")
    if not isinstance(bill, dict):
        return None
    return bill


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _bill_to_row(congress: int, b: dict) -> Optional[dict]:
    """Convert a bill summary + detail fetch into a CSV row dict."""
    bill_type = str(b.get("type",   "") or "").lower()
    number    = str(b.get("number", "") or "")

    if not bill_type or not number:
        return None

    detail = _fetch_bill_detail(congress, bill_type, number)
    if not detail:
        return None

    title       = str(detail.get("title", "") or "")
    # Summary can be a dict, list of dicts, or missing
    summary_obj = detail.get("summary") or {}
    if isinstance(summary_obj, dict):
        summary = str(summary_obj.get("text", "") or "")
    elif isinstance(summary_obj, list) and summary_obj:
        summary = str(summary_obj[0].get("text", "") or "")
    else:
        summary = ""
    summary = summary.strip() or title  # fall back to title if no summary

    sponsor_bioguide = _extract_sponsor_bioguide(detail.get("sponsors"))
    subjects_text    = _extract_subjects_text(detail.get("subjects"))

    combined_text = f"{title}\n{summary}\n{subjects_text}"
    topics        = classify_topics(combined_text)

    if not topics:
        return None

    return {
        "bill_id":             f"{bill_type.upper()}{number}",
        "title":               title,
        "summary":             summary,
        "sponsor_bioguide_id": sponsor_bioguide,
        "topics":              ",".join(topics),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_bills() -> None:
    """
    Fetch relevant bills for configured congresses and write to data/bills.csv.
    Pre-filters by title keyword to avoid fetching all ~20,000+ bills.
    """
    rows: list[dict] = []

    for congress in CONGRESSES:
        bill_list = _fetch_bill_list(congress)
        print(f"[BILLS] Fetching detail for {len(bill_list)} bills in congress {congress}…", flush=True)

        for i, b in enumerate(bill_list):
            row = _bill_to_row(congress, b)
            if row:
                rows.append(row)
            if i % 10 == 0:
                print(f"  {i}/{len(bill_list)} processed, {len(rows)} kept so far", flush=True)
            time.sleep(0.1)

    print(f"[BILLS] Total relevant bill rows: {len(rows)}", flush=True)
    _write_bills_csv(rows)


def _write_bills_csv(rows: list[dict]) -> None:
    """Write bill rows to the configured BILLS_CSV path."""
    fieldnames = ["bill_id", "title", "summary", "sponsor_bioguide_id", "topics"]
    with open(BILLS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[BILLS] Saved {len(rows)} rows to {BILLS_CSV}", flush=True)
