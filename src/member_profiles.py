# -*- coding: utf-8 -*-
"""
member_profiles.py
------------------
Fetches sponsored and cosponsored legislation for all current legislators
from the Congress.gov API, filters for healthcare/AI relevance, and saves
to data/member_profiles.csv.

This gives us a strong signal of each member's legislative priorities —
stronger than press releases, complementary to voting records.

Output: data/member_profiles.csv
Columns: bioguide_id | bill_id | title | type | congress | introduced_date | 
         policy_area | role | topics
"""

import csv
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_DIR,
    LEGISLATORS_CSV,
    CONGRESS_API_KEY,
    HEALTHCARE_KEYWORDS,
    AI_KEYWORDS,
)
from src.utils import classify_topics, congress_get

logger = logging.getLogger(__name__)

MEMBER_PROFILES_CSV = DATA_DIR / "member_profiles.csv"
FIELDNAMES = [
    "bioguide_id", "bill_id", "title", "type",
    "congress", "introduced_date", "policy_area", "role", "topics"
]

# Policy areas that are always relevant
RELEVANT_POLICY_AREAS = {
    "health", "science, technology, communications",
    "medicine", "public health", "medical research",
    "medicare", "medicaid", "health care",
}


def _is_relevant(title: str, policy_area: str) -> bool:
    """Return True if bill is healthcare or AI related."""
    combined = f"{title} {policy_area}".lower()
    # Also match short policy area names like "Health", "Technology"
    policy_lower = policy_area.lower()
    if policy_lower in ("health", "technology", "science", "medicine"):
        return True
    return any(k in combined for k in HEALTHCARE_KEYWORDS + AI_KEYWORDS)


def _fetch_member_legislation(bioguide_id: str, role: str = "sponsored") -> list[dict]:
    """
    Fetch sponsored or cosponsored legislation for one member.

    Args:
        bioguide_id: Member's bioguide ID.
        role: 'sponsored' or 'cosponsored'

    Returns:
        List of relevant bill dicts.
    """
    endpoint = f"/member/{bioguide_id}/{role}-legislation"
    rows = []
    url = endpoint

    max_pages = 8  # limit to 8 pages (160 items) per member for speed
    page_num  = 0

    while url and page_num < max_pages:
        page_num += 1
        data = congress_get(url)
        if not data:
            break

        items = data.get("sponsoredLegislation") or data.get("cosponsoredLegislation") or []

        for item in items:
            # Skip amendments (no 'number' or no 'type')
            if not item.get("number") or not item.get("type"):
                continue

            title       = str(item.get("title", "") or "")
            bill_type   = str(item.get("type",   "") or "")
            number      = str(item.get("number", "") or "")
            congress    = str(item.get("congress", "") or "")
            intro_date  = str(item.get("introducedDate", "") or "")
            policy_obj  = item.get("policyArea") or {}
            policy_area = str(policy_obj.get("name", "") if isinstance(policy_obj, dict) else "") or ""

            if not _is_relevant(title, policy_area):
                continue

            topics = classify_topics(f"{title} {policy_area}")
            if not topics:
                continue

            bill_id = f"{bill_type.upper()}{number}"

            rows.append({
                "bioguide_id":    bioguide_id,
                "bill_id":        bill_id,
                "title":          title,
                "type":           bill_type,
                "congress":       congress,
                "introduced_date": intro_date,
                "policy_area":    policy_area,
                "role":           role,
                "topics":         ",".join(topics),
            })

        # Pagination
        pagination = data.get("pagination") or {}
        next_url   = pagination.get("next")
        url        = next_url if next_url else None
        time.sleep(0.15)

    return rows


def build_member_profiles() -> None:
    """
    Fetch sponsored + cosponsored healthcare/AI legislation for all members.

    Saves to data/member_profiles.csv.
    """
    try:
        legislators = pd.read_csv(LEGISLATORS_CSV)
    except FileNotFoundError:
        logger.error("Legislators CSV not found: %s", LEGISLATORS_CSV)
        return

    all_rows: list[dict] = []
    total = len(legislators)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_member(args):
        i, member = args
        bio = str(member.get("bioguide_id", "") or "")
        if not bio:
            return []
        rows = []
        rows.extend(_fetch_member_legislation(bio, "sponsored"))
        rows.extend(_fetch_member_legislation(bio, "cosponsored"))
        return rows

    members_list = [(i, row) for i, (_, row) in enumerate(legislators.iterrows())]

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_fetch_member, args): args[0] for args in members_list}
        done = 0
        for future in as_completed(futures):
            try:
                all_rows.extend(future.result())
            except Exception as e:
                pass
            done += 1
            if done % 50 == 0:
                print(f"[PROFILES] {done}/{total} members processed, {len(all_rows)} rows so far", flush=True)

    # Deduplicate
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in all_rows:
        key = (r["bioguide_id"], r["bill_id"], r["role"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"[PROFILES] Total rows: {len(deduped)}", flush=True)

    with open(MEMBER_PROFILES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"[PROFILES] Saved to {MEMBER_PROFILES_CSV}", flush=True)


def get_member_profile_context(bioguide_id: str) -> str:
    """
    Return a formatted context string of a member's sponsored/cosponsored
    healthcare and AI legislation. Used in Q&A context building.
    """
    try:
        df = pd.read_csv(MEMBER_PROFILES_CSV)
    except FileNotFoundError:
        return ""

    member_bills = df[df["bioguide_id"] == bioguide_id]
    if member_bills.empty:
        return ""

    lines = [f"\n{bioguide_id} sponsored/cosponsored bills:"]

    for _, row in member_bills.head(10).iterrows():
        bill_id    = str(row.get("bill_id",    "") or "")
        title      = str(row.get("title",      "") or "")
        role       = str(row.get("role",       "") or "")
        topics     = str(row.get("topics",     "") or "")
        intro_date = str(row.get("introduced_date", "") or "")

        role_tag   = "sponsored" if role == "sponsored" else "cosponsored"
        topics_tag = f" [{topics}]" if topics else ""
        lines.append(f"  • {role_tag} {bill_id}{topics_tag}: {title[:100]} ({intro_date})")

    return "\n".join(lines)
