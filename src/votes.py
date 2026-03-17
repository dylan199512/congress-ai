# -*- coding: utf-8 -*-
"""
votes.py
--------
Fetches recorded congressional votes from two free, no-key-required sources:
  - House Clerk:  https://clerk.house.gov/evs/{year}/roll{num}.xml
  - Senate.gov:   https://www.senate.gov/legislative/LIS/roll_call_votes/...

Output: data/votes.csv
Columns: bill_id | bioguide_id | vote | date | chamber | source_url
"""

import csv
import logging
import time
from datetime import datetime
from typing import Optional
import xml.etree.ElementTree as ET

import requests

from src.config import VOTES_CSV, DATA_DIR, HEALTHCARE_KEYWORDS, AI_KEYWORDS

logger = logging.getLogger(__name__)

CUTOFF_DATE = datetime(2026, 3, 16)
HEADERS     = {"User-Agent": "CongressAI/1.0 (research project)"}


def _is_relevant(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in HEALTHCARE_KEYWORDS + AI_KEYWORDS)


# ---------------------------------------------------------------------------
# House Clerk
# ---------------------------------------------------------------------------

def _fetch_house_vote(year: int, roll_num: int) -> Optional[ET.Element]:
    url = f"https://clerk.house.gov/evs/{year}/roll{roll_num:03d}.xml"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        return ET.fromstring(r.text)
    except Exception:
        return None


def _parse_house_vote(root: ET.Element, year: int, roll_num: int) -> list[dict]:
    rows = []
    meta = root.find("vote-metadata")
    if meta is None:
        return []

    legis_num  = (meta.findtext("legis-num")    or "").strip()
    vote_date  = (meta.findtext("action-date")  or "").strip()
    question   = (meta.findtext("vote-question") or "").strip()
    vote_desc  = (meta.findtext("vote-desc")    or "").strip()

    if not _is_relevant(f"{legis_num} {question} {vote_desc}"):
        return []

    try:
        dt = datetime.strptime(vote_date, "%d-%b-%Y")
        if dt > CUTOFF_DATE:
            return []
        date_str = dt.date().isoformat()
    except Exception:
        date_str = f"{year}-01-01"

    source_url = f"https://clerk.house.gov/evs/{year}/roll{roll_num:03d}.xml"
    vote_data  = root.find("vote-data")
    if vote_data is None:
        return []

    for rv in vote_data.findall("recorded-vote"):
        legislator = rv.find("legislator")
        vote_cast  = rv.findtext("vote") or ""
        if legislator is None:
            continue
        bioguide = legislator.get("name-id", "")
        if not bioguide:
            continue
        rows.append({
            "bill_id": legis_num, "bioguide_id": bioguide,
            "vote": vote_cast, "date": date_str,
            "chamber": "House", "source_url": source_url,
        })
    return rows


def fetch_house_votes(years: list) -> list[dict]:
    all_rows = []
    for year in years:
        print(f"[VOTES] House {year}…", flush=True)
        roll_num = 1
        misses   = 0
        while misses < 5:
            root = _fetch_house_vote(year, roll_num)
            if root is None:
                misses += 1
            else:
                misses = 0
                rows = _parse_house_vote(root, year, roll_num)
                all_rows.extend(rows)
                if roll_num % 100 == 0:
                    print(f"  roll {roll_num}, {len(all_rows)} relevant so far", flush=True)
            roll_num += 1
            time.sleep(0.1)
    print(f"[VOTES] House total: {len(all_rows)}", flush=True)
    return all_rows


# ---------------------------------------------------------------------------
# Senate.gov
# ---------------------------------------------------------------------------

def _fetch_senate_vote_list(congress: int, session: int) -> list[dict]:
    url = f"https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_{congress}_{session}.xml"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        root  = ET.fromstring(r.text)
        votes = []
        for vote in root.findall(".//vote"):
            votes.append({
                "number": vote.findtext("vote_number") or "",
                "issue":  vote.findtext("issue")       or "",
                "title":  vote.findtext("title")       or "",
            })
        return votes
    except Exception as exc:
        logger.warning("Senate list error %s/%s: %s", congress, session, exc)
        return []


def _fetch_senate_vote_detail(congress: int, session: int, vote_num: str) -> Optional[ET.Element]:
    num = vote_num.zfill(5)
    url = f"https://www.senate.gov/legislative/LIS/roll_call_votes/vote{congress}{session}/vote_{congress}_{session}_{num}.xml"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        return ET.fromstring(r.text)
    except Exception:
        return None


def _parse_senate_vote(root: ET.Element, congress: int, session: int, vote_num: str) -> list[dict]:
    rows = []

    # Extract fields from root (roll_call_vote element)
    vote_date  = (root.findtext("vote_date")       or "").strip()
    title      = (root.findtext("vote_title")      or "").strip()
    question   = (root.findtext("vote_question_text") or "").strip()
    doc        = root.find("document")
    issue      = (doc.findtext("document_name") or "").strip() if doc is not None else ""

    # No keyword filter — match against bills dataset afterward



    # Date format: "January 9, 2025,  02:54 PM"
    try:
        dt = datetime.strptime(vote_date.split(",  ")[0] + "," + vote_date.split(",")[1], "%B %d, %Y")
        if dt > CUTOFF_DATE:
            return []
        date_str = dt.date().isoformat()
    except Exception:
        try:
            import re
            m = re.search(r"(\w+ \d+, \d{4})", vote_date)
            date_str = datetime.strptime(m.group(1), "%B %d, %Y").date().isoformat() if m else ""
        except Exception:
            date_str = ""

    num     = vote_num.zfill(5)
    src_url = f"https://www.senate.gov/legislative/LIS/roll_call_votes/vote{congress}{session}/vote_{congress}_{session}_{num}.xml"
    members = root.find("members")
    if members is None:
        return []

    for member in members.findall("member"):
        lis_id    = (member.findtext("lis_member_id") or "").strip()
        vote_cast = (member.findtext("vote_cast")     or "").strip()
        if not lis_id:
            continue
        rows.append({
            "bill_id": issue, "bioguide_id": lis_id,
            "vote": vote_cast, "date": date_str,
            "chamber": "Senate", "source_url": src_url,
        })
    return rows


def fetch_senate_votes(congresses: list) -> list[dict]:
    all_rows = []
    for congress in congresses:
        for session in [1, 2]:
            print(f"[VOTES] Senate {congress}/{session}…", flush=True)
            vote_list = _fetch_senate_vote_list(congress, session)
            relevant  = vote_list  # fetch ALL Senate votes, filter by bill match afterward
            print(f"  Fetching all {len(relevant)} Senate votes", flush=True)
            for i, v in enumerate(relevant):
                root = _fetch_senate_vote_detail(congress, session, v["number"])
                if root:
                    all_rows.extend(_parse_senate_vote(root, congress, session, v["number"]))
                if i % 20 == 0:
                    print(f"  {i}/{len(relevant)} processed, {len(all_rows)} votes so far", flush=True)
                time.sleep(0.1)
    print(f"[VOTES] Senate total: {len(all_rows)}", flush=True)
    return all_rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_votes() -> None:
    """Fetch House and Senate votes from free public APIs, save to votes.csv."""
    DATA_DIR.mkdir(exist_ok=True)

    house_rows  = fetch_house_votes(years=[2023, 2024, 2025, 2026])
    senate_rows = fetch_senate_votes(congresses=[118, 119])
    all_rows    = house_rows + senate_rows

    # Deduplicate
    seen, deduped = set(), []
    for r in all_rows:
        key = (r["bill_id"], r["bioguide_id"], r["date"], r["chamber"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"[VOTES] Total unique vote rows: {len(deduped)}", flush=True)

    fieldnames = ["bill_id", "bioguide_id", "vote", "date", "chamber", "source_url"]
    with open(VOTES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"[VOTES] Saved to {VOTES_CSV}", flush=True)
