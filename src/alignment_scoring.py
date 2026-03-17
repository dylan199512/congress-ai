# -*- coding: utf-8 -*-
"""
alignment_scoring.py
--------------------
Computes alignment scores for congressional members by comparing:
  1. Bills they SPONSORED (strong positive signal)
  2. How they VOTED on healthcare/AI bills (Yea=support, Nay=oppose)
  3. What they SAID in stances (keyword sentiment)

Produces a per-member, per-topic alignment score from -1.0 (strong opposition)
to +1.0 (strong support), plus a "says vs does" hypocrisy flag when votes
contradict stances.

Output: data/alignment_scores.csv
Columns: bioguide_id | topic | sponsor_score | vote_score | stance_score | 
         alignment_score | hypocrisy_flag | evidence_count
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import DATA_DIR, LEGISLATORS_CSV

logger = logging.getLogger(__name__)

ALIGNMENT_CSV    = DATA_DIR / "alignment_scores.csv"
PROFILES_CSV     = DATA_DIR / "member_profiles.csv"
VOTES_CSV        = DATA_DIR / "votes.csv"
STANCES_CSV      = DATA_DIR / "member_stances.csv"
BILLS_CSV        = DATA_DIR / "bills.csv"

TOPICS = ["healthcare", "ai"]

# Vote cast values → numeric score
VOTE_SCORE_MAP = {
    "yea": 1.0, "aye": 1.0,
    "nay": -1.0, "no": -1.0,
    "not voting": 0.0, "present": 0.0,
}


def _load_data() -> tuple:
    """Load all required dataframes."""
    try:
        profiles = pd.read_csv(PROFILES_CSV)
    except FileNotFoundError:
        profiles = pd.DataFrame()

    try:
        votes = pd.read_csv(VOTES_CSV)
    except FileNotFoundError:
        votes = pd.DataFrame()

    try:
        stances = pd.read_csv(STANCES_CSV)
    except FileNotFoundError:
        stances = pd.DataFrame()

    try:
        bills = pd.read_csv(BILLS_CSV)
    except FileNotFoundError:
        bills = pd.DataFrame()

    try:
        legislators = pd.read_csv(LEGISLATORS_CSV)
    except FileNotFoundError:
        legislators = pd.DataFrame()

    return profiles, votes, stances, bills, legislators


def _normalize_bill_id(bid: str) -> str:
    """Normalize bill ID for matching across datasets."""
    if not isinstance(bid, str):
        return ""
    return bid.replace(" ", "").replace(".", "").upper()


def _sponsor_score(bio: str, topic: str, profiles: pd.DataFrame) -> tuple[float, int]:
    """
    Score based on sponsored/cosponsored legislation.
    Sponsoring = +1.0, cosponsoring = +0.5 per bill (normalized).
    Returns (score, count).
    """
    if profiles.empty:
        return 0.0, 0

    member_profiles = profiles[
        (profiles["bioguide_id"] == bio) &
        (profiles["topics"].str.contains(topic, na=False))
    ]

    if member_profiles.empty:
        return 0.0, 0

    score = 0.0
    for _, row in member_profiles.iterrows():
        role = str(row.get("role", "") or "")
        if role == "sponsored":
            score += 1.0
        elif role == "cosponsored":
            score += 0.5

    # Normalize to 0-1 range (cap at 20 bills)
    count     = len(member_profiles)
    norm_score = min(score / 20.0, 1.0)

    return norm_score, count


def _vote_score(bio: str, topic: str, votes: pd.DataFrame, bills: pd.DataFrame,
                legislators: pd.DataFrame) -> tuple[float, int]:
    """
    Score based on voting record on healthcare/AI bills.
    Returns average vote score (-1 to +1) and vote count.
    """
    if votes.empty or bills.empty:
        return 0.0, 0

    # Build bill lookup: normalized_id -> topics
    bills = bills.copy()
    bills["norm_id"] = bills["bill_id"].apply(_normalize_bill_id)
    topic_bills = set(
        bills[bills["topics"].str.contains(topic, na=False)]["norm_id"].tolist()
    )

    if not topic_bills:
        return 0.0, 0

    # Get LIS ID for Senate votes
    lis_id = ""
    if not legislators.empty and "lis_id" in legislators.columns:
        match = legislators[legislators["bioguide_id"] == bio]
        if not match.empty:
            lis_id = str(match.iloc[0].get("lis_id", "") or "")

    # Get member votes
    member_votes = votes[votes["bioguide_id"] == bio].copy()
    if lis_id:
        senate_votes = votes[votes["bioguide_id"] == lis_id].copy()
        member_votes = pd.concat([member_votes, senate_votes], ignore_index=True)

    if member_votes.empty:
        return 0.0, 0

    # Filter to topic-relevant bills
    member_votes["norm_id"] = member_votes["bill_id"].apply(_normalize_bill_id)
    relevant_votes = member_votes[member_votes["norm_id"].isin(topic_bills)]

    if relevant_votes.empty:
        return 0.0, 0

    scores = []
    for _, row in relevant_votes.iterrows():
        vote_cast = str(row.get("vote", "") or "").lower().strip()
        score     = VOTE_SCORE_MAP.get(vote_cast, 0.0)
        scores.append(score)

    if not scores:
        return 0.0, 0

    return float(np.mean(scores)), len(scores)


def _stance_score(bio: str, topic: str, stances: pd.DataFrame) -> tuple[float, int]:
    """
    Score based on stance text sentiment (simple keyword approach).
    Returns score (-1 to +1) and stance count.
    """
    if stances.empty:
        return 0.0, 0

    member_stances = stances[
        (stances["bioguide_id"] == bio) &
        (stances["topic"] == topic)
    ]

    if member_stances.empty:
        return 0.0, 0

    SUPPORT_WORDS = [
        "support", "champion", "fight for", "protect", "expand", "strengthen",
        "invest", "improve", "advance", "promote", "ensure access", "affordable",
    ]
    OPPOSE_WORDS = [
        "oppose", "against", "repeal", "cut", "reduce", "block", "reject",
        "eliminate", "end", "defund", "halt", "stop",
    ]

    scores = []
    for _, row in member_stances.iterrows():
        text = str(row.get("text", "") or "").lower()
        support_count = sum(1 for w in SUPPORT_WORDS if w in text)
        oppose_count  = sum(1 for w in OPPOSE_WORDS  if w in text)

        if support_count + oppose_count == 0:
            scores.append(0.0)
        else:
            scores.append((support_count - oppose_count) / (support_count + oppose_count))

    if not scores:
        return 0.0, 0

    return float(np.mean(scores)), len(scores)


def _detect_hypocrisy(vote_score: float, stance_score: float,
                       vote_count: int, stance_count: int) -> bool:
    """
    Flag hypocrisy when vote score and stance score strongly disagree.
    Requires at least 2 data points on each side.
    """
    if vote_count < 2 or stance_count < 2:
        return False
    # Hypocrisy = one is clearly positive, other is clearly negative
    return (vote_score > 0.3 and stance_score < -0.3) or \
           (vote_score < -0.3 and stance_score > 0.3)


def compute_alignment_scores() -> pd.DataFrame:
    """
    Compute alignment scores for all members on all topics.

    Returns a DataFrame with one row per (member, topic).
    Also saves to data/alignment_scores.csv.
    """
    profiles, votes, stances, bills, legislators = _load_data()

    # Get all unique bioguide IDs across datasets
    all_bios: set[str] = set()
    for df, col in [(profiles, "bioguide_id"), (votes, "bioguide_id"),
                    (stances, "bioguide_id"), (legislators, "bioguide_id")]:
        if not df.empty and col in df.columns:
            all_bios.update(df[col].dropna().astype(str).tolist())

    # Remove LIS IDs (start with S + digits)
    import re
    all_bios = {b for b in all_bios if not re.match(r'^S\d+$', b)}

    rows = []
    total = len(all_bios)

    for i, bio in enumerate(sorted(all_bios)):
        if i % 100 == 0:
            print(f"[ALIGN] {i}/{total} members processed", flush=True)

        for topic in TOPICS:
            sp_score, sp_count = _sponsor_score(bio, topic, profiles)
            vt_score, vt_count = _vote_score(bio, topic, votes, bills, legislators)
            st_score, st_count = _stance_score(bio, topic, stances)

            # Weighted alignment score
            weights      = []
            score_parts  = []

            if sp_count > 0:
                weights.append(3.0)    # sponsorship is strongest signal
                score_parts.append(sp_score * 3.0)
            if vt_count > 0:
                weights.append(2.0)    # votes are strong
                score_parts.append(vt_score * 2.0)
            if st_count > 0:
                weights.append(1.0)    # stances weakest
                score_parts.append(st_score * 1.0)

            if not weights:
                continue   # no data at all for this member+topic

            alignment = sum(score_parts) / sum(weights)
            hypocrisy = _detect_hypocrisy(vt_score, st_score, vt_count, st_count)

            rows.append({
                "bioguide_id":    bio,
                "topic":          topic,
                "sponsor_score":  round(sp_score, 3),
                "vote_score":     round(vt_score, 3),
                "stance_score":   round(st_score, 3),
                "alignment_score": round(alignment, 3),
                "hypocrisy_flag": hypocrisy,
                "sponsor_count":  sp_count,
                "vote_count":     vt_count,
                "stance_count":   st_count,
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df.to_csv(ALIGNMENT_CSV, index=False)
        print(f"[ALIGN] Saved {len(df)} rows to {ALIGNMENT_CSV}", flush=True)

    return df


def get_top_supporters(topic: str, n: int = 10) -> pd.DataFrame:
    """Return top N members supporting a topic."""
    try:
        df = pd.read_csv(ALIGNMENT_CSV)
    except FileNotFoundError:
        return pd.DataFrame()
    return (df[df["topic"] == topic]
            .sort_values("alignment_score", ascending=False)
            .head(n))


def get_top_opponents(topic: str, n: int = 10) -> pd.DataFrame:
    """Return top N members opposing a topic."""
    try:
        df = pd.read_csv(ALIGNMENT_CSV)
    except FileNotFoundError:
        return pd.DataFrame()
    return (df[df["topic"] == topic]
            .sort_values("alignment_score", ascending=True)
            .head(n))


def get_hypocrisy_flags(topic: str) -> pd.DataFrame:
    """Return members whose votes contradict their stated positions."""
    try:
        df = pd.read_csv(ALIGNMENT_CSV)
    except FileNotFoundError:
        return pd.DataFrame()
    return df[(df["topic"] == topic) & (df["hypocrisy_flag"] == True)]
