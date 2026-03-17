# -*- coding: utf-8 -*-
"""
retrieval.py
------------
Semantic retrieval layer: given a free-text query, finds the most relevant
congressional stances and bills using FAISS approximate nearest-neighbour
search over pre-built embeddings.

Also provides context formatting for downstream LLM calls.

Public API
----------
    stance_hits, bill_hits = retrieve(query, model, stances_df, bills_df,
                                       stance_index, bill_index)
    context_str = build_context(stance_hits, bill_hits)
"""

import logging

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import RETRIEVAL_K_STANCES, RETRIEVAL_K_BILLS

logger = logging.getLogger(__name__)

# Maximum characters of stance text / bill summary shown in context
_MAX_STANCE_CHARS = 350
_MAX_SUMMARY_CHARS = 350


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    model: SentenceTransformer,
    stances_df: pd.DataFrame,
    bills_df: pd.DataFrame,
    stance_index: faiss.IndexFlatL2,
    bill_index: faiss.IndexFlatL2,
    k_stances: int = RETRIEVAL_K_STANCES,
    k_bills: int = RETRIEVAL_K_BILLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve the top-k most semantically similar stances and bills for a query.

    Args:
        query:        Free-text user question.
        model:        Loaded SentenceTransformer model.
        stances_df:   DataFrame of member stances.
        bills_df:     DataFrame of bills.
        stance_index: FAISS index over stance embeddings.
        bill_index:   FAISS index over bill embeddings.
        k_stances:    Number of stance hits to return.
        k_bills:      Number of bill hits to return.

    Returns:
        (stance_hits, bill_hits) as DataFrames (rows from the original DFs).
    """
    if not query.strip():
        logger.warning("[RETRIEVAL] Empty query — returning empty results.")
        return stances_df.iloc[:0], bills_df.iloc[:0]

    # Encode query to float32 (FAISS requires float32)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")

    # Clamp k to available rows so FAISS doesn't error on small datasets
    k_s = min(k_stances, len(stances_df))
    k_b = min(k_bills,   len(bills_df))

    if k_s == 0 or k_b == 0:
        logger.warning("[RETRIEVAL] One or both indexes are empty.")
        return stances_df.iloc[:0], bills_df.iloc[:0]

    _, stance_indices = stance_index.search(q_emb, k_s)
    _, bill_indices   = bill_index.search(q_emb, k_b)

    stance_hits = stances_df.iloc[stance_indices[0]]
    bill_hits   = bills_df.iloc[bill_indices[0]]

    return stance_hits, bill_hits


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _safe_str(val, max_chars: int = 0) -> str:
    """
    Safely convert a potentially NaN / None value to a clean string.
    Optionally truncate to max_chars (0 = no limit).
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s


def build_votes_context(bioguide_ids: list, votes_df, lis_map: dict, bills_df=None) -> str:
    """Build a votes context block, only showing votes on healthcare/AI bills with titles."""
    if votes_df is None or votes_df.empty:
        return ""

    # Build bill lookup: normalized_id -> (title, topics)
    bill_lookup = {}
    if bills_df is not None:
        def norm(b): return b.replace(" ","").replace(".","").upper() if isinstance(b,str) else ""
        for _, row in bills_df.iterrows():
            key = norm(str(row.get("bill_id","") or ""))
            bill_lookup[key] = (
                str(row.get("title","") or ""),
                str(row.get("topics","") or "")
            )

    output_lines = ["=== Member Votes on Healthcare/AI Bills ==="]

    for bio in bioguide_ids:
        # House votes (direct bioguide match)
        member_votes = votes_df[votes_df["bioguide_id"] == bio].copy()

        # Senate votes (via LIS ID)
        lis_id = lis_map.get(bio, "")
        if lis_id:
            senate_votes = votes_df[votes_df["bioguide_id"] == lis_id].copy()
            import pandas as pd
            member_votes = pd.concat([member_votes, senate_votes], ignore_index=True)

        if member_votes.empty:
            continue

        # Normalize bill IDs and filter to only healthcare/AI bills
        def norm(b): return b.replace(" ","").replace(".","").upper() if isinstance(b,str) else ""
        member_votes = member_votes.copy()
        member_votes["bill_norm"] = member_votes["bill_id"].apply(norm)

        if bill_lookup:
            member_votes = member_votes[member_votes["bill_norm"].isin(bill_lookup.keys())]

        if member_votes.empty:
            continue

        output_lines.append(f"\n{bio} votes:")
        for _, row in member_votes.head(10).iterrows():
            bill      = _safe_str(row.get("bill_id"))
            vote      = _safe_str(row.get("vote"))
            date      = _safe_str(row.get("date"))
            chamber   = _safe_str(row.get("chamber"))
            bill_norm = row.get("bill_norm","")
            title, topics = bill_lookup.get(bill_norm, ("",""))
            title_str  = f" — {title[:80]}" if title else ""
            topics_str = f" [{topics}]" if topics else ""
            if bill and vote:
                output_lines.append(f"  • {vote} on {bill}{title_str}{topics_str} ({date}) [{chamber}]")

    return "\n".join(output_lines) if len(output_lines) > 1 else ""


def build_context(stance_hits: pd.DataFrame, bill_hits: pd.DataFrame, votes_context: str = "") -> str:
    """
    Format retrieved stances and bills as a plain-text context block
    suitable for inclusion in an LLM prompt.

    Handles missing / NaN fields gracefully so a single bad row won't
    break the whole context.

    Args:
        stance_hits: DataFrame rows of retrieved member stances.
        bill_hits:   DataFrame rows of retrieved bills.

    Returns:
        Multi-line string with labelled sections for stances and bills.
    """
    lines: list[str] = ["=== Member Stances ==="]

    for _, row in stance_hits.iterrows():
        bioguide = _safe_str(row.get("bioguide_id"))
        date     = _safe_str(row.get("date"))
        topic    = _safe_str(row.get("topic"))
        text     = _safe_str(row.get("text"), _MAX_STANCE_CHARS)

        if not text:
            continue

        label = f"{bioguide} ({date})" if bioguide else "(unknown)"
        if topic:
            label += f" [{topic}]"
        lines.append(f"• {label}: {text}")

    lines += ["", "=== Related Bills ==="]

    for _, row in bill_hits.iterrows():
        bill_id = _safe_str(row.get("bill_id")) or "Unknown Bill"
        title   = _safe_str(row.get("title"))   or "No title"
        summary = _safe_str(row.get("summary"),  _MAX_SUMMARY_CHARS)
        topics  = _safe_str(row.get("topics"))

        topic_tag = f" [{topics}]" if topics else ""
        summary_fragment = f" — {summary}" if summary else ""
        lines.append(f"• {bill_id}{topic_tag}: {title}{summary_fragment}")

    if votes_context:
        lines += ["", votes_context]

    return "\n".join(lines)
