# -*- coding: utf-8 -*-
"""
qa.py
-----
Orchestrates the end-to-end question-answering pipeline:

    User question
        → retrieve relevant stances + bills  (retrieval.py)
        → format context block
        → call Claude claude-3-sonnet to synthesize an answer
        → return answer text + source evidence

Public API
----------
    client = make_claude_client()
    answer, stance_hits, bill_hits = answer_question(query, ..., client)
"""

import logging
from dataclasses import dataclass

import faiss
import pandas as pd
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

from src.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
)
from src.retrieval import retrieve, build_context, build_votes_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a nonpartisan congressional research assistant. "
    "Answer questions about US healthcare and AI legislation using ONLY "
    "the context provided. Cite specific members or bills where relevant. "
    "If the context does not contain enough information to answer confidently, "
    "say so rather than guessing."
)

_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer (be specific and cite sources from the context):"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    """
    Holds the full result of a single Q&A turn.

    Attributes:
        answer:       Claude's synthesised answer string.
        stance_hits:  DataFrame of retrieved member stance rows.
        bill_hits:    DataFrame of retrieved bill rows.
        context:      The formatted context string sent to Claude.
    """
    answer:      str
    stance_hits: pd.DataFrame
    bill_hits:   pd.DataFrame
    context:     str


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_claude_client() -> Anthropic:
    """
    Create and return an Anthropic client using the configured API key.

    The key is sourced from config.py, which itself reads the
    ANTHROPIC_API_KEY environment variable (with a hard-coded fallback).
    """
    return Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Core Q&A function
# ---------------------------------------------------------------------------

def answer_question(
    query: str,
    model: SentenceTransformer,
    stances_df: pd.DataFrame,
    bills_df: pd.DataFrame,
    stance_index: faiss.IndexFlatL2,
    bill_index: faiss.IndexFlatL2,
    client: Anthropic,
) -> QAResult:
    """
    Run the full retrieval-augmented generation (RAG) pipeline for one question.

    Steps:
      1. Encode the query and retrieve top-k stances + bills.
      2. Format a context block from the retrieved rows.
      3. Call Claude with a grounded prompt that forbids hallucination.
      4. Return the answer + evidence for UI display.

    Args:
        query:        The user's free-text question.
        model:        Loaded SentenceTransformer model.
        stances_df:   DataFrame of member stances.
        bills_df:     DataFrame of bills.
        stance_index: FAISS index over stance embeddings.
        bill_index:   FAISS index over bill embeddings.
        client:       Anthropic API client.

    Returns:
        QAResult containing the answer text and source evidence.
    """
    if not query.strip():
        return QAResult(
            answer="Please enter a question.",
            stance_hits=stances_df.iloc[:0],
            bill_hits=bills_df.iloc[:0],
            context="",
        )

    # --- Retrieval ---
    stance_hits, bill_hits = retrieve(
        query, model, stances_df, bills_df, stance_index, bill_index
    )

    # --- Votes context ---
    votes_context = ""
    try:
        import pandas as pd
        from src.config import VOTES_CSV, LEGISLATORS_CSV
        votes_df = pd.read_csv(VOTES_CSV)
        leg_df   = pd.read_csv(LEGISLATORS_CSV)

        # Build LIS ID map: bioguide -> lis_id
        lis_map = dict(zip(
            leg_df["bioguide_id"].fillna(""),
            leg_df["lis_id"].fillna("") if "lis_id" in leg_df.columns else [""] * len(leg_df)
        ))

        # 1. Get bioguide IDs from retrieved stances
        bio_ids = set(stance_hits["bioguide_id"].dropna().unique().tolist()) if not stance_hits.empty else set()

        # 2. Also search query for member names and look them up directly
        query_lower = query.lower()
        for _, row in leg_df.iterrows():
            full_name = str(row.get("full_name", "") or "").lower()
            last_name = str(row.get("last_name", "") or "").lower()
            if last_name and last_name in query_lower:
                bio_ids.add(str(row.get("bioguide_id", "") or ""))
            elif full_name and full_name in query_lower:
                bio_ids.add(str(row.get("bioguide_id", "") or ""))

        bio_ids = [b for b in bio_ids if b]
        if bio_ids:
            votes_context = build_votes_context(bio_ids, votes_df, lis_map, bills_df)
    except Exception as e:
        print(f"Votes context error: {e}")

    # --- Member profile context ---
    profile_context = ""
    try:
        from src.member_profiles import get_member_profile_context
        from src.config import LEGISLATORS_CSV
        import pandas as pd
        leg_df = pd.read_csv(LEGISLATORS_CSV)
        query_lower = query.lower()
        for _, row in leg_df.iterrows():
            last_name = str(row.get("last_name","") or "").lower()
            if last_name and last_name in query_lower:
                bio = str(row.get("bioguide_id","") or "")
                if bio:
                    profile_context += get_member_profile_context(bio)
    except Exception as e:
        pass

    # --- Context ---
    context = build_context(stance_hits, bill_hits, votes_context)
    if profile_context:
        context += "\n\n=== Sponsored/Cosponsored Legislation ===\n" + profile_context

    prompt = _USER_TEMPLATE.format(context=context, question=query)

    # --- Claude call ---
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text
    except Exception as exc:
        logger.error("[QA] Claude API error: %s", exc)
        answer = f"Error contacting Claude: {exc}"

    return QAResult(
        answer=answer,
        stance_hits=stance_hits,
        bill_hits=bill_hits,
        context=context,
    )
