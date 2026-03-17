# -*- coding: utf-8 -*-
"""
config.py
---------
Central configuration for the Congress AI Healthcare Chatbot.
All API keys, file paths, keyword lists, and tunable constants live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT_DIR   = Path(__file__).resolve().parent.parent   # project root
DATA_DIR   = ROOT_DIR / "data"
CACHE_DIR  = ROOT_DIR / "cache"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------

LEGISLATORS_CSV = DATA_DIR / "legislators-current.csv"
STANCES_CSV     = DATA_DIR / "member_stances.csv"
BILLS_CSV       = DATA_DIR / "bills.csv"
VOTES_CSV       = DATA_DIR / "votes.csv"

# Cached embedding/index artifacts
STANCE_EMBEDDINGS_NPY = CACHE_DIR / "stance_embeddings.npy"
BILL_EMBEDDINGS_NPY   = CACHE_DIR / "bill_embeddings.npy"
STANCE_INDEX_BIN      = CACHE_DIR / "stance_index.faiss"
BILL_INDEX_BIN        = CACHE_DIR / "bill_index.faiss"

# ---------------------------------------------------------------------------
# API keys  (override via environment variables in production)
# ---------------------------------------------------------------------------

CONGRESS_API_KEY  = os.getenv("CONGRESS_API_KEY", "")
GOVINFO_API_KEY   = os.getenv("GOVINFO_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# External API base URLs
# ---------------------------------------------------------------------------

CONGRESS_API_BASE = "https://api.congress.gov/v3"
GOVINFO_BASE      = "https://api.govinfo.gov"

# ---------------------------------------------------------------------------
# Congresses to fetch (list of ints)
# ---------------------------------------------------------------------------

CONGRESSES = [118, 119]

# ---------------------------------------------------------------------------
# Topic keyword lists
# ---------------------------------------------------------------------------

HEALTHCARE_KEYWORDS: list[str] = [
    "health", "healthcare", "health care", "medical", "medicine",
    "public health", "hhs", "human services", "medicare", "medicaid",
    "biotech", "biotechnology", "clinical", "clinical trials",
    "drug safety", "fda", "nih", "telehealth", "digital health",
    "hospital", "doctor", "nurse", "pharmacy", "insurance",
]

AI_KEYWORDS: list[str] = [
    "artificial intelligence",
    "machine learning",
    "large language model",
    "deep learning",
    "neural network",
    "generative ai",
    "ai regulation",
    "ai safety",
    "ai accountability",
    "algorithmic",
    "automated decision",
    "facial recognition",
    "chatbot",
    "foundation model",
]

TOPIC_WHITELIST: set[str] = {
    "healthcare", "ai", "immigration", "economy",
    "education", "environment", "veterans", "defense",
}

# ---------------------------------------------------------------------------
# Scraping / embedding settings
# ---------------------------------------------------------------------------

HTTP_TIMEOUT         = 15       # seconds per request
SCRAPE_SLEEP         = 0.3      # polite delay between member requests (seconds)
SCRAPE_MAX_WORKERS   = 8        # thread-pool size for parallel stance scraping
EMBED_MODEL_NAME     = "all-mpnet-base-v2"
EMBED_BATCH_SIZE     = 32
RETRIEVAL_K_STANCES  = 8
RETRIEVAL_K_BILLS    = 8
CLAUDE_MODEL         = "claude-sonnet-4-5"
CLAUDE_MAX_TOKENS    = 1000

# ---------------------------------------------------------------------------
# HTTP request headers (browser-like to avoid 403s)
# ---------------------------------------------------------------------------

HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
