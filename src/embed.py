# -*- coding: utf-8 -*-
"""
embed.py
--------
Builds and caches semantic embeddings for stance and bill text, then stores
them as FAISS nearest-neighbour indexes for fast retrieval.

Caching strategy
----------------
Embedding ~500 stances + thousands of bills takes 30-60 s the first time.
Once computed, the numpy arrays and FAISS index binaries are stored in
cache/ so subsequent runs skip the expensive encode step.  The cache is
invalidated automatically whenever the source CSV is newer than the cache.

Public API
----------
    model, stances_df, bills_df, stance_idx, bill_idx = load_indexes()
"""

import logging
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    STANCES_CSV,
    BILLS_CSV,
    STANCE_EMBEDDINGS_NPY,
    BILL_EMBEDDINGS_NPY,
    STANCE_INDEX_BIN,
    BILL_INDEX_BIN,
    EMBED_MODEL_NAME,
    EMBED_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh(cache_path: Path, source_path: Path) -> bool:
    """
    Return True if cache_path exists and is newer than source_path.
    This tells us whether we can skip re-encoding.
    """
    if not cache_path.exists():
        return False
    if not source_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_path.stat().st_mtime


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode a list of strings to float32 embeddings."""
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create a flat L2 FAISS index and add embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Build / load stances
# ---------------------------------------------------------------------------

def _get_stance_embeddings_and_index(
    model: SentenceTransformer,
    stances_df: pd.DataFrame,
) -> tuple[np.ndarray, faiss.IndexFlatL2]:
    """
    Return stance embeddings and a FAISS index, using cache when possible.
    """
    if _cache_is_fresh(STANCE_EMBEDDINGS_NPY, Path(STANCES_CSV)):
        logger.info("[EMBED] Loading stance embeddings from cache.")
        embeddings = np.load(str(STANCE_EMBEDDINGS_NPY))
        index      = faiss.read_index(str(STANCE_INDEX_BIN))
        return embeddings, index

    logger.info("[EMBED] Encoding %d stances…", len(stances_df))
    texts      = stances_df["text"].fillna("").tolist()
    embeddings = _encode(model, texts)

    index = _build_faiss_index(embeddings)

    # Persist
    np.save(str(STANCE_EMBEDDINGS_NPY), embeddings)
    faiss.write_index(index, str(STANCE_INDEX_BIN))
    logger.info("[EMBED] Stance cache written.")

    return embeddings, index


# ---------------------------------------------------------------------------
# Build / load bills
# ---------------------------------------------------------------------------

def _get_bill_embeddings_and_index(
    model: SentenceTransformer,
    bills_df: pd.DataFrame,
) -> tuple[np.ndarray, faiss.IndexFlatL2]:
    """
    Return bill embeddings and a FAISS index, using cache when possible.
    """
    if _cache_is_fresh(BILL_EMBEDDINGS_NPY, Path(BILLS_CSV)):
        logger.info("[EMBED] Loading bill embeddings from cache.")
        embeddings = np.load(str(BILL_EMBEDDINGS_NPY))
        index      = faiss.read_index(str(BILL_INDEX_BIN))
        return embeddings, index

    logger.info("[EMBED] Encoding %d bills…", len(bills_df))
    texts = (
        bills_df["title"].fillna("") + "\n" + bills_df["summary"].fillna("")
    ).tolist()
    embeddings = _encode(model, texts)

    index = _build_faiss_index(embeddings)

    # Persist
    np.save(str(BILL_EMBEDDINGS_NPY), embeddings)
    faiss.write_index(index, str(BILL_INDEX_BIN))
    logger.info("[EMBED] Bill cache written.")

    return embeddings, index


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_indexes() -> tuple[
    SentenceTransformer,
    pd.DataFrame,
    pd.DataFrame,
    faiss.IndexFlatL2,
    faiss.IndexFlatL2,
]:
    """
    Load (or build and cache) embeddings and FAISS indexes for stances and bills.

    Returns:
        (model, stances_df, bills_df, stance_index, bill_index)

    Raises:
        FileNotFoundError: if the stances or bills CSVs do not exist.
    """
    logger.info("[EMBED] Loading data CSVs…")

    stances_df = pd.read_csv(STANCES_CSV)
    bills_df   = pd.read_csv(BILLS_CSV)

    logger.info("[EMBED] Loading sentence-transformer model: %s", EMBED_MODEL_NAME)
    import torch
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    _, stance_index = _get_stance_embeddings_and_index(model, stances_df)
    _, bill_index   = _get_bill_embeddings_and_index(model, bills_df)

    logger.info("[EMBED] Indexes ready.")
    return model, stances_df, bills_df, stance_index, bill_index
