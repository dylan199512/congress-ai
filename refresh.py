#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refresh.py
----------
Runs the full data pipeline to refresh bills, stances, and embeddings.
Designed to be run daily via launchd (Mac) or cron.

Usage:
    python3.11 refresh.py

Logs to: logs/refresh.log
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]    = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Logging to both console and file
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "refresh.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def run_step(name: str, fn):
    """Run a pipeline step, logging success or failure."""
    logger.info("=" * 50)
    logger.info(f"Starting: {name}")
    logger.info("=" * 50)
    try:
        fn()
        logger.info(f"✓ Completed: {name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {name} — {e}")
        return False


if __name__ == "__main__":
    start = datetime.now()
    logger.info(f"Congress AI refresh started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Step 1: Bills
    from src.bills import build_bills
    results["bills"] = run_step("Fetch bills (Congress.gov)", build_bills)

    # Step 2: Stances
    from src.stances import build_member_stances
    results["stances"] = run_step("Scrape member stances", build_member_stances)

    # Step 3: Votes (skip if GovInfo is down)
    try:
        import requests
        r = requests.get(
            "https://api.govinfo.gov/collections/RCV",
            params={"api_key": os.getenv("GOVINFO_API_KEY", "")},
            timeout=10
        )
        if r.status_code == 200:
            from src.votes import build_votes
            results["votes"] = run_step("Fetch votes (GovInfo)", build_votes)
        else:
            logger.warning("GovInfo API unavailable (status %s) — skipping votes", r.status_code)
            results["votes"] = None
    except Exception as e:
        logger.warning("GovInfo check failed: %s — skipping votes", e)
        results["votes"] = None

    # Step 4: Rebuild embeddings
    import torch
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built     = lambda: False

    # Delete cache to force rebuild
    from src.config import STANCE_EMBEDDINGS_NPY, BILL_EMBEDDINGS_NPY, STANCE_INDEX_BIN, BILL_INDEX_BIN
    for cache_file in [STANCE_EMBEDDINGS_NPY, BILL_EMBEDDINGS_NPY, STANCE_INDEX_BIN, BILL_INDEX_BIN]:
        if Path(cache_file).exists():
            Path(cache_file).unlink()
            logger.info("Deleted cache: %s", cache_file)

    from src.embed import load_indexes
    results["embeddings"] = run_step("Rebuild embeddings", load_indexes)

    # Step 5: Send email digest
    try:
        from src.email_alerts import send_daily_digest
        send_daily_digest()
        logger.info("✓ Email digest sent")
    except Exception as e:
        logger.warning("Email digest failed: %s", e)

    # Summary
    elapsed = (datetime.now() - start).seconds // 60
    logger.info("=" * 50)
    logger.info(f"Refresh complete in {elapsed} minutes")
    for step, ok in results.items():
        status = "✓" if ok else ("skipped" if ok is None else "✗")
        logger.info(f"  {status} {step}")
    logger.info("=" * 50)
