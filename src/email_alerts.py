# -*- coding: utf-8 -*-
"""
email_alerts.py
---------------
Sends a daily digest email of new healthcare and AI bills introduced
in the last 24 hours.

Usage:
    python -c "from src.email_alerts import send_daily_digest; send_daily_digest()"

Or runs automatically via the daily refresh scheduler.

Config:
    Set GMAIL_USER and GMAIL_APP_PASSWORD as environment variables,
    or they will be read from src/config.py.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Email config — override via environment variables
# ---------------------------------------------------------------------------

GMAIL_USER     = os.getenv("GMAIL_USER",     "")
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "njfj qibs graq cvwv")
RECIPIENT      = os.getenv("ALERT_EMAIL",    "dylan@dylan-reid.com")

# ---------------------------------------------------------------------------
# HTML email template
# ---------------------------------------------------------------------------

def _build_email_html(new_bills: pd.DataFrame, date_str: str) -> str:
    """Build a styled HTML email body."""

    bill_rows = ""
    for _, row in new_bills.iterrows():
        bill_id = str(row.get("bill_id",  "") or "")
        title   = str(row.get("title",    "") or "No title")
        summary = str(row.get("summary",  "") or "")
        topics  = str(row.get("topics",   "") or "")
        sponsor = str(row.get("sponsor_bioguide_id", "") or "")

        if summary.lower() in ("nan", "none", ""):
            summary = ""

        topic_tags = ""
        for t in topics.split(","):
            t = t.strip()
            if t == "healthcare":
                topic_tags += f"<span style='background:#1a3d2b;color:#5fb88a;padding:2px 8px;border-radius:3px;font-size:11px;margin-right:4px'>{t.upper()}</span>"
            elif t == "ai":
                topic_tags += f"<span style='background:#0f2540;color:#4a9eff;padding:2px 8px;border-radius:3px;font-size:11px;margin-right:4px'>{t.upper()}</span>"

        sponsor_line = f"<div style='color:#6b7585;font-size:12px;margin-top:4px'>Sponsor: {sponsor}</div>" if sponsor and sponsor.lower() != "nan" else ""
        summary_line = f"<div style='color:#aaa;font-size:13px;margin-top:6px'>{summary[:200]}{'…' if len(summary)>200 else ''}</div>" if summary else ""

        bill_rows += f"""
        <div style='background:#14181e;border:1px solid #252b33;border-radius:6px;padding:16px;margin-bottom:12px'>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>
                <span style='font-family:monospace;color:#c8a96e;font-size:13px;font-weight:bold'>{bill_id}</span>
                {topic_tags}
            </div>
            <div style='color:#e8e4dc;font-size:15px;font-weight:500'>{title}</div>
            {summary_line}
            {sponsor_line}
        </div>"""

    if not bill_rows:
        bill_rows = "<p style='color:#6b7585'>No new bills introduced today.</p>"

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset='utf-8'></head>
<body style='background:#0d0f12;color:#e8e4dc;font-family:Georgia,serif;max-width:600px;margin:0 auto;padding:24px'>

  <div style='border-bottom:2px solid #c8a96e;padding-bottom:16px;margin-bottom:24px'>
    <h1 style='color:#c8a96e;font-size:24px;margin:0'>🏛️ Congress AI</h1>
    <p style='color:#6b7585;font-size:13px;font-family:monospace;margin:4px 0 0'>
      Daily Healthcare & AI Legislation Alert · {date_str}
    </p>
  </div>

  <h2 style='color:#e8e4dc;font-size:18px;margin-bottom:16px'>
    {len(new_bills)} New Bills Today
  </h2>

  {bill_rows}

  <div style='border-top:1px solid #252b33;margin-top:24px;padding-top:16px'>
    <p style='color:#6b7585;font-size:12px;font-family:monospace'>
      Data: Congress.gov · GovInfo · legislators-current.csv<br>
      To unsubscribe, remove your email from src/email_alerts.py
    </p>
  </div>

</body>
</html>"""


def _get_new_bills(days_back: int = 1) -> pd.DataFrame:
    """
    Return bills introduced in the last `days_back` days.
    Uses the member_profiles.csv introduced_date field.
    """
    from src.config import DATA_DIR

    profiles_path = DATA_DIR / "member_profiles.csv"
    bills_path    = DATA_DIR / "bills.csv"

    try:
        profiles = pd.read_csv(profiles_path)
        bills    = pd.read_csv(bills_path)
    except FileNotFoundError as e:
        logger.error("Could not load data for email alert: %s", e)
        return pd.DataFrame()

    # Filter to recent bills
    cutoff = (datetime.today() - timedelta(days=days_back)).date().isoformat()
    recent = profiles[profiles["introduced_date"] >= cutoff].copy()

    if recent.empty:
        return pd.DataFrame()

    # Merge with bills for full details
    bills["bill_id_norm"] = bills["bill_id"].str.replace(" ", "").str.upper()
    recent["bill_id_norm"] = recent["bill_id"].str.replace(" ", "").str.upper()

    merged = recent.merge(
        bills[["bill_id_norm", "summary", "sponsor_bioguide_id"]],
        on="bill_id_norm",
        how="left"
    )

    # Deduplicate
    merged = merged.drop_duplicates(subset=["bill_id"])

    # Sort by topic and date
    merged = merged.sort_values(["topics", "introduced_date"], ascending=[True, False])

    return merged[["bill_id", "title", "summary", "topics", "sponsor_bioguide_id", "introduced_date"]]


def send_daily_digest(
    gmail_user: str = GMAIL_USER,
    gmail_password: str = GMAIL_PASSWORD,
    recipient: str = RECIPIENT,
    days_back: int = 1,
) -> bool:
    """
    Send a daily digest email of new healthcare/AI bills.

    Args:
        gmail_user:      Gmail address to send from.
        gmail_password:  Gmail app password.
        recipient:       Email address to send to.
        days_back:       How many days back to look for new bills.

    Returns:
        True if email sent successfully, False otherwise.
    """
    import yagmail

    new_bills = _get_new_bills(days_back=days_back)
    date_str  = datetime.today().strftime("%B %d, %Y")

    logger.info("[EMAIL] Found %d new bills to report", len(new_bills))

    subject = f"Congress AI · {len(new_bills)} New Healthcare/AI Bills · {date_str}"
    html    = _build_email_html(new_bills, date_str)

    try:
        yag = yagmail.SMTP(gmail_user, gmail_password)
        yag.send(to=recipient, subject=subject, contents=html)
        logger.info("[EMAIL] Digest sent to %s", recipient)
        print(f"[EMAIL] Sent to {recipient} — {len(new_bills)} bills", flush=True)
        return True
    except Exception as exc:
        logger.error("[EMAIL] Failed to send: %s", exc)
        print(f"[EMAIL] Error: {exc}", flush=True)
        return False
