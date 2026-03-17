# -*- coding: utf-8 -*-
"""
app.py — Congress AI Streamlit UI
Tabs: Q&A | Member Lookup | Browse Bills

Run with:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    python3.11 -m streamlit run app.py
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
from src.config import STANCES_CSV, BILLS_CSV, LEGISLATORS_CSV

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

st.set_page_config(
    page_title="Congress AI · Healthcare & AI Legislation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
:root {
    --bg:#0d0f12; --surface:#14181e; --border:#252b33;
    --accent:#c8a96e; --accent2:#4a9eff; --text:#e8e4dc;
    --muted:#6b7585; --green:#5fb88a;
}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:'IBM Plex Sans',sans-serif!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
h1{font-family:'Playfair Display',serif!important;color:var(--accent)!important;}
h2,h3{font-family:'IBM Plex Sans',sans-serif!important;color:var(--text)!important;}
.stTextInput>div>div>input,.stTextArea textarea{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:4px!important;}
.stButton>button{background:var(--accent)!important;color:#0d0f12!important;font-weight:500!important;border:none!important;border-radius:4px!important;padding:0.5rem 1.5rem!important;letter-spacing:0.05em!important;text-transform:uppercase!important;}
.stButton>button:hover{background:#e0bf86!important;}
.answer-box{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:4px;padding:1.2rem 1.4rem;font-size:0.95rem;line-height:1.7;color:var(--text);margin-bottom:1.5rem;}
.evidence-card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;font-size:0.82rem;}
.meta{font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:var(--muted);margin-bottom:0.3rem;}
.topic-badge{display:inline-block;padding:1px 8px;border-radius:2px;font-size:0.68rem;font-weight:500;text-transform:uppercase;letter-spacing:0.08em;margin-right:6px;}
.topic-healthcare{background:#1a3d2b;color:var(--green);}
.topic-ai{background:#0f2540;color:var(--accent2);}
.bill-card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;}
.bill-id{font-family:'IBM Plex Mono',monospace;color:var(--accent);font-size:0.8rem;font-weight:500;}
.bill-title{font-size:0.88rem;color:var(--text);margin:0.25rem 0;}
.bill-summary{font-size:0.78rem;color:var(--muted);line-height:1.5;}
.member-card{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:4px;padding:1rem 1.2rem;margin-bottom:1rem;}
.context-box{background:#0a0c0f;border:1px solid var(--border);border-radius:4px;padding:1rem;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:var(--muted);line-height:1.6;max-height:300px;overflow-y:auto;white-space:pre-wrap;}
.divider{border:none;border-top:1px solid var(--border);margin:1.5rem 0;}
.stat-pill{display:inline-block;background:var(--surface);border:1px solid var(--border);border-radius:3px;padding:3px 10px;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:var(--muted);margin-right:6px;}
.subheadline{font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:2rem;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🏛️ Congress AI")
    st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#6b7585;letter-spacing:0.1em'>HEALTHCARE & AI LEGISLATION</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
**About**
Retrieval-augmented Q&A over:
- Congressional member **stances**
- **Bills** from Congress.gov (118th & 119th)
- **Votes** from GovInfo
    """)
    st.markdown("---")
    st.markdown("**Example questions**")
    for q in ["Which senators support AI regulation?","What healthcare bills passed in the 118th Congress?","Are there bills about AI in medical devices?","What is Congress doing about FDA drug approvals?"]:
        if st.button(q, key=f"ex_{q[:20]}"):
            st.session_state["prefill_query"] = q
    st.markdown("---")
    st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#6b7585'>Data: Congress.gov · GovInfo · legislators-current.csv</div>", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:0.3rem">
  <h1 style="margin:0">Congress AI</h1>
</div>
<div class="subheadline">Healthcare & Artificial Intelligence Legislation · 118th–119th Congress</div>
""", unsafe_allow_html=True)

# Session state
for key, default in [("history",[]),("prefill_query","")]:
    if key not in st.session_state:
        st.session_state[key] = default

# Loaders
@st.cache_resource(show_spinner="Loading embeddings and indexes…")
def _load_indexes():
    from src.embed import load_indexes
    return load_indexes()

@st.cache_resource(show_spinner="Connecting to Claude…")
def _load_client():
    from src.qa import make_claude_client
    return make_claude_client()

@st.cache_data(show_spinner=False)
def _load_legislators():
    try:
        df = pd.read_csv(LEGISLATORS_CSV)
        for col in ("bioguide_id","full_name","type","state","party"):
            if col not in df.columns: df[col] = ""
        return df
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_stances():
    try: return pd.read_csv(STANCES_CSV)
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_bills():
    try: return pd.read_csv(BILLS_CSV)
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def _dataset_stats():
    s = {}
    for k,f in [("stances",STANCES_CSV),("bills",BILLS_CSV),("members",LEGISLATORS_CSV)]:
        try: s[k] = len(pd.read_csv(f))
        except: s[k] = "—"
    return s

def get_resources():
    try:
        model, stances_df, bills_df, stance_idx, bill_idx = _load_indexes()
        client = _load_client()
        return model, stances_df, bills_df, stance_idx, bill_idx, client
    except FileNotFoundError as e:
        st.error(f"**Data files not found.**\n\n{e}\n\nRun the pipeline first.")
        return None

def _topic_badge(topic):
    cls = f"topic-{topic.lower()}" if topic.lower() in ("healthcare","ai") else ""
    return f"<span class='topic-badge {cls}'>{topic}</span>"

def _bill_card_html(row):
    bill_id = str(row.get("bill_id","") or "")
    title   = str(row.get("title","")   or "No title")
    summary = str(row.get("summary","") or "")
    topics  = str(row.get("topics","")  or "")
    sponsor = str(row.get("sponsor_bioguide_id","") or "")
    if summary.lower() in ("nan","none",""): summary = title
    badges = " ".join(_topic_badge(t.strip()) for t in topics.split(",") if t.strip())
    sp_tag = f"<span style='color:#6b7585;font-size:0.7rem'>Sponsor: {sponsor}</span>" if sponsor and sponsor.lower() not in ("nan","") else ""
    return f"<div class='bill-card'><div class='bill-id'>{bill_id} {badges} {sp_tag}</div><div class='bill-title'>{title}</div><div class='bill-summary'>{summary[:300]}{'…' if len(summary)>300 else ''}</div></div>"

# Stats bar
stats = _dataset_stats()
st.markdown(
    f"<span class='stat-pill'>📄 {stats['stances']} stances</span>"
    f"<span class='stat-pill'>📋 {stats['bills']} bills</span>"
    f"<span class='stat-pill'>👤 {stats['members']} members</span>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Tabs
tab_qa, tab_members, tab_bills = st.tabs(["💬 Q&A", "👤 Member Lookup", "📋 Browse Bills"])

# ── TAB 1: Q&A ──────────────────────────────────────────────────────────────
with tab_qa:
    prefill = st.session_state.pop("prefill_query","")
    query = st.text_input("Ask about Congress", value=prefill,
                          placeholder="e.g. Which senators support AI regulation?",
                          label_visibility="collapsed")
    col_ask, col_clear = st.columns([1,5])
    with col_ask:
        ask_clicked = st.button("Ask →")
    with col_clear:
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

    if ask_clicked and query.strip():
        resources = get_resources()
        if resources:
            model, stances_df, bills_df, stance_idx, bill_idx, client = resources
            from src.qa import answer_question
            with st.spinner("Retrieving context and consulting Claude…"):
                result = answer_question(query, model, stances_df, bills_df, stance_idx, bill_idx, client)
            st.session_state["history"].insert(0, (query, result))

    if not st.session_state["history"]:
        st.markdown("<div style='color:#6b7585;font-family:IBM Plex Mono,monospace;font-size:0.82rem;margin-top:2rem;text-align:center'>Enter a question above to begin.</div>", unsafe_allow_html=True)
    else:
        for q, result in st.session_state["history"]:
            st.markdown(f"#### Q: {q}")
            st.markdown(f"<div class='answer-box'>{result.answer}</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📣 Member Stances Retrieved**")
                if result.stance_hits.empty:
                    st.markdown("<div style='color:#6b7585;font-size:0.82rem'>No stances matched.</div>", unsafe_allow_html=True)
                else:
                    for _, row in result.stance_hits.iterrows():
                        bio=str(row.get("bioguide_id","") or ""); date=str(row.get("date","") or "")
                        topic=str(row.get("topic","") or ""); text=str(row.get("text","") or "")
                        url=str(row.get("source_url","") or "")
                        badge=_topic_badge(topic) if topic else ""
                        url_tag=f"<a href='{url}' target='_blank' style='color:#4a9eff;font-size:0.7rem'>source ↗</a>" if url.startswith("http") else ""
                        st.markdown(f"<div class='evidence-card'><div class='meta'>{badge} {bio} · {date} {url_tag}</div><div>{text[:300]}{'…' if len(text)>300 else ''}</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown("**📋 Related Bills Retrieved**")
                if result.bill_hits.empty:
                    st.markdown("<div style='color:#6b7585;font-size:0.82rem'>No bills matched.</div>", unsafe_allow_html=True)
                else:
                    for _, row in result.bill_hits.iterrows():
                        st.markdown(_bill_card_html(row), unsafe_allow_html=True)
            with st.expander("🔍 Raw context sent to Claude"):
                st.markdown(f"<div class='context-box'>{result.context}</div>", unsafe_allow_html=True)

            # PDF export button
            try:
                from src.pdf_export import generate_qa_pdf
                pdf_bytes = generate_qa_pdf(q, result)
                safe_q = q[:40].replace(" ", "_").replace("/", "-")
                st.download_button(
                    label="📄 Download as PDF",
                    data=pdf_bytes,
                    file_name=f"congress_ai_{safe_q}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{hash(q)}",
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── TAB 2: MEMBER LOOKUP ────────────────────────────────────────────────────
with tab_members:
    st.markdown("### Member Lookup")
    st.markdown("Search for a legislator and see their stances and sponsored bills.")

    legislators_df = _load_legislators()
    stances_df     = _load_stances()
    bills_df_raw   = _load_bills()

    col_s, col_st, col_ch, col_tp, col_yr = st.columns([3,1,1,1,1])
    with col_s:
        name_search = st.text_input("Name", placeholder="e.g. Sanders, Warren, Pelosi…", label_visibility="collapsed")
    with col_st:
        states = ["All states"] + sorted(legislators_df["state"].dropna().unique().tolist()) if not legislators_df.empty else ["All states"]
        state_filter = st.selectbox("State", states, label_visibility="collapsed")
    with col_ch:
        type_filter = st.selectbox("Chamber", ["All","sen","rep"], label_visibility="collapsed")
    with col_tp:
        topic_filter = st.selectbox("Topic", ["All","healthcare","ai"], label_visibility="collapsed")
    with col_yr:
        year_filter = st.selectbox("Year", ["All","2026","2025","2024","2023"], label_visibility="collapsed")

    filtered = legislators_df.copy() if not legislators_df.empty else pd.DataFrame()
    if not filtered.empty:
        if name_search.strip():
            filtered = filtered[filtered["full_name"].str.contains(name_search, case=False, na=False)]
        if state_filter != "All states":
            filtered = filtered[filtered["state"] == state_filter]
        if type_filter != "All":
            filtered = filtered[filtered["type"] == type_filter]

    if filtered.empty:
        st.markdown("<div style='color:#6b7585;font-size:0.82rem;margin-top:1rem'>No members found.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:#6b7585;font-family:IBM Plex Mono,monospace;font-size:0.75rem;margin-bottom:1rem'>{len(filtered)} members found — showing first 10</div>", unsafe_allow_html=True)
        for _, member in filtered.head(10).iterrows():
            bio   = str(member.get("bioguide_id","") or "")
            name  = str(member.get("full_name","")   or "Unknown")
            mtype = str(member.get("type","")        or "")
            state = str(member.get("state","")       or "")
            party = str(member.get("party","")       or "")
            chamber = "Senator" if mtype=="sen" else "Representative"

            m_stances = stances_df[stances_df["bioguide_id"]==bio].copy() if not stances_df.empty else pd.DataFrame()
            m_bills   = bills_df_raw[bills_df_raw["sponsor_bioguide_id"]==bio].copy() if not bills_df_raw.empty else pd.DataFrame()

            if topic_filter != "All":
                if not m_stances.empty: m_stances = m_stances[m_stances["topic"]==topic_filter]
                if not m_bills.empty:   m_bills   = m_bills[m_bills["topics"].str.contains(topic_filter, na=False)]
            if year_filter != "All" and not m_stances.empty:
                m_stances = m_stances[m_stances["date"].astype(str).str.startswith(year_filter)]

            with st.expander(f"**{name}** · {chamber} · {state} · {party} — {len(m_stances)} stances · {len(m_bills)} bills"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Stances**")
                    if m_stances.empty:
                        st.markdown("<div style='color:#6b7585;font-size:0.8rem'>No stances found.</div>", unsafe_allow_html=True)
                    else:
                        for _, row in m_stances.head(5).iterrows():
                            date=str(row.get("date","") or ""); topic=str(row.get("topic","") or "")
                            text=str(row.get("text","") or ""); url=str(row.get("source_url","") or "")
                            badge=_topic_badge(topic) if topic else ""
                            url_tag=f"<a href='{url}' target='_blank' style='color:#4a9eff;font-size:0.7rem'>source ↗</a>" if url.startswith("http") else ""
                            st.markdown(f"<div class='evidence-card'><div class='meta'>{badge} {date} {url_tag}</div><div>{text[:250]}{'…' if len(text)>250 else ''}</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("**Sponsored Bills**")
                    if m_bills.empty:
                        st.markdown("<div style='color:#6b7585;font-size:0.8rem'>No sponsored bills found.</div>", unsafe_allow_html=True)
                    else:
                        for _, row in m_bills.head(5).iterrows():
                            st.markdown(_bill_card_html(row), unsafe_allow_html=True)

# ── TAB 3: BROWSE BILLS ─────────────────────────────────────────────────────
with tab_bills:
    st.markdown("### Browse Bills")
    st.markdown("Filter and search all 5,000+ healthcare and AI bills.")

    bills_browse = _load_bills()
    if bills_browse.empty:
        st.error("Bills data not found.")
    else:
        col_bs, col_bt, col_btype = st.columns([3,1,1])
        with col_bs:
            bill_search = st.text_input("Search bills", placeholder="e.g. Medicare, artificial intelligence, drug pricing…", label_visibility="collapsed")
        with col_bt:
            bill_topic = st.selectbox("Topic", ["All","healthcare","ai"], key="btopic", label_visibility="collapsed")
        with col_btype:
            bill_type_f = st.selectbox("Type", ["All","HR","S","HRES","SRES","HJRES","SJRES"], key="btype", label_visibility="collapsed")

        fb = bills_browse.copy()
        if bill_search.strip():
            fb = fb[fb["title"].str.contains(bill_search, case=False, na=False) | fb["summary"].str.contains(bill_search, case=False, na=False)]
        if bill_topic != "All":
            fb = fb[fb["topics"].str.contains(bill_topic, na=False)]
        if bill_type_f != "All":
            fb = fb[fb["bill_id"].str.startswith(bill_type_f)]

        st.markdown(f"<div style='color:#6b7585;font-family:IBM Plex Mono,monospace;font-size:0.75rem;margin-bottom:1rem'>{len(fb):,} bills matching filters</div>", unsafe_allow_html=True)

        page_size = 20
        total_pages = max(1, (len(fb)-1)//page_size+1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page-1)*page_size

        for _, row in fb.iloc[start:start+page_size].iterrows():
            st.markdown(_bill_card_html(row), unsafe_allow_html=True)
