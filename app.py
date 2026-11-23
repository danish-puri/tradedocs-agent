"""Streamlit demo. Run it with:  streamlit run app.py"""

from __future__ import annotations

import json
import os

import anthropic
import streamlit as st

from tradedocs.agent import TradeDocsAgent
from tradedocs.checker import review_call
from tradedocs.config import EFFORT, MODEL, PAPER_BASE_URL
from tradedocs.store import IndexMissingError

st.set_page_config(page_title="TradeDocs Agent", page_icon="📄")

EXAMPLES = [
    "How do I buy $50 worth of AAPL using fractional shares?",
    "Is the stock market open right now?",
    "How much buying power does my account have?",
    "How do I place a trailing stop order to sell 1 share of AAPL that trails by 2%?",
]


@st.cache_resource
def get_agent() -> TradeDocsAgent:
    return TradeDocsAgent()


def md(text: str) -> str:
    # Streamlit reads $...$ as math, which garbles prices like $50.
    return (text or "").replace("$", "\\$")


def risk_badge(risk: dict) -> None:
    st.markdown(f":{risk['color']}-background[**{risk['level'].upper()}**] {risk['label']}")


def show_findings(findings: list[dict], heading: str = "Call checker") -> None:
    if not findings:
        st.success(f"{heading} found no problems.", icon="✅")
    for f in findings:
        show = st.error if f["level"] == "error" else st.warning
        show(f"{md(f['message'])}  \n*Source: {f['source']}*")


def show_sources(chunks: list[dict]) -> None:
    pages: dict[str, dict] = {}
    for c in chunks:
        pages.setdefault(c["url"], {"title": c["title"], "sections": []})["sections"].append(c["section"])
    with st.expander(f"Step 1. Searched the docs and found {len(pages)} pages"):
        for url, page in pages.items():
            st.markdown(f"- [{page['title']}]({url})  \n  {md(', '.join(dict.fromkeys(page['sections'])))}")


def show_record(number: int, record: dict) -> None:
    labels = {"approved": "approved and ran", "edited": "edited, then ran", "rejected": "rejected", "auto-rejected": "stopped by call limit"}
    if record.get("explanation"):
        st.markdown(md(record["explanation"]))
    with st.container(border=True):
        st.markdown(f"**API call {number}**, {labels[record['decision']]}")
        risk_badge(record["review"]["risk"])
        st.code(json.dumps(record.get("final_call", record["call"]), indent=2), language="json")
        if "result" in record:
            result = record["result"]
            st.caption(f"HTTP {result['status_code']} from {PAPER_BASE_URL}")
            st.json(result["data"], expanded=False)
        else:
            st.caption(f"Not run. Reason: {record.get('reason')}")


def approval_box(turn) -> None:
    pending = turn.pending
    if pending.get("explanation"):
        st.markdown(md(pending["explanation"]))
    with st.container(border=True):
        st.markdown(f"**Step {len(turn.state.get('calls', [])) + 2}. Approve this API call?**")
        risk_badge(pending["review"]["risk"])
        show_findings(pending["review"]["findings"])
        original = json.dumps(pending["call"], indent=2)
        key = f"edit-{turn.thread_id}-{len(turn.state.get('calls', []))}"
        edited_text = st.text_area("You can edit the call before approving it.", value=original, height=220, key=key)

        edited, problem = None, None
        try:
            edited = json.loads(edited_text)
        except json.JSONDecodeError as exc:
            problem = f"That isn't valid JSON yet: {exc}"
        if problem:
            st.error(problem)
        elif edited != pending["call"]:
            st.markdown("Your edited call:")
            risk_badge(review_call(edited, get_agent().endpoints)["risk"])
            show_findings(review_call(edited, get_agent().endpoints)["findings"], heading="Call checker on your edit")

        reason = st.text_input("Reason, if you reject it (optional)", key=f"reason-{key}")
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve and run", type="primary", disabled=problem is not None, use_container_width=True):
            decision = {"action": "approve"} if edited == pending["call"] else {"action": "edit", "call": edited}
            resume(turn.thread_id, decision)
        if reject_col.button("Reject", use_container_width=True):
            resume(turn.thread_id, {"action": "reject", "reason": reason})


def resume(thread_id: str, decision: dict) -> None:
    with st.spinner("Working..."):
        try:
            st.session_state.turn = get_agent().resume(thread_id, decision)
        except anthropic.AnthropicError as exc:
            st.session_state.error = f"Claude API error: {exc}"
    st.rerun()


def ask(question: str) -> None:
    st.session_state.error = None
    with st.spinner("Searching the docs and asking Claude..."):
        try:
            st.session_state.turn = get_agent().start(question)
        except IndexMissingError as exc:
            st.session_state.error = str(exc)
        except anthropic.AuthenticationError:
            st.session_state.error = "Claude rejected the API key. Check ANTHROPIC_API_KEY in .env."
        except anthropic.AnthropicError as exc:
            st.session_state.error = f"Claude API error: {exc}"


# ---------------------------------------------------------------- page

st.title("TradeDocs Agent")
st.caption("Ask a question about Alpaca's Trading API. Every API call waits for your OK, and only goes to the paper trading sandbox.")

with st.sidebar:
    st.subheader("Setup")
    for name in ("ANTHROPIC_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        st.markdown(f"{'✅' if os.getenv(name) else '❌'} `{name}`")
    st.markdown(f"Model `{MODEL}`, effort `{EFFORT}`")
    st.markdown(f"Alpaca host `{PAPER_BASE_URL}`")
    st.subheader("Try one")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            ask(example)

with st.form("ask"):
    question = st.text_input("Your question", placeholder="How do I place a fractional order?")
    if st.form_submit_button("Ask", type="primary") and question.strip():
        ask(question.strip())

if st.session_state.get("error"):
    st.error(st.session_state.error)

turn = st.session_state.get("turn")
if turn:
    st.markdown(f"### {md(turn.state['question'])}")
    show_sources(turn.state["chunks"])
    for number, record in enumerate(turn.state.get("calls", []), start=1):
        show_record(number, record)
    if turn.pending:
        approval_box(turn)
    else:
        st.markdown("#### Answer")
        st.markdown(md(turn.state.get("answer", "")))
