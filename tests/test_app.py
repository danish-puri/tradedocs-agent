"""Click through the real Streamlit page with a fake Claude and a fake Alpaca."""

import pytest
import streamlit as st
from conftest import CHUNKS, ENDPOINTS, FakeClient, FakeExecutor, answer, tool_call
from streamlit.testing.v1 import AppTest

import tradedocs.agent
from tradedocs.config import ROOT

APP = str(ROOT / "app.py")

ORDER = {"method": "POST", "path": "/v2/orders",
         "body": {"symbol": "AAPL", "notional": "50", "side": "buy", "type": "market", "time_in_force": "gtc"}}


@pytest.fixture
def fake_agent(monkeypatch):
    executor = FakeExecutor({"status_code": 200, "ok": True, "data": {"id": "order-1", "status": "accepted"}})
    real = tradedocs.agent.TradeDocsAgent

    class FakeAgent(real):
        def __init__(self):
            super().__init__(
                client=FakeClient([tool_call(ORDER, text="Here's a $50 notional order."), answer("Your order was accepted.")]),
                search=lambda q: CHUNKS, executor=executor, endpoints=ENDPOINTS,
            )

    monkeypatch.setattr(tradedocs.agent, "TradeDocsAgent", FakeAgent)
    st.cache_resource.clear()
    yield executor
    st.cache_resource.clear()


def ask(at, question):
    at.text_input[0].input(question)
    next(b for b in at.button if b.label == "Ask").click().run()


def test_edit_then_approve(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    ask(at, "How do I buy $50 of AAPL?")

    # The approval box shows the risk color and the checker's finding.
    assert any("ORANGE" in m.value.upper() or "orange-background" in m.value for m in at.markdown)
    assert any("time_in_force 'day'" in e.value for e in at.error)
    assert fake_agent.calls == []  # nothing ran yet

    # Fix the call in the editor, then approve.
    fixed = at.text_area[0].value.replace('"gtc"', '"day"')
    at.text_area[0].input(fixed).run()
    assert any("Call checker on your edit found no problems." in s.value for s in at.success)
    next(b for b in at.button if b.label == "Approve and run").click().run()

    assert fake_agent.calls[0]["body"]["time_in_force"] == "day"
    assert any("Your order was accepted." in m.value for m in at.markdown)
    assert any("edited, then ran" in m.value for m in at.markdown)


def test_reject(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    ask(at, "How do I buy $50 of AAPL?")
    next(b for b in at.button if b.label == "Reject").click().run()
    assert fake_agent.calls == []
    assert any("rejected" in m.value for m in at.markdown)
