"""Safety guards on the one tool that touches Alpaca."""

import pytest

from tradedocs import alpaca
from tradedocs.alpaca import CallRefused, call_paper_api, validate_call


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)


class FakeResponse:
    status_code = 200
    ok = True

    def json(self):
        return {"is_open": True}


def test_refuses_live_url(keys, monkeypatch):
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    sent = []
    monkeypatch.setattr(alpaca.requests, "request", lambda *a, **k: sent.append(a))
    with pytest.raises(CallRefused, match="only talks to https://paper-api.alpaca.markets"):
        call_paper_api({"method": "GET", "path": "/v2/account"})
    assert sent == []


@pytest.mark.parametrize(
    "call",
    [
        {"method": "PUT", "path": "/v2/orders"},
        {"method": "GET", "path": "https://api.alpaca.markets/v2/account"},
        {"method": "GET", "path": "/v1/account"},
        {"method": "GET", "path": "/v2/../v1/account"},
        {"method": "GET", "path": "/v2/orders?status=all"},
        {"method": "POST", "path": "/v2/orders", "body": ["not", "an", "object"]},
    ],
)
def test_refuses_bad_calls(call):
    with pytest.raises(CallRefused):
        validate_call(call)


def test_refuses_without_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(CallRefused, match="not set"):
        call_paper_api({"method": "GET", "path": "/v2/clock"})


def test_sends_to_paper_host_with_key_headers(keys, monkeypatch):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(method=method, url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(alpaca.requests, "request", fake_request)
    result = call_paper_api({"method": "get", "path": "/v2/clock", "query": {"x": "1"}})
    assert seen["method"] == "GET"
    assert seen["url"] == "https://paper-api.alpaca.markets/v2/clock"
    assert seen["headers"] == {"APCA-API-KEY-ID": "test-id", "APCA-API-SECRET-KEY": "test-secret"}
    assert seen["params"] == {"x": "1"}
    assert result == {"status_code": 200, "ok": True, "data": {"is_open": True}}
