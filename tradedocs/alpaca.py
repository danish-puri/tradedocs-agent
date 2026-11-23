"""The agent's one tool: send an HTTP request to Alpaca's paper trading API.

Three guards keep this safe.
1. The host is hard-coded to paper-api.alpaca.markets. If .env points
   anywhere else, the call is refused instead of quietly using it.
2. Only /v2/ paths and the four methods the docs use are allowed.
3. Nothing in this file decides to run a call. The agent graph only calls
   this after a person clicks approve.
"""

from __future__ import annotations

import os

import requests

from .config import PAPER_BASE_URL

ALLOWED_METHODS = {"GET", "POST", "PATCH", "DELETE"}


class CallRefused(Exception):
    """Raised when a call breaks a safety rule. Nothing was sent."""


def validate_call(call: dict) -> tuple[str, str, dict | None, dict | None]:
    method = str(call.get("method", "")).upper()
    path = str(call.get("path", ""))
    query = call.get("query") or None
    body = call.get("body") or None
    if method not in ALLOWED_METHODS:
        raise CallRefused(f"Method {method or '(missing)'} is not allowed.")
    if not path.startswith("/v2/") or "://" in path or ".." in path or "?" in path:
        raise CallRefused(f"Path {path!r} must start with /v2/ and must not hold a host or query string.")
    if query is not None and not isinstance(query, dict):
        raise CallRefused("query must be a JSON object.")
    if body is not None and not isinstance(body, dict):
        raise CallRefused("body must be a JSON object.")
    return method, path, query, body


def paper_base_url() -> str:
    configured = os.getenv("APCA_API_BASE_URL", PAPER_BASE_URL).strip().rstrip("/")
    if configured != PAPER_BASE_URL:
        raise CallRefused(
            f"APCA_API_BASE_URL is set to {configured!r}. This project only talks to {PAPER_BASE_URL}."
        )
    return PAPER_BASE_URL


def call_paper_api(call: dict, timeout: float = 20) -> dict:
    """Run one approved call. Returns status code, ok flag, and the JSON."""
    method, path, query, body = validate_call(call)
    base = paper_base_url()
    key_id = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key_id or not secret:
        raise CallRefused("APCA_API_KEY_ID and APCA_API_SECRET_KEY are not set in .env.")
    resp = requests.request(
        method,
        base + path,
        params=query,
        json=body,
        headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret},
        timeout=timeout,
    )
    try:
        data = resp.json()
    except ValueError:
        data = resp.text
    return {"status_code": resp.status_code, "ok": resp.ok, "data": data}
