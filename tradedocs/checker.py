"""The call checker and risk colors.

Before a person sees a proposed API call, it gets checked against the
endpoint definitions pulled from Alpaca's reference pages, plus a few order
rules the docs state in plain words. Problems show up on the approval
screen, so the person isn't the only line of defense.

Every rule here comes from the docs. The `source` on each finding names the
page it came from.
"""

from __future__ import annotations

import json
import re

from .config import ENDPOINTS_FILE

# The spec types these as strings, but the docs send them as numbers too
# ("Takes in string/number values"). Accept both, as long as it's a number.
NUMERIC_TEXT_FIELDS = {"qty", "notional", "limit_price", "stop_price", "trail_price", "trail_percent"}
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")

SPEC = "Create an Order (API reference)"
FRACTIONAL = "Fractional Trading"
PLACING = "Placing Orders"


def load_endpoints() -> list[dict]:
    if not ENDPOINTS_FILE.exists():
        return []
    return json.loads(ENDPOINTS_FILE.read_text())


def _finding(level: str, message: str, source: str) -> dict:
    return {"level": level, "message": message, "source": source}


def risk_level(method: str, endpoint: dict | None = None) -> dict:
    """Innovation #2: color each call by what it can do to the account."""
    method = str(method).upper()
    what = f" ({endpoint['summary']})" if endpoint else ""
    if method == "GET":
        return {"level": "read", "color": "green", "label": f"Read only. Changes nothing{what}."}
    if method == "DELETE":
        return {"level": "danger", "color": "red", "label": f"Cancels or closes something in your paper account{what}."}
    return {"level": "change", "color": "orange", "label": f"Places or changes something in your paper account{what}."}


def match_endpoint(method: str, path: str, endpoints: list[dict]) -> tuple[dict | None, dict]:
    """Find the endpoint whose template fits, like /v2/orders/{order_id}."""
    for ep in endpoints:
        if ep["method"] != str(method).upper():
            continue
        names = re.findall(r"{([^}]+)}", ep["path"])
        pattern = "^" + re.sub(r"\\{[^}]+\\}", r"([^/]+)", re.escape(ep["path"])) + "$"
        match = re.match(pattern, path)
        if match:
            return ep, dict(zip(names, match.groups()))
    return None, {}


def _looks_like_placeholder(value: str, name: str) -> bool:
    value = value.strip()
    return (
        not value
        or value[0] in "{<["
        or value.lower() == name.lower()
        or "your_" in value.lower()
        or (value.isupper() and "_" in value)
    )


def _type_ok(value, expected, *, from_url: bool = False) -> bool:
    if not expected:
        return True
    for kind in expected if isinstance(expected, list) else [expected]:
        if kind == "string" and isinstance(value, str):
            return True
        if kind in ("number", "integer") and isinstance(value, (int, float)) and not isinstance(value, bool):
            return kind == "number" or float(value).is_integer()
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
        # Query values travel in the URL as text, so "50" and "true" are fine.
        if from_url and isinstance(value, str):
            if kind in ("number", "integer") and _NUMBER.match(value):
                return True
            if kind == "boolean" and value.lower() in ("true", "false"):
                return True
            if kind == "array":
                return True
    return False


def _check_value(name: str, value, field: dict, findings: list, source: str, from_url: bool = False) -> None:
    if name.split(".")[-1] in NUMERIC_TEXT_FIELDS:
        if isinstance(value, bool) or not (isinstance(value, (int, float)) or _NUMBER.match(str(value))):
            findings.append(_finding("error", f"{name} should be a number, but got {value!r}.", source))
        return
    if not _type_ok(value, field.get("type"), from_url=from_url):
        findings.append(_finding("error", f"{name} should be {field.get('type')}, but got {value!r}.", source))
        return
    enum = [e for e in (field.get("enum") or []) if e != ""]
    if enum and str(value) not in [str(e) for e in enum]:
        findings.append(_finding("error", f"{name}={value!r} is not allowed. Use one of {', '.join(map(str, enum))}.", source))


def _decimals(value) -> int:
    text = str(value)
    return len(text.split(".", 1)[1]) if "." in text else 0


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def order_rules(body: dict) -> list[dict]:
    """Rules for POST /v2/orders that the docs state in plain words."""
    found = []
    order_class = body.get("order_class") or "simple"
    order_type = body.get("type")
    tif = body.get("time_in_force")
    has = lambda key: body.get(key) not in (None, "")  # noqa: E731

    if order_class != "mleg":
        for key in ("symbol", "side"):
            if not has(key):
                found.append(_finding("error", f"{key} is required for this order.", SPEC))
        if has("qty") and has("notional"):
            found.append(_finding("error", "Send qty or notional, not both.", SPEC))
        elif not has("qty") and not has("notional"):
            found.append(_finding("error", "Send qty (number of shares) or notional (dollar amount).", SPEC))
        if has("notional"):
            if tif and tif != "day":
                found.append(_finding("error", "Notional (dollar amount) orders need time_in_force 'day'.", SPEC))
            if _decimals(body["notional"]) > 2:
                found.append(_finding("error", "notional allows at most 2 decimal places. More gets the order rejected.", FRACTIONAL))
        if has("qty") and _num(body["qty"]) is not None and not float(body["qty"]).is_integer():
            if tif and tif != "day":
                found.append(_finding("error", "Fractional share orders need time_in_force 'day'.", FRACTIONAL))
            if _decimals(body["qty"]) > 9:
                found.append(_finding("warning", "qty keeps only 9 decimal places. Extra digits get cut off.", FRACTIONAL))

    if order_type in ("limit", "stop_limit") and not has("limit_price"):
        found.append(_finding("error", f"A {order_type} order needs limit_price.", SPEC))
    if order_type in ("stop", "stop_limit") and not has("stop_price"):
        found.append(_finding("error", f"A {order_type} order needs stop_price.", SPEC))
    if order_type == "trailing_stop":
        if not has("trail_price") and not has("trail_percent"):
            found.append(_finding("error", "A trailing_stop order needs trail_price or trail_percent.", SPEC))
        if tif and tif not in ("day", "gtc"):
            found.append(_finding("error", "Trailing stop orders only allow time_in_force 'day' or 'gtc'.", PLACING))

    if body.get("extended_hours") is True and (order_type != "limit" or tif not in ("day", "gtc")):
        found.append(_finding("error", "Extended hours orders must be limit orders with time_in_force 'day' or 'gtc'.", PLACING))

    if order_class == "bracket":
        take_profit = body.get("take_profit") or {}
        stop_loss = body.get("stop_loss") or {}
        if not isinstance(take_profit, dict) or take_profit.get("limit_price") in (None, ""):
            found.append(_finding("error", "A bracket order needs take_profit.limit_price.", PLACING))
        if not isinstance(stop_loss, dict) or stop_loss.get("stop_price") in (None, ""):
            found.append(_finding("error", "A bracket order needs stop_loss.stop_price.", PLACING))
        if tif and tif not in ("day", "gtc"):
            found.append(_finding("error", "Bracket orders only allow time_in_force 'day' or 'gtc'.", PLACING))
        if body.get("extended_hours") is True:
            found.append(_finding("error", "Bracket orders don't support extended hours.", PLACING))
        if isinstance(take_profit, dict) and isinstance(stop_loss, dict):
            tp, sl = _num(take_profit.get("limit_price")), _num(stop_loss.get("stop_price"))
            if tp is not None and sl is not None:
                if body.get("side") == "buy" and tp <= sl:
                    found.append(_finding("error", "For a buy bracket, take_profit.limit_price must be above stop_loss.stop_price.", PLACING))
                if body.get("side") == "sell" and tp >= sl:
                    found.append(_finding("error", "For a sell bracket, take_profit.limit_price must be below stop_loss.stop_price.", PLACING))
    return found


def review_call(call: dict, endpoints: list[dict] | None = None) -> dict:
    """Innovation #1: check one proposed call. Returns findings and risk."""
    endpoints = load_endpoints() if endpoints is None else endpoints
    method = str(call.get("method", "")).upper()
    path = str(call.get("path", ""))
    query = call.get("query") or {}
    body = call.get("body") or {}
    findings: list[dict] = []

    if not endpoints:
        findings.append(_finding("warning", "Endpoint list is missing, so the call can't be checked. Run the ingest step.", "setup"))
        return {"findings": findings, "risk": risk_level(method), "endpoint": None}

    ep, path_params = match_endpoint(method, path, endpoints)
    if ep is None:
        known = ", ".join(sorted({f"{e['method']} {e['path']}" for e in endpoints}))
        findings.append(_finding("error", f"{method} {path} isn't in the docs I have. Known endpoints are {known}.", "sources.txt"))
        return {"findings": findings, "risk": risk_level(method), "endpoint": None}

    source = f"{ep['summary']} (API reference)"
    for name, value in path_params.items():
        if _looks_like_placeholder(value, name):
            findings.append(_finding("error", f"{{{name}}} still holds a placeholder ({value!r}). Put a real value there.", source))

    query_params = {p["name"]: p for p in ep["params"] if p["in"] == "query"}
    if not isinstance(query, dict):
        findings.append(_finding("error", "query must be a JSON object.", source))
        query = {}
    for name, value in query.items():
        if name not in query_params:
            findings.append(_finding("warning", f"Query parameter {name!r} isn't in the docs for this endpoint.", source))
        else:
            _check_value(name, value, query_params[name], findings, source, from_url=True)
    for name, param in query_params.items():
        if param["required"] and name not in query:
            findings.append(_finding("error", f"Query parameter {name!r} is required.", source))

    spec_body = ep.get("body")
    if not isinstance(body, dict):
        findings.append(_finding("error", "body must be a JSON object.", source))
        body = {}
    if body and not spec_body:
        findings.append(_finding("warning", f"{method} {ep['path']} doesn't take a request body. It will be ignored.", source))
    elif spec_body:
        props = spec_body.get("properties", {})
        for name in spec_body.get("required", []):
            if body.get(name) in (None, ""):
                findings.append(_finding("error", f"Body field {name!r} is required.", source))
        for name, value in body.items():
            if name not in props:
                findings.append(_finding("warning", f"Body field {name!r} isn't in the docs for this endpoint.", source))
                continue
            _check_value(name, value, props[name], findings, source)
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    sub_props = props[name].get("properties") or {}
                    if sub_name in sub_props:
                        _check_value(f"{name}.{sub_name}", sub_value, sub_props[sub_name], findings, source)
        if method == "POST" and ep["path"] == "/v2/orders":
            already = {f["message"] for f in findings}
            findings.extend(f for f in order_rules(body) if f["message"] not in already)

    return {"findings": findings, "risk": risk_level(method, ep), "endpoint": ep["summary"]}
