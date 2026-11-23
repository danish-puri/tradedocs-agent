"""The call checker (innovation #1) and risk colors (innovation #2)."""

from tradedocs.checker import review_call


def order(**body):
    return {"method": "POST", "path": "/v2/orders", "body": body}


def errors(result):
    return [f["message"] for f in result["findings"] if f["level"] == "error"]


def warnings(result):
    return [f["message"] for f in result["findings"] if f["level"] == "warning"]


GOOD_NOTIONAL = dict(symbol="AAPL", notional="50", side="buy", type="market", time_in_force="day")


def test_clean_orders_pass(endpoints):
    assert review_call(order(**GOOD_NOTIONAL), endpoints)["findings"] == []
    limit = dict(symbol="AAPL", qty="1", side="buy", type="limit", limit_price="150", time_in_force="gtc")
    assert review_call(order(**limit), endpoints)["findings"] == []


def test_numbers_or_numeric_strings_both_pass(endpoints):
    # The docs send prices both ways ("Takes in string/number values").
    assert review_call(order(**{**GOOD_NOTIONAL, "notional": 50}), endpoints)["findings"] == []
    assert errors(review_call(order(**{**GOOD_NOTIONAL, "notional": "fifty"}), endpoints)) == [
        "notional should be a number, but got 'fifty'."
    ]


def test_notional_needs_day(endpoints):
    result = review_call(order(**{**GOOD_NOTIONAL, "time_in_force": "gtc"}), endpoints)
    assert "Notional (dollar amount) orders need time_in_force 'day'." in errors(result)


def test_fractional_qty_needs_day(endpoints):
    body = dict(symbol="TSLA", qty="0.5", side="buy", type="market", time_in_force="gtc")
    assert "Fractional share orders need time_in_force 'day'." in errors(review_call(order(**body), endpoints))


def test_qty_and_notional_together(endpoints):
    result = review_call(order(**{**GOOD_NOTIONAL, "qty": "1"}), endpoints)
    assert "Send qty or notional, not both." in errors(result)


def test_missing_prices_by_order_type(endpoints):
    base = dict(symbol="AAPL", qty="1", side="buy", time_in_force="day")
    assert "A limit order needs limit_price." in errors(review_call(order(**base, type="limit"), endpoints))
    stop_limit = errors(review_call(order(**base, type="stop_limit"), endpoints))
    assert "A stop_limit order needs limit_price." in stop_limit and "A stop_limit order needs stop_price." in stop_limit
    trailing = errors(review_call(order(**base, type="trailing_stop"), endpoints))
    assert "A trailing_stop order needs trail_price or trail_percent." in trailing


def test_bad_enum_and_missing_required(endpoints):
    result = review_call(order(symbol="AAPL", qty="1", side="hold", type="market"), endpoints)
    found = errors(result)
    assert any(m.startswith("side='hold' is not allowed") for m in found)
    assert "Body field 'time_in_force' is required." in found


def test_extended_hours_rules(endpoints):
    body = dict(symbol="AAPL", qty="1", side="buy", type="market", time_in_force="day", extended_hours=True)
    assert any("Extended hours" in m for m in errors(review_call(order(**body), endpoints)))
    body.update(type="limit", limit_price="150")
    assert review_call(order(**body), endpoints)["findings"] == []


def test_bracket_rules(endpoints):
    good = dict(symbol="SPY", qty="1", side="buy", type="market", time_in_force="gtc", order_class="bracket",
                take_profit={"limit_price": "301"}, stop_loss={"stop_price": "299"})
    assert review_call(order(**good), endpoints)["findings"] == []
    flipped = {**good, "take_profit": {"limit_price": "290"}}
    assert any("must be above" in m for m in errors(review_call(order(**flipped), endpoints)))
    missing = {k: v for k, v in good.items() if k != "stop_loss"}
    assert "A bracket order needs stop_loss.stop_price." in errors(review_call(order(**missing), endpoints))


def test_unknown_endpoint_and_unknown_field(endpoints):
    result = review_call({"method": "GET", "path": "/v2/options/contracts"}, endpoints)
    assert errors(result) and "isn't in the docs I have" in errors(result)[0]
    result = review_call(order(**{**GOOD_NOTIONAL, "color": "blue"}), endpoints)
    assert warnings(result) == ["Body field 'color' isn't in the docs for this endpoint."]


def test_placeholders_in_path(endpoints):
    for value in ("{order_id}", "<order_id>", "ORDER_ID", "your_order_id"):
        result = review_call({"method": "DELETE", "path": f"/v2/orders/{value}"}, endpoints)
        assert any("placeholder" in m for m in errors(result)), value
    real = review_call({"method": "DELETE", "path": "/v2/orders/61e69015-8549-4bfd-b9c3-01e75843f47d"}, endpoints)
    assert real["findings"] == []


def test_query_values_from_url_are_fine(endpoints):
    assert review_call({"method": "DELETE", "path": "/v2/positions/AAPL", "query": {"percentage": "50"}}, endpoints)["findings"] == []
    result = review_call({"method": "GET", "path": "/v2/orders", "query": {"status": "pending"}}, endpoints)
    assert any("status='pending' is not allowed" in m for m in errors(result))


def test_risk_colors(endpoints):
    assert review_call({"method": "GET", "path": "/v2/clock"}, endpoints)["risk"]["color"] == "green"
    assert review_call(order(**GOOD_NOTIONAL), endpoints)["risk"]["color"] == "orange"
    red = review_call({"method": "DELETE", "path": "/v2/positions/AAPL"}, endpoints)["risk"]
    assert red["color"] == "red" and "Close a Position" in red["label"]


def test_missing_endpoint_list_warns_instead_of_crashing():
    result = review_call({"method": "GET", "path": "/v2/clock"}, [])
    assert result["findings"][0]["level"] == "warning"
