"""Doc parsing and chunking, using small made-up pages."""

import json

from tradedocs.docs import (
    chunk_guide,
    chunk_page,
    clean_guide_markdown,
    endpoint_to_text,
    page_slug,
    parse_page,
    spec_to_endpoints,
)

GUIDE = """---
updatedAt: 2026-01-01
---

# Fractional Trading

Intro paragraph.

# Supported Order Types

Use `time_in_force` set to `day`.

```python
# Submit the order. This line is a code comment, not a heading.
client.submit_order(order)

print("done")
```

## Sample Requests

<HTMLBlock>{`
<style>.yes-cell { color: green; }</style>
<table>
  <tr><th class="h">TIF</th><th>Market</th></tr>
  <tr><td><b>DAY</b></td><td class="yes-cell">Yes</td></tr>
</table>
`}</HTMLBlock>
"""

SPEC = {
    "paths": {
        "/v2/orders": {
            "post": {
                "summary": "Create an Order",
                "description": "Places a new order.",
                "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Req"}}}},
                "responses": {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Order"}}}}},
            }
        },
        "/v2/positions/{symbol}": {
            "parameters": [{"name": "symbol", "in": "path", "required": True, "schema": {"type": "string"}}],
            "delete": {
                "summary": "Close a Position",
                "parameters": [{"name": "percentage", "in": "query", "schema": {"type": "number"}, "description": "0 to 100"}],
                "responses": {},
            },
        },
    },
    "components": {
        "schemas": {
            "Req": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"allOf": [{"$ref": "#/components/schemas/OrderType"}], "description": "Order type."},
                    "take_profit": {"type": "object", "properties": {"limit_price": {"type": "string"}}},
                },
            },
            "OrderType": {"type": "string", "enum": ["market", "limit"]},
            "Order": {"type": "object", "properties": {"id": {"type": "string", "description": "Order ID"}}},
        }
    },
}


def test_slug_and_title():
    page = parse_page("https://docs.alpaca.markets/us/docs/fractional-trading", GUIDE)
    assert page_slug("https://x/reference/postorder.md") == "postorder"
    assert page.title == "Fractional Trading" and page.kind == "guide"
    assert not page.body.startswith("---")


def test_html_tables_become_text_and_css_is_gone():
    text = clean_guide_markdown(GUIDE)
    assert "| TIF | Market |" in text and "| DAY | Yes |" in text
    assert "yes-cell" not in text and "<table>" not in text and "HTMLBlock" not in text


def test_code_comments_are_not_headings():
    page = parse_page("https://docs.alpaca.markets/us/docs/fractional-trading", GUIDE)
    chunks = chunk_guide(page, limit=200)
    sections = {c.section for c in chunks}
    assert not any("code comment" in s for s in sections)
    assert any("Supported Order Types" in s for s in sections)
    # Every chunk starts with the page title, so search knows where it's from.
    assert all(c.text.startswith("Fractional Trading") for c in chunks)


def test_chunks_respect_the_size_limit():
    long_page = parse_page("https://x/docs/long", "# Long\n\n" + "\n\n".join(["word " * 60] * 20))
    chunks = chunk_guide(long_page, limit=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 + len("Long\n\n") for c in chunks)


def test_spec_to_endpoints_resolves_refs():
    endpoints = spec_to_endpoints(SPEC, "https://x/reference/postorder")
    post = next(e for e in endpoints if e["method"] == "POST")
    assert post["body"]["required"] == ["type"]
    assert post["body"]["properties"]["type"]["enum"] == ["market", "limit"]
    assert post["body"]["properties"]["take_profit"]["properties"]["limit_price"]["type"] == "string"
    assert post["response_fields"] == [{"name": "id", "type": "string", "description": "Order ID"}]
    delete = next(e for e in endpoints if e["method"] == "DELETE")
    assert [p["name"] for p in delete["params"]] == ["symbol", "percentage"]


def test_endpoint_text_is_readable():
    post = spec_to_endpoints(SPEC, "https://x")[0]
    text = endpoint_to_text(post)
    assert text.startswith("POST /v2/orders (Create an Order)")
    assert "- type (string, required, one of: market, limit): Order type." in text
    assert "take_profit.limit_price" in text


def test_reference_page_gives_chunks_and_endpoints():
    body = "# Create an Order\n\n# OpenAPI definition\n\n```json\n" + json.dumps(SPEC) + "\n```\n"
    page = parse_page("https://docs.alpaca.markets/us/reference/postorder", body)
    chunks, endpoints = chunk_page(page)
    assert page.kind == "reference" and len(endpoints) == 2
    assert chunks[0].text.startswith("POST /v2/orders (Create an Order)")
    assert "{" not in chunks[0].text.split("\n")[2]  # plain text, not raw JSON
