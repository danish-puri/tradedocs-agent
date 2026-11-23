"""Shared test helpers. No test here needs an API key or the network."""

import copy
from types import SimpleNamespace

import pytest
from anthropic.types.beta import BetaMessage

# A tiny version of data/endpoints.json, shaped like what docs.py produces.
ENDPOINTS = [
    {
        "method": "GET", "path": "/v2/clock", "summary": "Get US Market Clock", "description": "",
        "params": [], "body": None, "response_fields": [], "doc_url": "https://example/legacyclock",
    },
    {
        "method": "GET", "path": "/v2/orders", "summary": "Get All Orders", "description": "",
        "params": [
            {"name": "status", "in": "query", "required": False, "type": "string", "enum": ["open", "closed", "all"], "description": ""},
            {"name": "limit", "in": "query", "required": False, "type": "integer", "enum": None, "description": ""},
        ],
        "body": None, "response_fields": [], "doc_url": "https://example/getallorders-1",
    },
    {
        "method": "DELETE", "path": "/v2/orders/{order_id}", "summary": "Delete Order by ID", "description": "",
        "params": [{"name": "order_id", "in": "path", "required": True, "type": "string", "enum": None, "description": ""}],
        "body": None, "response_fields": [], "doc_url": "https://example/deleteorderbyorderid-1",
    },
    {
        "method": "DELETE", "path": "/v2/positions/{symbol_or_asset_id}", "summary": "Close a Position", "description": "",
        "params": [
            {"name": "symbol_or_asset_id", "in": "path", "required": True, "type": "string", "enum": None, "description": ""},
            {"name": "qty", "in": "query", "required": False, "type": "number", "enum": None, "description": ""},
            {"name": "percentage", "in": "query", "required": False, "type": "number", "enum": None, "description": ""},
        ],
        "body": None, "response_fields": [], "doc_url": "https://example/deleteopenposition-1",
    },
    {
        "method": "POST", "path": "/v2/orders", "summary": "Create an Order", "description": "",
        "params": [],
        "body": {
            "type": "object", "required": ["type", "time_in_force"], "description": "",
            "properties": {
                "symbol": {"type": "string", "enum": None, "description": ""},
                "qty": {"type": "string", "enum": None, "description": ""},
                "notional": {"type": "string", "enum": None, "description": ""},
                "side": {"type": "string", "enum": ["buy", "sell"], "description": ""},
                "type": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit", "trailing_stop"], "description": ""},
                "time_in_force": {"type": "string", "enum": ["day", "gtc", "opg", "cls", "ioc", "fok"], "description": ""},
                "limit_price": {"type": "string", "enum": None, "description": ""},
                "stop_price": {"type": "string", "enum": None, "description": ""},
                "trail_price": {"type": "string", "enum": None, "description": ""},
                "trail_percent": {"type": "string", "enum": None, "description": ""},
                "extended_hours": {"type": "boolean", "enum": None, "description": ""},
                "order_class": {"type": "string", "enum": ["simple", "bracket", "oco", "oto", "mleg", ""], "description": ""},
                "take_profit": {"type": "object", "enum": None, "description": "", "required": [],
                                "properties": {"limit_price": {"type": "string", "enum": None, "description": ""}}},
                "stop_loss": {"type": "object", "enum": None, "description": "", "required": [],
                              "properties": {"stop_price": {"type": "string", "enum": None, "description": ""},
                                             "limit_price": {"type": "string", "enum": None, "description": ""}}},
            },
        },
        "response_fields": [], "doc_url": "https://example/postorder",
    },
]

CHUNKS = [
    {"id": "fractional-trading-001", "text": "Fractional Trading > Supported Order Types\n\nUse notional with day.",
     "slug": "fractional-trading", "url": "https://docs.alpaca.markets/us/docs/fractional-trading",
     "title": "Fractional Trading", "section": "Supported Order Types", "score": 0.5},
]


def message(content: list[dict], stop_reason: str) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
    )


def tool_call(call: dict, text: str = "Here is the request.", use_id: str = "toolu_1") -> BetaMessage:
    return message(
        [{"type": "text", "text": text}, {"type": "tool_use", "id": use_id, "name": "call_alpaca_api", "input": call}],
        "tool_use",
    )


def answer(text: str) -> BetaMessage:
    return message([{"type": "text", "text": text}], "end_turn")


class FakeClient:
    """Stands in for anthropic.Anthropic. Replays scripted responses and
    records every request so tests can inspect what was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


class FakeExecutor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"status_code": 200, "ok": True, "data": {"is_open": False}}

    def __call__(self, call):
        self.calls.append(call)
        return self.result


@pytest.fixture
def endpoints():
    return copy.deepcopy(ENDPOINTS)


@pytest.fixture
def make_agent(endpoints):
    from tradedocs.agent import TradeDocsAgent

    def build(responses, executor=None):
        client = FakeClient(responses)
        executor = executor or FakeExecutor()
        agent = TradeDocsAgent(client=client, search=lambda q: CHUNKS, executor=executor, endpoints=endpoints)
        return agent, client, executor

    return build
