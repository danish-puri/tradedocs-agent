"""The agent, built as a LangGraph graph.

    retrieve -> model --(answer)--> done
                  \\--(wants a call)--> check -> approve -> run -> model
                                                  \\--(rejected)--> model

`approve` pauses the graph with LangGraph's interrupt(). Nothing runs until
a person resumes it with approve, edit, or reject. Claude is called through
Anthropic's own Python SDK inside the `model` step.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Callable, TypedDict

import anthropic
import requests
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .alpaca import CallRefused, call_paper_api
from .checker import load_endpoints, review_call
from .config import EFFORT, MAX_CALLS, MODEL

SYSTEM_PROMPT = """You help developers use Alpaca's Trading API. Each question comes with excerpts from Alpaca's docs.

How to answer:
- Base your answer on the doc excerpts. If they don't cover the question, say so plainly instead of guessing.
- Cite the pages you used as markdown links, like [Fractional Trading](https://docs.alpaca.markets/us/docs/fractional-trading).
- Keep it short and practical. Developers want the request to send and what to watch out for.

Using the tool:
- When a live example would help, call call_alpaca_api with the exact request you recommend. It runs against Alpaca's paper trading sandbox (fake money), and only after the user approves it. The user may also edit or reject it.
- Use real values, never placeholders. If you need an ID you don't have, such as an order ID, get it first with a read call like GET /v2/orders.
- Make one call at a time. After a call runs, explain the important parts of the JSON response.
- If the user rejects a call, don't send the same call again. Finish your answer without it and show the request they could send themselves.
- Paths start with /v2/. Don't include the host."""

TOOL = {
    "name": "call_alpaca_api",
    "description": (
        "Send one HTTP request to Alpaca's paper trading API (https://paper-api.alpaca.markets) and get the "
        "JSON response. A person reviews every call first and can approve, edit, or reject it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PATCH", "DELETE"]},
            "path": {"type": "string", "description": "Path starting with /v2/, such as /v2/orders or /v2/assets/AAPL"},
            "query": {"type": "object", "description": "URL query parameters, if any"},
            "body": {"type": "object", "description": "JSON request body for POST or PATCH, if any"},
        },
        "required": ["method", "path"],
    },
}

# Models that accept server-side refusal fallbacks (a declined request gets
# retried on another model inside the same API call).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}

RESULT_CHARS = 6000  # how much of a JSON response the model gets to read


class AgentState(TypedDict, total=False):
    question: str
    chunks: list[dict]
    messages: list[dict]
    pending: dict | None
    calls: list[dict]
    answer: str
    stop_reason: str
    usage: list[dict]


@dataclass
class Turn:
    thread_id: str
    state: dict
    pending: dict | None  # what the approval screen should show, if paused

    @property
    def done(self) -> bool:
        return self.pending is None


def format_docs(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f'<doc index="{i}" title="{c["title"]}" url="{c["url"]}">\n{c["text"]}\n</doc>')
    return "\n\n".join(parts)


def to_echo_blocks(blocks) -> list[dict]:
    """Turn response blocks into plain dicts we can store and send back.

    If a refusal fallback happened mid-answer, thinking and tool_use blocks
    before the last "fallback" block must be left out when echoing.
    """
    dicts = [b.to_dict(mode="json", exclude_none=True) if hasattr(b, "to_dict") else dict(b) for b in blocks]
    fallback_at = [i for i, b in enumerate(dicts) if b.get("type") == "fallback"]
    if not fallback_at:
        return dicts
    cut = fallback_at[-1]
    drop = {"thinking", "redacted_thinking", "tool_use"}
    return [b for i, b in enumerate(dicts) if i >= cut or b.get("type") not in drop]


def _truncate(text: str, limit: int = RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (cut off, {len(text) - limit} more characters)"


class TradeDocsAgent:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        search: Callable[[str], list[dict]] | None = None,
        executor: Callable[[dict], dict] = call_paper_api,
        endpoints: list[dict] | None = None,
        model: str = MODEL,
        effort: str = EFFORT,
    ) -> None:
        self._client = client
        self._search = search
        self.executor = executor
        self.endpoints = load_endpoints() if endpoints is None else endpoints
        self.model = model
        self.effort = effort
        self.graph = self._build_graph()

    # Created on first use, so the app can load without keys in place.
    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def search(self) -> Callable[[str], list[dict]]:
        if self._search is None:
            from .store import DocSearch

            self._search = DocSearch().search
        return self._search

    # ------------------------------------------------------------ graph

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("model", self._model, destinations=("check", END))
        builder.add_node("check", self._check, destinations=("approve", "model"))
        builder.add_node("approve", self._approve, destinations=("run", "model"))
        builder.add_node("run", self._run, destinations=("model",))
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "model")
        return builder.compile(checkpointer=InMemorySaver())

    def _retrieve(self, state: AgentState) -> dict:
        chunks = self.search(state["question"])
        prompt = (
            f"Excerpts from Alpaca's docs:\n\n{format_docs(chunks)}\n\n"
            f"Question: {state['question']}"
        )
        return {"chunks": chunks, "messages": [{"role": "user", "content": prompt}], "calls": [], "pending": None}

    def _model(self, state: AgentState) -> Command:
        params = dict(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=[TOOL],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            output_config={"effort": self.effort},
            messages=state["messages"],
        )
        if self.model in FALLBACK_MODELS:
            params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        response = self.client.beta.messages.create(**params)

        content = to_echo_blocks(response.content)
        text = "\n\n".join(b["text"] for b in content if b.get("type") == "text").strip()
        update = {
            "messages": state["messages"] + [{"role": "assistant", "content": content}],
            "stop_reason": response.stop_reason,
            "usage": state.get("usage", []) + [response.usage.to_dict(mode="json", exclude_none=True)],
        }
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if response.stop_reason == "tool_use" and tool_uses:
            use = tool_uses[0]
            update["pending"] = {"id": use["id"], "call": use["input"], "explanation": text}
            return Command(goto="check", update=update)

        if response.stop_reason == "refusal":
            text = text or "The model declined to answer this question."
        elif response.stop_reason == "max_tokens":
            text += "\n\n(This answer was cut off because it hit the length limit.)"
        update.update(answer=text, pending=None)
        return Command(goto=END, update=update)

    def _check(self, state: AgentState) -> Command:
        pending = {**state["pending"], "review": review_call(state["pending"]["call"], self.endpoints)}
        if len(state.get("calls", [])) >= MAX_CALLS:
            reason = f"Call limit reached ({MAX_CALLS} per question). Finish your answer without more calls."
            return self._reject(state, pending, reason, decision="auto-rejected")
        return Command(goto="approve", update={"pending": pending})

    def _approve(self, state: AgentState) -> Command:
        pending = state["pending"]
        # The graph stops here until resume() is called with a decision.
        decision = interrupt(
            {"call": pending["call"], "review": pending["review"], "explanation": pending["explanation"]}
        )
        action = (decision or {}).get("action")
        if action == "approve":
            return Command(goto="run", update={"pending": {**pending, "final_call": pending["call"], "edited": False}})
        if action == "edit":
            new_call = decision["call"]
            edited = {**pending, "final_call": new_call, "edited": True, "final_review": review_call(new_call, self.endpoints)}
            return Command(goto="run", update={"pending": edited})
        if action == "reject":
            return self._reject(state, pending, decision.get("reason") or "No reason given.")
        raise ValueError(f"Unknown decision {decision!r}. Use approve, edit, or reject.")

    def _reject(self, state: AgentState, pending: dict, reason: str, decision: str = "rejected") -> Command:
        message = reason if decision == "auto-rejected" else f"The user rejected this call, so it did not run. Reason: {reason}"
        tool_result = {"type": "tool_result", "tool_use_id": pending["id"], "content": message, "is_error": True}
        record = {**pending, "decision": decision, "reason": reason}
        return Command(
            goto="model",
            update={
                "messages": state["messages"] + [{"role": "user", "content": [tool_result]}],
                "calls": state.get("calls", []) + [record],
                "pending": None,
            },
        )

    def _run(self, state: AgentState) -> Command:
        pending = state["pending"]
        call = pending["final_call"]
        try:
            result = self.executor(call)
        except CallRefused as exc:
            result = {"status_code": None, "ok": False, "data": f"Refused before sending: {exc}"}
        except requests.RequestException as exc:
            result = {"status_code": None, "ok": False, "data": f"Network error: {exc}"}

        text = ""
        if pending.get("edited"):
            text += f"The user edited your call before running it. This is the call that ran:\n{json.dumps(call)}\n\n"
        text += f"HTTP status: {result['status_code']}\n{_truncate(json.dumps(result['data'], indent=2))}"
        tool_result = {"type": "tool_result", "tool_use_id": pending["id"], "content": text, "is_error": not result["ok"]}
        record = {**pending, "decision": "edited" if pending.get("edited") else "approved", "result": result}
        return Command(
            goto="model",
            update={
                "messages": state["messages"] + [{"role": "user", "content": [tool_result]}],
                "calls": state.get("calls", []) + [record],
                "pending": None,
            },
        )

    # ------------------------------------------------------- public API

    def _turn(self, thread_id: str) -> Turn:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        pending = None
        for task in snapshot.tasks:
            for intr in task.interrupts:
                pending = intr.value
        return Turn(thread_id=thread_id, state=dict(snapshot.values), pending=pending)

    def start(self, question: str, thread_id: str | None = None) -> Turn:
        thread_id = thread_id or uuid.uuid4().hex
        self.graph.invoke({"question": question}, {"configurable": {"thread_id": thread_id}})
        return self._turn(thread_id)

    def resume(self, thread_id: str, decision: dict) -> Turn:
        """decision is {"action": "approve"}, {"action": "edit", "call": {...}},
        or {"action": "reject", "reason": "..."}."""
        self.graph.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})
        return self._turn(thread_id)
