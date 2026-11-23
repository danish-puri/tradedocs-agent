"""The approval flow: nothing runs until a person approves, edits, or rejects."""

from conftest import FakeExecutor, answer, message, tool_call

from tradedocs.config import EFFORT, MAX_CALLS, MODEL

CLOCK = {"method": "GET", "path": "/v2/clock"}


def test_answer_without_a_call(make_agent):
    agent, client, executor = make_agent([answer("Use notional with time_in_force day.")])
    turn = agent.start("How do I place a fractional order?")
    assert turn.done
    assert turn.state["answer"] == "Use notional with time_in_force day."
    assert executor.calls == []
    # The doc excerpts and the question both went to the model.
    prompt = client.requests[0]["messages"][0]["content"]
    assert "Fractional Trading" in prompt and "How do I place a fractional order?" in prompt


def test_graph_pauses_before_any_call(make_agent):
    agent, _, executor = make_agent([tool_call(CLOCK)])
    turn = agent.start("Is the market open?")
    assert not turn.done
    assert turn.pending["call"] == CLOCK
    assert turn.pending["review"]["risk"]["color"] == "green"
    assert executor.calls == []  # paused, nothing ran


def test_approve_runs_the_call_and_feeds_back_json(make_agent):
    agent, client, executor = make_agent([tool_call(CLOCK), answer("The market is closed.")])
    turn = agent.start("Is the market open?")
    turn = agent.resume(turn.thread_id, {"action": "approve"})
    assert turn.done
    assert executor.calls == [CLOCK]
    assert turn.state["calls"][0]["decision"] == "approved"
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert "HTTP status: 200" in tool_result["content"] and '"is_open": false' in tool_result["content"]
    assert tool_result["is_error"] is False


def test_edit_runs_the_edited_call(make_agent):
    bad = {"method": "POST", "path": "/v2/orders",
           "body": {"symbol": "AAPL", "notional": "50", "side": "buy", "type": "market", "time_in_force": "gtc"}}
    fixed = {**bad, "body": {**bad["body"], "time_in_force": "day"}}
    agent, client, executor = make_agent([tool_call(bad), answer("Order placed.")])
    turn = agent.start("Buy $50 of AAPL")
    # The checker caught the problem before the person saw it.
    messages = [f["message"] for f in turn.pending["review"]["findings"]]
    assert any("time_in_force 'day'" in m for m in messages)
    turn = agent.resume(turn.thread_id, {"action": "edit", "call": fixed})
    assert executor.calls == [fixed]
    record = turn.state["calls"][0]
    assert record["decision"] == "edited" and record["final_review"]["findings"] == []
    assert "The user edited your call" in client.requests[1]["messages"][-1]["content"][0]["content"]


def test_reject_never_runs_the_call(make_agent):
    agent, client, executor = make_agent([tool_call(CLOCK), answer("You can call GET /v2/clock yourself.")])
    turn = agent.start("Is the market open?")
    turn = agent.resume(turn.thread_id, {"action": "reject", "reason": "not now"})
    assert turn.done and executor.calls == []
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True and "not now" in tool_result["content"]
    assert turn.state["calls"][0]["decision"] == "rejected"


def test_thinking_blocks_are_sent_back_unchanged(make_agent):
    first = message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig-abc"},
            {"type": "tool_use", "id": "toolu_9", "name": "call_alpaca_api", "input": CLOCK},
        ],
        "tool_use",
    )
    agent, client, _ = make_agent([first, answer("Closed.")])
    turn = agent.start("Is the market open?")
    agent.resume(turn.thread_id, {"action": "approve"})
    assistant = client.requests[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}


def test_call_limit_stops_endless_calls(make_agent):
    responses = [tool_call(CLOCK, use_id=f"toolu_{i}") for i in range(MAX_CALLS + 1)] + [answer("Done.")]
    agent, _, executor = make_agent(responses)
    turn = agent.start("Is the market open?")
    for _ in range(MAX_CALLS):
        turn = agent.resume(turn.thread_id, {"action": "approve"})
    assert turn.done
    assert len(executor.calls) == MAX_CALLS
    assert turn.state["calls"][-1]["decision"] == "auto-rejected"


def test_failed_call_is_marked_as_error(make_agent):
    executor = FakeExecutor({"status_code": 403, "ok": False, "data": {"message": "forbidden"}})
    agent, client, _ = make_agent([tool_call(CLOCK), answer("It failed.")], executor=executor)
    turn = agent.start("Is the market open?")
    agent.resume(turn.thread_id, {"action": "approve"})
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True and "403" in tool_result["content"]


def test_request_uses_current_api_shape(make_agent):
    agent, client, _ = make_agent([answer("ok")])
    agent.start("hi")
    req = client.requests[0]
    assert req["model"] == MODEL
    assert req["output_config"] == {"effort": EFFORT}
    assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    if MODEL == "claude-opus-5-5":
        assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in req and "temperature" not in req
