"""Ask one question in the terminal and approve each API call by hand.

  python -m tradedocs.cli 'How do I buy $50 of AAPL?'
"""

from __future__ import annotations

import json
import sys

import anthropic

from .agent import TradeDocsAgent
from .store import IndexMissingError

ANSI = {"green": "\033[42;30m", "orange": "\033[43;30m", "red": "\033[41;97m", "reset": "\033[0m"}


def show_pending(pending: dict) -> None:
    review = pending["review"]
    risk = review["risk"]
    if pending.get("explanation"):
        print(f"\n{pending['explanation']}\n")
    print(f"{ANSI[risk['color']]} {risk['level'].upper()} {ANSI['reset']} {risk['label']}")
    print(json.dumps(pending["call"], indent=2))
    if not review["findings"]:
        print("Call checker found no problems.")
    for f in review["findings"]:
        print(f"  {f['level'].upper()}: {f['message']} (source: {f['source']})")


def ask_decision(call: dict) -> dict:
    while True:
        choice = input("\n[a]pprove, [e]dit, or [r]eject? ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve"}
        if choice in ("r", "reject"):
            return {"action": "reject", "reason": input("Reason (optional): ").strip()}
        if choice in ("e", "edit"):
            print("Paste the edited call as one line of JSON. The current call is:")
            print(json.dumps(call))
            try:
                return {"action": "edit", "call": json.loads(input("> "))}
            except json.JSONDecodeError as exc:
                print(f"That isn't valid JSON ({exc}). Try again.")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    question = " ".join(sys.argv[1:])
    try:
        agent = TradeDocsAgent()
        turn = agent.start(question)
        pages = {c["url"]: c["title"] for c in turn.state["chunks"]}
        print("Searched the docs and found:")
        for url, title in pages.items():
            print(f"  {title}  {url}")
        while not turn.done:
            show_pending(turn.pending)
            turn = agent.resume(turn.thread_id, ask_decision(turn.pending["call"]))
            last = turn.state["calls"][-1]
            if "result" in last:
                result = last["result"]
                print(f"\nHTTP {result['status_code']}")
                print(json.dumps(result["data"], indent=2)[:3000])
    except IndexMissingError as exc:
        sys.exit(str(exc))
    except anthropic.AuthenticationError:
        sys.exit("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.AnthropicError as exc:
        sys.exit(f"Claude API error: {exc}")
    print(f"\n{turn.state.get('answer', '')}")


if __name__ == "__main__":
    main()
