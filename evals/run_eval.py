"""Run the 20-question eval and print a score.

  python -m evals.run_eval                    full run (needs ANTHROPIC_API_KEY)
  python -m evals.run_eval --retrieval-only   only grades search (no keys needed)
  python -m evals.run_eval --ids q01 q05      run a few questions

Nothing is sent to Alpaca during the eval. Each proposed call is recorded and
graded, then rejected, so the agent finishes its answer without running it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tradedocs.config import DOCS_DIR, EFFORT, MODEL, TOP_K, read_sources
from tradedocs.docs import page_slug, parse_page
from tradedocs.grading import grade

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "questions.yaml"
RESULTS_DIR = HERE / "results"
EVAL_REASON = (
    "This is an automated grading run, so calls are not sent. Finish your answer without running it, "
    "and show the exact request the developer should send."
)

# List prices in dollars per million tokens: input, output, cache read, cache write.
# Used only to print a rough cost estimate.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
}


def load_questions(ids: list[str] | None = None) -> list[dict]:
    questions = yaml.safe_load(QUESTIONS_FILE.read_text())
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def page_titles() -> dict[str, str]:
    titles = {}
    for url in read_sources():
        path = DOCS_DIR / f"{page_slug(url)}.md"
        if path.exists():
            titles[page_slug(url)] = parse_page(url, path.read_text()).title
    return titles


def save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2))
    return path


# ------------------------------------------------------------ retrieval


def run_retrieval(questions: list[dict]) -> dict:
    from tradedocs.store import DocSearch

    searcher = DocSearch()
    search = searcher.search
    rows = []
    for q in questions:
        slugs = [hit["slug"] for hit in search(q["question"])]
        rank = next((i + 1 for i, slug in enumerate(slugs) if slug in q["pages"]), None)
        rows.append({"id": q["id"], "question": q["question"], "expected": q["pages"], "retrieved": slugs, "rank": rank})

    n = len(rows)
    found = [r for r in rows if r["rank"] is not None]
    summary = {
        "questions": n,
        "search_mode": searcher.mode,
        "top_k": TOP_K,
        "hit_at_k": len(found),
        "hit_at_3": sum(r["rank"] <= 3 for r in found),
        "hit_at_1": sum(r["rank"] == 1 for r in found),
    }
    print(f"\nRetrieval over {n} questions ({searcher.mode} search, top {TOP_K} chunks each)")
    print(f"  right page in top {TOP_K}: {summary['hit_at_k']}/{n}")
    print(f"  right page in top 3: {summary['hit_at_3']}/{n}")
    print(f"  right page first:    {summary['hit_at_1']}/{n}")
    misses = [r for r in rows if r["rank"] is None]
    if misses:
        print("\nMisses")
        for r in misses:
            print(f"  {r['id']} {r['question']}\n      wanted {r['expected']}, got {sorted(set(r['retrieved']))}")
    return {"summary": summary, "rows": rows}


# ----------------------------------------------------------- full agent


def run_agent(questions: list[dict]) -> dict:
    from tradedocs.agent import TradeDocsAgent

    agent = TradeDocsAgent()
    titles = page_titles()
    results = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    for q in questions:
        started = time.time()
        turn = agent.start(q["question"])
        while not turn.done:
            turn = agent.resume(turn.thread_id, {"action": "reject", "reason": EVAL_REASON})
        state = turn.state
        run = {
            "retrieved": [c["slug"] for c in state["chunks"]],
            "calls": [{"call": c["call"], "review": c.get("review", {})} for c in state.get("calls", [])],
            "answer": state.get("answer", ""),
        }
        for usage in state.get("usage", []):
            for key in usage_total:
                usage_total[key] += usage.get(key) or 0
        result = grade(q, run, titles)
        result.update(run=run, seconds=round(time.time() - started, 1), stop_reason=state.get("stop_reason"))
        results.append(result)
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"{mark} {q['id']} ({result['seconds']}s) {q['question']}")
        for reason in result["why"]:
            print(f"       {reason}")

    n = len(results)
    passed = sum(r["passed"] for r in results)
    check_names = ["retrieval", "citation", "call", "mentions"]
    per_check = {}
    for name in check_names:
        graded = [r for r in results if name in r["checks"]]
        per_check[name] = {"passed": sum(r["checks"][name] for r in graded), "graded": len(graded)}

    proposed = sum(len(r["run"]["calls"]) for r in results)
    flagged = sum(
        1 for r in results for c in r["run"]["calls"] if any(f["level"] == "error" for f in c["review"].get("findings", []))
    )
    cost = None
    if MODEL in PRICES:
        p_in, p_out, p_read, p_write = PRICES[MODEL]
        cost = round(
            (usage_total["input_tokens"] * p_in + usage_total["output_tokens"] * p_out
             + usage_total["cache_read_input_tokens"] * p_read + usage_total["cache_creation_input_tokens"] * p_write) / 1e6,
            2,
        )

    print(f"\nScore {passed}/{n} with {MODEL} (effort {EFFORT})")
    for name, row in per_check.items():
        print(f"  {name:10s} {row['passed']}/{row['graded']}")
    print(f"  call checker flagged {flagged} of {proposed} proposed calls")
    print(f"  tokens in {usage_total['input_tokens']:,} (+{usage_total['cache_read_input_tokens']:,} cached), out {usage_total['output_tokens']:,}")
    if cost is not None:
        print(f"  estimated cost ${cost} at list prices")

    summary = {
        "model": MODEL, "effort": EFFORT, "questions": n, "passed": passed, "per_check": per_check,
        "proposed_calls": proposed, "calls_flagged_by_checker": flagged, "usage": usage_total, "estimated_cost_usd": cost,
    }
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="grade search only, no API keys needed")
    parser.add_argument("--ids", nargs="*", help="only run these question ids")
    args = parser.parse_args()

    questions = load_questions(args.ids)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.retrieval_only:
        report = run_retrieval(questions)
        path = save("latest_retrieval.json", {"ran_at": stamp, **report})
    else:
        report = run_agent(questions)
        path = save("latest.json", {"ran_at": stamp, **report})
    print(f"\nSaved {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
