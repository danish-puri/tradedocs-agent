"""Grade one agent run against one test question.

Every check is plain code, not another LLM judging, so the same answer always
gets the same grade and every failure comes with a reason a person can read.
"""

from __future__ import annotations

import re


def _norm(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def values_match(expected, actual) -> bool:
    """Compare one expected value. Lists mean any of; dicts compare by key."""
    if isinstance(expected, list):
        return any(values_match(option, actual) for option in expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and values_match(value, actual[key]) for key, value in expected.items()
        )
    return _norm(expected) == _norm(actual)


def path_matches(expected: str, actual: str) -> bool:
    pattern = "^" + re.sub(r"\\{[^}]+\\}", "[^/]+", re.escape(expected.rstrip("/"))) + "$"
    return re.match(pattern, str(actual).rstrip("/"), re.I) is not None


def call_mismatch(expected: dict, actual: dict) -> str | None:
    """Return None if the call matches, otherwise the first difference."""
    if str(actual.get("method", "")).upper() != expected["method"].upper():
        return f"method {actual.get('method')} instead of {expected['method']}"
    if not path_matches(expected["path"], actual.get("path", "")):
        return f"path {actual.get('path')} instead of {expected['path']}"
    for part in ("query", "body"):
        for key, value in (expected.get(part) or {}).items():
            got = (actual.get(part) or {}).get(key)
            if got is None:
                return f"{part} is missing {key}"
            if not values_match(value, got):
                return f"{part}.{key} is {got!r}, expected {value!r}"
    return None


def missing_mentions(mentions: list, answer: str) -> list[str]:
    text = answer.lower()
    missing = []
    for item in mentions:
        options = item if isinstance(item, list) else [item]
        if not any(str(option).lower() in text for option in options):
            missing.append(" or ".join(map(str, options)))
    return missing


def cited(pages: list[str], answer: str, titles: dict[str, str]) -> bool:
    text = answer.lower()
    for slug in pages:
        if f"/{slug.lower()}" in text:
            return True
        title = titles.get(slug)
        if title and title.lower() in text:
            return True
    return False


def grade(question: dict, run: dict, titles: dict[str, str]) -> dict:
    """run holds: retrieved (list of slugs), calls (list of proposed calls with
    their checker review), and answer (the final text)."""
    checks: dict[str, bool] = {}
    why: list[str] = []
    pages = question.get("pages", [])

    checks["retrieval"] = any(slug in run["retrieved"] for slug in pages)
    if not checks["retrieval"]:
        why.append(f"search didn't return any of {pages} (got {sorted(set(run['retrieved']))})")

    checks["citation"] = cited(pages, run["answer"], titles)
    if not checks["citation"]:
        why.append(f"answer doesn't cite any of {pages}")

    expected = question.get("call")
    matched = None
    if expected:
        problems = []
        for record in run["calls"]:
            problem = call_mismatch(expected, record["call"])
            if problem is None:
                matched = record
                break
            problems.append(problem)
        checks["call"] = matched is not None
        if not checks["call"]:
            detail = "; ".join(problems) if problems else "no call was proposed"
            why.append(f"no proposed call matched {expected['method']} {expected['path']} ({detail})")

    missing = missing_mentions(question.get("mentions", []), run["answer"])
    checks["mentions"] = not missing
    if missing:
        why.append("answer doesn't mention " + ", ".join(repr(m) for m in missing))

    checker_errors = [
        f["message"] for record in run["calls"] for f in record.get("review", {}).get("findings", []) if f["level"] == "error"
    ]
    return {
        "id": question["id"],
        "question": question["question"],
        "passed": all(checks.values()),
        "checks": checks,
        "why": why,
        "checker_errors": checker_errors,
        "matched_call_clean": None if matched is None else not any(
            f["level"] == "error" for f in matched.get("review", {}).get("findings", [])
        ),
    }
