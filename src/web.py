"""Web search provider (D-027: an external search API returning results, never prose).

Tavily, behind one function -- `search()` -- so the provider can change without touching the
agent or the tools. Chosen over Claude's built-in web search because this code must SEE the
outgoing query before it leaves (the egress check) and READ the results it gets back (the
built-in returns page text encrypted), and over Brave because Tavily returns longer
query-matched passages and needs no card for the free tier.

Every live response is recorded under data/web_fixtures/, keyed by the exact request, so an
eval can replay precisely what the agent saw (D-027: recorded fixtures = eval mode, live =
demo). WEB_MODE=replay reads fixtures only and treats a miss as "web unavailable" -- an eval
must never quietly fall through to the live web.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from contracts import WebEvidence

TAVILY_URL = "https://api.tavily.com/search"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "web_fixtures"

# TUNABLE(5 results: enough to cross-check a claim across outlets, small enough that a search
#   adds ~2k tokens to the turn. Symptom wrong: answers lean on one outlet -> raise; turn input
#   tokens dominated by web text -> lower.)
MAX_RESULTS = 5
# TUNABLE(1,500 chars per result: Tavily joins up to 3 ~500-char passages per page; the cap
#   also bounds how much untrusted page text reaches the model. Symptom wrong: answers say
#   "reports suggest" without the specifics the page had -> raise.)
SNIPPET_CHARS = 1500
# TUNABLE(10 s: Tavily basic search usually answers in 1-3 s. Symptom wrong: web_unavailable
#   in traces with a timeout cause while the API is healthy -> raise.)
TIMEOUT_S = 10.0


class WebSearchError(RuntimeError):
    """Search could not run (no key, HTTP failure, replay miss). The tool turns this into an
    honest "web unavailable" result; it never crashes the turn."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _request(query: str, topic: str, time_range: str | None) -> dict:
    body = {
        "query": query,
        "topic": topic,
        "search_depth": "basic",          # 1 credit; "advanced" is 2
        "max_results": MAX_RESULTS,
        "include_answer": False,          # D-027: evidence only -- one writer in the system
        "include_raw_content": False,     # passages, not whole pages
        "include_published_date": True,
    }
    if time_range:
        body["time_range"] = time_range
    return body


def _fixture_path(body: dict) -> Path:
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return FIXTURE_DIR / f"{key}.json"


def _fetch_live(body: dict) -> dict:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise WebSearchError("TAVILY_API_KEY is not set")
    try:
        r = httpx.post(TAVILY_URL, json=body, timeout=TIMEOUT_S,
                       headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        raise WebSearchError(f"{type(exc).__name__}: {exc}") from exc
    if r.status_code != 200:
        raise WebSearchError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


# The model picks one of two kinds per search. The first probe (analysis/web_probe_*.md,
# 2026-10-13) showed each fixed setting failing one question shape: news-past-month went
# off-topic on a niche impact query (remittances -> IRS deadlines, oil prices) while general
# web stayed on-topic but surfaced 2025 explainers for "latest tariffs".
KINDS: dict[str, tuple[str, str | None]] = {
    # TUNABLE(past month for "what is happening now". Symptom wrong: news answers miss a
    #   development from 5-6 weeks ago that is still the story -> try a 60-day start_date
    #   ("year", the next preset, is too wide).)
    "news": ("news", "month"),
    "background": ("general", None),     # analysis and explainers, any date
}


def _iso_date(raw: str | None) -> str | None:
    """Tavily dates arrive as RFC 2822 ("Wed, 30 Sep 2026 14:04:46 GMT"); citations and
    "as of" phrasing want 2026-09-30. Unparseable -> None, never a guess."""
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _clean(text: str) -> str:
    # Pages leak headings and nav ("## Latest News", "Skip to content") into passages;
    # one line per snippet keeps the tool result readable for the model and the trace.
    return " ".join((text or "").split())


def search(query: str, kind: str = "news") -> list[WebEvidence]:
    """-> typed web evidence for one query. Raises WebSearchError when search cannot run."""
    if kind not in KINDS:
        raise WebSearchError(f"unknown search kind {kind!r}; expected one of {sorted(KINDS)}")
    body = _request(query, *KINDS[kind])
    path = _fixture_path(body)
    if os.environ.get("WEB_MODE", "live") == "replay":
        if not path.exists():
            raise WebSearchError(f"no recorded fixture for this request ({path.name})")
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        record = {"request": body, "retrieved_at": _now(), "response": _fetch_live(body)}
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")

    return [
        WebEvidence(
            url=r.get("url", ""),
            title=_clean(r.get("title")),
            snippet=_clean(r.get("content"))[:SNIPPET_CHARS],
            query=query,
            published=_iso_date(r.get("published_date")),
            retrieved_at=record["retrieved_at"],
        )
        for r in record["response"].get("results", [])
    ]
