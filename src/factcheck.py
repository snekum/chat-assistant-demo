"""Fact-check a drafted message before the member is asked to approve it (D-057).

A message goes out in the member's name to a real person, and the member cannot open the
recipient's dossier to check it. So before the approval card, a model from a DIFFERENT provider
(the drafter is Claude) reads the draft against the two dossiers and the member's own words and
returns every claim they do not support. The agent redrafts once without them; anything still
unsupported reaches the card marked.

Called over plain HTTP like the web search (no SDK). One result per draft: the approval filter,
the card text and the tool all ask about the same draft, so results are cached by content.
A failed call never blocks sending -- the card then says the check could not run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx

URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-5.4-mini"
# TUNABLE(low reasoning: the job is careful reading of ~5k tokens, not multi-step reasoning.
#   Symptom wrong: calibration misses presuppositions ("how you scaled X internationally") ->
#   raise to "medium" and re-measure latency.)
REASONING_EFFORT = "low"
# TUNABLE(15 s: a check measured at ~1-3 s; past this the card opens with "could not check" rather
#   than leaving the member waiting. Symptom wrong: timeouts in normal use -> raise.)
TIMEOUT_S = 15
# Secondary-source prices (USD per million tokens, 2026-06 listings); verify on OpenAI's page.
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.75, 0.075, 4.50
VERSION = "factcheck-v1"   # bump when SYSTEM changes: calibration results belong to one version

SYSTEM = """You check a short message that one member of a private business network is about to send to another member. Find every claim in the message that the sources do not support.

- Claims about the RECIPIENT must be supported by the recipient's profile.
- Claims about the SENDER (first person: I, we, my company, our) must be supported by the sender's profile or by what the sender said in the conversation.
- Count as claims: facts, numbers, roles, companies, places, dates and achievements; the sender's plans, projects, funding rounds, interests and past actions ("I've been following your work"); and presuppositions ("how you scaled the company internationally" claims it scaled internationally).
- Not claims: greetings and sign-offs, the request itself (a call, a meeting, advice), politeness, and the sender's reaction to a supported fact ("impressive").
- Wording may differ from the sources; judge meaning. A detail that goes beyond the sources (a bigger number, an extra country, a stated plan where the sender only asked a question) is unsupported.

Return each unsupported claim as an exact quote from the message, who it is about, and a short reason. Return an empty list if everything is supported."""

SCHEMA = {
    "name": "fact_check",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["unsupported"],
        "properties": {"unsupported": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["quote", "about", "reason"],
            "properties": {"quote": {"type": "string"},
                           "about": {"type": "string", "enum": ["recipient", "sender"]},
                           "reason": {"type": "string"}}}}},
    },
}


@dataclass
class FactCheck:
    ok: bool | None                        # None = the check could not run
    unsupported: list[dict] = field(default_factory=list)
    error: str | None = None
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    cost_usd: float = 0.0


def _clean(text: str) -> str:
    return re.sub(r"\[cite:[^\]]*\]", "", text)


def prompt(draft: str, recipient_doc: str, sender_doc: str, sender_words: list[str]) -> str:
    said = "\n".join(f"- {w}" for w in sender_words) or "- (nothing)"
    return (f"RECIPIENT PROFILE:\n{_clean(recipient_doc)}\n\nSENDER PROFILE:\n{_clean(sender_doc)}"
            f"\n\nWHAT THE SENDER SAID IN THIS CONVERSATION:\n{said}\n\nMESSAGE:\n{draft}")


_cache: OrderedDict[str, FactCheck] = OrderedDict()
CACHE_SIZE = 256


def check(draft: str, recipient_doc: str, sender_doc: str, sender_words: list[str],
          model: str = MODEL, use_cache: bool = True) -> FactCheck:
    """use_cache=False forces a fresh call (calibration measures run-to-run stability)."""
    user = prompt(draft, recipient_doc, sender_doc, sender_words)
    key = hashlib.sha256(f"{VERSION}|{model}|{user}".encode()).hexdigest()
    if use_cache and key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    t0 = time.perf_counter()
    try:
        r = httpx.post(URL, timeout=TIMEOUT_S,
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                       json={"model": model, "reasoning_effort": REASONING_EFFORT,
                             "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": user}],
                             "response_format": {"type": "json_schema", "json_schema": SCHEMA}})
        r.raise_for_status()
        body = r.json()
        found = json.loads(body["choices"][0]["message"]["content"])["unsupported"]
        u = body.get("usage", {})
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        usage = {"input_tokens": u.get("prompt_tokens", 0), "cached_tokens": cached,
                 "output_tokens": u.get("completion_tokens", 0)}
        usd = ((usage["input_tokens"] - cached) * PRICE_IN + cached * PRICE_CACHED
               + usage["output_tokens"] * PRICE_OUT) / 1e6
        result = FactCheck(ok=not found, unsupported=found, usage=usage,
                           seconds=time.perf_counter() - t0, cost_usd=round(usd, 6))
    except Exception as exc:  # the card says the check could not run; nothing is blocked
        # Cached too: the same draft is asked about three times, and three timeouts in a row
        # would keep the member waiting 45 s for one card.
        result = FactCheck(ok=None, error=f"{type(exc).__name__}: {str(exc)[:200]}",
                           seconds=time.perf_counter() - t0)
    _cache[key] = result
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return result
