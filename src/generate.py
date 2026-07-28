"""Generator (DECISIONS D-009 / FORKS F6): the answer-prompt contract.

The generation prompt is HALF the measurement instrument (F6), so its text is versioned
(PROMPT_CONTRACT_VERSION) and snapshotted into every run config. Contract (F6 option ii +
D-009):
  - answer ONLY from the provided reports; if the fact isn't there, refuse ("I don't know")
  - document-level citations: tag each claim with the report it came from ([<doc_id>]).
    Source-level [cite: N] mapping stays deferred (D-009).

Model = claude-haiku-4-5 (Default h; per-run variable, config-snapshotted). Haiku 4.5 accepts
temperature=0 (Default g) -- unlike the newer Sonnet-5/Opus tiers, which reject non-default
sampling (see eval/judge.py). No thinking: the dumbest baseline answers straight from context.

Contamination note (D-013): the contract says "the provided reports" and nothing else, so the
groundedness judge (D-010) can catch answers sourced from Haiku's pretraining instead.
"""
from __future__ import annotations

import os

# Bump this whenever the contract wording changes; a change re-defines the system-under-test,
# so runs under different versions are not comparable (config snapshot pins it).
PROMPT_CONTRACT_VERSION = "f6-v1"

# Default h. Not locked -- varying the generator is the point; the run config records it.
GENERATOR_MODEL = "claude-haiku-4-5"

# Short factual answers over <=3 whole docs; 1024 is ample and keeps us off the streaming path.
# TUNABLE(1024; revisit if answers truncate -- stop_reason == "max_tokens")
MAX_TOKENS = 1024

# The exact refusal sentence the contract mandates (rule 2 below). Extracted as a named
# constant so the deterministic refusal detector (eval/run.is_exact_refusal, D-019) checks for
# the SAME string the prompt demands -- the two can never silently drift apart. Interpolated in
# place below, so the SYSTEM text stays byte-identical to f6-v1 (contract version unchanged).
REFUSAL_STRING = "I don't know based on the provided reports."

SYSTEM = (
    "You answer questions about a corpus of professional dossiers on business leaders, using "
    "ONLY the reports provided in the user message.\n"
    "Rules:\n"
    "1. Use only the provided reports. Do not use any outside or prior knowledge, even if you "
    "recognize the person.\n"
    "2. If the answer is not stated in the provided reports, say you don't know -- reply "
    f'exactly: "{REFUSAL_STRING}" Do not guess or infer beyond '
    "what is written.\n"
    "3. Cite your source: after each claim, name the report(s) it came from in square "
    "brackets, e.g. [Jane Rivera]. Every factual sentence must carry a citation.\n"
    "4. Be concise."
)


def has_credentials() -> bool:
    """A run can score retrieval offline; only generate + judge need the API."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def format_context(retrieved: list[dict]) -> str:
    """Concatenate the retrieved whole-doc chunks, each tagged with its doc_id so the model
    can cite at document level (D-009). Order = retrieval rank. Public so the groundedness
    judge (D-010) scores the EXACT context string the generator saw."""
    blocks = [f"[{r['doc_id']}]\n{r['text']}" for r in retrieved]
    return "\n\n".join(blocks)


def usage_meta(resp, model: str, latency_s: float) -> dict:
    """Per-call token usage + latency, as a plain JSON-able dict (Tier-1 item 4). cache_* are
    0 here (these calls set no cache_control) but are recorded so Phase-5 monitoring gets them
    for free. Pricing lives in eval/cost.py, applied by the eval harness -- not here."""
    u = resp.usage
    return {
        "model": model,
        "latency_s": round(latency_s, 3),
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


class Generator:
    """Holds one Anthropic client. Answers a question from the retrieved context, or refuses."""

    def __init__(self, model: str = GENERATOR_MODEL):
        import anthropic  # imported lazily so offline (hit-rate-only) runs never need the SDK

        self.model = model
        self._client = anthropic.Anthropic()

    def answer(self, question: str, retrieved: list[dict]) -> tuple[str, dict]:
        """Returns (answer_text, meta). meta carries per-call token usage + wall-clock latency
        (Tier-1 item 4), as RAW numbers -- eval/cost.py turns them into dollars. Kept
        pricing-free so the generator (system-under-test) never imports the eval harness."""
        import time

        user = f"Reports:\n\n{format_context(retrieved)}\n\nQuestion: {question}"
        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0,  # Default g; accepted on Haiku 4.5
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        latency_s = time.perf_counter() - t0
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text, usage_meta(resp, self.model, latency_s)
