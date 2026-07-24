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

SYSTEM = (
    "You answer questions about a corpus of professional dossiers on business leaders, using "
    "ONLY the reports provided in the user message.\n"
    "Rules:\n"
    "1. Use only the provided reports. Do not use any outside or prior knowledge, even if you "
    "recognize the person.\n"
    "2. If the answer is not stated in the provided reports, say you don't know -- reply "
    "exactly: \"I don't know based on the provided reports.\" Do not guess or infer beyond "
    "what is written.\n"
    "3. Cite your source: after each claim, name the report(s) it came from in square "
    "brackets, e.g. [Aaron Silva]. Every factual sentence must carry a citation.\n"
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


class Generator:
    """Holds one Anthropic client. Answers a question from the retrieved context, or refuses."""

    def __init__(self, model: str = GENERATOR_MODEL):
        import anthropic  # imported lazily so offline (hit-rate-only) runs never need the SDK

        self.model = model
        self._client = anthropic.Anthropic()

    def answer(self, question: str, retrieved: list[dict]) -> str:
        user = f"Reports:\n\n{format_context(retrieved)}\n\nQuestion: {question}"
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0,  # Default g; accepted on Haiku 4.5
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
