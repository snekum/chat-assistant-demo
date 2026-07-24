"""LLM-as-judge (DECISIONS D-010): the measurement "ruler".

Pinned Sonnet-class judge, NEVER the generator's Haiku tier (self-preference guard, D-010).
Two lanes with DELIBERATELY DIFFERENT inputs -- this separation is the whole point:

  - GROUNDEDNESS lane (PRIMARY): sees question + retrieved context + answer, and NEVER the
    gold answer. If the answer key reached this lane it would leak into the ruler and we'd be
    scoring truth-in-the-world instead of support-in-the-provided-reports (the D-013
    contamination trap). Rubric wording: "supported by the provided reports ALONE." This lane
    also classifies is_refusal, which is a property of the answer alone -- so refusal
    detection never sees gold either.

  - CORRECTNESS lane (SECONDARY): sees question + gold + answer. For an abstention question the
    gold IS the expected refusal, so correctness there = "did it appropriately decline".

Rubric text is versioned (RUBRIC_VERSION) and snapshotted in the run config. Changing it
silently re-scores all history -> on a bump, re-score ALL runs under the new version, never
mix (D-010).

MODEL BEHAVIOR (verified against the named model, not assumed):
  claude-sonnet-5 REJECTS a non-default temperature (400), so D-010's literal "temp 0" is not
  settable on the current Sonnet. We omit temperature (temp 0 never guaranteed identical
  outputs anyway) and pin stability another way: fixed rubric + disabled thinking + structured
  output (json_schema). The judge stays Sonnet-class and never Haiku, so D-010's real
  invariant -- the ruler outranks the generator's tier -- holds. If literal temp-0
  determinism is required, the fallback is to pin the judge at claude-sonnet-4-6 (accepts
  temperature) and drop structured output; not taken here.
"""
from __future__ import annotations

# Bump on ANY rubric-wording change; re-score all runs under the new version (never mix).
RUBRIC_VERSION = "d010-v1"

# Sonnet-class, never the generator's Haiku tier (D-010). Config-snapshotted.
JUDGE_MODEL = "claude-sonnet-5"

MAX_TOKENS = 1024  # small JSON verdicts

_GROUNDEDNESS_SYSTEM = (
    "You are a strict evaluator of a retrieval-augmented answer. You are given a QUESTION, the "
    "REPORTS that were retrieved for it, and an ANSWER produced from them. You do NOT get a "
    "reference answer -- judge only against the reports.\n\n"
    "Return two judgments:\n"
    "- is_refusal: true if the ANSWER declines to answer (e.g. says it doesn't know, or that "
    "the information isn't in the reports) instead of asserting a substantive answer. An answer "
    "that states facts is NOT a refusal even if it also hedges.\n"
    "- grounded: true if EVERY factual claim in the ANSWER is directly supported by the "
    "provided REPORTS ALONE. Mark false if any claim is unsupported by the reports, "
    "contradicts them, or appears to come from outside knowledge -- even if it is true in the "
    "real world. A refusal makes no claims, so treat it as grounded (true).\n"
    "Give a one-sentence reason."
)

_GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "is_refusal": {"type": "boolean"},
        "grounded": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_refusal", "grounded", "reason"],
    "additionalProperties": False,
}

_CORRECTNESS_SYSTEM = (
    "You are evaluating whether an ANSWER matches a reference GOLD answer for a QUESTION.\n"
    "- If the gold answer is a refusal (the fact is absent from the corpus), then the answer "
    "is correct only if it also declines / says it doesn't know.\n"
    "- Otherwise the answer is correct if it conveys the same factual content as the gold, "
    "allowing paraphrase and extra citation markers. Ignore wording, formatting, and citation "
    "tags; judge the facts.\n"
    "Return correct (boolean) and a one-sentence reason."
)

_CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


class Judge:
    """Holds one Anthropic client; runs the two lanes. Thinking disabled + structured output
    for stability, since temperature is not tunable on the Sonnet-5 judge (see module docstring)."""

    def __init__(self, model: str = JUDGE_MODEL):
        import anthropic  # lazy: offline hit-rate-only runs never construct a judge

        self.model = model
        self._client = anthropic.Anthropic()

    def _call(self, system: str, user: str, schema: dict) -> dict:
        import json

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            thinking={"type": "disabled"},  # cheaper + steadier; accepted on Sonnet 5
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    def groundedness(self, question: str, context: str, answer: str) -> dict:
        """PRIMARY lane. Sees context + answer, NEVER gold. -> {is_refusal, grounded, reason}."""
        user = (
            f"QUESTION:\n{question}\n\nRETRIEVED REPORTS:\n{context}\n\nANSWER:\n{answer}"
        )
        return self._call(_GROUNDEDNESS_SYSTEM, user, _GROUNDEDNESS_SCHEMA)

    def correctness(self, question: str, gold: str, answer: str) -> dict:
        """SECONDARY lane. Sees gold + answer. -> {correct, reason}."""
        user = f"QUESTION:\n{question}\n\nGOLD ANSWER:\n{gold}\n\nANSWER:\n{answer}"
        return self._call(_CORRECTNESS_SYSTEM, user, _CORRECTNESS_SCHEMA)
