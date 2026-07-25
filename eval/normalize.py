"""Shared match-normalizer for D-011 (hit-rate span matching).

SINGLE SOURCE OF TRUTH: both the question validator and the hit-rate scorer import this,
so gold quotes are validated under the exact rule they'll later be scored under. Changing
this function silently re-scores all hit-rate history -- treat it like the judge rubric.

D-011 decision: normalize-then-exact CONTAINMENT. Normalize BOTH the gold quote and the
retrieved text (strip [cite:...] markers, lowercase, drop punctuation/markdown, collapse
whitespace), then require the full normalized quote to appear in the normalized text.

Corpus census (2026-07-18) forced the cite regex: ~20 marker variants exist -- [cite: N],
[cite: source: N], [cite: Internal Context], ... -- so we strip anything up to the closing
bracket, NOT the literal "[cite: N]". Cite-strip runs FIRST so the words inside a marker
(e.g. "cite", "source") never leak into the normalized text.
"""
from __future__ import annotations

import re

# Snapshot this like the judge rubric / prompt contract (repro item 8). The path string in
# config.json never changes when the LOGIC below does, so a normalizer edit would silently
# re-score all hit-rate history with no config diff to show for it. Bump this BY HAND whenever
# normalize_for_match changes; run.py records it, so a change is visible in a config diff.
# TUNABLE(hand-bumped on any normalize_for_match change; symptom skipped: two runs disagree on
#   hit@k with identical questions.sha256 + embedder + k -> a silent normalizer change is the suspect.)
NORMALIZER_VERSION = "norm-v1"

_CITE = re.compile(r"\[cite:[^\]]*\]")


def normalize_for_match(text: str) -> str:
    text = _CITE.sub(" ", text)
    text = text.lower()
    # Unicode-safe: keep letters/digits (incl. accented), everything else -> space.
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def quote_hits(quote: str, doc_text: str) -> bool:
    """True if the normalized gold quote appears in the normalized doc/chunk text."""
    nq = normalize_for_match(quote)
    return bool(nq) and nq in normalize_for_match(doc_text)
