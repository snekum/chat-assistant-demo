"""hit@k span-matching (DECISIONS D-011 / FORKS Default m).

A gold quote HITS if it appears -- normalize-then-exact containment (eval/normalize.py, the
single source of truth shared with the question validator) -- inside the GOLD doc's OWN
retrieved chunk (doc_id-anchored; D-011 amendment 2026-07-24 -- see hit_at_k).
This compares SOURCE TEXT against SOURCE TEXT (quote vs retrieved doc), distinct from
groundedness (D-010), which judges the paraphrased generated answer.

hit@k for a question = ALL of its gold spans appear within the top-k retrieved chunks.

Multi-hop 'all-spans' choice (the one open sub-fork D-011 didn't pin down):
  A multi-hop answer needs every cited doc, so a retrieval "hit" should mean retrieval handed
  the generator ALL required spans -- not just one. Consequence: a 2-doc multi-hop question
  can never hit@1 (two docs can't both rank #1), so multi-hop hit@1 is a structural 0 here.
  That is not a bug: it's exactly the "multi-hop needs k>1 / agentic retrieval" signal the
  eval exists to surface (a deferred fork). Report it honestly.
  # TUNABLE(all-spans for multi-hop; symptom it's wrong = the all-spans bar hides real
  #   partial-retrieval progress on multi-hop -> switch to per-span fraction or any-span,
  #   revisit when multi-hop hit@3 stays flat at 0 while single-hop is healthy)

Abstention questions have no gold span (evidence == []); their retrieval hit-rate is undefined
and they are EXCLUDED from hit-rate aggregation (Default m reports per type; abstention = N/A).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "eval")
from normalize import quote_hits  # noqa: E402


def hit_at_k(evidence: list[dict], retrieved: list[dict], k: int) -> bool:
    """True iff every gold span is found in some chunk among the top-k retrieved.

    evidence:  [{"doc_id", "quote"}, ...] from the gold set.
    retrieved: rank-ordered [{"text", ...}, ...] from store.search (index 0 == rank 1).
    """
    if not evidence:  # abstention -> no span to match; caller must not aggregate this
        raise ValueError("hit_at_k called on an evidence-less (abstention) question")
    topk = retrieved[:k]
    # doc_id + quote (D-011 amendment 2026-07-24): the gold span must appear in the GOLD doc's
    # OWN retrieved chunk -- not merely in some top-k chunk. Quote-only scored a FALSE HIT when a
    # generic gold phrase ("Santa Clara University", in 15 docs) sat in a DIFFERENT doc that got
    # retrieved while the gold doc did NOT -- hiding a real miss. doc_id pins the person; the quote
    # still pins the section (survives future section-chunking). Aligns hit@k with gold_rank, which
    # already matches on doc_id. Dormant until LLM-generated generic quotes appeared (unique
    # hand-authored quotes never triggered it).
    return all(
        any(r["doc_id"] == e["doc_id"] and quote_hits(e["quote"], r["text"]) for r in topk)
        for e in evidence
    )


def spans_found_at_k(evidence: list[dict], retrieved: list[dict], k: int) -> int:
    """How many of a question's gold spans appear in the top-k (0..len(evidence)).

    The partial-credit companion to the all-spans hit_at_k (Phase 1a, item 2). hit_at_k is
    binary -- a 2-span multi-hop that fetched 1 of 2 scores 0, identical to fetching neither;
    this returns 1, so span-recall = 1/2 exposes the partial progress the all-spans bar hides.
    Same normalize-then-exact span match (D-011); caller derives span_recall = found/total and
    the aggregate micro = sum(found)/sum(total)."""
    if not evidence:
        raise ValueError("spans_found_at_k called on an evidence-less (abstention) question")
    topk = retrieved[:k]  # doc_id + quote per the D-011 amendment (see hit_at_k)
    return sum(
        any(r["doc_id"] == e["doc_id"] and quote_hits(e["quote"], r["text"]) for r in topk)
        for e in evidence
    )
