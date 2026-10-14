"""Hybrid member search (D-051): meaning-based ranking and keyword ranking, merged.

Meaning search (the local embeddings) understands paraphrase -- "scaling a company" ~ "growing a
business" -- but blurs rare exact terms: a place, a regulator, "IPO". Keyword search (BM25) is the
reverse. Each ranks every person; reciprocal rank fusion merges the two lists by position, so
neither list's scores need calibrating against the other.

  meaning rank : a person's best SUM_TOP sections, similarities summed (rewards matching in several
                 sections -- the several-needs questions)
  keyword rank : a person's single best section by BM25
  evidence     : each top person's sections are fused the same way, so the passages the model is
                 shown include the one that actually ranked them (a "Vietnam" hit must be on screen)

ONE implementation, used by tools.find_members AND eval/recommend_eval.py: the eval measures exactly
what members get. Measured (2026-10-14, 34 questions, answer key v3): ndcg@5 0.408 -> 0.505 overall,
but 0.507 -> 0.530 on the 10 real-user questions and not better by the reviewer's own labels
(66% vs 62% pairwise) -- shipped on the product judgement that real users also use specific terms.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable

import store

# TUNABLE(the standard reciprocal-rank-fusion constant; it damps the top ranks so one list cannot
#   dominate. Symptom wrong: results ~= one of the two lists alone -> try 20 or 100.)
RRF_K = 60
# BM25 (Okapi) standard constants. TUNABLE(symptom wrong: long sections dominate keyword hits ->
#   raise B; repeated words over-reward -> lower K1.)
BM25_K1, BM25_B = 1.2, 0.75
# TUNABLE(sum of a person's best 3 section similarities; 2 scored lower on the key, 1 = the old
#   single-best ranking. Symptom wrong: people with one strong, on-point section lose to people
#   with three vaguely related ones -> lower to 2.)
SUM_TOP = 3
_STOP = set("""a an and are as at be but by can could do does for from has have he her his how i
if in into is it its me my of on or our she should so than that the their them they this to was we
what which who whom why will with would you your here there network members member people person
someone anyone help""".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    """Okapi BM25 in memory (4,006 sections: build <1 s, query ~10 ms). Twenty lines instead of a
    dependency; Postgres full-text ranking is not BM25."""

    def __init__(self, docs: list[str]):
        self.tf = [Counter(tokens(d)) for d in docs]
        self.lens = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(tokens(query))
        out = []
        for c, length in zip(self.tf, self.lens):
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg)
            out.append(sum(self.idf[t] * c[t] * (BM25_K1 + 1) / (c[t] + norm)
                           for t in terms if t in c))
        return out


def rrf(*rankings: list[str]) -> list[str]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for i, item in enumerate(ranking):
            fused[item] += 1 / (RRF_K + i + 1)
    return sorted(fused, key=lambda k: -fused[k])


_INDEX: dict[str, tuple[BM25, list[str], list[str]]] = {}


def keyword_index(conn, scheme: str) -> tuple[BM25, list[str], list[str]]:
    """-> (index, chunk_ids, person_ids), built once per process per scheme. The chat service
    builds it at startup so the first question does not pay for it."""
    if scheme not in _INDEX:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, person_id, text FROM chunks WHERE chunk_scheme = %s "
                        "ORDER BY chunk_id", (scheme,))
            rows = cur.fetchall()
        _INDEX[scheme] = (BM25([r[2] for r in rows]), [r[0] for r in rows], [r[1] for r in rows])
    return _INDEX[scheme]


def rank_people(conn, qvec, query: str, scheme: str,
                keep: Callable[[str], bool] = lambda p: True,
                top: int = 5, per_person: int = 3) -> tuple[list[str], dict[str, list[dict]]]:
    """-> (every kept person, best first; the `top` people's sections, best first, `per_person`
    each). Section rows are store.search rows (score = meaning similarity)."""
    dense_rows = store.search(conn, qvec, k=100_000, scheme=scheme)     # every section, best-first
    by_person: dict[str, list[dict]] = defaultdict(list)
    for r in dense_rows:
        by_person[r["person_id"]].append(r)
    index, chunk_ids, _ = keyword_index(conn, scheme)
    kw = dict(zip(chunk_ids, index.scores(query)))

    people = [p for p in by_person if keep(p)]                          # meaning order, ties stable
    meaning = sorted(people, key=lambda p: -sum(r["score"] for r in by_person[p][:SUM_TOP]))
    keyword = sorted(people, key=lambda p: -max(kw[r["chunk_id"]] for r in by_person[p]))
    order = rrf(meaning, keyword)

    sections = {}
    for p in order[:top]:
        rows = {r["chunk_id"]: r for r in by_person[p]}
        by_kw = sorted(rows, key=lambda c: -kw[c])
        chosen = rrf(list(rows), by_kw)[:per_person]
        # Fusion alone left the best keyword section off screen for 39 of 170 people shown: the
        # model was handed someone without the passage that ranked them. Guarantee it a slot.
        if kw[by_kw[0]] > 0 and by_kw[0] not in chosen:
            chosen = chosen[:per_person - 1] + [by_kw[0]]
        sections[p] = [rows[c] for c in chosen]
    return order, sections
