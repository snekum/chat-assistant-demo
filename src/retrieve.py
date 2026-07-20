"""Retrieval: query -> embed_query -> top-k cosine over chunks (optionally person-filtered).

This is the RAG retrieval step only (no generation yet). person_id, when supplied, is the
D-016 pre-filter a router subagent would set after resolving a name.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")
import store  # noqa: E402
from embedder import NomicLocal  # noqa: E402

# Whole-doc baseline: 3 docs ~= 8.5k tok context, fits any generator.
# TUNABLE(3, revisit when multi-hop needs facts from >3 subjects) -- FORKS.md Default f
DEFAULT_K = 3


class Retriever:
    """Holds the embedder + connection so the model loads once (a service would keep this
    alive; loading NomicLocal per call costs ~17s)."""

    def __init__(self):
        self.emb = NomicLocal()
        self.conn = store.connect()

    def retrieve(self, query: str, k: int = DEFAULT_K, person_id: str | None = None) -> list[dict]:
        qvec = self.emb.embed_query(query)
        return store.search(self.conn, qvec, k=k, person_id=person_id)


if __name__ == "__main__":
    r = Retriever()
    q = "Who negotiates core IT contracts for community banks?"
    print(f"Q: {q}\n")
    for i, hit in enumerate(r.retrieve(q), 1):
        print(f"{i}. {hit['doc_id']:<24} score={hit['score']:.4f}  chunk={hit['chunk_id']}")
