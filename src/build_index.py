"""Build the pgvector index: persons.jsonl + chunks.jsonl -> Postgres (persons + chunks).

Embeddings come from the on-disk cache (src/embedder.py), so re-runs are ~instant and
never re-pay the 49-min CPU embed. Idempotent (ON CONFLICT upsert) -> safe to re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import store  # noqa: E402
from embedder import NomicLocal  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def main() -> None:
    persons = load_jsonl("data/parsed/persons.jsonl")
    chunks = load_jsonl("data/parsed/chunks.jsonl")

    emb = NomicLocal()  # cache hit -> fast; no re-embed
    vecs = emb.embed_documents([c["text"] for c in chunks])
    assert vecs.shape == (len(chunks), emb.dim), vecs.shape

    conn = store.connect(init=True)
    np_persons = store.upsert_persons(conn, persons)
    n_chunks = store.upsert_chunks(conn, chunks, vecs, emb.model_id)

    got_p = conn.execute("SELECT count(*) FROM persons").fetchone()[0]
    got_c = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"upserted persons={np_persons} chunks={n_chunks}")
    print(f"db now holds persons={got_p} chunks={got_c} (embedder={emb.model_id} dim={emb.dim})")


if __name__ == "__main__":
    main()
