"""Build the pgvector index: persons.jsonl + a chunks file -> Postgres (persons + chunks).

Embeddings come from the on-disk cache (src/embedder.py), so re-runs are ~instant and never
re-pay the CPU embed. Idempotent (ON CONFLICT upsert on the composite PK) -> safe to re-run.

D-023: chunks now carry chunk_scheme, so the SAME DB holds whole_doc + section + fixed<N> at
once and retrieval filters by scheme. Pass the scheme's chunks file:

  ./.venv/Scripts/python.exe src/build_index.py                              # whole_doc (default)
  ./.venv/Scripts/python.exe src/build_index.py data/parsed/chunks.section.jsonl
  ./.venv/Scripts/python.exe src/build_index.py data/parsed/chunks.fixed.jsonl
  ./.venv/Scripts/python.exe src/build_index.py --recreate                   # DROP+recreate first

--recreate DROPs the chunks table before building -- needed ONCE to migrate the pre-D-023 table
(single-column PK) to the composite (chunk_id, chunk_scheme) PK. Run it with the FIRST arm, then
build the other arms without it. Run artifacts on disk preserve baseline history regardless of DB
state, so dropping the index is safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import store  # noqa: E402
from embedder import NomicLocal  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--recreate"]
    recreate = "--recreate" in sys.argv
    chunks_path = args[0] if args else "data/parsed/chunks.jsonl"

    persons = load_jsonl("data/parsed/persons.jsonl")
    chunks = load_jsonl(chunks_path)
    scheme = chunks[0].get("chunk_scheme", "whole_doc")
    assert all(c.get("chunk_scheme", "whole_doc") == scheme for c in chunks), \
        "mixed chunk_scheme in one file -- one scheme per build"

    emb = NomicLocal()  # cache hit -> fast; a NEW scheme's texts are cache misses -> real embed
    # Embed `embed_text` when a scheme sets it (section_hdr: name-header prepended, D-023), else the
    # plain text. The STORED text (upsert below) is always verbatim `text` -> D-011 scorer untouched.
    vecs = emb.embed_documents([c.get("embed_text", c["text"]) for c in chunks])
    assert vecs.shape == (len(chunks), emb.dim), vecs.shape

    conn = store.connect(init=False)
    if recreate:
        conn.execute("DROP TABLE IF EXISTS chunks")
        print("dropped chunks (composite-PK migration / clean rebuild)")
    conn.execute(store.SCHEMA)  # idempotent create (persons untouched if present)
    from pgvector.psycopg import register_vector
    register_vector(conn)

    store.upsert_persons(conn, persons)
    n_chunks = store.upsert_chunks(conn, chunks, vecs, emb.model_id)

    got = conn.execute(
        "SELECT chunk_scheme, count(*) FROM chunks GROUP BY chunk_scheme ORDER BY chunk_scheme"
    ).fetchall()
    print(f"upserted scheme={scheme!r} chunks={n_chunks} (embedder={emb.model_id} dim={emb.dim})")
    print(f"db chunks by scheme: {dict(got)}")


if __name__ == "__main__":
    main()
