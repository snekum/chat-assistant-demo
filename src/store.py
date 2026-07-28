"""pgvector store (DECISIONS D-014, D-016): persons + chunks, exact cosine search.

2-table schema (D-016): persons = resolution/entity anchor; chunks = retrieval unit that
carries the embedding. No ANN index yet -- exact seqscan is correct and sub-ms at this
scale (D-014); HNSW is deferred to the scale trigger. Distance = cosine (<=>) over
L2-normalized vectors, so score = 1 - cosine_distance.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

# TUNABLE(local dev creds; revisit when this leaves localhost -> secrets + TLS)
DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/portfolio")

EMBED_DIM = 768  # nomic-embed-text-v1.5 (D-008); must match vector(N) below

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS persons (
    person_id      text PRIMARY KEY,
    canonical_name text NOT NULL,
    source_doc_id  text NOT NULL,
    meta           jsonb NOT NULL DEFAULT '{{}}'
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     text PRIMARY KEY,
    person_id    text NOT NULL REFERENCES persons(person_id),
    doc_id       text NOT NULL,
    chunk_index  int  NOT NULL DEFAULT 0,
    text         text NOT NULL,
    embedding    vector({EMBED_DIM}) NOT NULL,
    token_est    int,
    model_id     text NOT NULL
);
"""


def connect(init: bool = False) -> psycopg.Connection:
    """Open a connection with the vector type registered. init=True creates the extension
    + tables (must run once before register_vector can see the vector type)."""
    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if init:
        conn.execute(SCHEMA)
    register_vector(conn)
    return conn


def upsert_persons(conn: psycopg.Connection, persons: list[dict]) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO persons (person_id, canonical_name, source_doc_id, meta)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (person_id) DO UPDATE SET
                 canonical_name = EXCLUDED.canonical_name,
                 source_doc_id  = EXCLUDED.source_doc_id,
                 meta           = EXCLUDED.meta""",
            [(p["person_id"], p["canonical_name"], p["source_doc_id"], Jsonb(p.get("meta", {})))
             for p in persons],
        )
    return len(persons)


def upsert_chunks(conn: psycopg.Connection, chunks: list[dict], vecs: np.ndarray,
                  model_id: str) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks
                 (chunk_id, person_id, doc_id, chunk_index, text, embedding, token_est, model_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (chunk_id) DO UPDATE SET
                 person_id=EXCLUDED.person_id, doc_id=EXCLUDED.doc_id,
                 chunk_index=EXCLUDED.chunk_index, text=EXCLUDED.text,
                 embedding=EXCLUDED.embedding, token_est=EXCLUDED.token_est,
                 model_id=EXCLUDED.model_id""",
            [(c["chunk_id"], c["person_id"], c["doc_id"], c["chunk_index"], c["text"],
              vecs[i], c.get("token_est"), model_id) for i, c in enumerate(chunks)],
        )
    return len(chunks)


def search(conn: psycopg.Connection, qvec: np.ndarray, k: int = 3,
           person_id: str | None = None) -> list[dict[str, Any]]:
    """Top-k by cosine. person_id set => metadata PRE-filter (the D-014/D-016 capability):
    Postgres narrows to that person BEFORE ranking, so it cannot return zero-after-filter."""
    where = "WHERE person_id = %(pid)s" if person_id else ""
    sql = f"""
        SELECT chunk_id, person_id, doc_id, text, 1 - (embedding <=> %(q)s) AS score
        FROM chunks {where}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"q": qvec, "pid": person_id, "k": k})
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def rank_all(conn: psycopg.Connection, qvec: np.ndarray,
             person_id: str | None = None) -> list[dict[str, Any]]:
    """FULL ranking (doc_id + score only, NO text) for the gold-rank / MRR diagnostics
    (D-018 option c). The exact seqscan already orders every chunk by distance -- search()
    just discards everything past LIMIT k; here we record the whole ordering so a buried-fact
    miss (gold at rank 67) is visible instead of collapsing to `hit@3: false`. Text is omitted
    to keep the artifact small -- the gold-quote span match already ran on the top-k in search().
    Distinct from search(): this returns ALL rows and no text; it is a diagnostic, not the
    context the generator sees."""
    where = "WHERE person_id = %(pid)s" if person_id else ""
    sql = f"""
        SELECT doc_id, 1 - (embedding <=> %(q)s) AS score
        FROM chunks {where}
        ORDER BY embedding <=> %(q)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"q": qvec, "pid": person_id})
        return [{"doc_id": doc_id, "score": float(score)} for doc_id, score in cur.fetchall()]
