r"""Ingestion: raw .md dossiers -> persons + chunks (data/parsed/{persons,chunks}.jsonl).

Per DECISIONS D-008 (whole-doc: 1 chunk == 1 doc), D-016 (person as first-class entity,
2-table persons+chunks):
  - read .md as UTF-8; filename stem = canonical_name (deterministic subject map, D-005)
  - person_id = slug(stem); chunk_id = "<person_id>#<chunk_index>" (whole-doc => "#0")
  - drop the trailing "**Sources:**" URL list (opaque vertexaisearch redirects = noise)
  - Unicode NFC; preserve text verbatim (NO lowercasing/stemming) for span fidelity,
    so D-011 gold quotes can be authored against THIS text, not the raw files
  - keep inline [cite: ...] markers verbatim. NB: the corpus has ~20 marker variants
    ([cite: N], [cite: source: N], [cite: Internal Context], ...); stripping them is a
    D-011 normalizer concern (regex \[cite:[^\]]*\]), NOT an ingestion concern.

Embeddings are NOT produced here -- they are added at index-build time (src/embedder.py)
from the chunk text. This file emits only text + metadata.

Anomaly handled: 1/268 docs has no "**Sources:**" block -> kept whole,
sources_dropped=false.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

RAW_DIR = Path("data/raw")
PERSONS_PATH = Path("data/parsed/persons.jsonl")
CHUNKS_PATH = Path("data/parsed/chunks.jsonl")

SOURCES_RE = re.compile(r"(?m)^\*\*Sources:\*\*[ \t]*$")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

# Diagnostic only (window-fit sanity vs nomic 8192), never a control threshold.
# TUNABLE(chars/4 rough proxy; revisit if a doc's real token count nears the window)
CHARS_PER_TOKEN = 4


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return _SLUG_STRIP.sub("-", s.lower()).strip("-")


def parse_body(path: Path) -> tuple[str, bool]:
    text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    matches = list(SOURCES_RE.finditer(text))
    if matches:
        return text[: matches[-1].start()].rstrip(), True
    return text.rstrip(), False


def main() -> None:
    paths = sorted(RAW_DIR.glob("*.md"))
    if not paths:
        raise SystemExit(f"no .md files under {RAW_DIR.resolve()}")

    persons: list[dict] = []
    chunks: list[dict] = []
    slug_counts: dict[str, int] = {}
    collisions: list[tuple[str, str]] = []

    for path in paths:
        canonical_name = path.stem  # the person's name, e.g. "Jane Rivera"
        body, sources_dropped = parse_body(path)

        base = slugify(canonical_name)
        if base in slug_counts:
            slug_counts[base] += 1
            person_id = f"{base}-{slug_counts[base]}"
            collisions.append((canonical_name, person_id))
        else:
            slug_counts[base] = 1
            person_id = base

        persons.append(
            {
                "person_id": person_id,
                "canonical_name": canonical_name,
                "source_doc_id": canonical_name,
                "meta": {},  # future: industry, company, aliases (resolution/filter)
            }
        )
        chunks.append(
            {
                "chunk_id": f"{person_id}#0",  # whole-doc: single chunk index 0
                "person_id": person_id,
                "doc_id": canonical_name,  # denormalized for the eval hit-rate join
                "chunk_index": 0,
                "text": body,
                "token_est": round(len(body) / CHARS_PER_TOKEN),
                "sources_dropped": sources_dropped,
            }
        )

    PERSONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PERSONS_PATH.open("w", encoding="utf-8") as f:
        for p in persons:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    toks = sorted(c["token_est"] for c in chunks)
    n = len(toks)
    pct = lambda q: toks[min(n - 1, int(q * n))]
    no_src = [c["doc_id"] for c in chunks if not c["sources_dropped"]]
    print(f"wrote {len(persons)} persons -> {PERSONS_PATH}")
    print(f"wrote {len(chunks)} chunks  -> {CHUNKS_PATH}")
    print(f"token_est: p50={pct(.5)} p90={pct(.9)} max={toks[-1]} min={toks[0]}")
    print(f"sources_dropped=false for {len(no_src)} doc(s): {no_src}")
    print(f"slug collisions (suffixed): {collisions if collisions else 'none'}")
    assert len({p['person_id'] for p in persons}) == len(persons), "person_id not unique!"


if __name__ == "__main__":
    main()
