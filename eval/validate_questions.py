"""Validate eval/questions.jsonl against the parsed corpus (D-011 / D-012 guardrail).

Every gold quote MUST appear (normalize-then-exact) in its cited doc's parsed chunk text,
or the question is unscoreable and would silently launder a miss. Run this after authoring
ANY question -- it's the fast feedback loop that keeps gold honest.

Usage:  ./.venv/Scripts/python.exe eval/validate_questions.py
Exit 0 = all rows valid; exit 1 = at least one row failed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "eval")
from normalize import quote_hits  # noqa: E402

QUESTIONS = Path("eval/questions.jsonl")
CHUNKS = Path("data/parsed/chunks.jsonl")
TYPES = {"single-hop", "multi-hop", "abstention"}


def load_corpus() -> dict[str, str]:
    return {
        json.loads(l)["doc_id"]: json.loads(l)["text"]
        for l in CHUNKS.read_text(encoding="utf-8").splitlines()
    }


def validate_row(q: dict, corpus: dict[str, str]) -> list[str]:
    errs: list[str] = []
    qtype = q.get("type")
    if qtype not in TYPES:
        errs.append(f"type {qtype!r} not in {TYPES}")
    if not q.get("question"):
        errs.append("missing question text")
    if not q.get("gold_answer"):
        errs.append("missing gold_answer")

    ev = q.get("evidence", [])
    docs = {e.get("doc_id") for e in ev}
    if qtype == "abstention":
        if ev:
            errs.append("abstention row must have empty evidence (answer is absent)")
    elif qtype == "single-hop":
        if len(ev) != 1:
            errs.append(f"single-hop needs exactly 1 evidence, got {len(ev)}")
    elif qtype == "multi-hop":
        if len(docs) < 2:
            errs.append(f"multi-hop needs >=2 distinct docs, got {len(docs)}")

    for e in ev:
        did, quote = e.get("doc_id"), e.get("quote", "")
        if did not in corpus:
            errs.append(f"doc_id {did!r} not in corpus")
        elif not quote_hits(quote, corpus[did]):
            errs.append(f"quote not found in {did!r}: {quote[:60]!r}")
    return errs


def main() -> None:
    corpus = load_corpus()
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]

    ids = [r.get("id") for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    n_fail = 0
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r.get("type", "?")] = by_type.get(r.get("type", "?"), 0) + 1
        errs = validate_row(r, corpus)
        if r.get("id") in dupes:
            errs.append("duplicate id")
        status = "PASS" if not errs else "FAIL"
        if errs:
            n_fail += 1
        print(f"[{status}] {r.get('id'):<10} {r.get('type','?'):<11} {r.get('question','')[:56]}")
        for e in errs:
            print(f"         - {e}")

    print(f"\n{len(rows)} questions  by-type={by_type}  failures={n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
