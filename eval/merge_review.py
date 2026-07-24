"""Merge approved review-queue rows into questions.jsonl (ROADMAP 1c / D-012 human-approve step).

Runs AFTER a human has reviewed eval/review_queue.jsonl. For each row it:
  - re-validates the gold quote (normalize-then-exact containment, D-011) -- aborts on any
    failure so a broken quote can never reach the gold set;
  - RECOMPUTES the depth tag from the final quote (so an edited quote gets correct tags);
  - assigns the next sequential sh-NNN id (never reuses one -- question ids are write-once);
  - appends to questions.jsonl (existing rows untouched, byte-for-byte) and clears the queue.

Usage:  ./.venv/Scripts/python.exe eval/merge_review.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "eval")

from depth import depth_of, load_corpus  # noqa: E402
from normalize import quote_hits  # noqa: E402

QUESTIONS = Path("eval/questions.jsonl")
QUEUE = Path("eval/review_queue.jsonl")


def next_id_num(rows: list[dict]) -> int:
    n = 0
    for r in rows:
        m = re.fullmatch(r"sh-(\d+)", r.get("id", ""))
        if m:
            n = max(n, int(m.group(1)))
    return n


def main() -> None:
    corpus = load_corpus()
    existing = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    queue = [json.loads(l) for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not queue:
        print("queue empty; nothing to merge")
        return

    n = next_id_num(existing)
    merged: list[dict] = []
    for row in queue:
        did, quote = row["evidence"][0]["doc_id"], row["evidence"][0]["quote"]
        if did not in corpus or not quote_hits(quote, corpus[did]):
            print(f"ABORT: quote fails containment for {did!r}: {quote[:60]!r}")
            sys.exit(1)
        tags = depth_of(quote, corpus[did]) or {}
        tags["fact_type"] = row.get("tags", {}).get("fact_type")
        n += 1
        merged.append({
            "id": f"sh-{n:03d}",
            "type": row["type"],
            "question": row["question"],
            "evidence": row["evidence"],
            "gold_answer": row["gold_answer"],
            "author": row.get("author", "llm-assisted"),
            "tags": tags,
        })

    raw = QUESTIONS.read_text(encoding="utf-8")
    with QUESTIONS.open("a", encoding="utf-8") as f:
        if raw and not raw.endswith("\n"):
            f.write("\n")
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    QUEUE.write_text("", encoding="utf-8")
    print(f"merged {len(merged)} rows: {merged[0]['id']}..{merged[-1]['id']}; queue cleared")


if __name__ == "__main__":
    main()
