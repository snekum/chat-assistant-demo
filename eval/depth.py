"""Depth metadata for a gold quote (ROADMAP item 13 + the split-scoring decision).

The split-scoring call (2026-07-24): single-hop hit@1 is reported PER DEPTH BUCKET, not
blended, because a blended rate just measures the easy:hard mix of questions you happened to
author (see DECISIONS D-012 sizing note). This module answers "how deep in the report does
this fact sit?" so both (a) the question generator can sample by depth and (b) run.py can
group hit@1 by bucket.

Everything runs in NORMALIZED space (eval/normalize.py) so it matches the exact rule the
gold quote was validated + scored under (D-011). Position = index of the normalized quote in
the normalized doc, as a fraction of doc length. Cheap, deterministic, torch-free.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "eval")
from normalize import normalize_for_match  # noqa: E402

CHUNKS = Path("data/parsed/chunks.jsonl")

# Depth buckets by position fraction. Terciles = the dumbest defensible split; the reports
# carry ~15 sections so a fact's fractional position is a fair proxy for "how far you must
# read to reach it." top = identity/summary region (near-always retrieved), buried = the
# deep-dive tail where whole-doc embedding dilution bites (ex-sh-2 "Antananarivo" ranked 60).
# TUNABLE(tercile cutoffs 0.33/0.66; revisit when the buried bucket's hit@1 doesn't separate
# from top -- i.e. depth stops predicting misses -> re-cut against the gold-rank-vs-depth curve.)
TOP_MAX = 0.33
BURIED_MIN = 0.66

_HEADING = re.compile(r"(?m)^#{2,3}\s*(\d+)\.")


def load_corpus() -> dict[str, str]:
    return {
        json.loads(l)["doc_id"]: json.loads(l)["text"]
        for l in CHUNKS.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }


def bucket_of(pos_frac: float) -> str:
    if pos_frac < TOP_MAX:
        return "top"
    if pos_frac >= BURIED_MIN:
        return "buried"
    return "mid"


def _section_at(doc_text: str, norm_pos: int, norm_len: int) -> int | None:
    """The `### N.` / `## N.` section number whose text encloses norm_pos (item 13 extra).
    Maps each heading's raw offset into normalized space by the length of its normalized
    prefix, then takes the last heading at or before the quote."""
    best: int | None = None
    for m in _HEADING.finditer(doc_text):
        head_norm_pos = len(normalize_for_match(doc_text[: m.start()]))
        if head_norm_pos <= norm_pos:
            best = int(m.group(1))
        else:
            break
    return best


def depth_of(quote: str, doc_text: str) -> dict | None:
    """-> {pos_frac, bucket, section} for a gold quote, or None if the quote isn't found
    (a validated question should never hit None -- validate_questions.py is the gate)."""
    ndoc = normalize_for_match(doc_text)
    nq = normalize_for_match(quote)
    if not nq or not ndoc:
        return None
    pos = ndoc.find(nq)
    if pos < 0:
        return None
    frac = pos / len(ndoc)
    return {"pos_frac": round(frac, 4), "bucket": bucket_of(frac),
            "section": _section_at(doc_text, pos, len(ndoc))}


if __name__ == "__main__":  # quick characterization of the current question set by depth
    corpus = load_corpus()
    rows = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if r["type"] != "single-hop":
            continue
        e = r["evidence"][0]
        d = depth_of(e["quote"], corpus.get(e["doc_id"], ""))
        print(f"{r['id']:<8} {r['evidence'][0]['doc_id']:<16} "
              f"pos={d['pos_frac'] if d else '?':<7} §{d['section'] if d else '?':<3} "
              f"bucket={d['bucket'] if d else 'NOT FOUND'}")
