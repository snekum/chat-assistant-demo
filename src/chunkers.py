r"""Experimental chunk schemes for the Phase 2 A/B (DECISIONS D-023).

Whole-doc (1 chunk == 1 doc) lives in ingest.py (D-008). This module adds the two arms the
A/B compares against it, both reading the same raw dossiers and reusing ingest.parse_body
(drops the opaque **Sources:** URL list) + ingest.slugify:

  section   -- split at `^#{2,3} \d+\.` headings; ONE chunk per section. The section's OWN
               heading line is kept verbatim (native document text, NOT the deferred synthetic
               contextual-header -- so D-011 span-matching's "clean corpus text" invariant
               holds). Census (2026-07-29): 252/268 docs use `###`, 16 use `##` -> the `#{2,3}`
               pattern catches both; a `###`-only splitter would silently leave those 16 whole
               INSIDE the section arm and dilute the A/B. Section counts: 262 docs=15, 5 docs=12
               (truncated), 1 doc=16 -> assert count in {12,15,16}, everything else -> variance
               log. Preamble (the "# Deep Research Report: <Name>" title before section 1) folds
               into the section-1 chunk. LOSSLESS: chunks partition the body at heading offsets,
               so every character lands in exactly one chunk and no gold quote can fall in a gap.

  fixed<N>  -- structure-BLIND ~N-token windows, NO overlap, cut at the next whitespace to avoid
               mid-word slicing. N is size-matched to the MEASURED section-chunk mean (fork 3),
               so the A/B isolates boundary PLACEMENT (structure vs ruler) from chunk SIZE. Text
               is sliced as a verbatim substring of the body (span fidelity, D-011). A gold quote
               severed across a window boundary becomes unfindable -- that is the intended
               EVIDENCE for structure-awareness (sections cut at topic breaks; a ruler doesn't),
               not a bug (D-023 pre-registered prediction 4).

Emits data/parsed/chunks.<scheme>.jsonl with a `chunk_scheme` field. Embeddings are added at
index-build time (build_index.py) from chunk text; this file emits only text + metadata.

Usage:
  ./.venv/Scripts/python.exe src/chunkers.py section
  ./.venv/Scripts/python.exe src/chunkers.py fixed          # window = measured section mean
  ./.venv/Scripts/python.exe src/chunkers.py fixed --chars 760
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from ingest import CHARS_PER_TOKEN, RAW_DIR, parse_body, slugify  # noqa: E402

SECTION_CHUNKS_PATH = Path("data/parsed/chunks.section.jsonl")
FIXED_CHUNKS_PATH = Path("data/parsed/chunks.fixed.jsonl")

# A section heading: 2-3 '#' then a number and a dot, at line start. `#{2,3}` (NOT `###`)
# because 16/268 docs use `## N.` (census 2026-07-29) -- the documented A/B-diluting trap.
HEAD_RE = re.compile(r"(?m)^#{2,3}\s+(\d+)\.")

# Expected section counts (census 2026-07-29): 15 is the template; 12 = truncated docs, 16 = the
# one over-sectioned doc. Anything outside this set is logged, not silently accepted.
# TUNABLE(expected section counts {12,15,16}; revisit if the corpus template changes or new docs
#   land -> re-census. Symptom wrong: the variance log fills with a count that is actually normal.)
EXPECTED_SECTION_COUNTS = {12, 15, 16}


def split_sections(body: str) -> list[tuple[int, str]]:
    """Partition `body` into (section_number, text) at heading offsets. Lossless: the union of
    chunk texts is exactly `body` (preamble folded into section 1). Returns [] only if the doc has
    no `N.` heading (never, per census) -- caller treats that as whole-doc + logs it."""
    heads = list(HEAD_RE.finditer(body))
    if not heads:
        return []
    starts = [m.start() for m in heads]
    nums = [int(m.group(1)) for m in heads]
    bounds = starts + [len(body)]
    chunks: list[tuple[int, str]] = []
    for i, num in enumerate(nums):
        lo = 0 if i == 0 else bounds[i]  # fold the pre-section-1 preamble into section 1
        hi = bounds[i + 1]
        text = body[lo:hi].strip()
        if text:
            chunks.append((num, text))
    return chunks


def split_fixed(body: str, target_chars: int) -> list[tuple[int, str]]:
    """Structure-blind ~target_chars windows, no overlap, extended to the next whitespace so a
    window never ends mid-word. Each chunk is a verbatim substring of `body` (span fidelity)."""
    chunks: list[tuple[int, str]] = []
    i, n, idx = 0, len(body), 0
    while i < n:
        while i < n and body[i].isspace():  # don't start a chunk on whitespace
            i += 1
        if i >= n:
            break
        end = min(i + target_chars, n)
        while end < n and not body[end].isspace():  # extend to a word boundary
            end += 1
        text = body[i:end].strip()
        if text:
            chunks.append((idx, text))
            idx += 1
        i = end
    return chunks


def _person_id(name: str, counts: dict[str, int]) -> str:
    base = slugify(name)
    if base in counts:
        counts[base] += 1
        return f"{base}-{counts[base]}"
    counts[base] = 1
    return base


def build_section() -> None:
    paths = sorted(RAW_DIR.glob("*.md"))
    counts: dict[str, int] = {}
    out: list[dict] = []
    variance: list[tuple[str, int]] = []  # (name, n_sections) for counts not in {12,15,16}
    sec_tokens: list[int] = []
    for path in paths:
        name = path.stem
        pid = _person_id(name, counts)
        body, _ = parse_body(path)
        secs = split_sections(body)
        assert secs, f"{name}: no `N.` heading found (census says impossible) -- inspect this doc"
        if len(secs) not in EXPECTED_SECTION_COUNTS:
            variance.append((name, len(secs)))
        for num, text in secs:
            tok = round(len(text) / CHARS_PER_TOKEN)
            sec_tokens.append(tok)
            out.append({
                "chunk_id": f"{pid}#s{num}",       # per-scheme id; composite PK (id, scheme) in DB
                "chunk_scheme": "section",
                "person_id": pid,
                "doc_id": name,                     # unchanged -> hit@k doc_id join still works
                "chunk_index": num,                 # the real section number (traceable diagnostic)
                "text": text,
                "token_est": tok,
            })
    _write(SECTION_CHUNKS_PATH, out)
    n = len(sec_tokens)
    st = sorted(sec_tokens)
    mean = round(sum(sec_tokens) / n)
    pct = lambda q: st[min(n - 1, int(q * n))]
    print(f"wrote {len(out)} section chunks over {len(paths)} docs -> {SECTION_CHUNKS_PATH}")
    print(f"chunks/doc: {len(out) / len(paths):.1f}")
    print(f"section token_est: mean={mean} p50={pct(.5)} p90={pct(.9)} min={st[0]} max={st[-1]}")
    print(f"MEAN section chars ~= {mean * CHARS_PER_TOKEN} -> use as the fixed-window size")
    if variance:
        print(f"VARIANCE LOG ({len(variance)} docs off the 15-section template): {variance}")
    else:
        print("variance log: none (all docs in {12,15,16})")


def build_fixed(target_chars: int | None) -> None:
    if target_chars is None:
        # size-match to the measured section mean (fork 3) -- requires the section split first
        if not SECTION_CHUNKS_PATH.exists():
            raise SystemExit("run `chunkers.py section` first, or pass --chars N (size-match source)")
        secs = [json.loads(l) for l in SECTION_CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
        target_chars = round(sum(len(c["text"]) for c in secs) / len(secs))
        print(f"size-matched window = {target_chars} chars (= mean section chunk length)")
    paths = sorted(RAW_DIR.glob("*.md"))
    counts: dict[str, int] = {}
    out: list[dict] = []
    fixed_tokens: list[int] = []
    for path in paths:
        name = path.stem
        pid = _person_id(name, counts)
        body, _ = parse_body(path)
        for idx, text in split_fixed(body, target_chars):
            tok = round(len(text) / CHARS_PER_TOKEN)
            fixed_tokens.append(tok)
            out.append({
                "chunk_id": f"{pid}#f{idx}",
                "chunk_scheme": f"fixed{round(target_chars / CHARS_PER_TOKEN)}",
                "person_id": pid,
                "doc_id": name,
                "chunk_index": idx,
                "text": text,
                "token_est": tok,
            })
    _write(FIXED_CHUNKS_PATH, out)
    n = len(fixed_tokens)
    ft = sorted(fixed_tokens)
    pct = lambda q: ft[min(n - 1, int(q * n))]
    print(f"wrote {len(out)} fixed chunks over {len(paths)} docs -> {FIXED_CHUNKS_PATH}")
    print(f"chunks/doc: {len(out) / len(paths):.1f}  scheme={out[0]['chunk_scheme']}")
    print(f"fixed token_est: mean={round(sum(fixed_tokens)/n)} p50={pct(.5)} p90={pct(.9)} "
          f"min={ft[0]} max={ft[-1]}")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scheme", choices=["section", "fixed"])
    ap.add_argument("--chars", type=int, default=None, help="fixed window size (default = section mean)")
    args = ap.parse_args()
    if args.scheme == "section":
        build_section()
    else:
        build_fixed(args.chars)


if __name__ == "__main__":
    main()
