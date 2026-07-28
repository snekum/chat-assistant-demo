---
name: eval-goldset-review
description: Validate and review the eval GOLD SET (the question ground truth) for the RAG project. Use whenever a contributor authors, edits, or asks to check questions in eval/questions.jsonl. Runs the deterministic gold-quote validator, then applies the judgment checks a script can't (absence-proof for abstention, aggregation-vs-multihop, fuzzy gold, name resolution) and reports type balance and sizing. This reviews the QUESTION SET only; it does NOT run evals (that is a separate eval-run skill).
---

# Review the eval gold set

A two-layer review for `eval/questions.jsonl` (the gold ground truth). Layer 1 is a
deterministic script; layer 2 is judgment that needs reading the corpus. Run BOTH. Never
declare a question "ready" on the script alone — the script proves a quote matches; it
cannot prove an abstention is truly unanswerable or that a "multi-hop" isn't secretly
aggregation.

## Layer 1 — deterministic validator (run first)

```
./.venv/Scripts/python.exe eval/validate_questions.py
```

Checks: type in {single-hop, multi-hop, abstention}; evidence-count rules; every gold quote
matches its cited doc under the D-011 normalizer; unique ids. If ANY row FAILs, report the
failures and STOP — fix the gold quotes before judgment review (a broken quote can't be
scored).

## Layer 2 — judgment checks (per new/changed row; the script can't do these)

Read the cited docs (`data/raw/<doc_id>.md` or `data/parsed/chunks.jsonl`) and apply the
check for each row's type:

- **abstention -> PROVE ABSENCE.** Valid only if the fact is absent from ALL 268 docs, not
  just the obvious one. Grep the raw corpus for every phrasing (`grep -riE ... data/raw/`).
  Ignore matches inside `vertexaisearch` source URLs (false positives). If any real doc
  contains the answer, the row is INVALID. Record the grep in `notes`.

- **multi-hop -> reject aggregation and fuzzy gold.**
  - If the honest answer is a SET ("who are all…", "recommend people in X", "which
    leaders…"), it is AGGREGATION, not multi-hop -> a completeness trap. Reject from this
    hit@k set; it belongs in the deferred recommendation track (see eval/README.md).
  - True multi-hop = a few SPECIFIC docs joined by reasoning. In this corpus (dossiers do
    not cross-reference each other) the realistic form is a comparison of two NAMED people
    with >=2 evidence docs. Confirm >=2 docs are genuinely needed — if one doc alone answers
    it, it is single-hop mislabeled.
  - If no exact quote settles the answer (opinion/judgment, e.g. "who can help me expand in
    China"), the gold is FUZZY -> reject.

- **single-hop / all types -> person resolution + answer sanity.**
  - Each name mentioned must resolve to exactly one `person_id` in
    `data/parsed/persons.jsonl`. Flag ambiguous names (bare surnames, or a surname shared by
    two subjects) as a resolution risk — the scorer does not yet handle a clarify/confirm
    outcome.
  - `gold_answer` must actually answer the question and be supported by the evidence quotes,
    not by world knowledge (D-013 contamination).
  - **hit@k is doc_id-anchored** (D-011 amendment 2026-07-24): a gold quote counts as a HIT
    only if the GOLD doc is retrieved AND contains it — so a non-unique quote ("Santa Clara
    University", in 15 docs) can no longer cause a false hit, and you needn't reject generic
    quotes. Still prefer questions that NAME their subject so the query resolves to the gold doc.

## Report back

- Per-question verdict: PASS / FIX (with the specific reason).
- Type balance (single-hop / multi-hop / abstention counts).
- Sizing note: resolving small config differences needs a bigger N — cite the CI table in
  eval/METRICS.md (N=40 -> +-0.14). Do not silently imply the set is big enough.
