# Chat-assistant RAG — a measured, decision-logged retrieval system

A retrieval-augmented Q&A system over a corpus of long-form dossiers, built as a **learning
and interview artifact**. The point of this repo is not the code — it is the **architectural
judgment behind it**: every non-trivial choice (chunking, retrieval, index, judge, eval
design) is logged with its alternatives, my reason, and the measured symptom that would prove
it wrong.

If you are reviewing this to gauge how I work, start with **[DECISIONS.md](DECISIONS.md)** and
**[ROADMAP.md](ROADMAP.md)** — the code is downstream of those.

---

## What this is (and the honest caveat)

- **Corpus:** 268 deep-research dossiers about real, semi-public people (one report each).
- **Because the subjects are real, the corpus is NOT in this repo.** `data/` and the eval gold
  set (`eval/questions.jsonl`) are gitignored — they carry verbatim personal facts. What you
  *can* see and run is documented below.
- **What IS committed and reviewable:** all pipeline code (`src/`, `eval/`), the full decision
  log, the roadmap, the eval schema + validator, a public schema example
  ([eval/questions.example.jsonl](eval/questions.example.jsonl)), and the **metrics-only run
  artifacts** (`config.json` / `summary.json` per run — see `runs/`; the leaky
  `results.jsonl` / `rankings.jsonl` are gitignored).

So a fresh clone can: read every decision, inspect the committed measured results, and stand up
the full infrastructure (Postgres + pgvector + the harness). It **cannot** reproduce a run
end-to-end without the private corpus — that is a deliberate privacy tradeoff, not a missing
piece. See [DECISIONS D-013](DECISIONS.md) for why the corpus stays real (and why the eval is
still provably not just testing the model's memory of these people).

---

## Architecture

```
raw .md dossiers
      │  ingest (src/ingest.py)         whole-doc chunks, 1 chunk == 1 doc (D-008)
      ▼                                 person = first-class entity (D-016)
persons.jsonl + chunks.jsonl
      │  embed (src/embedder.py)        nomic-embed-text-v1.5, 768d, local/CPU
      │                                 disk cache keyed (model_id, role, sha256(text)) (D-015)
      ▼
Postgres + pgvector (src/store.py)      2-table: persons + chunks; exact cosine (D-014)
      │                                 person_id PRE-filter built for name-scoped retrieval
      ▼
retrieve top-k=3 (src/retrieve.py)
      │
      ▼
generate (src/generate.py)             claude-haiku-4-5; versioned refuse-if-absent
      │                                 contract "f6-v1"; document-level citations (D-009)
      ▼
answer + citations
      │
      ▼
LLM-as-judge (eval/judge.py)           claude-sonnet-5 (never the generator's tier);
                                        versioned rubric "d010-v1"; gold-blind
                                        groundedness lane + correctness lane (D-010)
```

The **eval harness** (`eval/run.py`) wraps this and writes an immutable, fully config-snapshotted
run per question set: offline retrieval metrics (hit@k, span-recall, gold-rank + MRR) always,
plus the online generation/judge metrics when an Anthropic key is present.

---

## Measured baseline

From the current baseline-of-record `runs/20260726T091259Z-2158c98/` (n=59 questions, clean
tree; Wilson 95% intervals). Full numbers in that run's `summary.json`.

| Metric | Value | Reading |
|---|---|---|
| single-hop hit@1 | **0.63** [0.48, 0.76] | retrieval finds the right dossier ~2/3 of the time at k=1 |
| single-hop hit@3 | 0.71 [0.56, 0.82] | |
| multi-hop hit@1 | **0.00** (structural) | the all-spans rule can't score hit@1 on 2-doc questions — expected; the before-picture for Phase 3 |
| multi-hop span-recall@3 | 0.67 | partial credit shows the progress the all-spans bar hides |
| **groundedness (primary)** | **1.00** (33/33) [0.90, 1.00] | every non-refusal answer supported by the reports alone |
| false-refusal rate | 0.30 | **decomposes to 0 real generation false-refusals** — it's the retrieval miss-rate surfacing as honest "I don't know" |
| closed-book correctness | **0.00** | with empty context the bot refused all 47 answerable questions → **zero pretraining contamination**; correctness is retrieval-driven, not memory (D-013) |
| cost | ~$0.05 / question | Haiku generation + Sonnet judge; ~$3 for the full n=59 run |

**What makes the measurement trustworthy** (this is the actual deliverable):
- The **groundedness judge never sees the gold answer** — it scores "supported by the reports
  alone," so it can't be fooled into grading truth-in-the-world instead of retrieval support.
- The **judge is calibrated**: 6/6 adversarial ungrounded plants caught, 100% agreement with
  blind human labels on genuinely-grounded answers, 0% flip-rate across repeat runs, with an
  armed asymmetric trip rule for any future rubric change (D-022 / GAPS G-002).
- **Versioned rulers**: the generation contract and judge rubric are version-stamped and
  snapshotted into every run, so two runs are only ever compared under the same instrument.
- **Write-once runs** with full config snapshots (git SHA, question-set hash, normalizer
  version, embed-stack versions) — a repro-hole audit closed in D-021.

---

## Design philosophy: no one-sided metric

Every headline metric here is paired with a counter-metric that catches the way the headline
lies. A number without its counter is theatre — this pairing is the spine of the eval design.

| Headline | What it can't see alone | Counter-metric |
|---|---|---|
| **abstention accuracy** (did it correctly decline?) | a bot that refuses *everything* scores 100% | **false-refusal rate** — punishes over-refusal on answerable questions |
| **groundedness** (answer supported by retrieved context?) | says nothing about whether retrieval fetched the *right* document — a fluent answer grounded in a mis-retrieved dossier passes | **correctness + hit@k**, jointly — hit@k proves the gold doc was retrieved; correctness checks the answer against gold |
| **correctness** (answer matches gold?) | could be the model reciting pretraining memory of a real person, not using retrieval at all | **closed-book control** — same questions, empty context; correctness came back **0.00**, so correctness is retrieval-driven, not memory (D-013) |
| **fabricated-citation rate** (cited a doc it wasn't shown?) | an answer that cites *nothing* fabricates nothing yet is untraceable | **zero-citation rate** — the sparse-citation direction (D-020) |
| **judge groundedness score** | the ruler itself can be miscalibrated or unstable | **blind human calibration + flip-rate** — the judge is measured, not trusted (D-022) |

This is the direct answer to the interview question *"abstention accuracy is 100% for a system
that always refuses — where's the counter-metric?"* — it's `false_refusal_rate`, and the
`is_refusal` label that drives both is deterministic (D-019), so a drifting model never sits
inside the primary denominator.

---

## Decision log — the differentiator

Every choice below is a row in [DECISIONS.md](DECISIONS.md) with options, my reason, and a
revisit trigger. Open questions I couldn't defend under interrogation live in
[GAPS.md](GAPS.md). This index is the fast path:

**Anonymizer (parked — building on real data first):** D-001..D-007 — public-company allowlist,
fake-name generation, filename policy, surname matching, NER posture, locations, domain fuzzing.

**RAG baseline + eval harness:**
- **D-008** whole-doc chunking + local nomic embedder · **D-009** refuse-if-absent answer
  contract · **D-010** LLM-as-judge (pinned Sonnet, gold-blind groundedness) · **D-011**
  normalize-then-exact hit-rate · **D-012** hand-seed + verified LLM-expansion question set ·
  **D-013** real-entity contamination (accept + measure via closed-book)
- **D-014** pgvector 2-table store (pre-filter over FAISS post-filter) · **D-015** embedder
  abstraction + cache · **D-016** person as first-class entity · **D-017** single-provider
  (Anthropic), cross-provider judge pre-registered behind bias evidence

**Instrument hardening (Phase 1):**
- **D-018** full gold-rank + MRR recording · **D-019** deterministic refusal cross-check ·
  **D-020** citation validity parser · **D-021** repro-hole closure · **D-022** judge
  calibration design (closes G-002)

---

## Repo layout

```
src/           ingest → embed → store → retrieve → generate (the pipeline)
eval/          run.py harness, judge, hit-rate, calibration, question schema + validator
runs/          write-once run artifacts (metrics-only files committed; leaky files gitignored)
calibration/   judge-calibration reports (G-002 / D-022)
notes/         interview-prep talk-track (self-quiz scaffold)
DECISIONS.md   every choice: options, my reason, revisit trigger
GAPS.md        questions I couldn't fully answer under interrogation
ROADMAP.md     the governing plan (6 phases to interview-ready)
FORKS.md       the original fork enumeration (historical record)
```

---

## Setup

Prereqs: **Docker** (for Postgres + pgvector) and **Python 3.14** with a virtualenv.

```bash
# 1. Postgres + pgvector
docker compose up -d

# 2. Python deps
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# 3. Keys (only needed for the generation + judge lanes)
cp .env.example .env        # then paste your ANTHROPIC_API_KEY

# 4. Verify the setup (no corpus, no paid API calls needed)
.venv/Scripts/python scripts/smoke.py
```

`scripts/smoke.py` proves the plumbing a fresh clone *can* check — deps import, Postgres +
pgvector is reachable and the schema applies, the local embedder loads and returns a 768-d
unit vector — and reports whether your `ANTHROPIC_API_KEY` is set. It never touches the
private corpus and spends nothing.

### Running (requires the private corpus in `data/raw/`)

```bash
.venv/Scripts/python src/ingest.py            # raw .md → persons.jsonl + chunks.jsonl
.venv/Scripts/python src/build_index.py       # embed (cached) + load into pgvector
.venv/Scripts/python eval/run.py              # score → runs/<timestamp>-<sha>/
```

`eval/run.py` runs the offline retrieval metrics with no key; add `ANTHROPIC_API_KEY` to also
run the generation + judge lanes. See [eval/README.md](eval/README.md) for the question schema
and the gold-quote validator.

---

## Status

Phase 1 (eval core) is complete: instrument hardening, gold set (n=59), baseline-of-record,
closed-book contamination control, and judge calibration all done and committed. Next up is
Phase 2 (section-chunking A/B) per [ROADMAP.md](ROADMAP.md). The roadmap is the source of
truth for what's built, what's deferred, and why.
