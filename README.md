# Grounded RAG Q&A over long-form dossiers — with a versioned evaluation harness

A retrieval-augmented question-answering system over a corpus of long-form professional
dossiers. The emphasis of this project is **measurement**: a rigorous, versioned evaluation
harness that can prove whether the retrieval and generation actually work — not just a pipeline
that returns plausible text.

The interesting engineering is in the eval layer: gold-blind judging, deterministic refusal and
citation checks, a calibrated LLM-as-judge, paired statistics on a small gold set, and
write-once run artifacts you can diff.

---

## About the data

The corpus is 268 long-form dossiers about real executives, generated from publicly available
information. Although every fact is individually public, aggregating them into per-person
profiles makes the collection sensitive — so **the corpus is not included in this repo** and is
gitignored. What *is* here: all pipeline and eval code, the question-set schema and validator, a
public schema example ([eval/questions.example.jsonl](eval/questions.example.jsonl)), and the
**metrics-only run artifacts** (`config.json` / `summary.json` per run — the files carrying
report text or names are withheld).

So a fresh clone can read the code, inspect the measured results, and stand up the full
infrastructure. Reproducing a full run end-to-end additionally requires the private corpus.

---

## Architecture

```
raw .md dossiers
      │  ingest              whole-doc chunks (1 chunk / doc, baseline);
      ▼                      person modeled as a first-class entity
persons.jsonl + chunks.jsonl
      │  embed               nomic-embed-text-v1.5, 768-d, local/CPU
      │                      disk cache keyed (model_id, role, sha256(text))
      ▼
Postgres + pgvector          2-table: persons + chunks; exact cosine search
      │                      person_id PRE-filter for name-scoped retrieval
      ▼
retrieve top-k=3
      │
      ▼
generate                     Haiku generator; versioned "answer only from the
      │                      reports, else refuse" contract; document-level citations
      ▼
answer + citations
      │
      ▼
LLM-as-judge                 Sonnet-class judge (never the generator's tier);
                             versioned rubric; gold-blind groundedness lane +
                             correctness lane
```

The eval harness wraps this and writes an immutable, fully config-snapshotted run per question
set: offline retrieval metrics (hit@k, span-recall, gold-rank + MRR) always, plus online
generation/judge metrics when an API key is present.

---

## Measured baseline

From the current baseline-of-record (n=59 questions, clean tree; Wilson 95% intervals). Full
numbers live in that run's `summary.json` under `runs/`.

| Metric | Value | Reading |
|---|---|---|
| single-hop hit@1 | **0.63** [0.48, 0.76] | retrieval finds the right dossier ~2/3 of the time at k=1 |
| single-hop hit@3 | 0.71 [0.56, 0.82] | |
| multi-hop hit@1 | **0.00** (structural) | the all-spans rule can't score hit@1 on 2-doc questions — expected; the before-picture for the routing work |
| multi-hop span-recall@3 | 0.67 | partial credit shows progress the all-spans bar hides |
| **groundedness (primary)** | **1.00** (33/33) [0.90, 1.00] | every non-refusal answer supported by the reports alone |
| false-refusal rate | 0.30 | decomposes to **0 real generation false-refusals** — it's the retrieval miss-rate surfacing as honest "I don't know" |
| closed-book correctness | **0.00** | with empty context the model refused all 47 answerable questions → correctness is retrieval-driven, not pretraining memory |
| cost | ~$0.05 / question | Haiku generation + Sonnet judge; ~$3 for the full n=59 run |

---

## Design philosophy: no one-sided metric

Every headline metric is paired with a counter-metric that catches the way the headline lies. A
number without its counter is theatre — this pairing is the spine of the eval design.

| Headline | What it can't see alone | Counter-metric |
|---|---|---|
| **abstention accuracy** (correctly declined?) | a bot that refuses *everything* scores 100% | **false-refusal rate** — punishes over-refusal on answerable questions |
| **groundedness** (supported by retrieved context?) | says nothing about whether retrieval fetched the *right* document — a fluent answer grounded in a mis-retrieved dossier passes | **correctness + hit@k**, jointly — hit@k proves the gold doc was retrieved; correctness checks the answer against gold |
| **correctness** (matches gold?) | could be the model reciting memory of a real person, not using retrieval | **closed-book control** — same questions, empty context; came back **0.00**, so correctness is retrieval-driven |
| **fabricated-citation rate** (cited an unseen doc?) | an answer that cites *nothing* fabricates nothing yet is untraceable | **zero-citation rate** — the sparse-citation direction |
| **judge groundedness score** | the ruler itself can be miscalibrated or unstable | **blind human calibration + flip-rate** — the judge is measured, not trusted |

The refusal label that drives both abstention metrics is a deterministic string check, so a
drifting model never sits inside the primary denominator.

---

## Key design decisions

- **Whole-doc chunking as the baseline; section-aware chunking as the measured next step.** Each
  dossier follows a fixed 15-section template. The baseline embeds one chunk per document to
  establish a measurable floor; the known failure mode (a buried fact dilutes inside a ~2.8k-token
  whole-doc embedding) is the motivation for the section-chunking A/B, run as a paired
  before/after rather than assumed.
- **pgvector over a two-table schema (persons + chunks).** Keeping vectors and metadata in one
  store enables a native *pre-filter* — resolve a name to a person, then search only that
  person's chunks — which avoids the retrieve-then-filter failure mode of a separate vector index
  plus metadata database (a filter can otherwise leave zero relevant results).
- **Local embedding model behind a thin interface, with an on-disk cache.** Embeddings are cached
  by `(model_id, role, sha256(text))`, so re-runs never re-pay the CPU embed cost; the interface
  hides backend asymmetries so an embedder swap is a config change, not a rewrite.
- **LLM-as-judge with a gold-blind groundedness lane.** The primary metric is scored by a judge
  that never sees the answer key, so it measures *"supported by the retrieved reports"* rather
  than *"true in the world."* A separate correctness lane sees the gold. The judge is a
  higher tier than the generator, never the same model.
- **The judge is calibrated, not trusted.** Blind human labels, adversarial planted-ungrounded
  answers, and a repeat-run flip-rate measure it; an asymmetric trip rule forces a rubric-version
  bump and a full re-score of history if the judge ever passes an ungrounded answer as grounded.
- **Deterministic refusal and citation checks.** Both are string operations, so a drifting model
  stays out of the primary metric; a fabricated citation (a cited document that was never
  retrieved) is flagged automatically.
- **Versioned measurement instruments.** The generation prompt contract and the judge rubric are
  version-stamped and snapshotted into every run — two runs are only ever compared under the same
  instrument.
- **Write-once run artifacts with full config snapshots.** Git SHA, question-set hash, normalizer
  version, and embedding-stack versions are pinned per run, so any metric difference between two
  runs is explainable from the snapshots alone.
- **Contamination control.** Because the subjects are real, a closed-book control (same questions,
  empty context) checks that correctness comes from retrieval rather than the model's pretraining
  memory of these people. It came back 0.00.
- **Hand-authored, validator-checked gold set.** Abstention and multi-hop questions are
  hand-authored (an abstention question requires *proving* the fact is absent from all 268 docs);
  single-hop is LLM-assisted with every gold span verified against the parsed corpus. Gold is
  anchored to document id + verbatim quote, so re-chunking can never invalidate the question set.

---

## Repo layout

```
src/       ingest → embed → store → retrieve → generate (the pipeline)
eval/      run harness, LLM judge, hit-rate, calibration, question schema + validator
runs/      write-once run artifacts (metrics-only files committed)
scripts/   infra smoke check
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
pgvector is reachable and the schema applies, the local embedder loads and returns a 768-d unit
vector — and reports whether your `ANTHROPIC_API_KEY` is set. It never touches the corpus and
spends nothing.

### Running (requires the private corpus in `data/raw/`)

```bash
.venv/Scripts/python src/ingest.py            # raw .md → persons.jsonl + chunks.jsonl
.venv/Scripts/python src/build_index.py       # embed (cached) + load into pgvector
.venv/Scripts/python eval/run.py              # score → runs/<timestamp>-<sha>/
```

`eval/run.py` runs the offline retrieval metrics with no key; add `ANTHROPIC_API_KEY` to also run
the generation + judge lanes. See [eval/README.md](eval/README.md) for the question schema and the
gold-quote validator.

---

## Status

The retrieval and evaluation core is complete: hardened metrics (Wilson intervals, rank/MRR
diagnostics, per-stage cost/latency), a hand-authored and validated question set, a
baseline-of-record, a closed-book contamination control, and a calibrated judge. Next up: a
section-chunking A/B (measured before/after), then a multi-agent routing layer (name resolution +
a web-freshness lane) and a production-monitoring simulation.
