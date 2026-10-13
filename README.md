# Network Assistant — a grounded chat agent over long-form dossiers, with a versioned eval harness

A chat assistant that helps members of a private business network find and connect with other
members. It answers only from long-form member dossiers: when the dossiers can't support an
answer it says so, and when a question refers to something that isn't in the conversation it
asks instead of guessing.

The emphasis is **measurement**. Every architectural change in this repo was made against a
versioned evaluation harness — gold-blind judging, deterministic refusal and citation checks, a
calibrated LLM-as-judge, paired statistics on a small gold set, and write-once run artifacts you
can diff — not on how plausible the output looked.

---

## About the data

The corpus is 268 long-form dossiers about real executives, generated from publicly available
information. Although every fact is individually public, aggregating them into per-person
profiles makes the collection sensitive — so **the corpus is not included in this repo** and is
gitignored. What *is* here: all agent, pipeline and eval code, the question-set schema and
validator, a public schema example ([eval/questions.example.jsonl](eval/questions.example.jsonl)),
and the **metrics-only run artifacts** (`config.json` / `summary.json` per run — files carrying
dossier text or names are withheld).

A fresh clone can read the code, inspect the measured results, and stand up the infrastructure.
Running the assistant or a full evaluation additionally requires the private corpus.

---

## Architecture

```
browser chat (Gradio)            member picker stands in for login; tool steps shown while it works
      │
      ▼
LangChain agent                  create_agent on the LangGraph runtime
(src/agent.py)                   · one system prompt: answer only from dossiers, exact refusal
      │                            sentence, clarify on missing context, cite every claim
      │                          · per-conversation memory: LangGraph Postgres checkpointer
      │                          · guardrail hooks (code, not prompt):
      │                              - after the model: a reply naming a member no tool returned
      │                                is replaced with the refusal sentence before it is saved
      │                              - inside the tools: a person_id the conversation never
      │                                showed is rejected (the model must not construct ids)
      ▼
tools (src/tools.py)             find_members        discover members by criterion, network-wide
      │                          get_person_evidence facts about named people / companies
      ▼
resolver (src/resolve.py)        names and company mentions → person_ids, deterministically;
      │                          an ambiguous first name becomes a clarifying question
      ▼
Postgres + pgvector              section-header chunks (adopted via a measured A/B, D-023),
                                 exact cosine search, person_id pre-filter for scoped retrieval
```

Claude Haiku 4.5 runs the agent; a Sonnet-class model judges, never the same tier. Citations
(`[profile: …]`) are kept in every stored reply — they drive fabricated-citation scoring and let
any complaint be traced to the excerpt that produced it — but are hidden on screen, because a
member cannot open another member's dossier to check one (D-043).

---

## Measured results

Same 59-question gold set, paired by question. Wilson 95% intervals. Full numbers live in each
run's `summary.json`.

| Metric | Single-call pipeline | Agent (tool loop) | Reading |
|---|---|---|---|
| answerable correctness | 0.89 [0.77, 0.95] | **0.96** [0.86, 0.99] | 0 questions got worse, 3 got better (exact McNemar p = 0.25 — direction, not proof) |
| false-refusal rate | 0.09 | **0.04** [0.01, 0.14] | the agent searches network-wide when a question names only a company |
| abstention accuracy | 1.00 | 1.00 | unanswerable questions still refused |
| **groundedness** (primary) | 1.00 [0.92, 1.00] | 0.93 [0.82, 0.98] | 3 of 45 answers flagged; the judge was calibrated on the older answer format, so re-calibration comes before treating this as a regression |
| fabricated citations | 0 | 0 | no answer cited a dossier it was not shown |
| cost / question (incl. judge) | ~$0.05 | ~$0.04 | |

Pipeline: run `20260730T022029Z-5d5bcf2`. Agent: run `20261012T125451Z-aeae465-agent`, measured
on the coordinator tool loop that preceded the current LangChain agent (same tools, retrieval and
guardrails); re-measuring the LangChain agent is the next evaluation run.

Retrieval (pipeline baseline): single-hop hit@1 0.66, hit@3 0.78. **Closed-book control: 0.00** —
with empty context the model refused every answerable question, so correctness comes from
retrieval, not from pretraining memory of these real people. **Prompt caching** cut input cost
42% by the third turn of a conversation (each turn re-reads earlier turns at ~0.1×); it does
nothing for single-question evals, where Haiku's 4,096-token cache minimum is never reached.

---

## How it evolved

Each step was a recorded decision with a measured reason ([DECISIONS.md](DECISIONS.md)):

1. **Single-call RAG pipeline + eval harness** (D-008–D-022). Retrieve top-k, one generation
   call, a calibrated judge. Established the metrics before optimizing anything.
2. **Section-header chunking** (D-023). A four-arm paired A/B: plain section chunking *lost* to
   whole-document chunks; a decomposition arm showed the cause was context loss (sections without
   the person's name), not corpus size — prepending the name recovered +0.20 hit@1 (p = 0.039).
3. **Agent tool loop** (D-024–D-038). Real conversations refer back to earlier results ("the
   first one", "anyone else among them"), which no fixed per-query workflow could enumerate, so
   the model moved into the control seat over typed, deterministic tools. Built hand-rolled at
   first, with a separate answer-writing call so answer quality could be measured in isolation.
4. **Standard LangChain agent + chat interface** (D-042). Measurement showed the separate
   answer call cost one model call per turn without improving groundedness, and the loop,
   memory and provider plumbing were exactly what an agent framework's runtime provides. The
   earlier rejection of LangGraph had judged it by graph size; its value is the runtime
   (per-conversation persistence, streaming, resume, tracing), which even a one-step chatbot
   needs. The earlier systems are kept in [src/baselines/](src/baselines/) because every recorded
   run came from one of them.

---

## Design philosophy: no one-sided metric

Every headline metric is paired with a counter-metric that catches the way the headline lies.

| Headline | What it can't see alone | Counter-metric |
|---|---|---|
| **abstention accuracy** (correctly declined?) | a bot that refuses *everything* scores 100% | **false-refusal rate** — punishes over-refusal on answerable questions |
| **groundedness** (supported by retrieved context?) | a fluent answer grounded in a mis-retrieved dossier passes | **correctness + retrieval hit**, jointly |
| **correctness** (matches gold?) | could be the model reciting memory of a real person | **closed-book control** — same questions, empty context; came back **0.00** |
| **fabricated-citation rate** | an answer that cites *nothing* fabricates nothing yet is untraceable | **zero-citation rate** |
| **clarify accuracy** (asked when context was missing?) | a bot that asks "what do you mean?" all the time scores 100% | **clarify-on-answerable rate** — asking instead of searching is a false refusal by another name |
| **judge groundedness score** | the ruler itself can be miscalibrated | **blind human calibration + flip-rate** — the judge is measured, not trusted |

The refusal label is a deterministic string check, so a drifting model never sits inside the
primary denominator.

---

## Key design decisions

- **Guardrails in code, not in the prompt.** A prompt is a request; a hook is a law. Name
  provenance and person-id provenance are checked by code on every turn and fail safe (the reply
  becomes the refusal sentence) rather than relying on the model to comply.
- **Two kinds of "can't answer".** Missing *context* ("which of these leaders?" with no list
  shown) is user-fixable, so the agent asks. Missing *data* (a clear question the dossiers don't
  answer) is not, so it refuses with one exact sentence (D-040).
- **Standard agent over a custom graph — for now.** A custom LangGraph graph would keep the
  separate answer-writing step and declared outcomes; the standard agent was chosen to get a
  usable product, with a stated condition for dropping one layer down within the same stack
  (D-042).
- **Section-header chunking, adopted after a measured A/B and a decomposition (D-023).**
  Reversible via a scheme-tagged store: whole-document chunks are one config flag away.
- **pgvector over a two-table schema (persons + chunks).** Vectors and metadata in one store
  enable a native pre-filter — resolve a name to a person, then search only that person's
  chunks — avoiding the retrieve-then-filter failure where a filter leaves zero results.
- **Local embedding model with an on-disk cache** keyed by `(model_id, role, sha256(text))`, so
  re-runs never re-pay the CPU embed cost and an embedder swap is a config change.
- **LLM-as-judge with a gold-blind groundedness lane**, calibrated against blind human labels,
  planted ungrounded answers and a repeat-run flip rate, with an asymmetric trip rule that forces
  a rubric bump if the judge ever passes an ungrounded answer.
- **Versioned measurement instruments and write-once runs.** Prompt contracts and the rubric are
  hashed into every run with the git SHA, question-set hash and embedding-stack versions, so any
  difference between two runs is explainable from the snapshots alone.
- **Hand-authored, validator-checked gold set.** Abstention questions require *proving* the fact
  is absent from all 268 dossiers; every gold span is verified against the parsed corpus and
  anchored to document id + verbatim quote, so re-chunking can never invalidate a question.

---

## Repo layout

```
src/            the product: agent.py (LangChain agent), tools, resolver, store, ingest/embed
src/baselines/  earlier systems kept runnable because every recorded eval run came from them
prompts/        versioned prompt files, hashed into run configs
eval/           run harnesses, LLM judge, citation parser, calibration, gold schema + validator
runs/           write-once run artifacts (metrics-only files committed)
scripts/        chat_app.py (the interface), infra smoke check
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

# 3. Keys
cp .env.example .env        # then paste your ANTHROPIC_API_KEY

# 4. Verify the setup (no corpus, no paid API calls needed)
.venv/Scripts/python scripts/smoke.py
```

### Running (requires the private corpus in `data/raw/`)

```bash
.venv/Scripts/python src/ingest.py            # raw .md → persons.jsonl + chunks.jsonl
.venv/Scripts/python src/build_index.py       # embed (cached) + load into pgvector
.venv/Scripts/python scripts/chat_app.py      # the chat interface, at the printed local URL
.venv/Scripts/python eval/run.py              # pipeline baseline eval → runs/<timestamp>-<sha>/
.venv/Scripts/python eval/agent_run.py        # tool-loop baseline eval (paid: ~$2 for the full gold set)
```

Self-checks that need no API key: `python src/agent.py` (scripted model, real tools) and, from
`src/`, `python -m baselines.coordinator`.

---

## Status

Working: the chat interface over the LangChain agent, with per-conversation memory, guardrail
hooks and missing-context clarification. Measured: the pipeline and agent baselines above.

Next: point the agent eval at the LangChain agent and re-measure; re-calibrate the groundedness
judge on the agent's answer format; stream replies token by token; a web-search tool for current
events, whose sources will be shown as links; then a production-monitoring simulation.
