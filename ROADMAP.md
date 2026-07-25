# ROADMAP — from here to interview-ready

> Governing plan for the rest of this project. Deliverable: architectural judgment you can
> defend in senior AI-engineer interviews — measured claims, owned decisions, named tradeoffs.
> Scope settled 2026-07-22 after four rounds of revision: private talk-track repo, production
> capability non-negotiable, evals as the learning spine (hand-author the new, automate the
> plateaued), everything else descoped or deferred with a pre-registered trigger.

## 0. How to use this document

- **No fork below is decided.** Each phase lists its genuine design forks with tradeoffs and,
  where earned, a *flagged default*. The decision happens at build time as a DECISIONS.md row
  with YOUR reason, per CLAUDE.md protocol. If your reason parrots the tradeoff text here,
  that's the tell it isn't yours yet.
- **Every number here is a proposal**, carrying its rationale and the symptom that would prove
  it wrong. When implemented, it becomes a `# TUNABLE(reason, revisit when <condition>)`.
- **Phases carry six elements each:** what gets built · why it's worth building (padding named
  as padding) · design forks · interview questions with the exposing follow-up · what to read
  first · focused hours.
- Corrections to FORKS.md discovered during planning are stated inline where load-bearing;
  FORKS.md itself stays untouched as historical record.

## 1. Where the project actually is

**Built (Components 1+2):** ingest → whole-doc chunks (268 dossiers, p50 ~2.8k tok) → local
`nomic-embed-text-v1.5` (768d, disk cache keyed model+text-hash) → pgvector 2-table
(`persons` + `chunks`, `person_id` PRE-filter built but unused) → k=3 cosine → Haiku generator
(versioned refuse-if-absent contract `f6-v1`, document-level citations) → Sonnet-5 judge
(versioned rubric `d010-v1`; gold-blind groundedness lane + correctness lane; two-sided
abstention design). Write-once run artifacts with full config snapshots.

**What run `20260721T023555Z-c232836` actually says (the only run; OFFLINE — the
generation+judge lanes have NEVER executed):**

| Metric | Value | Honest reading |
|---|---|---|
| single-hop hit@1 | 0.60, Wald ±0.43 (n=5) | Wilson gives ≈[0.23, 0.88] — this is a direction, not a number |
| single-hop hit@3 | 0.80 (n=5) | consistent with expected hit@3 saturation over 268 docs |
| multi-hop hit@1/@3 | 0.00 / 0.00 (n=3) | hit@1 is a STRUCTURAL zero (all-spans rule); hit@3 = real failure |
| multi-hop span-recall@3 | 2/6 ≈ 0.33 (recomputed from results.jsonl) | the all-spans bar hides partial progress — hitrate.py's own pre-registered TUNABLE **has already fired** |
| groundedness / correctness / abstention | null | `api_available: false` — half the instrument is untested code |

**The multi-hop zero is two different failures, not one** (results.jsonl):
- `mh-001`/`mh-002` — named 2-person comparisons that fetched exactly one of the two people.
  mh-001 fetched Craig Hunter at rank 1 and the **wrong Ross** at rank 2 (Ross Woolley over
  Ross Fernandes) — a live exhibit of the G-001 resolution gap. Fixable by person-scoped
  retrieval (Phase 3).
- `ex-mh-1` — an unnamed join ("two leaders in the Alliance of CEOs") that fetched NEITHER
  gold doc. That is a rare-exact-phrase / hybrid-BM25 signature, NOT a decomposition problem —
  and the question itself is borderline aggregation under the goldset skill's rules. Re-review
  it before it anchors any claim.

**Corrections to the written record this roadmap must carry:**
- FORKS.md says "100% follow one fixed 15-section `### N.` template (16 off-template
  deleted)." **Stale.** Census: 252/268 use `### N.`, 16 use `## N.`, ≥5 stop at §12, one has
  freeform subsections. A `###`-keyed splitter silently leaves 16 docs whole-doc *inside* the
  "section-chunked" config and dilutes the A/B. Where one documented claim is wrong, expect
  undocumented variance.
- `summary.json` prints `ci95_halfwidth: 0.0` for multi-hop at n=3 — the Wald formula
  collapses at p∈{0,1}. An interviewer who opens the file sees a zero-width confidence
  interval. Cheapest credibility fix in the repo.
- Retrieval ranks below top-3 are discarded, so METRICS.md's own "rank 67" example cannot be
  reproduced from any run artifact. Runs are write-once → **instrument before baselining**,
  or the before-picture is permanently missing its diagnostics. (hit@1 stays the ONLY
  headline; ranks are the zero-cost diagnostic that routes the next build — miss at rank 2–5
  ⇒ ranking problem ⇒ rerank/bump-k; miss at rank 40+ ⇒ representation problem ⇒ re-chunk.
  Opposite prescriptions, currently indistinguishable as `hit@3: false`.)
- Fresh clone cannot run: no README, no requirements/pyproject, no compose, no .env.example.
  Single commit; the only run was made from a dirty tree (`"dirty": true` — the SHA pins
  nothing).
- Cost of running evals is NOT a constraint: ≈$0.04–0.05/question full pipeline (Haiku 4.5
  generation ≈$0.01, Sonnet-5 groundedness ≈$0.03, correctness ≈$0.002 — estimates; verify
  against recorded `usage` once Phase 1a lands). An n=80 full run ≈ $3–4.
- Iteration tax that IS a constraint: ~49 min CPU per full re-embed (new chunk text = cache
  miss). Batch chunking experiments; run overnight.

**Open debts to close at Phase 1 start:** confirm the judge temp-0 resolution as a D-010
amendment note (Sonnet-5 rejects temperature; shipped = omit temp + structured output +
thinking disabled), run the D-013 closing drill (the joint signature: correctness high while
hit-rate flat = retrieval bypassed), confirm/replace D-017's drafted reason. Plus the two
logged gaps this roadmap closes by construction: G-001 (resolution eval, Phase 1c/3) and
G-002 (calibration protocol, Phase 1e — designed below, decided by you).

## 2. The eval ladder — the learning spine

Your fact-based retrieval questions have stopped teaching you anything; that is not a flaw in
them (checkable gold is a feature — the goldset skill's fuzzy-gold rejection exists precisely
to keep them fact-based), it means that rung is learned. The ladder: **hand-author a few of
each NEW eval type — that's where the learning is — then automate expansion under your
supervision** (which is what D-012 pre-registered: LLM-assist, human-verify every gold span).

| Rung | Eval type | You hand-author (the learning) | Automated under your supervision |
|---|---|---|---|
| 1 Retrieval | fact-based hit@k, span-recall, rank diagnostics | done — the 12 existing + validator design taught this; a few seeds for new strata (buried-fact, clean comparisons) | `eval/generate_questions.py`: LLM drafts single-hop from sampled docs/sections → auto-validated (quote containment) → review queue → you approve/reject; gated by the eval-goldset-review skill. Paraphrase variants same path |
| 2 Generation | groundedness / correctness via LLM judge | ~24 blind calibration labels (G-002) — inherently manual; NEW skills: judging the judge, flip-rate, asymmetric trip rules | judge scores at scale once calibrated; recalibration on a cadence |
| 3 Behavior | two-sided abstention; resolution third state (G-001) | 6–8 resolution questions (answer/refuse/**clarify** — NEW outcome; the Solaru pair and the wrong-Ross collision are ready-made); new abstention sub-types with absence proofs | LLM proposes abstention candidates; you prove absence (grep campaign, patterns recorded in `notes`) |
| 4 Agent | routing lanes, why-multi-agent rows, trajectories | ~12 routed questions with lane labels (corpus/web/both/neither/ambiguous) — NEW type; fixture-relative web gold | lane-labeled expansion later via the same generator+review path |
| 5 Production | online monitoring WITHOUT gold labels | detector and threshold design, canary selection, incident-drill design — NEW discipline | canary replays on schedule; sampled async judge; deterministic proxies run themselves |

Supervision protocol for the generator (numbers are proposals): batch of 10 drafts per review
round; **rejection-rate is the health metric** — if you reject <10% you're rubber-stamping
(tighten review), if >50% the drafting prompt is junk (fix it before scaling). Provenance
field `author: llm-assisted` per D-012; every gold span validator-proven before merge.

## 3. Eval gap analysis — what a serious setup has that this one doesn't yet

### 3.1 What is already strong (and earned — say it plainly in interviews)
Versioned measurement instruments (prompt contract + judge rubric, snapshotted per run);
gold-blind groundedness lane (the judge that defines your primary metric never sees the
answer key); write-once run artifacts with full config snapshots; gold anchored to
doc_id+verbatim-quote so re-chunking can never invalidate the question set; two-sided
abstention design; pre-registered deferred forks with trigger signatures; embedding cache.
This is above typical portfolio work. The gaps below are execution and statistics, not
architecture.

### 3.2 The gaps, ranked by (interview-signal × cheapness-to-close)

**Tier 1 — sub-2h instrument fixes (Phase 1a; do before ANY baseline-of-record):**
1. **Wilson intervals** replacing Wald; report `[lo, hi]` (Wilson is asymmetric). Kills the
   ±0.00 artifact.
2. **span-recall@k** as a secondary column next to all-spans hit@k (all-spans stays headline;
   the TUNABLE's own revisit condition fired).
3. **Full gold-rank + MRR recording** per question (exact search already scores all 268 rows
   — record instead of discard; cannot be backfilled into write-once history).
4. **Per-stage tokens / cost / latency** — `resp.usage` is currently discarded by both
   generate.py and judge.py; add wall-clock per stage. Feeds Phase 5 from day one.
5. **Deterministic refusal cross-check** — the contract mandates an exact refusal string;
   record `refusal_exact` alongside the judge's `is_refusal` and count disagreements.
   `is_refusal` errors contaminate TWO metrics: the abstention lanes and the groundedness
   denominator (`non_refusal`).
6. **Citation coverage/validity parse** — the prompt mandates `[doc_id]` tags; the harness
   never reads them. That is D-009's own warning verbatim: "an unverified citation launders
   hallucinations." Report per answer: fraction of sentences tagged, cited-doc ∈ retrieved
   set, and the fabricated-citation flag (cited doc ∉ retrieved). Becomes the Phase 4 gate
   and the Phase 5 sensor.
7. **Per-type correctness breakout** (currently blended across abstention + answerable).
8. **Repro holes:** `sha256(questions.jsonl)` into config (edit a quote without changing n
   and two runs silently score different gold); `NORMALIZER_VERSION` constant snapshotted
   like the rubric; rule: comparison-grade runs from a clean tree only.

**Tier 2 — the trust layer (before any judge number is used comparatively):**
9. **G-002 calibration protocol, concretely** (your decision row; this is the proposal):
   24 verdicts per rubric version, stratified 8 groundedness-true / 8 groundedness-false
   (oversampled — rarer and higher-stakes; take all if fewer exist) / 4 is_refusal-positive /
   4 correctness-on-abstention, drawn across question types. **Blind mechanic**: a script
   emits shuffled CSV with judge columns stripped; you label before opening results. Stat:
   raw agreement now, Cohen's kappa once ≥50 cumulative labels. **Asymmetric trip rule**:
   agreement <90% trips, AND any single case of judge-passed-but-human-says-ungrounded trips
   on its own — a false "grounded" on the primary metric is the reputational-risk direction
   D-009 exists for. On trip: bump `RUBRIC_VERSION`, re-score ALL runs, never mix (per
   D-010). Symptom the 90% is wrong: trips dominated by trivial phrasing disagreements →
   loosen with a documented reason.
10. **Judge flip-rate** — Sonnet-5 rejects temperature, so stability must be *measured*, not
    asserted: call each lane 3× on the 24 calibration items; flip-rate = fraction
    non-unanimous (≈$2–3). If >5–10% (TUNABLE; symptom: A/B groundedness deltas smaller than
    flip-rate), use majority-of-3 for comparison runs. Position bias is N/A here (boolean
    lanes, nothing pairwise); verbosity bias: eyeball on the 24; family bias: D-017 owns it.
11. **Paired comparison machinery (McNemar)** — `eval/compare.py <run_a> <run_b>`, exact
    binomial on discordant pairs, printed WITH the b/c discordant counts. The n-arithmetic
    that makes a small gold set defensible: independent CIs need ≈42/arm to detect δ=0.30,
    ≈94 for δ=0.20, ≈167 for δ=0.15 (80% power, α=.05) — but the chunking A/B is paired on
    the same questions and near-one-directional (section chunks rescue buried-fact misses,
    rarely break top-of-doc hits), and one-way McNemar reaches significance at ~8/p_flip
    discordant questions: **~30–45 paired single-hops suffice** where independent comparison
    demands hundreds. This argument IS the interview answer to "n=45 looks tiny."

**Tier 3 — dataset and coverage:**
12. **Gold growth n=12 → ~45** via the ladder's supervised automation. Strata targets
    (proposals): ~15 buried-fact single-hop (sections 8–15 / bottom depth-tercile — the
    stratum the whole A/B is *about* currently has 2 questions), coverage including the 16
    `##`-variant docs and ≥2 §12-truncated docs, ~10–12 multi-hop clean 2-person comparisons
    (re-review ex-mh-1), ~12–14 abstention across sub-types (private-fact / off-domain
    contamination-trap / empty-set / not-in-corpus person), 6–8 resolution questions, ~10
    single-hop × 2 paraphrase variants (`paraphrase_of` field; is hit@1 a property of the
    retriever or of the twelve sentences you happened to type?). Growth to ~75 is
    FUTURE-gated: only if A/B deltas land inside the CIs (F5's own revisit rule).
13. **Auto-derived section/depth metadata** — the validator already locates every quote; make
    it emit section number + char-depth percentile per evidence item, plus a coverage matrix
    (sections × types, persons touched). Zero authoring cost; enables "hit@1 on buried facts"
    — the headline the chunking A/B needs.
14. **Closed-book contamination control** — `--no-context` run per gold-set version: generate
    with empty context, judge correctness only; mark each question contamination-prone.
    Activates D-013 as a *measurement* ("how much of correctness is retrieval vs Haiku's
    memory of these real people") instead of a passive trigger. ≈$1.
15. **G-001 resolution lane** — schema gains `expected_behavior ∈ {answer, refuse, clarify}`;
    the judge's groundedness lane extends its boolean `is_refusal` to a `response_mode` enum
    (same call, no new lane); run.py scores the third state. Confidently-wrong resolution is
    the cardinal failure (your own D-009 reason).
16. **Failure-taxonomy sidecar** — `eval/triage.py` writing `runs/<id>/analysis.json` (never
    mutate write-once results): auto-derived buckets `wrong-person@1-gold-in-top3`,
    `gold-near-miss (4–10)`, `gold-deep-miss (>10)`, `multi-hop-partial`, plus judged buckets
    `ungrounded-claim`, `false-refusal`, `fabricated-citation`, `contamination-suspect`,
    `judge-flip`. This is how "groundedness dropped 8 points" becomes a diagnosis instead of
    80 transcripts re-read.

### 3.3 Anti-bloat — eval work you deliberately refuse (the one-line defense is the deliverable)

| Skipped | Because |
|---|---|
| nDCG / MAP | relevance here is binary with 1–2 gold docs — the machinery collapses to MRR, which rank-recording gives free |
| RAGAS / TruLens adoption | the harness's value IS the hand-rolled versioned-ruler discipline; opaque framework prompts would replace an instrument you can defend line-by-line with one you can't |
| Hundreds of synthetic questions | D-012 already found LLM sets come out generic, and the McNemar math shows ~40 stratified paired questions resolve the deltas you actually act on |
| Judge ensembles | triples the ruler's cost before measuring whether the single judge is miscalibrated — G-002 is the evidence-first version of the same move |
| Cross-provider judge now | D-017 pre-registers it behind calibration evidence of family bias; early adoption pays a second SDK/key/drift surface for a hunch |
| BERTScore / embedding-sim metrics | D-010 rejected them for the right reason — can't score abstention, reward overlap over support; re-adding is a second, worse groundedness |
| Batch API for eval runs | 50% off a $4 run buys an async polling loop in a harness whose point is a tight synchronous feedback cycle |
| CI/CD pass-fail gating of OFFLINE evals | a red/green gate fires on noise at n≈45; the runs/ ledger is the offline record. (Distinct from Phase 5's PRODUCTION monitoring — in scope and non-negotiable) |
| Multi-seed variance studies | the pipeline is deterministic except the judge; flip-rate measures exactly that residual without n× reruns |
| Trajectory-eval scaffolding now | eval for a system that doesn't exist yet; it arrives with Phase 3's traces |

## 4. Phases

### Phase 1 — Eval core (~15–18h)

**What gets built** (order forced by write-once runs — instrument, then gold, then baseline):
- **1a Instrument (~6–7h):** Tier-1 items 1–8 above.
- **1b Smoke-run the online lanes on n=12 (~1h, ≈$0.55):** first-ever execution of
  generate+judge — expect breakage (structured-output path, refusal edge cases). This is a
  debugging session, NOT the baseline-of-record.
- **1c Gold growth to ~45 (~4–5h):** build `eval/generate_questions.py` + review queue
  (~2h), supervise expansion (~2–3h). Hand-author only the new types: resolution 6–8,
  abstention sub-types, 2–3 clean comparisons.
  - **STATUS (2026-07-25) — HALF DONE; do NOT read "gold grown to 47" as 1c complete.** The
    LLM-assisted SINGLE-HOP half is done (pipeline built + pushed; single-hop 12→41). The
    HAND-AUTHORED half is NOT started — current counts: multi-hop **2**, abstention **4**,
    resolution **0**. Targets: this bullet says "2–3 clean comparisons"; Tier-3 item 12 says
    ~10–12 multi-hop and ~12–14 abstention — the two numbers disagree, so the multi-hop/
    abstention target is itself a sizing call (apply the D-012 n-logic when authoring, as with
    single-hop n=45). Resolution (~6–8) pulls to Phase 3 (the clarify state doesn't exist yet).
    This is the eval-ladder LEARNING (owner's own authoring, D-012), the main gold work before
    the 1d baseline, and abstention needs an absence-proof per question (~20–30 min each).
    Multi-hop will score ~0 until Phase-3 person-scoped retrieval — expected; it's the
    before-picture.
- **1d Baseline-of-record + closed-book control (~1–2h):** full pipeline on ~45 questions,
  clean tree, plus the `--no-context` contamination run.
- **1e G-002 execution (~3–4h):** 24 blind labels + flip-rate; decide the trip rule as a
  DECISIONS row.
- Phase start: close the open debts — judge temp-0 confirmation (D-010 amendment note),
  D-013 closing drill, D-017 reason confirmation.

**Why it's worth building:** every later phase claims a measured before/after; today no
comparison is legitimate (±0.43) and the ruler is uncalibrated. This phase is also the honest
correction to "I'm ~halfway on evals" — the online half has run zero times. It converts your
least-confident area into your strongest interview material, because "show me your eval
methodology" is where most candidates die.

**Design forks (yours to decide):**
1. Calibration sample design: random-20 (easy, near-zero information on refusal calls) vs
   stratified (proposed 8/8/4/4) vs disagreement-oversampled (max alarm sensitivity, biased
   agreement estimate — report which you measured).
2. Trip semantics: fixed %-agreement vs kappa vs the proposed asymmetric rule (any single
   false-"grounded" trips).
3. Refusal instrument: judge-only (drift-prone model inside a denominator) vs string-only
   (misses hedged refusals) vs both-with-divergence-log (nearly free — flagged default).
4. Generator-supervision protocol: batch size, rejection-rate bounds, when automation earns
   larger batches.

**Interview questions this phase makes you answerable for:**
1. "Your groundedness judge never sees the gold answer — why?" → *"Then what catches a fluent
   answer grounded in a mis-retrieved wrong document?"* (lane separation; correctness +
   hit-rate jointly cover what groundedness can't)
2. "n=45 looks tiny — why is your comparison legitimate?" → *"What exactly makes it paired,
   and what were your discordant counts?"*
3. "Abstention accuracy is 100% for a system that always refuses — where's the counter-metric?"
   → *"Who labels is_refusal, and which OTHER metric silently breaks if that label is wrong?"*
   (the groundedness denominator)
4. "Prove 'Aaron Silva's wife' is absent from all 268 documents." → *"What's the
   false-negative risk of your procedure?"* (paraphrase absence; grep campaigns recorded)
5. "You can't set temperature on your judge — how do you know the ruler reads the same twice?"
   (flip-rate, majority-of-3 fallback)

**Read first:** Cohen's kappa vs raw agreement (when %-agreement misleads on skewed labels);
Zheng et al., "Judging LLM-as-a-Judge" (bias catalog → stratification design); McNemar's test
mechanics (discordant pairs).

### Phase 2 — Section chunking A/B (~5–6h; +1–2h optional control arm)

**What gets built:** splitter keyed `^#{2,3} \d+\.` (NOT `###`-only — the 16 `##` files), with
per-doc section-count assertions and a variance log (§12-truncated docs handled explicitly);
~3.5–4k chunk rows (schema needs no migration — D-016 designed for this); one re-embed
(overnight); **retrieval-lane comparison immediately** (offline, judge-free — D-008's revisit
trigger already fired on run-1 evidence, so re-chunking needs no further permission),
groundedness-lane comparison after 1e passes; McNemar verdicts with discordant counts;
buried-fact case study (the rank-67 probe: where does it land now?); **pre-registered
prediction, written before the run:** named-comparison multi-hop all-spans stays ~0 under any
global ranking (that failure is per-person coverage, which no chunking fixes — it pulls
Phase 3), while span-recall may rise. Abstention/false-refusal held steady is itself a
finding ("chunking didn't break abstention").

**The chunking-strategy map** (your question "is section-wise the only one that'll work?" —
answer: no; it's the only one that exploits structure the corpus declares. This table is
interview ammunition):

| Strategy | Verdict for THIS corpus |
|---|---|
| Whole-doc | measured baseline; fails on buried facts (rank 67) — the dilution thesis |
| Section-aware | uses free template boundaries; its own pre-registered failure is pronoun identity loss → contextual headers (future) |
| Fixed-window 512 | would function; rejected in F1+F2 — severs `[cite: N]` from claims, ignores declared boundaries. (F1's "corpus is uniform" premise now corrected: structure present at TWO heading levels — rejection survives) |
| Semantic chunking | pays compute to *discover* boundaries the template *declares*; adds a breakpoint tunable with no falsifying story here |
| Proposition/sentence | too fine for dossier Q&A — k explodes, citation integrity dies |
| Small-to-big / parent-doc | not an alternative — a retrieval-unit vs generation-unit decoupling composable WITH sections (fork below) |
| Late chunking | context-preservation via full-doc token pooling; headers test the same hypothesis cheaper — future-bucket grade |

**Design forks:**
1. Chunk unit: pure sections (~200 tok) vs min-length merge (tiny sections) vs small-to-big
   (embed sections, hand generator the parent doc — richer generation, but hit@k then
   measures something the generator doesn't see; that metric/system divergence must be
   defensible).
2. k for section chunks: 5–8 (Default f's pre-registered range; ~5× context-size change —
   tie to measured token budgets). Symptom k is wrong: a multi-hop gold span sits at k+1.
3. **Optional fixed-512 control arm** (+1–2h + one overnight embed): converts
   "structure-awareness helps" from assumption into measurement and pre-empts *"how do you
   know sections beat naive 512-token chunks?"*. Gold survives any chunking by design — extra
   arms are cheap. Your call at build time.

**Descope alternative, flagged honestly:** this is the only NOW-phase not pulled by the
evals/production/multi-agent triad. Deferring saves 5–6h; Phase 3 works on whole docs
(per-person k=1 fetches both dossiers). Cost: the buried-fact story stays designed-not-run.
Recommendation: keep — cheapest phase, fired trigger, and it's the vehicle for learning
paired-comparison statistics. You decide.

**Interview questions:**
1. "hit@1 rose after re-chunking, but the unit of 'hit' changed — why is the before/after
   still valid?" → *"What breaks if gold had been anchored to chunk ids?"*
2. "Why sections over fixed 512-token windows — and how do you KNOW?" → *(without the control
   arm the honest answer is "by argument, not measurement" — know which one you're giving)*
3. "The buried fact went from rank 67 to rank N — what was the whole-doc embedding averaging
   away?" → *"Why did hit@3 barely move while hit@1 moved a lot?"* (saturation logic)
4. "Sixteen dossiers use a different heading level and five truncate at §12 — what does your
   parser do with a doc it can't split, and how would you NOTICE silent chunk loss six weeks
   later?" (assertions + per-doc counts)

**Read first:** Anthropic's Contextual Retrieval post (to defend deferring headers);
parent-document/small-to-big pattern (to argue fork 1 honestly).

### Phase 3 — Multi-agent + routed evals (~15–18h)

*(Absorbs the formerly-standalone agentic-retrieval phase: decomposition is the RAG agent's
job, measured inside this phase.)*

**What gets built:**
- **Requirements memo first** (~30 min of writing that shapes everything): dossiers are
  point-in-time snapshots → freshness requires a web lane; answers must carry per-source
  provenance; ambiguous person references require interactive clarification; generation must
  be guarded. These requirements *necessitate* the architecture — then defend it against the
  strongest alternative (a single agent with a tool-use loop).
- **Coordinator** = the router D-016 deferred: name-spot against 268 canonical names +
  surname index → resolve person_ids → ambiguous → **clarify** (G-001's third outcome,
  finally scoreable; bare "Ross" hitting Fernandes AND Woolley is the feature, not a bug) →
  lane routing {corpus / web / both / neither} → dispatch → synthesize with per-source
  attribution.
- **RAG agent** wraps existing retrieve+generate, adding per-person PARALLEL pre-filtered
  sub-retrieval for comparisons (the built-but-unused D-016 capability becomes load-bearing —
  which also rescues D-014's justification now that the fake-corpus motive is gone: the
  pre-filter fixed the headline failure). Multi-hop before/after measured here.
- **Web agent** with **recorded fixtures as the eval mode** (live = demo flag). Web-lane gold
  is authored FIXTURE-RELATIVE (correct *given the recorded results*) or scored on
  routing+grounding only — world-truth drifts daily and would break run comparability.
- **Routed gold set**: ~12 hand-authored questions with lane labels (ladder rung 4).
- **Versioned handoff schema** (typed envelope: query, resolved person_ids + ambiguity flag,
  lane + rationale, per-agent evidence with provenance, outcome ∈ {answer, clarify, refuse});
  snapshotted like prompt/rubric versions — the routed eval SCORES fields of this schema, so
  schema design is eval design.
- **Per-step traces in run artifacts from day one** (agent debugging without traces is
  archaeology; these traces are also Phase 5's observability feed).
- **The why-multi-agent defense pack** — the two rows that answer the latency question:
  (i) capability rows where single-agent fails (web-lane, clarify); (ii) the overhead row —
  p50/p95/tokens of agency on plain corpus questions, where the router fast-path should
  collapse to near-direct latency. Measure, don't assert.

**Design forks:**
1. **Substrate — four-way: hand-rolled vs LangGraph vs Google ADK vs Claude Agent SDK.**
   Context: your original project used ADK with no justified reason — re-litigating that
   inherited default with explicit criteria is itself the portfolio story ("I used ADK by
   default; here's my re-evaluation and why I stayed/switched"), and hands-on experience with
   TWO frameworks is rarer interview currency than either alone. Criteria matrix to fill in
   your DECISIONS row: JD frequency (LangChain/LangGraph dominate — and note the layer
   distinction JDs blur: LangChain = components library, LangGraph = orchestration runtime);
   control + drift risk (a framework's release cadence sits inside your measurement
   instrument — the D-010 ruler-drift problem transplanted); state/checkpointing need (a
   3-node workflow needs ~none — so "what would make me need LangGraph" is a flip condition
   you can state: parallel fan-out, resumable long runs, >~5 nodes); ecosystem fit
   (generator/judge are Anthropic; ADK is Gemini-first though model-agnostic via LiteLLM);
   learning ROI (hands-on defense beats name-dropping).
2. Routing policy: rules-first (deterministic, free, eval-able; brittle to phrasing) vs LLM
   router (robust; latency + a versioned prompt that drifts) vs hybrid cascade (rules carry
   the unambiguous majority, LLM the residue — production-realistic, two code paths).
3. Web integration: Anthropic server-side web_search (fewest moving parts; engine outside
   your control; fixtures recorded at message level) vs external API (Tavily/Brave —
   explicit request/response you snapshot; one more key) vs fixtures-only (perfect
   reproducibility; no live path ever demonstrated).
4. Decomposition mechanism: deterministic name-spot (transparent, zero cost; "fails" into
   exactly the clarify behavior you want) vs LLM decomposer (handles phrasing; adds latency,
   a versioned prompt, and hallucinated-person risk — must validate against `persons`) —
   LLM version pre-registered behind a phrasing-failure signature.

**Interview questions:**
1. "Why multi-agent at all when it adds latency?" → *"Show me the failure row AND the
   overhead row."* (the defense pack, by construction)
2. "'Compare Ross and Craig Hunter' — your corpus has two Rosses and your own run log shows
   the wrong one outranking the right one. Walk me through what happens, and what your eval
   scores as correct." (clarify third state; confidently-wrong-is-worse policy)
3. "You used ADK before and considered LangGraph — steelman the one you didn't choose. What
   flips your decision?" → *"Your orchestrator is part of the measurement instrument — what's
   the drift story on a framework that ships monthly?"*
4. "Your eval consumed recorded fixtures — what have you and haven't you proven about the
   live system?" → *"Are your web-lane gold answers true, or true-as-of-the-fixture?"*
5. "Routing accuracy came back 100% on your routed set — why should I not believe it?"
   (n tiny; router and test set share an author; adversarial paraphrases needed)

**Read first:** Anthropic "Building Effective Agents" (this system is a *workflow* — router +
workers; naming it precisely is signal); Anthropic's multi-agent research-system engineering
post; enough Claude Agent SDK + LangGraph hands-on to defend fork 1 in either direction.

### Phase 4 — Runtime verification, sensor-minimal (~2–3h)

**What gets built:** the Phase-1a citation-validity parser becomes a runtime GATE — on
fabricated citation (cited doc ∉ retrieved) or unverifiable claim, the response is gated —
plus **false-block rate** measurement (grounded claims wrongly suppressed — the two-sided
habit applied to the newest component), and the before/after groundedness delta with its
latency price. Its per-response pass/fail stream is Phase 5's primary hallucination sensor.

**Why (and why so small):** a standalone "hallucination detection" phase would duplicate the
already-built offline judge (D-010). Runtime *enforcement* of D-009's own unenforced
guardrail, measured, is the real content. The LLM-entailment cascade is FUTURE, behind the
trigger "deterministic checks false-block paraphrases beyond X%."

**Design forks:** gate action (strip-and-answer vs regenerate-once-with-feedback — better UX,
2× worst-case latency, loop risk — vs abstain-entirely — safest; false-refusal rate spikes,
which Default l's metrics already price); verifier independence when the cascade lands
(same-model-as-judge = the judge grades its own guardrail — circularity; tier-gap mirrors
D-010's logic; cross-family contradicts D-017 unless calibration evidence demands it; the
Citations API alternative changes the answer contract → new PROMPT_CONTRACT_VERSION →
comparability break — a genuine tradeoff, not trivia).

**Interview questions:**
1. "The model cites [Aaron Silva] for a claim that's actually in the Fernandes doc — does
   your check pass it?" → *"So what does your check assert: claim truth, claim support, or
   citation correctness?"* (the distinction IS the phase)
2. "Groundedness rose after gating — show me the metric that catches you suppressing TRUE
   claims, and what the gate costs at p95."
3. "Why doesn't the runtime verifier replace the offline judge?" → *"What measures the
   verifier?"* (guardrail vs ruler; the judge prices the guardrail)

**Read first:** Anthropic Citations API docs; MiniCheck-class cheap-NLI fact-checking and its
failure modes (for the future cascade's design).

### Phase 5 — Production simulation: serving, observability, incident response (~8–12h)

**What gets built** — the demo that answers "how would you detect the model suddenly
hallucinating in production, and how would you fix it":
- **(a) Serving surface:** thin service (FastAPI-class) wrapping the coordinator + a traffic
  generator replaying gold/paraphrase/out-of-scope questions at modest QPS with **seeded
  anomaly windows** (you can't demonstrate detection without incidents to detect).
- **(b) Instrumentation:** per-stage/per-agent spans, tokens, cost — the same fields the run
  artifacts carry, now emitted live.
- **(c) Online monitoring WITHOUT gold labels — the detector streams.** This framing is
  itself the interview answer: *the dashboard is the surfacing layer, never the detector.*
  Streams, layered: (i) inline guardrail/verifier firing rates — Phase-4 citation-validity
  failures, refusal rate, retrieval-score distribution, schema violations — cheap, 100%
  coverage, catch sudden systemic breakage (bad deploy, silently updated model); (ii)
  **sampled async judge**: the Phase-1-calibrated judge scoring N% of traffic as a time
  series (this is what LangSmith online evaluators / Langfuse model-based evals /
  Arize-Datadog LLM observability productize — mainstream practice, not exotic); (iii)
  **scheduled canary replays** of a fixed labeled subset against the prod config — the ONLY
  labeled stream; catches prompt/model/config regressions; (iv) user-feedback aggregates —
  the fourth real-world stream, absent here (no users); the traffic generator stands in.
- **(d) ONE dashboard + 2–3 alert rules** on top of those streams, wired to a real
  notification. Latency/cost alerts are classic SRE golden signals (p95 > SLO, cost/query >
  budget); quality alerts ride the detector streams (proposal: citation-validity 15-min rate
  drops >X points → alert; refusal-rate spike; each threshold a TUNABLE whose symptom is its
  false-alarm rate).
- **(e) ONE incident drill, end-to-end:** inject a fault (pick one: swap generator to a
  weaker model / disable the person pre-filter / truncate retrieved context) → watch
  detection fire → diagnose from traces → mitigate (config rollback / gate tighten) → write
  a half-page postmortem. The postmortem is a talk-track artifact.
- Plus **ONE data-driven optimization** with before/after: generator sweep (Haiku↔Sonnet
  groundedness×cost frontier — Default h anticipated varying the generator) or prompt-caching
  with honest math (per-question contexts defeat context caching; the shared system
  prefix is the real surface) or Batch API for judge lanes (50% of eval cost, zero
  query-path effect).

**Design forks:**
1. Observability substrate: LLM-native (Langfuse self-hosted / LangSmith / Arize Phoenix —
   traces+cost+judge integration out of the box; tool-name recognition in interviews) vs
   generic (OTel SDK → Prometheus/Grafana — the stack most orgs actually run; infra-fluency
   signal) vs layered (OTel emit + exporter to either — architecturally cleanest, most
   setup). Whichever you pick, the OTel GenAI semantic conventions are worth speaking to.
2. Detection policy: deterministic-only (free, misses subtle quality drift) vs sampled-judge
   % (cost curve: at ≈$0.03/judgment, 10% of 1k queries/day ≈ $3/day — pick the % with the
   math shown) vs hybrid. And: who watches the watcher — the production judge inherits
   G-002's recalibration cadence.
3. Alert thresholds + canary cadence: each a TUNABLE; the falsifying symptom is the
   false-alarm rate (an alert that cries wolf is worse than no alert in an interview demo).
4. Fault-injection set: which failure classes to simulate — each should map to a named
   real-world incident type (model swap = silent provider update; pre-filter off = index
   regression; truncated context = serving bug).

**Interview questions:**
1. "You have no gold labels in production — what exactly fires when the model starts
   hallucinating, how long until it fires, and what's your false-alarm rate?"
2. "Walk me through your last incident: detection → diagnosis → mitigation. Which trace
   fields did you actually use?" (if the answer isn't specific, the drill didn't happen)
3. "Your canary set passed but users are seeing garbage — what does the canary not cover?"
   (distribution shift; canary staleness; the traffic the canary doesn't represent)
4. "Monitoring added latency and cost — defend the budget or roll it back." (price-of-safety
   framing: detection latency vs sampling cost vs coverage)
5. "Which stage dominates cost per query, and what did you change because of it?"

**Read first:** OTel GenAI semantic conventions (skim); your chosen backend's quickstart; SRE
golden-signals/SLO vocabulary (one chapter's worth); one solid "LLM monitoring without
labels / online evals" writeup.

### Phase 6 — README + reproducibility bootstrap (~3h: scaffold 1h EARLY, finalize last)

**What gets built:** root README with architecture sketch, numbers table from the latest
clean-tree runs, decision-log index (D-001..D-017 one-liners linking to DECISIONS.md), the
two-sided-metrics story, dashboard screenshot + incident postmortem; plus the repro bootstrap
a fresh clone currently lacks entirely: requirements/pyproject freeze, docker-compose
(Postgres+pgvector, observability backend), `.env.example`, a smoke script. Scaffold early —
writing the narrative frame first forces clarity about what each phase must produce.

**Why:** this is the artifact you actually screen-share. Cheapest phase, disproportionate
interview value. **Forks:** spine (numbers-first vs architecture-first vs decision-log-first
— the decision-log index is the differentiator no other candidate will have); demo mode
(live CLI — needs the bootstrap, riskier — vs recorded transcript vs none).

**Interview questions:** "Two runs disagree by 5 points — from your config snapshot alone,
enumerate everything that could have moved." / "I clone your repo fresh — what happens?"

## 5. Future bucket — deferred WITH pre-registered triggers, not dropped

*(The FORKS.md pattern applied forward: each item names the signature that would pull it.
"What about X?" in an interview is answered by pointing at its trigger.)*

- **Contextual headers** ← groundedness mis-attribution on pronouns in section chunks
  (headers go in the EMBEDDED text only, stored text verbatim, or D-011 span matching
  breaks — the design constraint is already known).
- **LLM decomposer** ← deterministic name-spot fails on phrasings/nicknames.
- **Hybrid BM25 + dense** ← ex-mh-1's unnamed-join class; rare-exact-token misses (company
  names, "GCC", tickers). The most likely trigger to actually fire.
- **Verification cascade (LLM entailment)** ← deterministic gate false-blocks paraphrases
  beyond threshold.
- **Additional incident classes + second optimization** ← after the first drill is solid.
- **Scale story** ← your instinct that 268 docs is a toy count is RIGHT: hit@1 is
  corpus-size-relative (0.60@268 ≠ 0.60@26k) — say so in interviews as the honest caveat.
  Cheap entry: VECTORS-ONLY tier (~4–5h): perturbed-embedding distractors to 10k–100k
  vectors measure hit@1-under-crowding on real questions with no text generation (a fake
  row's text never quote-matches — correct distractor behavior; perturbation scale is the
  hardness knob; distractors clustered too far away void the test — check inter-corpus
  similarity first). Then exact-scan latency curve + HNSW recall/latency sweep including the
  filtered-ANN interaction (recall UNDER person_id filter, not just global). 10x real-text =
  one overnight embed; 100x text infeasible (≈3.5 CPU-days). Without this phase, pgvector is
  defended via: the pre-filter became load-bearing in Phase 3 + one-store-no-drift vs your
  real FAISS post-filter scar + open concession of the numpy steelman logged in D-014.
- **Vector-store swap (Qdrant/Chroma-class)** ← swap surface is already just store.py's three
  functions (per D-015's own logic: NO speculative VectorStore abstraction). Pre-registered
  costs: `persons` has no home in a collection store (re-opens D-014's two-store-drift wound,
  or payload denormalization); exact→HNSW breaks eval-history comparability (mitigation:
  store-as-config-field + recall-parity paired McNemar run; Qdrant's `exact:true` exists for
  such audits); the pre-filter itself SURVIVES (Qdrant filterable HNSW / Chroma `where` —
  your FAISS scar was POST-filter, a different failure). Triggers: native sparse+dense hybrid
  if the hybrid item fires (most likely); filtered-ANN recall at scale; JD-signal via a
  measured migration. Today it buys zero capability over filtered-exact pgvector.
- **Cross-provider judge** ← D-017's trigger (calibration shows family bias).
- **Multi-turn memory** ← no signature fired; no phase exists for it.
- **Gold growth ~45 → ~75** ← A/B deltas land inside the CIs (F5's own rule).
- **Reranking** ← taxonomy shows right-person-wrong-rank (near-miss bucket) dominating.

## 6. Sequencing & scope arithmetic

**Order:** Phase 1 → Phase 6 scaffold (1h) → Phase 2 (or defer — flagged above) → Phase 3 →
Phase 4 → Phase 5 → Phase 6 finalize.

**Edges:** everything rests on Phase 1 (trusted ruler + big-enough n; the calibrated judge is
also Phase 5's sampled monitor). Phase 2 soft-before Phase 3 (build the merge policy once, on
final chunking — but Phase 3 works on whole docs if 2 is deferred). Phase 4's sensor feeds
Phase 5. Phase 5 needs a serveable pipeline (Phase 3).

**Hours:**

| Phase | Hours |
|---|---|
| 1 Eval core | 15–18 |
| 2 Chunking A/B (lean; optional control arm +1–2) | 5–6 |
| 3 Multi-agent + routed evals | 15–18 |
| 4 Runtime gate (sensor-minimal) | 2–3 |
| 5 Production simulation | 8–12 |
| 6 README + bootstrap | 3 |
| **NOW total** | **≈48–60 (plan for ~55)** |

Stated honestly: this is the floor with production + multi-agent non-negotiable. "Focus on
evals" shows up as Phase 1's size and the ladder across every phase — not as a smaller total.
If it must shrink further, the descope order is: defer Phase 2 (−5–6h) → drop the Phase 5
optimization item (−1–2h) → slim Phase 3's web agent to one fixture set + rules-only router
(−3–4h) → **never Phase 1 or Phase 5's core.**

## 7. Where you're underestimating, where you'd be overengineering

**Underestimating:**
- "I'm ~halfway on evals" — the online half (generation + judge) has executed zero times;
  the first API run is a debugging session, and Phase 1 is your biggest phase.
- Absence proofs: ~20–30 min each even with LLM-proposed candidates — grep campaigns across
  268 docs per phrasing, recorded in `notes`.
- Judge calibration is recurring, not one-shot: every rubric bump re-scores all history
  (D-010), and the production monitor inherits the same recalibration cadence.
- Web fixtures are a design problem, not a caching chore: what exactly is pinned, staleness
  policy, and fixture-relative gold (world-truth drifts under you).
- Agent debugging without day-one traces is archaeology — the trace format exists BEFORE the
  coordinator or Phase 3 inflates 1.5–2×.
- Section-parser unknown unknowns: one documented corpus claim is already wrong; assume more
  variance until the per-doc assertions say otherwise.
- The 49-min re-embed tax on every chunking/header variant — batch experiments, run
  overnight, never iterate interactively.
- Observability stack friction: backend setup, Docker-on-Windows networking, and alert
  tuning against false alarms is where "quick Grafana setup" dies. Budget real hours.

**Overengineering / padding (named per your request):**
- Dashboards without traffic, alerts, and an incident — a static screenshot is theater; the
  drill is what makes it real.
- A standalone "hallucination detection" phase — the offline judge exists (D-010); runtime
  sensor + measured gate + drill is the actual content.
- Framework adoption purely to name-drop — adopting one to defend the fork hands-on is
  legitimate; adopting without being able to steelman the alternative is the padding version.
- 100x fake corpus; LLM-generated distractor text at any scale before the vectors-only tier.
- Fancy fuzzy person-resolution at 268 people — lookup table + surname index + clarify
  policy is the entire justified scope.
- Multi-turn memory (no fired signature), web-agent productionization polish (retries, rate
  limits, engine comparisons) beyond one working path + fixtures, and hundreds of synthetic
  questions (the McNemar math says ~40 stratified paired questions resolve what you act on).
