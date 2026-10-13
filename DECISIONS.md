# Architecture Decision Record

This log records the architecture decisions for a grounded assistant over long-form
dossiers — first a retrieval-augmented Q&A pipeline, later a conversational agent —
together with its evaluation harness. Superseded entries are kept and marked, so the
log reads as the system's actual evolution. Each entry captures
the context, the options weighed, the decision, the rationale, and the condition
that would force a revisit. Numbering begins at D-008 because earlier identifiers
belong to a separate, parked data-preparation effort outside the scope of the
retrieval and evaluation system documented here. The D-0xx identifiers are stable
and are referenced from the code.

---

## D-008: Chunking unit + embedder

**Context.** Chunk size and embedder context window are a single joint decision:
a whole document runs ~2,840 tokens at p50 (~5,300 max), which does not fit the
small local embedders (bge-small at 512 tokens, MiniLM at 256), so choosing
whole-document chunks forces a long-context embedder.

**Options.**
- Whole-doc + long-context embedder (Voyage-3 32k, OpenAI 8,191, or local nomic 8,192). Simplest indexing unit; hides intra-document attribution.
- Section-aware (~200 tok) + small local embedder. Sharper hit-rate, but more moving parts than a baseline needs.
- Fixed-window (512 tok) + small local. Template-agnostic, but severs the inline `[cite: N]` markers.

**Decision.** Whole-document chunks (one per document). The embedder is resolved
as a procedure rather than a second standing decision: attempt local
`nomic-embed-text-v1.5` once; if the install is blocked in this environment, fall
back to the Voyage-3 API.

**Rationale.** The documents share a fixed 15-section template, so section-aware
chunking would be straightforward. Whole-doc was chosen first deliberately, to
observe its failure modes directly before adding structure. Local embedder first
because it is free and fast to set up — worth one attempt before paying for an API.

**Note — a quiet failure mode.** Whole-doc chunking fails silently: single-hop
hit-rate saturates near 100% regardless. The metrics that actually move are
groundedness and multi-hop hit-rate, so those are watched, not the headline
hit-rate.

**Revisit when.** (Chunking) groundedness is poor and traceable to
mis-attribution inside a ~2.8k-token blob, or multi-hop hit-rate floors low → move
to section-aware chunking (bge-small/MiniLM become viable again). (Embedder) the
local download is blocked → Voyage-3.

---

## D-009: Answer-prompt contract

**Context.** Whether the bot *can* abstain is a property of the generation prompt,
not just the model. The prompt is therefore half the measurement instrument.

**Options.**
- Bare "answer from context." The model answers even when the context lacks the fact; abstention becomes accidental and unmeasurable.
- Answer-only-from-context + may-refuse. The minimum that makes abstention measurable.
- The above + mandatory citations, either document-level (cheap) or source-level `[cite: N]` mapping (substantially more work).

**Decision.** "Answer only from the provided reports; if the answer isn't there,
say you don't know," plus document-level citations (each claim tagged with the
report it came from). Source-level `[cite: N]` mapping is deferred. Prompt text is
versioned and snapshotted with each run's config.

**Rationale.** Even the simplest version must be genuinely grounded in the source
documents rather than eyeballed as plausible. A confidently false answer about a
corpus subject is the reputational failure the system exists to prevent.

**Guardrail.** A citation only counts if the harness *verifies* it — does the
cited report actually contain the claim? An unverified citation launders
hallucination and defeats the purpose.

**Revisit when.** Groundedness or abstention errors trace to prompt phrasing →
tighten the contract or pull source-level citations forward.

**Known limitation (deferred).** The contract's user-facing voice exposes the
internal "reports" framing — answers open with "Based on the provided reports…"
and refusals read "I don't know based on the provided reports." That framing is
load-bearing internally: the groundedness rubric ("supported by the provided
reports alone") and the deterministic refusal detector (D-019) both depend on it.
Re-skinning the presentation voice belongs at a later coordinator/synthesis layer
that owns user-facing output, not in this contract, where a reword would reset
baseline comparability.

---

## D-010: LLM-as-judge

**Context.** The judge is the measurement ruler; changing it silently re-scores all
prior runs.

**Options.**
- LLM judge with a fixed rubric. Handles abstention nuance; risks drift and self-preference.
- Deterministic overlap (ROUGE / embedding similarity). No drift, but cannot score abstention and rewards word overlap over genuine support.
- Human-only. Highest trust, does not scale.

**Decision.** A pinned Claude Sonnet judge (never the generator's Haiku tier),
with a versioned, snapshotted rubric, plus ~20 human-checked judgments per rubric
version as a drift alarm. The groundedness judge sees the retrieved context and the
answer only (never the gold answer); the correctness judge sees the gold and the
answer. Groundedness is the primary metric, correctness secondary; the rubric
wording is "supported by the provided reports alone."

**Rationale.** Judging whether a refusal is appropriate is semantic and subjective;
word-overlap scoring would flag correct answers that share no vocabulary with the
gold. An LLM judge handles this, with humans spot-checking a sample as insurance
against drift.

**Revisit when.** Human/judge agreement on the calibration sample diverges, or the
rubric changes — in which case all runs are re-scored under the new version and
versions are never mixed.

**Later refinement — temperature is not settable.** The intended "temperature 0"
is not accepted by the shipped judge model (it rejects a non-default temperature
with a 400). Resolution in `judge.py`: omit temperature and pin stability instead
via the fixed rubric, disabled extended thinking, and a `json_schema` structured
output. The judge stays Sonnet-class and above the generator's Haiku tier, so the
real invariant — the ruler outranks the thing it grades — holds. Determinism was
never guaranteed by temperature anyway; the residual non-determinism is *measured*
by the judge flip-rate (D-022), not asserted.

---

## D-011: Span-matching / hit-rate

**Context.** How the grader decides retrieval fetched the right passage. This
compares a true quote *from the document* against the *retrieved document text*
(both are source text, no paraphrase) — distinct from groundedness (D-010), which
judges the paraphrased generated answer.

**Options.**
- Quote-substring exact. Simple, but brittle to formatting and `[cite: N]` noise.
- Char-offset intersection. Precise, but any normalization change breaks every label.
- Fuzzy: normalize-then-exact (safe) or token-overlap threshold (false-hit risk).

**Decision.** Normalize-then-exact containment: normalize both the gold quote and
the retrieved text (lowercase, collapse whitespace, strip punctuation and
`[cite: N]` markers), then require the full quote to still appear. No loose
token-overlap threshold. Gold quotes are authored from the parsed/normalized text
the system actually sees, not the raw files.

**Rationale.** Source documents carry stray whitespace and special characters, so
text is cleaned before matching. Because this compares retrieved source text
against the true source sentence (not the paraphrased answer — that is the LLM
judge's job), an exact match after cleaning is correct and avoids false-positive
inflation of hit-rate.

**Revisit when.** Real quotes still fail to match after normalization → fall back
to a token-overlap threshold with an explicit tunable value and a documented
false-hit symptom.

**Later refinement — doc_id anchoring.** The original `hit_at_k` matched the gold
quote against *any* top-k chunk's text with no check that the chunk came from the
gold document. That silently assumed each gold quote uniquely identifies its
document — true for the distinctive hand-authored seed quotes, false once
LLM-assisted single-hop generation introduced generic quotes (e.g. a university
name appearing in 15 documents, a job title in 5). Failure mode: retrieval *misses*
the gold document, but a *different* document containing the same phrase lands in
top-k → scored a false hit, inflating hit-rate and hiding the miss. Fix: hit@k (and
span-recall) now require `retrieved.doc_id == gold.doc_id AND quote in that chunk` —
the quote pins the section (surviving section-chunking) and the doc_id pins the
subject. This also aligns hit@k with gold_rank, which already matched on doc_id.
Surfaced by a cross-document quote-collision scan of the first LLM batch; caught
before any baseline-of-record existed, so there was no history to re-score.

---

## D-012: Question-set construction

**Context.** Abstention questions need proof the answer is *absent* from all 268
reports; multi-hop questions need facts spanning several. Neither is safe to
auto-generate.

**Options.**
- Hand-seed + verified LLM expansion. Trust where it matters, scale where it is safe.
- Fully hand-authored. Maximum trust, small n → noisy metrics.
- LLM-generated + verified. Scales, but weak precisely on abstention and multi-hop — the types that matter most.

**Decision.** Hand-author the seed set — all abstention and all multi-hop by hand —
then LLM-assist single-hop drafting with every gold span human-verified. LLMs
brainstorm alongside the author; they never author unattended.

**Rationale.** LLM-generated eval sets tend to come out generic; deliberate human
thinking is what makes the set exhaustive, with LLMs used only to brainstorm.

**Revisit when.** The hand-authored n is too small to separate two configurations
(confidence intervals overlap) → grow single-hop via verified LLM expansion.

**Sizing — single-hop (n=12 pilot → ~45 target).** The target is computed, and two
independent calculations agree:
- *Precision (one config's band).* A Wilson/normal half-width `w` at rate `p` needs `n ≈ (1.96/w)²·p(1−p)`. At the pilot's single-hop p≈0.6: ±0.20→~23, ±0.14→~47, ±0.10→~94, ±0.07→~188. 45 buys ≈±0.14. Halving the band costs ~4× the sample (the 1/√n wall), so the stopping point is "narrow enough to decide," not "narrow."
- *Power (the chunking A/B — the binding calc).* Pre-registered δ=0.30, α=.05, power=.80 → ~42/arm with independent CIs. This drops to ~30–45 *total* because the A/B is paired (same questions both configs) and near one-directional, so McNemar counts only discordant pairs (~8/p_flip). This is the "why 45, not 84" argument.
- Growth trigger is pre-committed: grow toward ~75 *only if* the A/B delta lands inside the CIs (i.e. the true effect is smaller than the 30 points powered for, so the test is underpowered for *this* effect). Not "if the bands still look broad" — that is optional-stopping / p-hacking. `# TUNABLE(n=45 powered for δ=0.30 paired; recompute from observed p_flip if the measured A/B effect < 0.30. Symptom too-small: discordant count < ~8/p_flip ⇒ McNemar can't reach significance.)`

**Sizing — multi-hop (2 → 6).** Multi-hop measures a different thing than
single-hop (a person-scoped-retrieval before/after, not the chunking A/B), so it
sizes differently:
- *Not power-limited.* The effect is large and near-deterministic — before ≈ 0 (a global top-3 rarely fetches *both* named documents), after ≈ 1 (per-person retrieval fetches each directly). For a paired before/after where nearly every question flips miss→hit, one-way McNemar reaches p<0.05 at ~6 discordant pairs (0.5⁶·2 ≈ 0.03). So ~6 already proves the feature worked; more buys almost no certainty.
- *Coverage-limited on two axes*, 2 per mode (n=1/mode can't separate a real mode-effect from a fluke). Axis 1 = retrieval difficulty {different-domain, same-domain, same-name}, where the same-name tier tests the disambiguating resolver, not just retrieval. Axis 2 = attribute/reasoning type, including a numeric mode {magnitude/units, year-inversion where older = earlier = smaller number}. The 6 questions cover both axes via double-duty, so numeric costs no extra rows and stays diagnostic (a numeric-only drop says *which* trap failed).
- Grow trigger is pre-registered per tier and per mode: add to a specific tier/mode only if its before/after is *ambiguous* (e.g. same-name flips 1-of-2, or numeric fails one trap but not the other). Not "bands look wide" and not "it's cheap, add more." `# TUNABLE(6 = 2 per retrieval-tier + numeric via double-duty; grow a tier/mode only on ambiguous before/after. Symptom too-small: a tier's flip rate is neither ~0 nor ~1.)`

**Sizing — abstention (4 → 12).** Same 2-per-mode discipline; the mode axis is the
*reason* the bot should refuse, and the binding cost is the absence proof, which
varies enormously by mode:
- *Cheap (no full-corpus grep):* not-in-corpus (proof = a subject-registry lookup), structural-absence (proof = the 15-section template has no contact section, corroborated by 0/268 phones and 1/268 emails), superlative/computed (proof is conceptual — a global max/min over 268 subjects is not computable from a k=3 retrieval; these test locally-grounded-but-globally-wrong, where the failure is answering "youngest of the 3 retrieved"), and off-domain.
- *Medium (one targeted grep) — empty-set:* the riskiest cheap mode, because its proof depends on "nobody matches," knowable only by grep. Concrete example of the false-negative risk: a proposed "no one is from university X" was rejected when a grep found a subject who studied there; it was replaced with an institution the grep confirmed at 0/268.
- *Expensive (full grep campaign) — private-fact:* the hard ones were already authored, so this round was almost entirely cheap.
- *Deferred — advice-reframe* ("how should I pivot to SaaS?"): the intended gold behavior is to *redirect* to relevant subjects, but the harness scores a binary refuse-vs-answer, and a redirect reads as a non-refusal and would mis-score. This needs a response-mode enum {answer, refuse, clarify, redirect} that does not yet exist; forcing gold=refuse would bake in the opposite of the intended behavior. `# TUNABLE(12 = 2 per refusal-mode; grow a mode only if its refuse-rate is ambiguous. advice-reframe blocked on the response-mode enum.)`

---

## D-013: Base-model contamination of correctness

**Context.** Corpus subjects may appear in the base model's pretraining data, so
the generator could answer a question correctly *without* retrieval — masking a
broken RAG pipeline behind memorized facts.

**Options.**
- Mitigate + pre-register a formal trigger for a names-only anonymizer.
- Anonymize now (names-only). Cleanest, but invests work in a deliberately parked component.
- Accept + lean on groundedness. Simplest; document the blind spot.

**Decision.** Accept the contamination; do not anonymize now. Rely on groundedness
(already the primary metric, scored "supported by the reports alone" per D-010) to
catch claims sourced from pretraining rather than the reports. The full anonymizer
stays parked. This entry documents the blind spot.

**Rationale.** The pipeline already runs groundedness checks that catch claims not
supported by the retrieved reports, and at scale per-corpus anonymization is
impractical when grounding enforcement already catches ungrounded claims.

**A qualifier the analysis exposed.** The rationale above is incomplete as first
written: groundedness catches a pretraining-sourced claim only when the retrieved
context does *not* already contain the fact. When pretraining and context agree,
groundedness returns true — proving *support*, not that retrieval was
*load-bearing*. Neither groundedness nor hit-rate can establish load-bearingness;
only a retrieval ablation (closed-book) can.

**Metric exposure to contamination.** hit-rate is contamination-*proof* (embedder +
gold quote only, no generator in the calculation — memory cannot move it);
groundedness is contamination-*robust* (checks answer ⊆ context, so a memorized
answer that also sits in context is genuinely grounded — it never gives a false
green on hallucination, it just can't reveal source); correctness is
contamination-*vulnerable* (memory inflates the pass rate). Consequently an
all-green row (hit + grounded + correct) is not a contamination worry — it is
carried by the two resistant metrics, and hit-rate independently certifies the
retrieval capability memory was suspected of masking. The dangerous false green is
correctness-high × hit-rate-*low* (a green answer with retrieval absent). This is
the architectural reason correctness stays secondary and caveated.

**Closed-book control — result.** A closed-book control (`eval/closed_book.py`)
runs the generator with *empty* context, keeping the full answer contract (rule 1:
"no prior knowledge, even if you recognize the person"; rule 2: "refuse if
absent"). This measures *operational* contamination — memory leaking past the
guardrail, the same decision the bot faces open-book on a retrieval miss — rather
than raw memory capacity (an upper bound that never occurs operationally, since the
guardrail is always present). Executed result: closed-book correctness = **0.00**
across all types (single 0/41, multi 0/6, overall 0/47); guardrail-hold rate 1.00
[0.92, 1.00] — the bot refused all 47 answerable questions with empty context,
never leaking memory even for recognizable subjects. This confirms zero
contamination two ways: open-book correctness is correct-iff-retrieved (HIT 29/29 =
1.00, MISS 0/12 = 0.00, zero correct-on-miss), and closed-book is 0.00. The gap
open − closed = 0.71 is retrieval's full lift (100% of single-hop correctness is
retrieval, 0% memory). The names-only anonymizer trigger does not fire; it stays
parked. Related finding: the 30% "false-refusal" rate decomposes to 0 real
generation false-refusals (gold retrieved but refused anyway) + 14 correct-given-miss
(retrieval missed the gold) — the generator's abstention discipline is perfect, and
the 30% is simply the retrieval miss-rate surfacing as refusals.

**What anonymization is for — a scope clarification.** Anonymizing the corpus is a
*measurement* instrument, not a *production* defense; conflating the two is the
common error. Two distinct questions need two different tools:
- *Q1 (measurement):* "does the RAG actually work, or is the model's memory masking a broken pipeline?" Anonymization answers this — fake names the model cannot know make any correctness provably retrieval-driven. Eval-set only; never touches production.
- *Q2 (production):* "when a user asks about a real subject, does the answer come from the report or from stale pretraining memory?" Grounding enforcement answers this — contract rule 1 + the groundedness judge + refuse-if-absent. It works identically on real and fake names, so it *is* the production defense; anonymization is not.

The closed-book control is a second, dominant way to answer Q1: anonymization
removes the model's *memory* (fake names); closed-book removes the *context* and
checks whether correctness survives. It came back 0.00 on the real names, so (a)
the eval set is already memory-clean → anonymization buys nothing for measurement,
and (b) it demonstrates Q2 directly (the model refused all 47 subject questions with
no context → grounding holds even on subjects it recognizes). Anonymization would
have earned its cost only if closed-book were *high*, and even then it fixes only
Q1, never Q2.

**Revisit when.** Correctness reads high while hit-rate stays flat (answers
bypassing retrieval) → reconsider the cheap names-only swap. That
(high-correctness × low-hit-rate) cell is the contamination signature; it is cheap
but n-starved, firing only on questions retrieval happens to miss. For the current
corpus and generator this decision is closed as a measurement.

---

## D-014: Vector index / store layer

**Context.** With 268 vectors, a brute-force numpy scan over an in-memory matrix is
sub-millisecond and an ANN index is premature. But a prior project's stack (managed
embeddings → FAISS → Postgres metadata) had exhibited a post-filter failure mode
worth designing against.

**Options.**
- numpy brute-force. Exact, zero dependencies, correct until ~10⁵ vectors; no metadata-filtering story; nothing to migrate, so no scale narrative.
- FAISS. Fast ANN at millions of vectors, but *post*-filter only (retrieve-then-filter can return zero relevant rows after metadata filtering); two stores (vectors + Postgres metadata) that can drift.
- pgvector. Vectors and metadata in one Postgres store; native *pre*-filter (`WHERE … ORDER BY embedding <=> q`); scales; DB operational cost is dead weight at 268 rows.

**Decision.** pgvector from the start (Postgres + pgvector extension, Docker-local).
No ANN index yet — exact search until the corpus grows; HNSW pre-registered below.

**Rationale.** Growth is planned, not hypothetical: there is a concrete near-term
plan to generate many synthetic profiles to simulate a much larger corpus.
Building pgvector in now avoids a later re-plumb and frees effort for the chunking
and retrieval work, which is where the interesting problems are. Not everything
needs to be maximally minimal.

**Alternative considered (numpy).** The strongest scale narrative is the
*migration* itself (numpy → pgvector/HNSW at the measured wall), which building
pgvector up front forgoes. If the synthetic-profile corpus never materializes and
the DB stays a 268-row toy, the numpy path would have been cheaper — but the stated
plan makes that unlikely.

**Revisit when.** (Index type) row count climbs to where exact seqscan latency
exceeds budget → add an HNSW index and tune `m` / `ef_search`. `# TUNABLE(exact is
best <~10⁴ rows; HNSW trades recall for speed; revisit when filtered-query p99 >
budget)`. Wrong-symptom: a known-nearest document drops out of top-k after adding
HNSW (approximation error) → raise `ef_search` or revert to exact.

---

## D-015: Embedder abstraction layer

**Context.** D-008 leaves the embedder unresolved as a procedure (attempt local
nomic, else Voyage-3) — a named, imminent variation point.

**Decision.** A thin `Embedder` interface (`embed_documents` / `embed_query`,
exposing `model_id` and `dim`, hiding the nomic prefix and Voyage `input_type`
asymmetry), with two implementations (NomicLocal, VoyageAPI), over an embedding
cache keyed by `(model_id, sha256(normalized_text))`.

**Rationale.** Justified not by speculative future-proofing but by the concrete
unresolved embedder choice — the shipping embedder is not yet known, so a swap is
near-certain. Precise scope: the interface makes the *code* swap cheap; it does not
make embeddings survive a model change (a different vector space means a full
re-embed and invalidation of all hit-rate history). The *cache*, not the interface,
is what makes a swap operationally cheap (a new `model_id` is a cache miss → auto
re-embed).

**Revisit when.** A third embedder or a reranker enters → widen the interface.
Never let the abstraction imply a model swap is "free"; the re-embed and
comparability break are inherent.

---

## D-016: Person as a first-class entity (2-table schema)

**Context.** RAG is one subagent of a larger chatbot. A query about a named person
must resolve name → person_id, then fetch that person's chunks (a metadata
*pre*-filter — the capability that justified pgvector over numpy, D-014). The data
has three grains — person / document / chunk — currently 1:1:1.

**Options.**
- Denormalized (subject as a text column on chunks). Simplest; no resolution anchor; drifts once one person has >1 chunk.
- 2-table (persons + chunks; embedding on chunks, person_id FK). Models the resolution target and the retrieval unit distinctly; section-chunking becomes a data change (more chunk rows, same person_id), not a schema migration.
- 3-table (persons + documents + chunks). Fully normalized but a redundant 1:1:1 join today.

**Decision.** 2-table (persons + chunks). `person_id` slug (not `user_id` — "user"
is reserved for the chatbot's end-users). Embedding lives on chunks. A `meta jsonb`
column on persons holds future resolution/filter metadata (industry, company,
aliases). No `documents` table until a person has >1 report. Resolution *logic*
(name→person_id, disambiguation) is deferred to a router subagent; the table is
only the anchor.

**Rationale.** Since RAG is one subagent of a chatbot, a query about a specific
person should resolve to a person_id and then fetch that person's chunks — model
the person as first-class now rather than migrate later.

**Revisit when.** A person gains multiple reports → insert a `documents` table
between. Resolution ambiguity bites (two subjects sharing a surname, per D-011's
collision case; real-world name collisions, e.g. a dossier that flags two distinct
public figures sharing a name) → build the disambiguating resolver.

---

## D-017: Model provider (single-provider default; cross-provider judge deferred)

**Context.** The generator (`claude-haiku-4-5`) and the judge (`claude-sonnet-5`)
both run on Anthropic Claude. Provider was an inherited default, never justified
against alternatives until now. The embedder is local nomic, so this touches only
generation and judging — not retrieval.

**Options.**
- Single-provider (Anthropic) for both. One SDK, key, and billing surface, fewer things that can drift; but the judge shares the generator's model family, so a mild same-family self-preference cannot be ruled out (only partly mitigated by the Haiku↔Sonnet tier gap in D-010).
- Cross-provider *judge* (a GPT-/Gemini-class judge grading the Claude generator). Strongest kill for family bias; costs a second SDK, key, and failure mode.
- Multi-provider generator too. Maximum diversity, but the generator is already a per-run variable and multiplying providers there is scope with no measurement payoff.

**Decision.** Single-provider (Anthropic) for the baseline. A cross-provider judge
is pre-registered as the response *if* calibration shows family bias — not adopted
blind.

**Rationale.** Stay single-provider for the baseline: cheapest tier for the
generator (Haiku), one tier up for the judge (Sonnet), so the ruler outranks what
it grades. Escalate only on evidence, and note the two escalations are different
axes: bump the *generator* tier if measured numbers show it underperforming (a
cost/quality call); move the *judge* cross-family only if calibration shows
same-family bias (an independence call). A second provider is a second key and a
second drift surface — not worth buying for diversity's sake, and earns its cost
only at the judge, once calibration data asks for it.

**Alternative considered (cross-provider judge).** Judge independence is the one
axis where provider diversity is a real quality gain: a cross-family judge cannot
share the generator's blind spots by construction. The only reason to defer is that
calibration has not yet shown same-family judging to be measurably biased here.

**Revisit when.** The judge-calibration sample shows the Sonnet judge systematically
over-rating same-family (Claude) outputs versus the blind human labels → move the
*judge* cross-family (the generator can stay Claude; it is the ruler that must be
independent). Also revisit on a provider outage or model deprecation that forces a
swap.

---

## D-018: Ranking-record depth

**Context.** `store.search` discards everything past `LIMIT k`, and `results.jsonl`
kept only the top-3. But the exact seqscan already ranks all 268 chunks (D-014), so
gold-rank, MRR, and a near/deep-miss taxonomy are *free* to record and impossible to
backfill into write-once runs. hit@1 stays the only headline; ranks are the
zero-cost diagnostic that routes the next build (a miss at rank 2–5 is a ranking
problem → rerank/bump-k; a miss at rank 40+ is a representation problem → re-chunk).

**Options.**
- (a) gold-rank-only via a per-gold count query. Cheapest, exact rank + MRR, but records nothing about *who* outranked the gold.
- (b) top-N (doc_id + score). Near-miss neighborhood + gold-rank if gold ≤ N; a deep gold (rank 67) records only as ">N", losing the exact rank; adds a silent N.
- (c) full 268-row ranking (doc_id + score, no text). Exact gold rank always, full MRR, complete taxonomy, re-analysable forever; ~10 KB/question (~450 KB/run).

**Decision.** (c) full-ranking now, doc_id + score only (no text — the gold-quote
match already ran offline). Pre-registered migration to (b) threshold-N once the
chunking strategy is frozen.

**Rationale.** While chunking is still unsettled, a miss must be classifiable as a
near-miss (ranking fix) or way off (re-chunk), so everything is ranked — reasonable
even on a few hundred documents at company scale. Once chunking is settled a
threshold N suffices, and starting with (c) is what *gives* the gold-rank
distribution to set that N empirically instead of guessing.

**Threshold-N derivation (applied at migration).** Set N from the observed
gold-rank distribution of the *frozen* config, not a priori: N = max( near-miss-band
tail [~95th pct of gold-rank among questions a reranker could still rescue],
distractor-head [top ~5–10, to see who outranked the gold] ) — both modest, ~15–20
here. Past N collapses to one "deep-miss / representation" bucket because the action
(re-chunk) is identical for rank 40 vs 200. `# TUNABLE(N read off the frozen-config
gold-rank curve; re-measure when the embedder or chunk scheme changes. Symptom
too-small: a non-trivial fraction of gold lands in ">N" and you keep needing the
exact deep rank.)`

**MRR note.** MRR is single-gold-natural; multi-hop records each gold document's
rank and feeds min-rank (best-placed gold) as the MRR input, so the metric survives
re-chunking (a document's rank = best rank among its chunks).

**Revisit when.** Chunking + embedder frozen → cut full-ranking to threshold-N; or
the corpus grows past ~10⁴ where per-question full-ranking artifacts bloat and
option (a)'s targeted count becomes the scalable path (premature at 268).

---

## D-019: Refusal label — deterministic string vs judge

**Context.** `is_refusal` does double duty — it scores both abstention lanes
(abstention_accuracy, false_refusal_rate) *and* filters the primary groundedness
denominator (only non-refusals are graded for grounding). It shipped as the judge's
semantic boolean: a drift-prone model sitting inside the primary metric. But the
generation contract already mandates an *exact* refusal sentence ("I don't know
based on the provided reports."), so a deterministic detector is available for free.
Whether the generator actually emits the exact string was unmeasured before the
first online run.

**Options.**
- Judge-only (status quo). Catches hedged/off-script refusals, but a model boolean drives the primary denominator and can drift; a judge false-"refusal" would wrongly exclude a possible hallucination from groundedness (the reputational-risk direction).
- String-only. Deterministic normalized-equality match against the mandated sentence. Drift-proof, contamination-proof, and cannot wrongly pull a real answer out of the denominator; but a refusal in the bot's own words scores as a non-refusal (undercounts abstention if the bot goes off-script).
- Both (chosen). String-authoritative + judge cross-check + divergence log.

**Decision.** Record both. `refusal_exact` (normalized equality, reusing the D-011
normalizer) is the official label for abstention and the groundedness filter; the
judge's `is_refusal` is kept as a recorded cross-check; disagreements are counted
(`judge_divergence_n/rate/divergent_ids`) as an alarm that the bot went off-script.
`REFUSAL_STRING` is a named constant in `generate.py` so the detector and the prompt
cannot drift. No extra API call — both labels come from data already collected.

**Rationale.** Start with the simplest thing that still measures — a plain string
check on the exact sentence the bot is told to use — keep a drifty model out of the
primary metric, then graduate to the judge's semantic label only if the divergence
log shows the bot won't obey "reply exactly." Whether it goes off-script is unknown
up front; the divergence count is what will tell.

**Alternative considered (judge-authoritative).** For abstention questions
specifically, a refusal in the bot's own words is genuinely correct behavior, and
string-only scores it as a non-refusal — so if the bot hedges often, the judge
measures the metric that matters most (did it correctly decline?) more faithfully.
The judge wins the moment the divergence log shows the bot won't obey the
exact-string contract.

**Revisit when.** The first online run shows material divergence *and* inspection
confirms the divergent cases are genuine off-script refusals (not judge errors) →
either tighten the contract or promote the judge to official. `# TUNABLE(equality
not containment; symptom wrong: divergence fills with genuine refusals that merely
appended a citation/token → loosen to containment.)`

---

## D-020: Citation instrument — validity-only vs coverage

**Context.** The generation contract mandates a document-level `[doc_id]` tag after
every claim, but the harness never read them — D-009's own "an unverified citation
launders hallucinations" was unenforced. A `doc_id` equals the subject's full name,
and the citation form is `[Full Name]`, so a cited tag matches a retrieved `doc_id`
by name with no resolution layer needed.

**Options.**
- A (validity + counts). Deterministic parse: extract `[…]` tags, split plural brackets, normalize, classify each as valid (name is in the *retrieved* set for this question) or fabricated. Record counts + `has_fabricated` + `has_any_citation`. No API. Catches the two real failures — a fabricated citation (cited a document it was never shown, the laundering signal and a contamination tell) and a zero-citation answer. Blind to graded per-sentence coverage.
- B (A + coverage). Also split the answer into sentences and score the fraction of *factual* sentences carrying a tag. Adds a sentence-splitter tunable and a fuzzy "what is a factual sentence" denominator (a mini-judge problem).
- C (B + support). Judge that the cited document actually *contains* the claim. Rejected — that is the groundedness judge's job (D-010), duplicated per-citation and expensive.

**Decision.** A. A deterministic per-answer citation parser (`parse_citations`): a
`citations` block per answer plus a `citations` summary section
(`fabricated_citation_rate`, `zero_citation_rate`, mean citations/answer) measured
over non-refusal answers. Validity is checked against the retrieved set for that
question, not the whole corpus.

**Rationale.** The simplest and most important guarantee is that whatever the bot
cites is actually one of the reports it was handed — a fabricated citation is the
dangerous failure, catchable with plain name-matching and no AI, so it runs on every
answer. Fuller sentence-by-sentence coverage is not worth the fuzzy machinery until
something shows it is needed.

**Alternative considered (B).** Coverage is what makes citations *useful*, not
merely non-fabricated — an answer that cites nothing scores zero-fabricated yet
zero-traceable, and only graded coverage catches a wall of grounded-but-uncited
claims. If a later gate becomes "every claim traceable," coverage is its real input
and this parser reopens.

**Revisit when.** (Matching) a surname-only tag, an odd separator, or a non-name
bracket gets mis-flagged fabricated → extend the splitter/matcher. `# TUNABLE(full-
name normalized-exact match + comma/semicolon split for plural brackets.)` (Scope)
add B's per-sentence coverage when a later gate needs "every claim traceable" or
runs show grounded-but-uncited answers.

**Later refinement — the `cite:`-prefix fix.** The generator sometimes copies the
corpus's own bracket forms — `[cite: Name]`, `[Name, Section 5]` — instead of the
contract's `[Name]`, which the parser mis-flagged as fabricated (baseline
fabricated-rate 0.061, from two answers both citing *valid* retrieved names). Fix in
`_citation_names`: strip a leading `cite:` prefix and drop bare-numeric
(`[cite: N]`) and `Section N` brackets — all non-doc citations; a real fabrication
(a name not in the retrieved set) still surfaces. Offline re-parse of the baseline:
0.061 → 0.000 (zero real fabrication). The baseline `summary.json` keeps 0.061
(write-once, D-021); the corrected 0.000 is verified via re-parse, and every future
run uses the fixed parser.

---

## D-021: Reproducibility-hole closure

**Context.** Runs are write-once and used as a comparison ledger, but four things a
run's reproducibility depends on were unpinned in `config.json` — a run could differ
from another with no config diff to explain it. Three are mechanical; the fourth
(dirty-tree handling) is the one genuine fork.

**Mechanical closures (no alternatives; recorded for the audit trail).**
- `question_set.sha256`. `n` is a weak fingerprint: a gold quote can be edited (changing which gold is scored) without changing the row count, so two runs could silently score different gold under a same-looking config. Hash the raw bytes; any edit → different hash → visible in a diff.
- `normalizer.version` (`NORMALIZER_VERSION="norm-v1"`). The config named the normalizer as a path string that never changed when the function body did; changing the body silently re-scored all hit-rate history. A hand-bumped version, snapshotted like the rubric and prompt versions, makes the change a config diff.
- `embed_stack` (torch / sentence-transformers / numpy versions). These determine the embeddings but are invisible to the cache key (`model_id + role + sha256(text)`). Read via `importlib.metadata` (package metadata, not `import torch`) so a fully-cached offline run still never loads torch (preserving the D-015 cache payoff). Honesty limit recorded in-code: this is the version *installed now*; for a cache-hit run it is not necessarily the version that produced the cached vector. Recording makes a mismatch auditable; it does not fix the cache-key blindness (that fix — folding the stack version into the cache key — is deferred, no signature having fired).

**The one fork — dirty-tree handling** (config already records `git.dirty`).
- Options: (a) record-only (silent flag, no nudge); (b) warn, don't block (loud banner at launch, run proceeds); (c) enforce (hard-block unless `--allow-dirty`).
- Decision: (b) warn, don't block. `warn_if_dirty()` prints a banner at launch; nothing is blocked; `git.dirty` remains the permanent audit trail.

**Rationale.** Throwaway offline retrieval runs happen constantly and vastly
outnumber baseline-of-record runs; a hard block would tax the tight synchronous
feedback loop that is the harness's whole point, and `--allow-dirty` would become
reflex muscle-memory anyway — so enforce pays friction daily and still erodes to a
warning. The banner nudges at the one moment that matters (cutting a baseline) while
the recorded flag keeps any dirty baseline auditable after the fact.

**Alternative considered (enforce).** Closing repro holes *structurally* so they
don't depend on human vigilance is exactly why the sha256 and version constants beat
"remember to check." A warning is itself a vigilance-dependent guard, and an
accidental baseline from a dirty tree is precisely the hole the mechanical closures
attack. Only enforce makes the guarantee structural rather than behavioral. It loses
solely because the block would fire on every throwaway offline run, where the
friction cost is real.

**Revisit when.** A dirty-tree baseline-of-record slips through and pollutes a
comparison despite the banner → promote to enforce for comparison-grade runs (keep
offline iteration unblocked, e.g. gate only when the API lanes run). `# TUNABLE(warn-
not-enforce; symptom: a dirty baseline gets compared anyway.)`

---

## D-022: Judge calibration design

**Context.** The open question this closes is that the measurement ruler (the judge)
was uncalibrated. Two pieces: human-vs-judge agreement with a trip rule, and a judge
flip-rate (since the Sonnet judge rejects a set temperature, stability must be
*measured*, not asserted — see D-010). The binding data fact discovered at design
time: the baseline's groundedness pool is **33 non-refusal answers, all
grounded=true, zero grounded=false**. The bot refuses instead of hallucinating (the
correct-iff-retrieved + closed-book=0.00 discipline from D-013), so the run
*structurally* cannot produce an ungrounded answer. A pre-registered
8-grounded-true / 8-grounded-false stratification is therefore impossible from real
data, and raw agreement on the natural set is ~100% by construction (Cohen's kappa
undefined — no variance in the human labels).

**Three sub-decisions.**
1. *Sample design* (natural-only / injected-only / hybrid). **Hybrid.** Label a natural slice of the real run (the correctness lane *has* negatives — 14 false-refusals plus one fluent-but-wrong multi-hop case — plus a sample of grounded-true for the false-alarm direction), *and* author ~8 injected adversarial ungrounded answers (a real question + real retrieved context with a *planted* unsupported claim) to stress-test the direction the run can't produce. The two are reported as separate measurements, never pooled into one agreement number. Budget ~24 labels total (≈16 natural + ≈8 injected).
2. *Trip semantics* (fixed-% / kappa / asymmetric). **Asymmetric.** On the injected groundedness set, zero tolerance — the judge passing even one injected-ungrounded answer as grounded trips the alarm (→ bump the rubric version, re-score all runs, never mix). On the natural set, report-only disagreement, no gate (a judge that wrongly *fails* a grounded answer only makes the bot look worse than it is — the safe/pessimistic direction). Cohen's kappa deferred until ≥50 cumulative mixed labels exist.
3. *Flip-rate* (mechanical). Judge each item 3×; flip-rate = fraction non-unanimous; if > ~5–10%, switch official runs to majority-of-3. Measured on the injected + borderline items (multi-hop / the ambiguous case), *not* the slam-dunk grounded-true answers — an obviously grounded answer never makes the judge wobble, so measuring flips there gives a falsely reassuring 0%.

**Rationale.**
- *Sample.* Test the false refusals (the correctness lane) and — more importantly — the ungrounded case; but every answer is currently grounded, so inject false facts and see whether the judge marks them ungrounded.
- *Trip.* Zero tolerance on an ungrounded answer being judged grounded, because wrong information about a subject is the failure least tolerated and the whole reason for the check. The judge wrongly failing a grounded answer is annoying but safe; wrong refusals matter too, but that is the *app's* behavior (a retrieval concern), not the ruler, and false_refusal_rate is measured deterministically and judge-independently.
- *Flip.* Temperature is not controllable, so run it a few times and check for stability; if it drifts, take best-of-3 — and run it on the hard cases (injected / almost-right), because the obviously-correct ones the judge always gets right.

**Injection taxonomy grounded in real generator behavior.** All 33 non-refusal
baseline answers were read before authoring the plants. Findings that reshaped the
set: (a) the generator is fact-dense and number-heavy (percentages, dollar figures,
areas) → off-by-a-number is the most realistic plant, promoted to a primary,
unambiguous trip item; (b) it does *not* manufacture superlatives — the only
superlatives in all 33 are ones it *relays* in quotes from the report — so an
"unsupported inference / most-experienced" plant was demoted from a trip item to a
borderline (report-only) item; (c) realistic drift homes are multi-document answers
(blending two subjects' figures → a wrong-attribution plant) and the occasional
added context sentence (an elaboration-drift plant). Gray-zone items never trip,
because the judge cannot be zero-tolerance-failed on a case where careful humans
themselves disagree.

**Result — the judge passes; the alarm did not trip.**
- *Trip test:* the judge marked all 6 confirmed-ungrounded plants (off-by-a-number ×2, outside-knowledge ×2, wrong-attribution, blatant) grounded=false → 0 misses → no trip. The ruler catches the dangerous direction the run itself never produces.
- *Natural false-alarm:* 8/8 (100%) human/judge agreement — the judge never wrongly failed a genuinely grounded answer, including the two relayed-superlative cases it correctly kept grounded. Strict without being trigger-happy.
- *Gray zone:* on both borderline plants the human labeler independently labeled ungrounded — the same as the judge; the predicted disagreement did not appear. Honest caveat: n=1 labeler, so this shows the labeler and *this* judge share the line, not that the line is objectively right. The class stays gray in principle; report-only, never trips.
- *Flip-rate:* 0% non-unanimous over 9 hard items × 3 runs → the single-call judge is stable enough; no majority-of-3 needed at this rubric version (re-measure on any rubric bump). This is the measured replacement for the unsettable temperature-0.
- *Correctness lane:* 7/8 (88%) human/judge agreement. The sole disagreement is the genuinely ambiguous multi-hop case: the bot named the right person but only because retrieval fetched one subject and missed the other, so it never actually performed the comparison (it abstained on the missing subject). Outcome-correct vs process-correct — the textbook reason correctness is secondary: the real failure is a retrieval miss, already caught load-bearingly by hit@1=false / span-recall@3=0.5, while grounded=true is also right (the bot didn't lie, it was honest about the missing report). The disagreement is kept as the more honest artifact; report-only, no action. That the lone split lands on the one ambiguous item is itself a calibration positive.

**Blind-labeling protocol** (encoded in `eval/calibrate.py`). Label *first*;
`judge-injected` prints no verdicts, ids, or kinds (only a progress counter); the
judge's outcome and the set composition are never narrated before labeling. Revealing
verdicts or composition beforehand would anchor the labeler and contaminate the
independence of the labels.

Net: the ruler is calibrated in both directions and stable for this rubric version;
the asymmetric trip rule is armed for every future rubric bump.

**Revisit when.** (Trip) an injected-ungrounded answer is judged grounded → the
ruler is unreliable in the dangerous direction; bump the rubric and re-score.
(Kappa) ≥50 cumulative mixed labels accumulate → switch the natural-set statistic
from raw agreement to kappa. (Flip) flip-rate > ~5–10% → majority-of-3 becomes the
official aggregation for comparison runs.

**Alternative considered (natural-only).** An injected bad answer is blatant and
real drift is subtle — catching an obvious plant proves nothing about catching the
plausible-number-slightly-off that happens in the wild, so injection can buy false
confidence, and the natural 33 (no false alarms on genuinely grounded answers) is a
real result in the direction the data went. This loses only because a guardrail
never tested in the failure direction can't be claimed to work at all (the
dead-battery smoke detector). The point survives as a design constraint: the plants
must be realistic (subtle number/inference/attribution errors), not cartoonish, or
the stress-test is theater.

**TUNABLEs.**
`# TUNABLE(~8 injected = 2 each across {off-by-a-number, unsupported-inference, wrong-attribution, outside-knowledge} + 1 blatant-fabrication control; grow a KIND only if the judge misses it or a suspected kind is untested. Symptom too-small: judge catches all 8 yet you distrust an unrepresented failure shape.)`
`# TUNABLE(flip: 3 runs, majority-of-3 above ~5–10% non-unanimous; symptom wrong: A/B groundedness deltas you act on are SMALLER than the measured flip-rate ⇒ the ruler's wobble dominates the signal.)`

---

## D-023: Section-chunking A/B — section-header chunking adopted

**Context.** The whole-doc baseline (D-008: one chunk per document) showed a handful
of single-hop misses where the correct dossier ranked deep (rank 40–60) — the
hypothesised "a buried fact dilutes inside a ~2.8k-token whole-doc embedding" failure.
Section-aware chunking was run as a paired A/B to test whether splitting on the
template's declared section boundaries rescues those buried facts, rather than assuming
it would. Each dossier follows a fixed ~15-section template; a census confirmed 252/268
use one heading level and 16 use another (so the splitter keys `^#{2,3} \d+\.`, not a
single level — otherwise 16 documents stay whole *inside* the section arm and dilute the
comparison), with section counts in {12, 15, 16}.

**Forks and decisions.**
- *Chunk unit:* pure sections (one chunk per heading), over min-length merge or
  small-to-big. Rationale: the dumbest unit that measures the hypothesis; let the failure
  signature pull complexity (context loss → contextual headers; tiny weak chunks →
  min-length merge) rather than pre-solving. Each section keeps its own native heading
  line verbatim (document text, not a synthetic header), so span-matching (D-011) is
  untouched; the splitter partitions the body losslessly so no gold quote can fall in a gap.
- *Retrieval k:* sweep k ∈ {1,3,5,8} offline (free — retrieval is a similarity search, no
  generation), read the effect at *matched* k so "structure helped" is separated from "we
  fetched more chunks," and pick any generation-run k from the curve.
- *Control arm:* a size-matched fixed-window splitter (window = the measured section-chunk
  mean, ~268 tokens; no overlap), not the conventional 512, so the comparison isolates
  boundary *placement* (structure vs a blind ruler) from chunk *size*.
- *Storage:* a `chunk_scheme` column (composite primary key) so all arms coexist in one
  store and retrieval filters by scheme — a chunking change is a config diff, and future
  schemes (contextual headers, re-embeds) are a filter away, not a rebuild. Every query
  filters by scheme, so a forgotten filter returns one defined scheme, never a blend.
- *Cost:* on this experiment run, defer the expensive groundedness judge — it is the lane
  with a principled reason not to move under chunking; generation plus the deterministic
  refusal/citation checks plus the cheap correctness judge suffice to catch the real risk
  (context-loss false-refusals). The frozen-config baseline runs the full judge.

**Result — the A/B, then a decomposition that changed the conclusion.** The first sweep
(offline retrieval, single-hop n=41) showed *plain* section chunking did **not** beat whole-doc:

| scheme | hit@1 | hit@3 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|
| whole-doc | 0.63 | 0.71 | 0.83 | 0.85 | 0.71 |
| section (plain) | 0.46 | 0.68 | 0.73 | 0.76 | 0.90 |
| fixed (size-matched) | 0.37 | 0.54 | 0.63 | 0.71 | 0.93 |
| **section + name header** | **0.66** | **0.78** | **0.85** | 0.85 | 0.91 |

Paired McNemar on the first three arms is all underpowered at n=41 (whole vs section hit@1:
+0.17, 13 vs 6, p=0.17; hit@3 a tie, 7 vs 8, p=1.0; section vs fixed hit@3: +0.15 for
structure, 10 vs 4, p=0.18) — none significant. Plain chunking lost, but *why* was the
interesting question, with two candidate causes: **context loss** (sections are name-poor, so
a query naming the person can't surface the right section) versus **corpus-size / fit** (each
~2.8k-token dossier fits the 8k embedding window, so whole-doc pools a name-rich vector that a
268-doc corpus has few distractors to beat).

**Decomposition — a name header isolates the cause.** A fourth arm (`section + name header`)
prepends the person's name to the *embedded* text only; the stored text stays verbatim, so
span-matching (D-011) is untouched. This restores the single piece of context a section lost —
*whose* section it is. Result: section vs plain section at hit@1 is **+0.20, discordant 10 vs 2,
p = 0.039 — significant** (the only significant result in the phase). **Context loss was the
dominant cause; the corpus-size hypothesis is refuted** — the missing name, not the corpus size,
was the bottleneck. Against whole-doc the header arm ties at hit@1 (0.66 vs 0.63, p=1.0), edges
it at hit@3 (+0.07, p=0.55) and on multi-hop span-recall (0.83 vs 0.67). A fairness note: the
header lives only in the embedding, so hit@k still requires the gold quote in the verbatim stored
chunk — the lift is a real retrieval gain, not a scorer artifact; and queries naming the person
are the real product query distribution, so exploiting the name is legitimate.

**Decision — adopt section+header at k=5.** k=5 is the hit@k plateau (5→8 adds nothing).
Rationale: it ties whole-doc at this scale (no measured quality lost), it is the config that
*wins* as the corpus grows (whole-doc's averaged centroids crowd together at scale while a sharp
chunk does not), and adopting now avoids a later re-plumb. A tie at this scale does not by itself
justify the added complexity — the adoption rests on the scale trajectory and forward-compatibility,
not on the (non-significant) lean. Reversible cheaply via the `chunk_scheme` column: revert is a
one-line config flag, whole-doc stays in the store, and the gold set survives on doc_id+quote.

**Generation validation.** Because section+header hands the generator small (~1.3k-token) chunks
instead of a full ~2.8k-token document, a generation-quality regression was the one live risk. A
generation+correctness run at k=5 showed the *opposite* — the lane improved: **false-refusal rate
0.30 → 0.09, single-hop correctness 0.71 → 0.90**, abstention 1.00 unchanged, zero fabricated
citations. Mechanism: false-refusals track retrieval misses (the generator declines when the gold
was not fetched), and correctness is correct-iff-retrieved — so the retrieval gain lifted both. A
full baseline then measured the primary metric: **groundedness 1.00 (43/43) [0.92, 1.00]** — held,
over *more* non-refusal answers than whole-doc's 33 (section+header refuses less), with a tighter
lower bound. Notably the full section+header run cost **$0.63** vs whole-doc's $2.99, because the
small chunks that were the config's risk are also what make its judge contexts ~5× cheaper.

**Revisit when.** A future full baseline shows groundedness regress below 1.0 → revert (the store
keeps whole-doc a flag away). At much larger corpus scale, re-run the arm comparison to confirm the
header advantage widens as predicted. `# TUNABLE(section+name header at k=5 = the hit@k plateau;
symptom to revert: a measured groundedness regression, or a multi-hop gold section pushed past k=5
by one person's chunks crowding the top — then raise k or move to person-scoped retrieval.)`

---

## D-024: Orchestration architecture — router + workers workflow

**Status.** Superseded by D-036.

**Context.** Requirements were fixed before architecture: answer lanes (corpus
named-person, corpus aggregate, web for out-of-corpus knowledge, both, neither),
response modes (answer, refuse, clarify, redirect), session-scoped memory, and
evidence-only retrieval returns feeding a single synthesis step. The substrate choice
(D-025) presupposes the workflow shape, so the shape came first.

**Options.**
- A — single agent + tool loop. Cheapest and flexible on unanticipated phrasings, but never-auto-resolve (D-016) becomes a prompt hope, routing lives in model behaviour (scoreable only by trajectory evals, silently changed by model updates), and every serial tool decision costs an LLM call.
- B — router + workers workflow. Deterministic resolver and lane router in code, per-lane retrieval flows, one synthesis step, a versioned handoff schema at the seams. Fast and parallel for known shapes; rigid on unanticipated ones, and lanes proliferate at 10x.
- C — autonomous multi-agent. Rejected: the lanes are known and no open-ended research task exists, so autonomy adds latency and unmeasurable seams for no required capability.

**Decision.** B — an orchestrator-workers workflow, not autonomous multi-agent.

**Rationale.** The corpus and web paths are independent, and the anticipated
dossier-side flows (meeting prep for one person, two-person comparison, people
discovery, member brief) each map to an explicit step sequence. Separation of concerns
alone would not force a workflow — tools separate concerns too. What decides it is that
the steps per query type are known in advance, so paying a model to re-derive them per
query buys unneeded flexibility at the cost of measurability, latency and drift.

**Alternative considered (A).** A capable model with three tools covers unanticipated
shapes for free, while every new query style is a lane to code and defend. It lost
because the flows were known and never-auto-resolve had to live in code.

**Revisit when.** A persistent class of real queries fits no lane, or lane count passes
~6–7. D-036 acted on exactly this trigger.

---

## D-025: Orchestration substrate — LangGraph

**Status.** Superseded by D-037, before any LangGraph code was written. The LangGraph
runtime later returned, for different reasons, in D-042.

**Context.** The workflow (D-024) needs routing on deterministic checks, a pause for
user confirmation when a name is ambiguous with resume exactly where it stopped, and
persistent per-thread state. A distinction set here carried forward: an LLM *as a step*
(called by code, returning a schema-validated, scoreable label) keeps orchestration
deterministic; an LLM *as controller* (deciding what happens next) is the tool loop
D-024 rejected.

**Options.** LangChain alone (no graph, state or interrupt primitives). The Claude Agent
SDK (centred on the autonomous loop the design rejected). Google ADK 2.0 (an initial
assessment that ADK lacked deterministic branching proved out of date — 2.0, GA May
2026, added GraphAgent with conditional edges, interrupts and checkpoints). Hand-rolled
(~200 well-understood lines, but durable interrupt/resume is the costly part).
LangGraph.

**Decision.** LangGraph, with two commitments: *thin graph, thick pure functions* —
rules, resolver, retrieval, synthesis and validation live in plain modules with no
framework import, so nothing a metric touches depends on the framework; and framework
versions pinned and snapshotted into run config (D-021), since the orchestrator is part
of the measurement instrument.

**Rationale.** Routing, interrupt/resume and persistence are native to LangGraph. ADK
now matches the capability, but its graph engine was weeks old against roughly three
years of production history for LangGraph's checkpoint core, and fresh-churn components
are kept out of the measurement instrument. Secondary: checkpoint replay, and fit with
an all-Anthropic generation and judge stack.

**Alternatives considered.** ADK 2.0 lost on maturity, replay and ecosystem fit;
hand-rolled lost because two flip conditions had fired (parallel fan-out, resumable
multi-turn state).

**Revisit when.** ADK's checkpoint/interrupt machinery has about a year of stable
releases, or a Google-stack requirement appears. Failure symptom: a framework bump
forces graph-code rewrites.

---

## D-026: Routing policy — hybrid rules + LLM cascade

**Status.** Superseded by D-036 — the agent architecture has no router. The LLM-as-step
principle survives in schema-validated tool decisions, and stage attribution survives as
per-decision trace spans.

**Context.** What produces each query's lane label? D-025's step-vs-controller
distinction makes an LLM classifier inside a deterministic router legitimate.

**Options.** Rules-only (free, drift-proof, fully evaluable; brittle at the corpus-topic
vs web boundary). LLM-only (robust, but an LLM call of latency on every query, including
"brief me on <member>", which a name lookup routes for free). Hybrid cascade (rules take
the unambiguous majority, an LLM step classifies the residue).

**Decision.** Hybrid cascade, with both stages built up front rather than adding the LLM
stage after measuring residue.

**Rationale.** Not every query needs a model call, but natural-language queries were
expected to leak across the topic boundary from day one, so rules alone would not
suffice.

**Alternative considered (staged build).** Adding the LLM stage only on measured residue
is complexity-on-evidence, and "X% routes deterministically" is a number worth having —
so the routed eval always scores stage-1-alone coverage, and every trace records which
stage decided.

**Design.** Stage 1: name/alias spotting plus unambiguous structural checks,
deliberately minimal. Stage 2: a schema-forced label from a versioned prompt on Haiku.
`# TUNABLE(Haiku for closed-label classification; symptom wrong: label accuracy on
routed gold materially below the gold's own authoring agreement → bump tier before
redesigning)`

**Revisit when.** Residue ≈ 0 on real traffic, the LLM stage mislabels a class a cheap
rule could catch, or the label distribution drifts.

---

## D-027: Web integration — external search API returning evidence

**Context.** The web lane answers out-of-corpus questions under two constraints: workers
return evidence only, and web evidence is never attributed to a member.

**Options.** Provider server-side web search (search inside a model call, returning
cited prose; fewest moving parts; message-level fixtures only). An external search API
of the Tavily/Brave class (explicit request → JSON results; request/response fixtures;
live path behind a flag). Fixtures only (perfect reproducibility, no live path).

**Decision.** External search API. The provider is chosen when the web lane is built
(JSON with URL, date and snippet; an adequate free tier; a stable API).

**Rationale.** Symmetric envelopes: the corpus worker returns JSON chunks, so the web
worker should return evidence too. Prose at the web worker plus prose at synthesis is
double generation and extra latency; with JSON results there is exactly one generation
step.

**Alternative considered (server-side search).** No new vendor or key, and native
citations — but it returns generated prose, re-importing double generation and blurring
the evidence/synthesis boundary, and its fixtures are coarser.

**Implementation note.** Until the web lane exists, the web tool is withheld from the
agent behind `WEB_SEARCH_ENABLED`: the first agent baseline (D-036) had three refusals
after the model called only the unimplemented web tool.

**Revisit when.** Snippets prove too thin to ground synthesis (→ a richer API or a
fetch-page step), or key and cost management becomes real at production traffic.

---

## D-028: Query decomposition — deterministic name-spotting; nominate vs appoint

**Status.** Carried forward by D-036 as a harness rule at the tool layer.

**Context.** Never-auto-resolve (D-016) depends on person resolution being a lookup, not
a guess.

**Decision.** Deterministic name-spotting (canonical names, surname index, company
aliases) is the only path to a `person_id`. *Nominate vs appoint:* a model may propose
mention strings ("the <company> guy"), but each is validated against `persons` before
becoming a `person_id` — the model nominates, only the lookup appoints.

**Rationale.** Resolution is on the critical path, and a model guessing it risks
confidently discussing the wrong person (the D-009 reputational failure). At 268 people
a lookup is both the safe and the cheap option.

**Alternative considered (LLM decomposer now).** Handles nicknames and implicit
references from day one. It lost because no phrasing failure had been observed and it
puts hallucinated-person risk on the critical path.

**Later hardening.** Under the agent loop, the model was seen *inventing* person_ids
(they are name-derived, hence guessable) while the appoint step accepted any valid id,
so it searched an unrelated member and refused. Now a raw `person_id` is accepted only
if the conversation has already shown it (asker, tool results, user text, finished
replies — never the model's own tool inputs). In the baseline-of-record run, which
predates the fix, 2 of 59 turns started with an invented id.

**Revisit when.** Real phrasings (nicknames, role references, misspellings) are missed →
fuzzy matching first, then an LLM proposer — appoint-by-lookup in every variant.

---

## D-029: Handoff schema — envelope + typed evidence; flow classified, lane derived

**Status.** Amended by D-036: the envelope survives unchanged; lane derivation moved
from control flow to eval-side metadata on gold rows.

**Context.** The routed eval scores this schema's fields and a planned citation gate
reads it, so schema design is eval design. Implemented in `src/contracts.py`, with no
framework import.

**(a) One envelope, two strict evidence kinds** (over one loose dict, or per-seam
contracts). `CorpusEvidence` carries person, document, chunk, section, confidence grade
and score; `WebEvidence` carries url, title, snippet, query and dates — and has no
`person_id` field at all. The invariant is identity scoping, not information isolation:
the web worker may receive asker-derived attributes (an industry, a region) to shape a
query, but never a person's identity. It prevents a wire-service claim being rendered as
"Jane Rivera's profile notes…", and an absent field is the one thing a refactor cannot
quietly reintroduce.

**(b) Flow classified, lane derived** (over 4 or 6 predicted lanes). `Flow`
(person_fact, comparison, brief, aggregate, topic, advice, composition, meta,
off_domain) is the only label the router classifies; `Lane` (corpus, web, both, neither)
is computed by `FLOW_TO_LANE`. Four lanes would hide four different retrieval paths
under one unscored `corpus` label; six would trip D-024's proliferation trigger on day
one; and a predicted lane would report a routing accuracy that is 100% by construction.
Routing accuracy therefore means flow accuracy.

**(c) Topic flow fans out to corpus ‖ web in parallel.** A similarity-threshold cascade
(corpus first, web on a weak match) was ruled out by measurement: top-1 cosine spans
0.586–0.849 for answerable questions and 0.642–0.831 for unanswerable ones — an
unanswerable question (ex-ab-1, 0.831) outscores ~90% of answerable ones, and an
answerable one (sh-019, 0.586) scores below every unanswerable one. Dense retrieval has
no natural empty result, so any threshold misroutes both ways. A cascade gated on the
measured refusal detector would pay for web only on a genuine miss, but its worst case
is two LLM calls and a web round-trip in series. Named-person flows are corpus by
construction, so the parallel cost lands on a narrow slice. `# TUNABLE(always-both on
topic; symptom wrong: web evidence uncited on >70% of topic queries → predict the lane
instead; uncited on <30% → the corpus half is dead weight)`

**Consequences for synthesis.** An unsupportive source means silence on that source; the
exact refusal string marks whole-answer refusal only; retrieved-but-uncited corpus
evidence becomes the per-source false-refusal signal.

**Revisit when.** A third evidence source appears, a flow needs a lane the table cannot
express, or flows pass ~9–10.

---

## D-030: Question scope as a routing signal

**Status.** Amended by D-036: with no router, the protection became a synthesis rule
(web evidence is never framed as the network's view) plus an eval check.

**Context.** Fanning a network-scoped topic question ("what do our members think
about…") out to the web can weave non-member commentary into an answer the user reads as
the network's view. Every existing instrument passes that answer: the web claim is
grounded in its URL, the citation is valid, and nothing triggers the refusal detector.
It is a "locally grounded, globally wrong" failure.

**Options.** A — a synthesis rule against network framing, checked by scanning for
framing phrases (a closed phrase list policing open-ended output). B — suppress the web
half at routing when the question scopes itself to the network. C — a new judged
source-appropriateness lane.

**Decision.** B, with C pre-registered. The router emits `flow` plus `Scope` ∈ {network,
open} — a property of the user's words, not a routing verdict — and `lane =
derive_lane(flow, scope)`. Only topic flow is affected; the other eight flows have one
possible lane. This is not lane prediction returning: D-029 removed guessing a source,
while scope reads a source the user declared.

**Rationale.** Prevention over post-hoc diagnosis: decide the source up front instead of
fanning out and later trying to tell where the answer should have come from. The model
supplies signals and code supplies the verdict. Scope is a scoreable envelope field, so
the cue list is measured rather than trusted.

**Alternative considered (C).** The only option that catches misuse nobody enumerated.
It lost on sequencing: a full judge-calibration cycle before any trace showed B missing
anything. Trigger: traces show network-scoped phrasings B missed.

**Revisit when.** Scope accuracy on routed gold falls materially below flow accuracy, or
C's trigger fires.

---

## D-031: Trace format — OpenTelemetry spans, confined; local viewer

**Context.** Per-step traces were required from the first orchestrated build — debugging
an agent without them is archaeology — and the same stream feeds later observability
work and user-facing progress lines.

**Options.** 1 — hand-rolled JSONL events (no dependency; a homegrown shape every viewer
must translate). 2 — LangGraph's native stream events (near-zero effort, but the
framework's shape, node-level only). 3 — OpenTelemetry spans with the GenAI semantic
conventions (standard vocabulary; one dependency).

**Decision.** Option 3, confined to the SDK and exporters — no collector or hosted
backend. (An early belief that OTel needs a collector was wrong; the SDK writes files
directly.)

**Rationale.** The GenAI conventions are the standard vocabulary for LLM observability,
and with OTel any viewer is one endpoint line, keeping the later platform choice open.
Option 2 lost because traces become a metric source, and D-025 keeps metrics independent
of the framework.

**Design commitments.** The file is the record, the UI a lens: spans append to
`runs/<id>/trace.jsonl`, with `TRACE_SCHEMA_VERSION` snapshotted into run config. Tokens
only — no dollars in spans. One step vocabulary feeds the file, the viewer and progress
lines. Tool arguments are recorded, since trajectories are undebuggable without them.
Spans carry dossier text and subject names, so hosted trace services are ruled out and
trace files are gitignored. Arize Phoenix was intended but would not install (a
dependency has no Python 3.14 Windows wheel); Jaeger shipped as an optional
docker-compose service — a one-line swap, which is what OTel bought. The code now lives
in `src/baselines/tracing.py`.

**Measured trap.** The first `query` span took 210 ms while its children summed to under
1 ms: a synchronous span processor on the OTLP exporter was posting inside the parent
span, counting export as work. Moving the viewer exporter to a batch processor (the file
keeps the synchronous one for deterministic ordering) brought it to 2.87 ms.
Instrumentation that runs inside the span it measures reports itself — check a parent's
duration against its children before trusting a latency number.

---

## D-032: Synthesis contract (synth-v1) — core prompt; prompts as hashed files

**Status.** Amended twice: the per-flow sections collapsed into the core (see (a)), and
D-042 retired the separate answer-writer call from the product path, merging these rules
into the agent prompt.

**Context.** With evidence-only workers, synthesis is the system's one generation step,
so its contract is half the measurement instrument.

**(a) Core + per-flow sections** (over one prompt, or one per flow). The core holds what
the metrics read (grounding, citations, refusal, confidence, voice); sections hold
output shape only. Rationale: no duplicated rules and no single overlong prompt — plus
comparability, since a version covers input bytes. Under one blended prompt, editing an
email-drafting rule changes what a factual question receives. Sized honestly, this was
comparability insurance, not cost (~2,511 input tokens per answer). Pre-registered tell:
collapse if sections average under ~5 lines. **It fired** — only two sections were ever
written (40 and 53 words), and with the router gone (D-036) nothing selected them. They
became core rules, plus a redirect rule (advice → point at members), in one core prompt
edited in place before any scored run.

**(b) Prompts as markdown files, hashed into run config** — core, section, and the exact
system bytes the model saw — for the same reason the question set is hashed (D-021). The
refusal string stays a code constant interpolated into the template, so detector and
prompt cannot drift; line endings are normalised and files are ASCII-only, so a checkout
cannot change a hash.

**(c) Contract content.**
- Refusal: "I don't have information on that." — whole-answer only; per-source silence emits no string.
- Citations: `[profile: <member name>]` and `[web: <publisher>, <date>]`, so the two evidence kinds are classifiable.
- Profiles are point-in-time, so a member's current situation is phrased "as of".
- Voice: warm, second person, conversational, length matched to the question; next steps only where evidence supports them. A "thinking partner" framing was not adopted — the product connects people, and advice questions redirect.
- Only profile evidence establishes membership or facts about a member; a web-named person is never presented as a member. (An early draft wrongly claimed web evidence never names members.)
- Facts about the asker are tagged `[self]` and stripped at display only. A marker, not an omission, because an uncited asker fact would look like dropped evidence in the uncited-evidence residue.

**Citation parser v2** (`eval/citations.py`). `[self]` is valid only if the asker's own
evidence was retrieved that turn; `[stated]` (user-stated facts) is counted but never
scored fabricated; a bare `[Name]` counts as a profile citation, flagged off-contract.

---

## D-033: Name spotting — case-insensitive with a derived stop-list

**Context.** Which strings in a free-text query are candidates for the D-028 lookup at
all? Chat input is unreliably cased.

**Measured first.** All 268 canonical names are two tokens. 35.1% of members (94) share
a first name with someone else (the most common is shared by six), so clarify is the
happy path, not an edge case; 5.2% (14) share a surname. 19 of 469 name tokens are also
ordinary lowercase English words. (A first count using `[a-z]+` matched lowercase tails
of capitalised words and inflated these several-fold; the decision used corrected
word-boundary counts.)

**Options.** A — require capitalisation (zero false positives; fails lowercase chat). B
— case-insensitive plus a measured stop-list. C — case-insensitive with no exceptions
(an ordinary word that is also a member's name resolves a person).

**Decision.** B. Tokens that are ordinary lowercase words *and* resolve to exactly one
member need corroboration — capitalisation or a full-name pair. Only those 15 of the 19
are listed: a false spot on an ambiguous token lands in clarify and self-corrects, while
one on a unique token auto-resolves to the wrong person.

**Rationale.** Names must resolve in either case. The error asymmetry decides the edge:
a missed name degrades to another flow, which is recoverable; a false name is the D-009
reputational failure. Bias toward missing.

**Alternative considered (A).** Zero false spots, and name autocomplete in the UI would
make casing reliable. It lost because users lowercase names in a chat box.

**Design commitments.** The stop-list is derived at build time (`build_stop_tokens`),
never hard-coded — it is a list of real surnames. `# TUNABLE(corpus lowercase usage as a
proxy for query usage; symptom wrong: a false spot on a token it missed)` Clarify text
is deterministic (`Resolver.clarify_question()`); when the agent path was first wired it
had drifted to model-written text and was restored. The resolver is pure (no DB,
framework or model). Ambiguity outranks everything: answering half a comparison while
dropping the ambiguous half is the confidently-wrong shape.

**Bug from real output.** Self-reference ("me", "my") first resolved to the asker, so
"Who can help me expand to China?" — an aggregate — targeted the asker. Fix:
`Resolution.self_reference` marks the asker's dossier as context, not a target.
Authenticated identity is context, not an entity to extract.

**Revisit when.** A false spot slips past the list (if recurring, switch the proxy to a
general word list or query logs), or the UI starts autocompleting names.

---

## D-034: Company aliases (derived, scoped); non-subject index built and dropped

**Context.** Two inputs the name index cannot serve: company queries ("who at
<company>…") and people named inside a dossier who have none of their own.

**Company aliases.** Each member's *current* company only, parsed from the
current-company header: 268/268 covered, 251 companies, 9 shared. Prior employers are
excluded — "connected to <vendor>" usually means a supplier or client relationship, and
aliasing every prior employer adds noise. Coverage was measured, not assumed: four
undocumented header phrasings took it from 194 to 268, and `build_company_index()`
returns coverage so callers can assert on it.

**Governing rule.** Prompted by a review question ("shouldn't semantic search do
this?"): semantic search finds information, but it cannot produce an id to filter on and
cannot detect ambiguity — it always returns top-k with no confidence signal (D-029(c)).
A deterministic index earns its place only when it produces ids or detects ambiguity;
everything else belongs to retrieval. Company aliases qualify: a shared company would
otherwise silently resolve to one member, and company-phrased comparisons need
person-scoped retrieval.

**Non-subject index — dropped the day it was built.** An index of non-member Key
Personnel was arbitrary and incomplete, barely changed behaviour, and is covered by a
simpler rule: on `not_found`, retrieve anyway and let refuse-if-absent decide. Deleting
it exposed a bug it had masked: "Who is <non-member full name>?" fell back to a
first-name match and asked the user to choose between two unrelated members. Fix: a
capitalised pair that is not a member is consumed as a non-member mention, never
partially matched.

**Derived, never committed.** Both indexes hold real names, so the source carries the
algorithm only.

**Revisit when.** Company mentions fail to resolve where a prior-employer alias would
have caught them. (A ten-way clarify on one widely shared company led to D-035.)

---

## D-035: Company mentions resolve to a set, never a clarify

**Context.** The resolver asked a ten-way clarify for the network's umbrella
organisation. Across the 9 shared companies, one has 10 members and eight have 2 — 9.7%
of members — against 35.1% of members ambiguous by first name.

**Options.** A — clarify at any count (never wrong; asks the user to pick between
colleagues when they asked about the firm). B — aggregate above a threshold N (with one
data point at 10 and eight at 2, any N from 3 to 9 is unfalsifiable). C — by mention
type: person mentions clarify at any count; company mentions resolve to the member set.

**Decision.** C.

**Rationale.** Multiplicity is not ambiguity. Two members sharing a first name: the user
meant one, and only they know which — clarify. Two members at one company: the user
meant the company, both are correct answers, nothing is missing — clarify would answer a
question nobody asked. True at 2 as at 10.

**Consequences.** "Anything mentioning the company" is plain retrieval, which also finds
the company in other dossiers as vendor, partner or client. The alias index serves
company → members for scoped flows and now stands on that one justification. Large sets
fall under the existing top-5 cap with an honest-scope statement.

**Alternative considered (A).** Never returns someone the user did not ask about, keeps
one rule for all mentions, and degrades gracefully if two different companies normalise
identically. It lost because that case does not occur in this corpus, while a menu
instead of an answer would hit 9.7% of members.

**Revisit when.** A normalised company key maps to dossiers describing different
businesses → clarify for that case only.

---

## D-036: Coordinator agent tool loop replaces the router + workers workflow

**Status.** Supersedes D-024. The tool-loop architecture carries forward into D-042 on a
standard agent runtime; the hand-rolled coordinator remains in `src/baselines/` as the
measurement instrument.

**Context.** D-024's own trigger (lane count past ~6–7) was visible at design time — the
contracts ship nine flows — and the costliest confusions (aggregate vs topic) sit where
rules are blind. To test rather than assume, eight representative multi-turn
conversations were written out to map turn-to-turn dynamics, where a workflow and a loop
actually differ. The patterns match observed usage of a comparable production chatbot.

**Options.**
- A — LLM turn-planner on the workflow spine: one schema-validated interpretation per turn, with deterministic code executing the known flows. Bounded and cheap to evaluate; D-025 survives.
- B — coordinator agent tool loop: an LLM controller over typed deterministic tools, with invariants enforced by the harness.
- C — build the workflow and let the routed eval's residue fire the trigger. Most rigorous, most hours.

**Decision.** B.

**Rationale.** The actions are few, but what should happen when in a multi-turn
conversation is not deterministic. The workflow had enumerable flows per *query* — which
is why D-024 was right when made — but enumeration breaks per *conversation*: the
operations are known, while the sequence and the operand are chosen by the conversation.
The scenarios that carried it: a vague opener where the user rejects two proposals and
the strategy pivots (industry → adjacency → alumni affinity); follow-ups that operate on
a prior result set ("who else among the people you suggested…", re-ranking under a
changed criterion); repair of the bot's interpretation; recall of an earlier session's
results.

**Alternative considered (A).** One bounded, cheaply evaluable call instead of an open
loop with costlier trajectory evals. It lost because a per-turn planner cannot react to
intermediate results (thin retrieval, collisions, rejected proposals) without becoming
the loop.

**Dispositions.** D-026 dies (no router). D-028 becomes a harness rule: the coordinator
nominates, only the resolver appoints, and an unresolved or ambiguous mention returns a
structured error that steers toward clarify. The D-029 envelope is untouched; lane
derivation becomes eval metadata; D-030's protection becomes a synthesis rule plus an
eval check. Traces gain trajectory spans. With the LLM as controller, this is now
genuinely an agent.

**Measurement.** A separate runner (`eval/agent_run.py`) keeps the earlier runner, and
every prior baseline, reproducible. Retrieval is scored as evidence recall (gold spans
anywhere in the turn's tool evidence), and harness-rule exceptions score as incorrect.
Comparisons are against the section-header baseline (D-023) on the same 59 gold rows.
- Run 1: answerable correctness 0.79 vs 0.89, false-refusal 0.13 vs 0.09, abstention and groundedness 1.00 in both, fabricated citations 0. Paired 7 regressions vs 2 improvements, McNemar p = 0.18 — not beyond noise. Five regressions were mechanical: two name-check false positives and three refusals after calling only the unimplemented web tool.
- Fixes: a full-name-pair name check (D-038); the web tool withheld (D-027); the coordinator prompt — with no member named, search the network, read the asker's profile only on self-reference, search before refusing (7 of 8 failures fixed on re-test); invented person_ids closed by the unseen-id rule (D-028).
- Run 3, the agent baseline of record: answerable correctness **0.96** [0.86, 0.99] vs 0.89 (0 regressions, 3 improvements; p = 0.25, not significant); false-refusal **0.04** vs 0.09; abstention 1.00 (n=11; one row errored, see D-040); fabricated citations 0. **Groundedness 0.93** (3/45 ungrounded) vs 1.00: a premise accepted from the question (borderline), a public figure's current title asserted without retrieved support (likely parametric knowledge, the D-013 class), and a mischaracterised shareholding structure. The judge was calibrated on the earlier contract's answers, so re-calibration (D-022 protocol) is owed before 0.93 is treated as real.
- Open: evidence recall is 0.73, yet several recall-0 multi-hop rows were judged correct and grounded — a possible mismatch between gold quotes and section-header chunk text.
- Cost shape: $0.079 per question in a smoke run; coordinator rounds dominate through thread replay (caching: D-037).

**Revisit when.** Single-turn correctness regresses beyond noise (add deterministic fast
paths for those shapes only), or coordinator rounds dominate latency or cost beyond what
conversational capability justifies.

---

## D-037: Substrate — hand-rolled loop + thin provider adapter

**Status.** Superseded by D-042.

**Context.** With control flow in the model (D-036), a LangGraph graph would be a single
looping node, which — as the framework was then assessed — removed its justification. No
LangGraph code had been written under D-025. Candidates: a LangGraph-hosted loop, a
hand-rolled loop with a provider adapter, the Claude Agent SDK, or a pydantic-ai-class
typed-tool framework.

**Decision.** A hand-rolled loop (~150 lines), a thin provider adapter (~50 lines;
Anthropic now, a local model as a drop-in behind the same interface), a session/artifact
store, and streaming passthrough.

**Rationale.** A one-node graph seemed to defeat the purpose of a graph framework; a
vendor agent SDK locks in one provider when a local model must stay possible; the loop
is small and fully customisable; and the model decides what happens when. Multi-turn
referents ("the first person") are resolved by the model reading the replayed thread,
token streaming comes from the provider SDK, and progress events come from the D-031
spans. Portability is option value, not a cost driver: a Haiku coordinator round costs
about $0.002.

**Alternatives considered.** LangGraph-hosted loop — checkpointing and streaming;
rejected because with one node its main asset seemed gone. Claude Agent SDK —
provider-locked, and at odds with the envelope/trace/prompt-hash discipline. Pydantic-ai
class — imports framework drift into the instrument, while everything distinctive here
stays custom anyway.

**Prompt caching.** Automatic 5-minute caching on both provider calls. Single-turn eval
questions get no benefit: the ~1.3k-token static prefix is below Haiku 4.5's 4,096-token
minimum, and two-round turns write then end (a ~$0.002/question write premium). In a
three-turn chat, cache reads grow from 4.3k to 10.4k tokens per call and input cost
falls **42%** ($0.046 vs $0.079), with the saving growing per turn. Kept on in evals to
measure the path that ships. Because caching is prefix-matched, the separate answer
writer can never share the coordinator's cache entry.

**In hindsight** (D-042). LangGraph was judged here by node count, but its value for
this system is the runtime — per-thread persistence, streaming, resume, tracing — which
a single-node chatbot needs too. The custom turn-to-turn context and persistence
plumbing was re-implementing it.

**Revisit when** (as written). A real multi-provider requirement appears, or the loop
grows framework-shaped features (checkpoint/replay, parallel sub-orchestration) past
~300 lines.

---

## D-038: The social turn ("Hi") and the identity rule it exposed

**Status.** Greeting mechanics revised by D-041; the separate answer writer retired from
the product path by D-042.

**Context.** In the first live smoke of the answer path, "Hi" made the coordinator
retrieve the asker's profile, and the writer replied in the first person *as the member*
("Hello! I'm <member name>, founder and CEO of <company>…"). What "Hi" should do had to
be decided first.

**Options.** A — greet and stop (no retrieval; cannot be wrong). B — greet plus a canned
capability line. C — greet plus a proposal drawn from the asker's profile.

**Decision.** C, as a fallback chain: (1) offer to resume an unresolved thread by name;
else (2) offer an introduction angle from the asker's profile; else (3) a warm greeting
by name and an invitation. Each rung degrades to the next, so the behaviour is defined
even when nothing is known about the member.

**Rationale.** The product's value is that it already knows who is asking; a generic
"how can I help?" throws that away. So reading the asker's profile on a greeting was
intended — the defects were in synthesis, which recited the profile instead of proposing
from it, in the first person. Rung (1) works within a thread for free (the thread is
replayed); across sessions it needs recall, first provided by D-041.

**Alternative considered (A).** Cannot misfire — no retrieval on a contentless turn, and
no misreading of a member's priorities from a point-in-time dossier. Rung (3) is option
A, so the retreat path exists.

**The identity rule.** The contract never said who the bot is, so a turn carrying the
asker's own dossier (tagged `[self]`, in a second-person voice) left first person open —
on any turn, not only greetings. New core rules: the bot speaks for the network and is
never a member (no first-person claims about any member's company, career or
achievements), and on a social turn it proposes rather than briefs. The writer now also
receives the asker's name, outside the pinned bytes.

**Holes closed alongside.** The greeting had shipped as a `redirect`, bypassing the
writer with an uncited claim about the member; the coordinator now never writes
user-facing text, so every sentence a user sees is a fixed string or the measured
contract. The post-generation name check now covers every model-written reply — only
names from this turn's evidence, the thread, the user's words or the asker are allowed —
and was later narrowed to full-name pairs after false positives on surname-like words in
prose. `# TUNABLE(full-name pairs only; symptom wrong: a violation on an answer a reader
judges correct → inspect the stop-list before loosening)`

**Citation placement.** Two runs of one prompt differed — one tag annotated its
sentence, the other replaced the subject's name — so tags now go at sentence end and
never substitute for a name; otherwise D-020's per-claim counts would measure something
else.

**Revisit when.** Members ignore or correct the opening proposal more often than they
take it up (default to rung 3), or cross-session recall can be measured.

---

## D-039: Real-person data removed from git history

**Context.** A sweep of the public repository's history against the corpus's subject
names found real-person data that had been committed in earlier commits; it had since
been removed from HEAD, but not from history.

**Options.** A — make the repository private, rewrite, re-publish. B — rewrite history
while public. C — accept.

**Decision.** B. History was rewritten with `git filter-repo` (affected files removed
from every commit; remaining name occurrences replaced with a redaction marker) and
force-pushed. A re-scan of every history blob found no matches, and the HEAD tree was
byte-identical before and after.

**Rationale.** Proportionality: an in-place rewrite closes the exposure within minutes,
so taking the repository private first would add little.

**Alternative considered (A).** Going private closes the window during the rewrite
itself; it lost because that window is minutes long.

**Related hygiene.** Leftover occurrences in code comments were replaced in HEAD; the
sweep was extended to name-derived identifier forms and history re-scanned clean; the
asker is recorded as a hash in committed run configs; and trace files, which carry
dossier text, are gitignored.

**Since then.** The sweep became a pre-push hook (`scripts/hooks/`): every push is scanned
for member names in any separator form and for person ids, read at run time from the
private persons file, and blocked on a hit; it fails closed if the list is missing. It
caught a name in an unpushed prompt example on its first run.

**Revisit when.** The hook blocks a push → fix and rewrite the unpushed commit, never
bypass.

---

## D-040: Clarify for missing context, refuse for missing data

**Context.** In a fresh thread, "Which of these leaders has raised the most venture
capital?" (ab-009) made the coordinator declare clarify, which the harness allowed only
for person ambiguity, so the turn crashed.

**Options.** A — clarify only for ambiguous person references. B — also clarify when the
user presupposes conversation context that does not exist.

**Decision.** B.

**Rationale.** Missing *context* is user-fixable, so ask; missing *data* is not, so
refuse.

**Alternative considered (A).** Refusal is deterministic (D-019), while clarify is a
model-written turn that can be over-triggered — asking instead of searching is a false
refusal by another name.

**Implementation.** `clarify_reason` ∈ {ambiguous_person, missing_context} on the
respond schema and envelope (handoff schema v3). Missing-context text comes from the
measured writer and names no member. A harness precondition forbids it while a result
set exists in the thread (TUNABLE), and a new metric, `clarify_on_answerable`, tracks
over-triggering. The D-042 agent has no code-level equivalent of the precondition, so
after first live use the prompt rule was narrowed to "nothing in the conversation could
be the referent".

**Gold set.** The row was split so each tests one failure mode: ab-009 became "Which
member of the network has raised the most venture capital?" (still a refusal), the
original wording became clarify row cl-001, and two more clarify rows were
hand-authored. Gold is now 62 rows (41 single-hop, 6 multi-hop, 12 abstention, 3
clarify); the file hash changed, so the next run compares per
question id.

**Revisit when.** `clarify_on_answerable` > 0 on a full run, or users answer a
missing-context clarify with "you know what I mean".

---

## D-041: One-call greeting — pure greetings skip the coordinator

**Status.** Retired from the product path by D-042; it remains in the coordinator path
(`src/baselines/`), kept as the measurement instrument. In the D-042 agent a greeting
takes one model call because no tool is called.

**Context.** A greeting — often a member's first message — should not cost several model
calls. "Hi" cost three: the coordinator retrieved the asker's profile, read the result
in a second round, and the separate writer wrote. Any tool call forces a second round,
and a separate writer puts the floor at two calls.

**Options.** A — a deterministic pure-greeting check skips the coordinator (one writer
call with a SQL preload). B — the coordinator writes the greeting in round 1 (one call,
but outside the measured writer, reopening a hole D-038 closed). C — accept two calls.

**Decision.** A.

**Rationale.** A contentless greeting should not spend latency and per-turn cost on
several model calls. B needs no vocabulary and recognises any phrasing, but would put
greeting text outside the measured contract.

**Implementation.** `src/baselines/greeting.py`: every token from a greeting vocabulary,
at least one core greeting word, at most 6 tokens — fail-safe, since a miss costs calls,
never a wrong answer (TUNABLE). In 48 ms, `build_session_brief` loads by SQL the asker's
executive summary (as evidence on the envelope, visible to the judge) and the latest
search criterion from the member's most recent other thread within 7 days, as a callback
topic — the first cross-session recall, a read of data already stored. `# TUNABLE(7-day
window; symptom wrong: members reject the callback as stale → shorten it, or require the
thread to have ended on an open question)` Greetings that miss the check call no tools
(two calls, not three). A greeting cannot name other members, since that needs
retrieval. Verified offline only.

**Revisit when.** Pure greetings show up on the multi-call path (a vocabulary gap), or
members reject the callback.

---

## D-042: Standard LangChain agent + chat interface now; rework when complexity grows

**Status.** Supersedes D-037. Retires from the product path the separate answer writer
(D-032), declared response outcomes, and the greeting fast path (D-041); the coordinator
stays in `src/baselines/` as the instrument the agent baselines were cut with.

**Context.** After a long design stretch there was still no interactive interface, and
much of the code was custom plumbing for turn-to-turn context and persistence. That
exposed the error in D-037: the earlier rejection of LangGraph judged it by graph node
count, while its value is the runtime — per-thread persistence, streaming, resume,
tracing — that even a single-node chatbot needs. The lesson: choose frameworks from
requirements that will not change, start at the highest abstraction that fits, and when
it is outgrown, drop one layer within the same stack.

**Options.** A — a standard agent (`create_agent`) with guardrail hooks. B — a custom
LangGraph graph keeping the current design. C — stay hand-rolled.

**Evidence weighed.** The separate writer did not improve groundedness (1.00 with the
earlier single-call contract vs 0.93 with it); the correctness gains came from search
fixes (D-036); and it costs a model call per turn.

**Decision.** A.

**Rationale.** Reach a usable interface quickly; rework when complexity grows, not
before.

**Alternative considered (B).** Keeps measurement precision (an answer prompt isolated
from tool-prompt edits; declared outcomes for scoring clarify). It lost because a
product that cannot be used cannot be improved from use.

**Implementation.**
- `src/agent.py`: `create_agent` on the LangGraph runtime; Haiku 4.5 via ChatAnthropic with automatic prompt caching (not yet verified live); tools wrap the unchanged toolbox and enforce the unseen-id rule (D-028); an after-model name check swaps a leaking reply for the refusal sentence before it is saved; at most 6 model calls per turn; `PostgresSaver` checkpoints on the existing database; per-request asker context with a profile summary.
- `prompts/agent/core-v1.md` merges the coordinator and synthesis rules.
- `scripts/chat_app.py`: Gradio, with a member picker in place of login, status lines during tool calls, the reply shown on completion (no token streaming yet), and a New chat button.
- Pinned: langchain 1.4.5, langgraph 1.2.15, langgraph-checkpoint-postgres 3.1.3, langchain-anthropic 1.7.6, gradio 6.30.0. The anthropic SDK moved across a major version (0.117 → 1.13); the older provider and judge paths are untested on 1.x.
- Verified offline with a scripted model against the real database and tools: one-call greeting, search, name check, unseen-id rule, thread memory.
- Deferred: last-topic recall in greetings, and re-pointing the eval runner (still on the coordinator) at the agent before the next measured run.

**First live use.** On turn 3, "hmm maybe more about the first guy?" — after a list of
fintech members and a turn about the fourth — the reply claimed to lack the earlier
context, then listed the members. The prompt merge had dropped the coordinator's
conversation rules (ordinals and pronouns resolve against earlier results; corrections;
on a vague need, propose), and the coordinator's code-level guard against
missing-context clarify while a result list exists has no agent equivalent. Fix: the
rules were restored ("first" = first in the most recent list; with a plausible reading,
take it and say so) and missing-context clarify was narrowed to "nothing in the
conversation could be the referent". Lesson: when collapsing prompts, diff them rule by
rule.

**Revisit when.** A requirement the standard loop cannot express (a separate writer,
declared outcomes) becomes necessary → a custom LangGraph graph in the same stack.

---

## D-043: Profile citations hidden on screen; kept in stored text

**Context.** In the first live chat, a `[profile: <name>]` tag after every sentence read
as clutter.

**Decision.** Strip profile citation tags at the display boundary only
(`agent.render_reply`). The tags stay in the model's output and in the stored thread,
and the display text is never fed to the eval, the judge or the name check. Web
citations stay visible and become links when the web lane lands.

**Rationale.** A citation earns screen space only when the reader can verify it. Members
cannot open other members' dossiers — the chat is their only view — so profile tags are
unverifiable for them. Web citations differ: they are clickable, and they mark claims
from outside the network (the D-030 concern).

**Consequences.** The stored tags still power fabricated-citation scoring (D-020),
future claim-level verification, and complaint tracing.

**Revisit when.** Members gain a way to open the underlying profiles, at which point
profile citations become verifiable and earn their screen space.
