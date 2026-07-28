# Architecture Decision Record

This log records the architecture decisions for a retrieval-augmented Q&A system
over long-form dossiers, together with its evaluation harness. Each entry captures
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
