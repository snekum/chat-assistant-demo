# Decisions

## D-001: Third-party public companies
- Date: 2026-07-05
- Context: Step 1 — pseudonymizer
- Options:
  - Keep — max semantic/retrieval value, but kept facts can combine into a re-identification vector.
  - Redact all — safest, but kills the industry signal that makes the corpus a realistic RAG target and makes QA harder.
  - Partial (curated keep-list) — keep genuine public entities, map the rest.
- Decision: Keep public companies, via an explicit allowlist.
- My reason: Keep public company names since they can't be uniquely used to identify a person and we need those public company names that are well known to understand what domain and big industry the user belongs to. We need connection to real big industries or else it'll be hard for retrieval.
- Revisit when: a kept org is regional/small enough to fingerprint a subject, or a retrieval eval shows leakage via org co-occurrence.

## D-002: Fake-name generation
- Date: 2026-07-05
- Context: Step 1 — pseudonymizer
- Options:
  - Random (faker) — trivial, but destroys signal (gender flips, org quirks vanish), harder eyeball-QA.
  - Themed — low value, collision-prone.
  - Structured to preserve signal (gender, org morphology) — realistic + QA-able, more effort.
- Decision: Random-ish names from built-in pools, no gender preservation, minimal effort. (Org *shape* still lightly preserved only so fuzzed domains stay plausible — see D-007.)
- My reason: I think since it's only a demo project, we don't need this level of anonymyzation. It's okay if gender is mismatched.
- Revisit when: the corpus moves beyond a demo, or eyeball-QA needs gender/name-shape realism to catch bad mappings.

## D-003: Clean filenames
- Date: 2026-07-05
- Context: Step 1 — pseudonymizer
- Options:
  - Mapped fake name (`Marcus Reyes.md`) — readable/traceable, but a leak surface and collision-prone.
  - Opaque ID (`report_001.md`) — leak-proof, collision-proof, but unreadable during QA.
- Decision: Mapped fake name, with a numeric suffix (1/2) appended on collision. Filename is run through the same verification grep.
- My reason: Mapped fake name and add a suffix of 1 or 2 to address collision.
- Revisit when: fake-name collisions become frequent, or a filename leak slips past verification.

## D-004: Partial / bare-surname matching
- Date: 2026-07-05
- Context: Step 1 — pseudonymizer
- Options:
  - Full-name-only — precise, but under-redacts vs. the sample ("Silva" alone must map).
  - Global bare-surname replace — high recall, but merges the two different `Solaru` subjects and corrupts finance words (e.g. "Sharpe ratio" for subject Becky Sharpe).
  - Per-file, subject-scoped — matches the sample with no cross-document collateral.
- Decision: Per-file bare-surname replacement, scoped to the document's own subject; ambiguous/common-word surnames matched full-name-only via a stoplist.
- My reason: The surnames are only referred within document, so we just need to make sure that the replaced surname is used throughout the document. Reports don't cross reference other reports — each report contains information about that particular person only, so we can do per-file.
- Revisit when: reports start cross-referencing other subjects, or a subject's surname is a common/finance word that appears inside their own document.

## D-005: NER engine & precision/recall posture
- Date: 2026-07-05
- Context: Step 1 — pseudonymizer
- Options:
  - spaCy trf/lg — best recall, heavy, native-code install/DLL risk.
  - spaCy small — lighter, still native.
  - Mapping/regex-first, deterministic — NER used only as a discovery aid, pure-Python fallback if the model can't load.
- Decision: Mapping/regex-first. The subject is mapped deterministically from the filename (bulletproof). spaCy `en_core_web_sm` runs as a discovery aid *if it loads*; a pure-Python capitalized-phrase heuristic is the fallback. NER never silently redacts — its label wins nothing over the known map.
- My reason: Lets do mapping with regex if spacy might not install. But we can also look at other alternatives for spacy.
- Revisit when: NER misses materially hurt recall, or a non-native NER option (heuristic/LLM-local) is worth adding.
- Reality note (2026-07-05): spaCy's training/download path is blocked on this machine by Windows Application Control; inference loaded fine, and on the sample it mislabeled "Paladin" (ORG) as PERSON and dropped "fs" — confirming discovery-only is the right posture.

## D-006: Locations
- Date: 2026-07-05
- Context: Step 1 — pseudonymizer
- Options:
  - Curated city/state substitution table — consistent fakes, effort + a pool to maintain.
  - NER GPE mapping — automatic, imprecise.
  - Leave locations real — zero effort, but the hand-redacted sample did map them.
- Decision: Leave locations real; no city/state replacement.
- My reason: No need to replace locations. We don't need fake city or any kinda city replacement.
- Revisit when: a location + kept-org combination becomes a re-identification vector, or the corpus leaves demo scope.
- Note: This diverges from the hand-redacted sample, which mapped Austin/Dallas, Texas -> Denver/Boulder, Colorado.

## D-007: Domain / URL fuzzing depth
- Date: 2026-07-05
- Context: Step 1 — pseudonymizer
- Options:
  - Org roots only — covers company domains, misses personal-name domains.
  - Org roots + subject surname — covers both common cases seen in Sources.
  - Aggressive (all tokens) — max recall, high false-positive risk inside URLs.
- Decision: Fuzz domain fragments derived from mapped org roots + the subject surname (hyphen/concat/underscore/dot variants). Leave the opaque `vertexaisearch...` redirect strings untouched.
- My reason: let's do org mapping and surname mapping for domains.
- Revisit when: verification finds a real domain fragment that isn't covered by an org root or the subject surname.

## Status (2026-07-05): PARKED
The pseudonymizer is intentionally deferred. Decision: build the RAG project on
the real raw reports first, then swap in fully fake users/data later. The
architectural decisions above (D-001..D-007) stand and are the deliverable.

Open item when resumed — auto-map scope beyond the deterministic subject:
- Option A: subject-only auto-map; everything else -> manual override loop.
- Option B (leaning): auto-map an org ONLY if its root appears as a real domain
  in that file's `Sources:` block (uses the corpus's own structure as a
  precision filter; makes spaCy discovery-only and optional).
- Why parked here: the first cut used spaCy as a silent redactor of everything
  it tagged -> 11,596 false "orgs" for 284 people, over-redacting the corpus.
  That's the precision failure to fix before regenerating.
`scripts/pseudonymize.py` holds the working draft (deterministic subject
mapping is sound; discovery/verification need the precision + perf rework above).

---

# RAG baseline + eval harness (Step 2)
Decisions D-008..D-013 fill the critical-path forks enumerated in FORKS.md. Each
carries the owner's own reason (not a replay of the fork tradeoffs) per the CLAUDE.md
protocol. Step-3 interrogation runs against these.

## D-008: Chunking unit + embedder (joint — F1+F2)
- Date: 2026-07-05
- Context: Step 2 — RAG baseline. Chunk size and embedder context window are one
  decision: whole-doc (~2,840 tok p50, ~5,300 max) does NOT fit the small local
  embedders (bge-small 512 tok, MiniLM 256 tok), so a long-context embedder is forced.
- Options:
  - Whole-doc + long-context embedder (Voyage-3 32k / OpenAI 8,191 / local nomic 8,192).
  - Section-aware (~200 tok) + small local embedder (bge-small/MiniLM revive) — sharper
    hit-rate, more moving parts than baseline scope calls for.
  - Fixed-window (512 tok) + small local — template-agnostic but severs `[cite: N]`.
- Decision: Whole-doc chunks (1/doc). Embedder resolved as a procedure, not a second
  decision: attempt local `nomic-embed-text-v1.5` ONCE as the single test of the D-005
  wall; if blocked like spaCy was, fall back to Voyage-3.
- My reason: The reports all share the same 15 sections, so section-aware would be
  easy and fine — but I deliberately want to build whole-doc first to feel its pitfalls
  firsthand, so I can explain the tradeoffs properly when I watch evals fail. Local
  embedder first because it's free and easy to set up — worth one shot before paying.
- Revisit when: (chunking) groundedness poor AND traceable to mis-attribution inside a
  ~2.8k-tok blob, OR multi-hop hit-rate floors low → section-aware (bge-small/MiniLM
  then revive as local options). (embedder) nomic download blocked → Voyage-3.
- Guardrails: whole-doc's failure is QUIET — single-hop hit-rate saturates ~100%; watch
  groundedness + multi-hop, not the headline hit-rate. Timebox the nomic attempt
  (~15 min); do not rabbit-hole documenting the Windows wall.

## D-009: Answer-prompt contract (F6)
- Date: 2026-07-05
- Context: Step 2 — whether the bot CAN abstain is a property of the generation prompt,
  not just the model; the prompt is half the measurement instrument.
- Options:
  - Bare "answer from context" — model answers even when context lacks the fact;
    abstention becomes accidental/unmeasurable.
  - Answer-only-from-context + may-refuse — minimum that makes abstention measurable.
  - + mandatory citations (document-level cheap / source-level `[cite: N]` = deferred beast).
- Decision: "Answer only from the provided reports; if the answer isn't there, say you
  don't know," PLUS document-level citations (tag each claim with the report it came
  from). Source-level `[cite: N]` mapping stays deferred. Prompt text versioned +
  config-snapshotted.
- My reason: Even the simplest version must be properly grounded in the actual reports,
  not "it looks okay to my eyes." These are real people — someone may ask the bot
  important questions before meeting them, and confident false info would ruin our
  reputation.
- Revisit when: groundedness/abstention errors trace to prompt phrasing → tighten the
  contract / consider pulling source-level citations.
- Guardrail: a citation only counts if the harness VERIFIES it (does the cited report
  actually contain the claim?). An unverified citation launders hallucinations — the
  opposite of the goal.

## D-010: LLM-as-judge (F3)
- Date: 2026-07-05
- Context: Step 2 — the judge is the "ruler"; changing it silently re-scores history.
- Options:
  - LLM judge with fixed rubric — handles abstention nuance; risks drift + self-preference.
  - Deterministic overlap (ROUGE/embedding-sim) — no drift, but can't score abstention
    and rewards word-overlap over real support.
  - Human-only — highest trust, doesn't scale.
- Decision: Pinned Claude Sonnet judge (never the generator's Haiku tier), temp 0,
  rubric versioned + snapshotted; ~20 human-checked judgments per rubric version as a
  drift alarm. Groundedness judge sees retrieved context + answer ONLY (never the gold
  answer); correctness judge sees gold + answer. Groundedness is the PRIMARY metric,
  correctness secondary; rubric wording = "supported by the provided reports alone."
- My reason: Judging whether a refusal is appropriate is subjective and needs semantic
  understanding; relying on word overlap would flag correct answers that share no words
  with the gold as wrong (high error rate). So an LLM judge, with humans spot-checking a
  sample as insurance against its drift.
- Revisit when: human/judge agreement on the calibration sample diverges, OR the rubric
  changes (then re-score ALL runs under the new version — never mix versions).
- Temp-0 amendment (2026-07-22): D-010's literal "temp 0" is NOT settable on the shipped judge —
  claude-sonnet-5 rejects a non-default temperature (400). Shipped resolution (already in
  judge.py): omit temperature; pin stability via fixed rubric + disabled thinking + json_schema
  structured output. Judge stays Sonnet-class, never the generator's Haiku tier, so D-010's real
  invariant (ruler outranks generator) holds. temp-0 determinism was never guaranteed anyway;
  the residual non-determinism is MEASURED by the Phase-1e judge flip-rate (roadmap Tier-2
  item 10), not asserted.

## D-011: Span-matching / hit-rate (F4)
- Date: 2026-07-05
- Context: Step 2 — how the grader decides retrieval fetched the right passage. This
  compares a true quote FROM THE DOCUMENT against the RETRIEVED document text (both
  source text, no paraphrase) — distinct from groundedness (D-010), which judges the
  paraphrased generated answer.
- Options:
  - Quote-substring exact — simple, but brittle to formatting/`[cite: N]` noise.
  - Char-offset intersection — precise, but any normalization change breaks all labels.
  - Fuzzy: normalize-then-exact (safe) / token-overlap threshold (false-hit risk).
- Decision: Normalize-then-exact containment — normalize BOTH the gold quote and the
  retrieved text (lowercase, collapse whitespace, strip punctuation + `[cite: N]`
  markers), then require the full quote to still appear. No loose token-overlap
  threshold. Gold quotes authored from the parsed/normalized text the system sees, not
  the raw files.
- My reason: Internet-sourced reports carry extra whitespace and special characters, so
  clean before matching; and because this compares retrieved document text against the
  true source sentence (not the paraphrased answer — that's the LLM judge's job), it
  should be an exact match after cleaning, with no false-positive inflation of hit-rate.
- Revisit when: real quotes still fail to match after normalization → fall back to a
  token-overlap threshold with an explicit TUNABLE value and a false-hit symptom.
- Amendment — doc_id anchoring (2026-07-24): the shipped hit_at_k matched the gold quote
  against ANY top-k chunk's text, with NO check that the chunk was the GOLD doc. That silently
  assumed each gold quote uniquely identifies its doc — true for the 12 hand-authored seeds
  (distinctive quotes), FALSE once LLM-assisted single-hop generation (D-012) introduced generic
  quotes ("Santa Clara University" is in 15 docs, "Vice President of Operations" in 5). Failure
  mode: retrieval MISSES the gold doc but a DIFFERENT doc containing the same phrase lands in
  top-k → scored a FALSE HIT, inflating hit-rate and hiding the miss. Fix: hit@k (and
  span-recall) now require `retrieved.doc_id == gold.doc_id AND quote in that chunk` — the quote
  pins the section (survives section-chunking), the doc_id pins the person. This aligns hit@k with
  gold_rank, which already matched on doc_id (the two used inconsistent rules before). Surfaced by
  a cross-doc quote-collision scan while reviewing the first LLM batch. No baseline-of-record
  existed yet, so no history to re-score; the amended definition is the one the baseline uses.
  The eval-goldset-review skill did NOT catch this — it reviews the QUESTION SET, not the scorer,
  and checked quote containment, never uniqueness (a dormant risk absent from the seed data).

## D-012: Question-set construction (F5)
- Date: 2026-07-05
- Context: Step 2 — abstention questions need proof the answer is ABSENT from all 268
  reports; multi-hop needs facts spanning several. Neither is safe to auto-generate.
- Options:
  - Hand-seed + verified LLM expansion — trust where it matters, scale where it's safe.
  - Fully hand-authored — max trust, small n → noisy metrics.
  - LLM-generated + verified — scales, but weak on abstention/multi-hop (the key types).
- Decision: Hand-author the seed set — all abstention + multi-hop by hand — then
  LLM-assist single-hop drafting with every gold span human-verified. LLMs used to
  brainstorm alongside me, never to author unattended.
- My reason: I've built LLM-generated eval sets before and they come out generic; I need
  my own thinking to make the set exhaustive, using LLMs only to brainstorm together.
- Revisit when: hand-authored n is too small to separate two configs (confidence
  intervals overlap) → grow single-hop via verified LLM expansion.
- Sizing derivation (2026-07-24; n=12 pilot → ~45 target). "Why 45" is COMPUTED, not
  guessed, and two independent calcs agree:
  - Precision (Job A, one config's band): a Wilson/normal half-width w at rate p needs
    n ≈ (1.96/w)² · p(1−p). At the pilot's single-hop p≈0.6: ±0.20→~23, ±0.14→~47,
    ±0.10→~94, ±0.07→~188. 45 buys ≈±0.14. Halving the band costs ~4× n (the 1/√n wall),
    so we stop at "narrow enough to decide," not "narrow."
  - Power (Job B, the Phase-2 chunking A/B — the binding calc): pre-registered δ=0.30,
    α=.05, power=.80 → ~42/ARM with independent CIs. Drops to ~30–45 TOTAL because the
    A/B is PAIRED (same Qs both configs) and near-one-directional, so McNemar counts only
    discordant pairs (~8/p_flip). This is the "why 45 not 84" argument (see item 11).
  - The n=12 run was the PILOT that supplies p and the rough effect for both formulas.
  - Growth trigger is PRE-COMMITTED, not eyeballed: → ~75 ONLY IF the A/B delta lands
    inside the CIs (effect smaller than the 30 pts we powered for ⇒ underpowered for THIS
    effect). NOT "if the bands still look broad" — that trigger is optional-stopping /
    p-hacking (peek-and-stop manufactures false positives). `# TUNABLE(n=45 powered for
    δ=0.30 paired; revisit when the measured A/B effect < 0.30 → recompute n from the
    observed p_flip, grow toward ~75. Symptom too-small: A/B discordant count < ~8/p_flip
    ⇒ McNemar can't reach significance.)`

## D-013: Real-entity contamination
- Date: 2026-07-05
- Context: Step 2 — subjects are real, semi-public people the generator may already know
  from pretraining, so it can answer correctly WITHOUT retrieval, masking a broken RAG.
  (Flagged as "D-008" in the external critique / FORKS.md.)
- Options:
  - Mitigate + pre-register a formal un-park trigger for the names-only anonymizer.
  - Anonymize now (names-only) — cleanest, but work now on a deliberately parked thing.
  - Accept + lean on groundedness — simplest; document the blind spot.
- Decision: Accept the contamination; do NOT anonymize now. Rely on groundedness
  (already primary and scored "supported by the reports alone" per D-010) to catch
  claims sourced from pretraining rather than the reports. The full anonymizer stays
  PARKED (D-001..D-007). This row itself documents the blind spot.
- My reason: The pipeline already runs groundedness checks that catch claims sourced
  from pretraining rather than the actual reports, so stressing over correctness
  contamination is pointless. And at scale (thousands of users) per-corpus anonymization
  is impractical when the grounding checks already catch ungrounded claims.
- Revisit when: correctness reads high while hit-rate stays flat (answers bypassing
  retrieval) → reconsider the cheap names-only swap. That (high-correctness × low-hit-rate)
  cell is the contamination signature caught in the wild — gold doc not retrieved yet the
  answer is right, so it came from pretraining — but it is cheap and n-starved (only fires on
  questions retrieval happens to miss).
- Trigger, measured (Phase 1d, closed-book control = roadmap item 14): run the generator with
  EMPTY context and score correctness on the contamination-prone answerable questions.
  Closed-book correctness = the share of correctness that is retrieval-INDEPENDENT (pure
  memory); open−closed = retrieval's actual lift. High closed-book correctness ⇒ correctness
  is grading Haiku's memory of these real people, not the RAG → fire the names-only swap.
  Threshold = TUNABLE set at 1d against the observed closed-book distribution; no cutoff
  pre-committed before data.
- Drill-closed (2026-07-22): owner articulated the joint signature and derived the closed-book
  control. Qualifier the drill exposed — the "My reason" above is INCOMPLETE as written:
  groundedness catches a pretraining claim only when the retrieved context does NOT already
  contain the fact; when pretraining and context agree, groundedness returns true, so it
  proves SUPPORT, not that retrieval was load-bearing. Neither groundedness nor hit-rate can
  establish load-bearing; only retrieval ablation (closed-book) can. Hence item 14 is the real
  closer for D-013, with groundedness covering the unsupported-claim case.
- Metric contamination-exposure (2026-07-22): hit-rate is contamination-PROOF (embedder + gold
  quote only, no generator in the calc — memory can't move it); groundedness is contamination-
  ROBUST (checks answer ⊆ context, so a memory answer that also sits in context is genuinely
  grounded — it never gives a false green on hallucination, it just can't reveal source);
  correctness is contamination-VULNERABLE (memory inflates the pass rate). Consequence: an
  all-green row (hit + grounded + correct) is NOT a contamination worry — it is carried by the two
  resistant metrics, and hit-rate independently certifies the retrieval capability memory was
  suspected of masking. The dangerous false green is correctness-high × hit-rate-LOW (green answer,
  retrieval absent). This is the architectural reason correctness stays SECONDARY + caveated, not a
  headline.

## D-014: Vector index / store layer (Default e — vetoed by exception)
- Date: 2026-07-18
- Context: Step 2 build. FORKS.md Default (e) was brute-force numpy over an in-memory
  matrix (268 vectors → sub-ms; ANN premature). User raised their prior stack (Gemini
  embeddings → FAISS → Postgres metadata) and its post-filter failure mode.
- Options:
  - numpy brute-force — exact, zero deps, correct until ~10^5 vectors; no metadata
    filtering story; nothing to migrate = no scale narrative.
  - FAISS — fast ANN at millions of vectors, but POST-filter only (retrieve-then-filter
    can return zero relevant after metadata filtering — the user's real-project scar);
    two stores (vectors + Postgres metadata) that drift.
  - pgvector — vectors + metadata in one Postgres store; native PRE-filter
    (`WHERE ... ORDER BY embedding <=> q`); scales; DB operational cost is dead weight
    at 268 rows.
- Decision: pgvector from the start (Postgres + pgvector extension, Docker-local).
  No ANN index yet — exact search until the corpus grows; HNSW pre-registered (below).
- My reason: This is a portfolio/interview artifact where working on large data is the
  expected competency, and I have a concrete near-term plan to generate many fake
  profiles to simulate a much larger corpus — so growth is planned, not hypothetical.
  Building pgvector in now avoids a later re-plumb and frees my energy for the chunking
  and retrieval work, which is where the interesting RAG problems are. Not everything
  needs to be maximally minimal.
- Revisit when: (index type) row count climbs to where exact seqscan latency exceeds
  budget → add an HNSW index and tune `m` / `ef_search`
  `# TUNABLE(exact is best <~10^4 rows; HNSW trades recall for speed, revisit when
  filtered-query p99 > budget)`. Symptom wrong: a known-nearest doc drops out of top-k
  after adding HNSW (approximation error) → raise `ef_search` or revert to exact.
  (whole choice) if the fake-profile corpus never materializes and the DB stays a
  268-row toy, the numpy path was cheaper — but the stated plan makes that unlikely.
- Steelman (numpy, logged once): the strongest scale interview story is the *migration*
  (numpy → pgvector/HNSW at the measured wall), which building pgvector up front forgoes.

## D-015: Embedder abstraction layer
- Date: 2026-07-18
- Context: Step 2 build. D-008 leaves the embedder unresolved as a procedure (attempt
  local nomic, else Voyage-3) — a named, imminent variation point.
- Decision: A thin `Embedder` interface (`embed_documents` / `embed_query`, exposing
  `model_id` + `dim`, hiding the nomic prefix / Voyage `input_type` asymmetry), two
  implementations (NomicLocal, VoyageAPI), over an embedding cache keyed by
  `(model_id, sha256(normalized_text))` (Default i).
- My reason: justified NOT by speculative future-proofing but by the concrete
  unresolved D-008 fork — we don't yet know which embedder ships, so a swap is
  near-certain. Correction pinned: the interface makes the CODE swap cheap; it does
  NOT make embeddings survive a model change (different vector space → full re-embed +
  all hit-rate history invalidated, per D-008 guardrail). The CACHE, not the interface,
  is what makes a swap operationally cheap (new model_id → cache-miss → auto re-embed).
- Revisit when: a third embedder or a reranker enters → widen the interface; never let
  the abstraction imply a model swap is "free" (the re-embed + comparability break is
  inherent).

## D-016: Person as first-class entity (2-table schema)
- Date: 2026-07-18
- Context: Step 2 build, pgvector schema. RAG will be ONE subagent of a chatbot; a query
  about a named person must resolve name → person_id, then fetch that person's chunks
  (a metadata PRE-filter — the capability that justified pgvector over numpy, D-014).
  The data has three grains — person / document / chunk — currently 1:1:1.
- Options:
  - Denormalized (subject as a text column on chunks) — simplest; no resolution anchor;
    drifts once one person has >1 chunk.
  - 2-table (persons + chunks; embedding on chunks, person_id FK) — models the
    resolution target and the retrieval unit distinctly; section-chunking becomes a data
    change (more chunk rows, same person_id), not a schema migration.
  - 3-table (persons + documents + chunks) — fully normalized but a redundant 1:1:1 join
    today.
- Decision: 2-table (persons + chunks). `person_id` slug (NOT `user_id` — reserve "user"
  for the chatbot's end-users). Embedding lives on chunks. `meta jsonb` on persons for
  future resolution/filter metadata (industry, company, aliases). No `documents` table
  until a person has >1 report. Resolution LOGIC (name→person_id, disambiguation) is
  DEFERRED to a router subagent; the table is only the anchor.
- My reason (user): eventually this is a chatbot with RAG as one subagent, so a user
  asking about a specific person should resolve to a person_id and then fetch that
  person's chunks/report — model the person as first-class now rather than migrate later.
- Revisit when: a person gains multiple reports → insert a `documents` table between;
  resolution ambiguity bites (two Solaru subjects per D-004; real-world name collisions,
  e.g. the Ross Fernandes dossier flags an academic + a footballer) → build the
  disambiguating resolver.

## D-017: Model provider (single-provider default; cross-provider judge deferred)
- Date: 2026-07-21
- Context: The generator (D-009 / Default h, `claude-haiku-4-5`) and the judge (D-010,
  `claude-sonnet-5`) both run on Anthropic Claude. Provider was an INHERITED default, never
  justified against alternatives until now (surfaced under interrogation, 2026-07-21). The
  embedder is local nomic, so this touches only generation + judging — not retrieval.
- Options:
  - Single-provider (Anthropic) for both gen + judge — one SDK / key / billing surface, fewer
    things that can silently drift; but the judge shares the generator's model family, so a
    mild "Claude prefers Claude-shaped output" self-preference / family bias can't be ruled out
    (the F3 conflict, only partly resolved by the Haiku↔Sonnet tier gap in D-010).
  - Cross-provider JUDGE (a GPT-/Gemini-class judge grading the Claude generator) — strongest
    kill for family bias; F3 lists cross-family as a legitimate option. Costs a second SDK /
    key / failure-mode and a second thing that can change under you.
  - Multi-provider generator too — max diversity, but the generator is already a per-run
    variable (Default h) and multiplying providers there is scope with no measurement payoff.
- Decision: Single-provider (Anthropic) for the baseline. A cross-provider judge is
  PRE-REGISTERED as the response IF calibration shows family bias — not adopted blind.
- My reason (confirmed 2026-07-22): Stay single-provider (Claude family) for the baseline —
  cheapest tier for the generator (Haiku), one tier up for the judge (Sonnet), so the ruler
  outranks the thing it grades. Escalate only on evidence, and note the two escalations are
  different axes: bump the GENERATOR tier if measured numbers show it underperforming (a
  cost/quality call, Default h), and move the JUDGE cross-family only if G-002 calibration shows
  same-family bias (an independence call). I have OpenAI + Gemini keys on hand, but a second
  provider is a second key and a second drift surface — not worth buying for diversity's sake; it
  only earns that cost at the judge, once calibration data asks for it.
- Revisit when: the G-002 judge-calibration sample shows the Sonnet judge systematically
  over-rating Haiku-family (Claude) outputs vs the blind human labels → move the JUDGE
  cross-family (the generator can stay Claude; it's the ruler that must be independent). Also
  revisit on a provider outage / model deprecation that forces a swap.
- Steelman (cross-provider judge, logged once): judge independence is the one axis where
  provider diversity is a real quality gain, not just added complexity — a cross-family judge
  cannot share the generator's blind spots by construction. The only reason to defer is that
  G-002 hasn't yet told us whether same-family judging is measurably biased here.

## D-018: Ranking-record depth (Phase 1a instrument, Tier-1 item 3)
- Date: 2026-07-22
- Context: store.search discards everything past `LIMIT k`; results.jsonl keeps only top-3. The
  exact seqscan already ranks all 268 chunks (D-014), so gold-rank + MRR + a near/deep-miss
  taxonomy are FREE to record and impossible to backfill into write-once runs. hit@1 stays the
  only headline; ranks are the zero-cost diagnostic that routes the next build (miss at rank 2-5
  = ranking problem -> rerank/bump-k; miss at rank 40+ = representation problem -> re-chunk).
- Options:
  - (a) gold-rank-only via a per-gold count query — cheapest, exact rank + MRR, but records
    nothing about WHO outranked the gold (no wrong-person@1 taxonomy).
  - (b) top-N (doc_id+score) — near-miss neighborhood + gold-rank if gold<=N; a deep gold (rank
    67) records only as ">N", losing the exact rank; adds a silent N.
  - (c) full 268-row ranking (doc_id+score, no text) — exact gold rank always, full MRR, complete
    taxonomy, re-analysable forever; ~10 KB/question (~450 KB/run).
- Decision: (c) full-ranking now, doc_id+score only (no text — the gold-quote match already ran
  offline). Pre-registered migration to (b) threshold-N once the chunking strategy is frozen.
- My reason: while I'm still figuring out chunking I need to see whether a miss is a near-miss
  (ranking fix) or way off (re-chunk), so I rank everything — I'd do this at a company too, on a
  few hundred docs even if the corpus is thousands. Once chunking is settled there's no reason to
  rank everything; a threshold N suffices. And starting with (c) is what GIVES me the gold-rank
  distribution to set that N empirically instead of guessing.
- Threshold-N derivation (pre-registered, applied at migration): set N from the observed
  gold-rank distribution of the FROZEN config, not a priori. N = max( near-miss-band tail [~95th
  pct of gold-rank among questions a reranker could still rescue], distractor-head [top ~5-10, to
  see who outranked the gold] ) — both modest, ~15-20 here. Past N collapses to ONE
  "deep-miss / representation" bucket because the action (re-chunk) is identical for rank 40 vs
  200. `# TUNABLE(N read off the frozen-config gold-rank curve; revisit when the embedder or
  chunk scheme changes -> representation shift moves the distribution -> re-measure N or go full
  again. Symptom too-small: a non-trivial fraction of gold lands in ">N" AND you keep needing the
  exact deep rank to decide something.)`
- MRR note: MRR is single-gold-natural; multi-hop records each gold doc's rank and feeds min-rank
  (best-placed gold) as the MRR input, so the metric survives Phase-2 re-chunking (a doc's rank =
  best rank among its chunks).
- Revisit when: chunking + embedder frozen -> cut full-ranking to threshold-N (above); or the
  corpus grows past ~10^4 where full-ranking-per-question artifacts bloat and option (a)'s
  targeted count becomes the scalable path (premature at 268).

## D-019: Refusal label — deterministic string vs judge (Tier-1 item 5)
- Date: 2026-07-24
- Context: `is_refusal` is the per-answer label that does DOUBLE duty — it scores both abstention
  lanes (abstention_accuracy, false_refusal_rate) AND filters the PRIMARY groundedness
  denominator (only non-refusals are grounded, run.py). It shipped as the JUDGE's semantic
  boolean: a drift-prone model sitting inside the primary metric. But the generation contract
  (f6-v1) already MANDATES an exact refusal sentence ("I don't know based on the provided
  reports."), so a deterministic detector is available for free. The online lanes have never
  executed, so whether Haiku actually emits the exact string is unmeasured.
- Options:
  - Judge-only (status quo): semantic `is_refusal` drives the metrics. Catches hedged/off-script
    refusals, but a model boolean drives the primary denominator and can drift across runs; a
    judge false-"refusal" would exclude a possible hallucination from groundedness (the
    reputational-risk direction, D-009).
  - String-only: deterministic normalized-EQUALITY match against the mandated sentence.
    Drift-proof, contamination-proof, and can't wrongly pull a real answer out of the denominator
    (a substantive answer never normalizes to exactly the refusal sentence); but a refusal in the
    bot's own words scores as a non-refusal (undercounts abstention IF the bot goes off-script).
  - Both — string-authoritative + judge cross-check + divergence log (CHOSEN): string drives the
    metrics; judge `is_refusal` is recorded; disagreements are counted (judge_divergence_n/rate/
    divergent_ids) as the alarm that the bot went off-script, and as a G-002 calibration feed.
- Decision: Record both. `refusal_exact` (normalized equality, reusing the D-011 normalizer) is
  the OFFICIAL label for abstention + the groundedness filter; judge `is_refusal` kept as a
  recorded cross-check; divergence logged in the abstention block. REFUSAL_STRING extracted as a
  named constant in generate.py so the detector and the prompt can't drift (SYSTEM text
  byte-identical -> f6-v1 unchanged). Cost/latency are non-differentiators (no extra API call —
  both labels come from data already collected).
- My reason: Start with the dumbest thing that still measures — a plain string check on the exact
  sentence I already tell the bot to use — keep a drifty model out of my primary metric, then
  measure and only graduate to the judge's semantic label if the divergence log shows the bot
  won't obey "reply exactly." I don't yet know if it goes off-script; the divergence count is
  what will tell me.
- Revisit when: the smoke run (1b) shows material divergence AND inspection shows the divergent
  cases are genuine off-script refusals (not judge errors) -> either tighten the contract (bump
  f6-v2) or promote the judge to official. No cutoff pre-committed before data (same posture as
  D-013's closed-book threshold). `# TUNABLE(equality not containment; symptom wrong: divergence
  fills with genuine refusals that merely appended a citation/token -> loosen to containment.)`
- Steelman (judge-authoritative, logged once): for abstention questions specifically, a refusal
  in the bot's own words is genuinely CORRECT behavior, and string-only scores it as a
  non-refusal — so if the bot hedges often, the judge measures the metric I care most about (did
  it correctly decline?) more faithfully than the string. The judge wins the moment the
  divergence log shows the bot won't obey the exact-string contract.

## D-020: Citation instrument — validity-only vs coverage (Tier-1 item 6)
- Date: 2026-07-24
- Context: the generation contract (f6-v1) mandates a document-level [doc_id] tag after every
  claim, but the harness never reads them — D-009's own "an unverified citation launders
  hallucinations" is unenforced. doc_id == the person's full name, and the citation form is
  [Full Name] (contract allows "report(s)" -> plural brackets), so a cited tag matches a
  retrieved doc_id BY NAME with no resolution layer needed.
- Options:
  - A (validity + counts): deterministic parse — extract [..] tags, split plural brackets,
    normalize, classify each as valid (name is in the RETRIEVED set for this question) or
    fabricated (not). Record counts + has_fabricated + has_any_citation. No API. Catches the two
    real failures: a fabricated citation (cited a doc it was never shown — the laundering signal,
    and a contamination tell per D-013) and a zero-citation answer. Blind to graded per-sentence
    coverage.
  - B (A + coverage): also split the answer into sentences and score the fraction of FACTUAL
    sentences carrying a tag. Adds a sentence-splitter TUNABLE and a fuzzy "what is a factual
    sentence" denominator (a mini-judge problem — a naive splitter counts "Here's what I found:").
  - C (B + support): judge that the cited doc actually CONTAINS the claim. Rejected — that is the
    groundedness judge's job (D-010), duplicated per-citation and expensive.
- Decision: A. A deterministic per-answer citation parser (`parse_citations`, sibling of
  is_exact_refusal): a `citations` block per answer + a `citations` summary section
  (fabricated_citation_rate, zero_citation_rate, mean citations/answer) measured over NON-REFUSAL
  answers. Validity is checked against the RETRIEVED set for that question, not the whole corpus.
- My reason: the simplest and most important thing is to make sure whatever the bot cites is
  actually one of the reports it was handed — a made-up citation is the dangerous failure, and I
  can catch it with plain name-matching, no AI, so it runs on every answer. Fuller
  sentence-by-sentence coverage isn't worth the fuzzy machinery until something shows I need it.
- Revisit when: (matching) a surname-only [Silva], an odd separator [X and Y], or a non-name
  bracket gets mis-flagged fabricated -> extend the splitter/matcher. `# TUNABLE(full-name
  normalized-exact match + comma/semicolon split for plural brackets; symptom above.)` (scope) ->
  add B's per-sentence coverage when the Phase-4 gate needs "every claim traceable" OR runs show
  grounded-but-uncited answers (valid citations but sparse).
- Steelman (B, logged once): coverage is what makes citations USEFUL, not merely non-fabricated —
  an answer that cites nothing scores zero-fabricated yet zero-traceable, and only graded coverage
  catches a wall of grounded-but-uncited claims. If Phase-4's gate becomes "every claim
  traceable," coverage is its real input and Phase-4 reopens this parser.

## D-021: Repro-hole closure (Tier-1 item 8; last instrument item before 1b)
- Date: 2026-07-25
- Context: runs are write-once and used as a comparison ledger, but four things a run's
  reproducibility depends on were unpinned in config.json — a run could differ from another with
  no config diff to explain it. Three are mechanical (no real alternative); the fourth (dirty-tree
  handling) is the one genuine fork.
- Mechanical closures (no D-fork; recorded for the audit trail):
  - `question_set.sha256` — `n` is a WEAK fingerprint: a gold quote can be edited (changing which
    gold is scored) without changing the row count, so two runs silently score different gold under
    a same-looking config. Hash the raw bytes; any edit -> different hash -> visible in a diff.
  - `normalizer.version` (`NORMALIZER_VERSION="norm-v1"` in normalize.py) — the config named the
    normalizer as a PATH STRING that never changes when the function body does; changing the body
    silently re-scores all hit-rate history (the module's own warning). Hand-bumped version,
    snapshotted like RUBRIC_VERSION / PROMPT_CONTRACT_VERSION, makes the change a config diff.
  - `embed_stack` (torch / sentence-transformers / numpy versions) — these determine the
    EMBEDDINGS but are INVISIBLE to the cache key (model_id + role + sha256(text), the torch
    caveat). Read via importlib.metadata (package metadata, NOT `import torch`) so a fully-cached
    offline run still never loads torch (preserves the D-015 cache payoff). HONESTY LIMIT recorded
    in-code: this is the version INSTALLED NOW; for a cache-HIT run it is not necessarily the
    version that produced the cached vector. Recording makes a mismatch AUDITABLE; it does not FIX
    the cache-key blindness (that fix = folding the stack version into the cache key, deferred —
    no signature has fired).
- The one fork — dirty-tree handling (config already records `git.dirty`):
  - Options: (a) record-only (status quo — silent flag, no nudge); (b) WARN, don't block (loud
    banner at launch, run proceeds); (c) ENFORCE (hard-block unless `--allow-dirty`).
  - Decision: (b) warn, don't block. `warn_if_dirty()` prints a banner at launch; nothing is
    blocked; `git.dirty` remains the permanent audit trail.
- My reason: I do throwaway offline retrieval runs constantly and they vastly outnumber
  baseline-of-record runs; a hard block would tax the tight synchronous feedback loop that is the
  harness's whole point, and `--allow-dirty` would become reflex muscle-memory anyway — so enforce
  pays friction daily and still erodes to a warning. The banner nudges at the one moment that
  matters (cutting a baseline) while the recorded flag makes any dirty baseline auditable after.
- Revisit when: a dirty-tree baseline-of-record slips through and pollutes a comparison despite the
  banner -> promote to ENFORCE for comparison-grade runs (keep offline iteration unblocked, e.g.
  gate only when the API lanes run). `# TUNABLE(warn-not-enforce; symptom: a dirty baseline gets
  compared anyway.)`
- Steelman (enforce, logged once): item 8's whole job is closing repro holes STRUCTURALLY so they
  don't depend on human vigilance — that is exactly why sha256 / version constants beat "remember
  to check." A warning is itself a vigilance-dependent guard, and an accidental baseline from a
  dirty tree is precisely the "SHA pins nothing" hole the roadmap flagged about the original run.
  Only enforce makes the guarantee structural rather than behavioral. It loses solely because the
  block would fire on every throwaway offline run too, where the friction/erosion cost is real.
