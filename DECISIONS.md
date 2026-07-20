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
  retrieval) → reconsider the cheap names-only swap.

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
