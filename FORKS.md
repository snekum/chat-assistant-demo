# Forks — Step 2: RAG baseline + eval harness

> Rev. 2 — revised after external critique. Changes: F1 and F2 merged into one
> joint fork (whole-doc forces a long-context embedder — the "fits any window"
> claim was wrong for the small local models); new fork F6 (answer-prompt
> contract); F3 judge conflict resolved; abstention made two-sided; candidate
> D-013 added (real-entity contamination); small fixes to gold-quote authoring
> and metric reporting.

Scope: (a) dumbest-possible single-turn stateless Q&A over `data/raw/`;
(b) the eval harness that judges it. Multi-turn memory is out of baseline scope
(see Deferred). Corpus: 268 deep-research reports, p50 ~2,186 words (~2,840 tok),
tightly clustered (p10 1,911 / p90 2,448), max 4,085; 100% follow one fixed
15-section `### N.` template with `[cite: N]` markers (16 off-template deleted).

Fixed constraints (NOT forks — do not reopen):
- Gold labels anchor to source spans (doc id + quote/offsets), never chunk ids.
- Hit-rate = overlap(retrieved chunk text, gold span). Chunking changes never
  invalidate the question set.
- Question schema has `type ∈ {single-hop, multi-hop, abstention}`. Abstention is
  scored (two-sided — see F3 / Default l).
- Retrieval hit-rate and answer groundedness computed & reported independently.
- Every run writes an immutable artifact: run_id, git SHA, full config snapshot,
  per-question results JSONL, summary metrics.

No prior DECISIONS.md row constrains any fork below (D-001..D-007 are the PARKED
pseudonymizer; unrelated, but D-013 below pre-registers a trigger to un-park a
names-only subset). Carry-over environment fact: per D-005, Windows Application
Control blocks native model downloads on this machine — load-bearing in F1+F2.

---

## 1. CRITICAL-PATH FORKS

Candidate DECISIONS.md rows. `Decision` and `My reason` left blank for you to fill.
Reasons must add something beyond the tradeoffs written here (your rule). Step-3
interrogation runs against the filled rows.

### F1+F2 (joint) — Chunking unit + embedding representation
These are one decision, not two: **the chunk size determines the embedder context
window you are allowed to use.** A whole-doc chunk (~2,840 tok p50, ~5,300 max)
does NOT fit the small local embedders — `bge-small-en-v1.5` caps at 512 tok,
`all-MiniLM-L6-v2` at 256 tok. Whole-doc + bge-small would silently embed only the
first ~18% of each dossier; sections 4–15 become unrepresented, questions about
them collapse, and per-doc hit-rate cannot localize the cause. So the embedder must
be chosen jointly with the chunk unit.

- **Options / tradeoffs:**
  - **Whole-doc + long-context embedder:** dumbest chunker; requires an embedder
    whose window ≥ max doc (~5,300 tok). Long-context options: **Voyage-3 (32k,
    Anthropic's recommended partner)**, OpenAI `text-embedding-3-small` (8,191),
    or local **`nomic-embed-text-v1.5` (8,192)** — but nomic is an HF download →
    same D-005 wall as spaCy. Coarse hit-rate (a "hit" = right person, not right
    passage); generator gets the whole doc.
  - **Section-aware (~200 tok/chunk) + small local embedder:** the 100%-uniform
    `### N.` template gives clean chunks that fit `bge-small` (512) / MiniLM (256)
    → the small local models become viable again, offline and free. Sharp hit-rate.
    But sections lack subject identity ("he/his"); more moving parts than the scope
    calls for at baseline.
  - **Fixed-window (512 tok) + small local embedder:** template-agnostic, fits
    local models, but splits mid-section and severs `[cite: N]` from its claim; no
    upside now that the corpus is uniform.
- **Recommend:** Whole-doc, per the earlier F1 decision (dumbest baseline, let a
  measured failure pull section-aware). Because whole-doc forces a long-context
  embedder, resolve the embedder as a **procedure, not a second decision:** attempt
  **one** local long-context download (`nomic-embed-text-v1.5`) as the single test
  of the D-005 wall; if it's blocked like spaCy was, fall back to **Voyage-3**.
  Decide now — the index and all hit-rate history are built on one embedder;
  switching re-embeds everything and breaks cross-run comparability.
- **Revisit when:** (chunking) groundedness poor AND traceable to mis-attribution
  inside a ~2.8k-tok blob, OR multi-hop hit-rate floors low → split to section-aware,
  **at which point `bge-small`/MiniLM revive as local embedder options** (200-tok
  chunks fit their windows). (embedder) the nomic download is blocked → Voyage-3.

### F3 — LLM-as-judge: model + rubric (the measurement instrument)
- **Options / tradeoffs:**
  - **LLM judge (Claude) with a fixed rubric** scoring groundedness (is each answer
    claim supported by the retrieved context?), correctness (vs gold), and abstention
    (correct refuse/answer per `type`): handles the nuance abstention needs; the
    judge *defines* the metrics, so a rubric/model change silently re-scores history.
  - **Deterministic overlap metrics (ROUGE/embedding-sim):** cheap, no drift, but
    can't tell "correctly abstained" from "wrongly answered" → wrong instrument.
  - **Same family judging its own output:** self-preference bias (named and, in
    Rev. 1, left unresolved — resolved below).
- **Recommend:** LLM judge, with the self-preference conflict *managed, not
  waved away:*
  - **Judge pin:** hold the judge FIXED across runs at a **Sonnet-class model,
    never the generator's tier** (generator default = Haiku). If the generator is
    later raised to Sonnet-class, the judge must move up or cross-family so it never
    grades its own tier.
  - **Calibration:** a scheduled **human-agreement sample (~20 judgments per rubric
    version)**; record the residual disagreement in the run ledger instead of
    pretending bias is zero.
  - **Per-lane judge inputs (important):** the **groundedness** judge sees only
    *retrieved context + answer* — **never the gold answer**, or the answer key
    leaks into the ruler. The **correctness** judge sees *gold + answer*.
  - **Rubric wording:** groundedness is scored as "supported by the provided
    context **alone**" (see D-013), rubric versioned + pinned in the config snapshot,
    judge temp 0.
  - **Primary/secondary:** **groundedness is the primary metric; correctness is
    secondary** (see D-013 for why).
- **Revisit when:** human/judge agreement on the calibration sample diverges, OR you
  bump the rubric (then re-score all runs under the new version — never mix).

### F4 — Hit-rate span-matching definition
- **Options / tradeoffs:**
  - **Offset-range intersection:** precise but breaks whenever normalization shifts
    offsets.
  - **Quote-substring containment (gold stores a verbatim quote; hit = quote found
    in retrieved chunk text):** robust to offset drift and re-chunking, human-legible.
  - **Fuzzy/normalized containment:** most robust but adds a threshold to defend.
- **Recommend:** Quote-substring containment. **Author each gold quote from the
  parsed text the system actually sees — post Sources-drop, post-NFC (Default a/c) —
  never from the raw file**, or a quote lifted from a dropped/renormalized region
  will never match its chunk. Decouples gold from offsets and chunk boundaries with
  no fuzz threshold. Defines the hit-rate metric — lock before authoring questions.
- **Revisit when:** verbatim quotes fail to match because normalization mangles them
  → move to fuzzy containment with an explicit, TUNABLE threshold.

### F5 — Question-set construction method
- **Options / tradeoffs:**
  - **Hand-authored (~30–50 Qs):** highest trust; you know the gold span exists (or
    provably doesn't, for abstention). Slow; small n → noisy metrics.
  - **LLM-generated then human-verified:** scales, but the LLM writes mostly
    single-hop lookups and can't reliably author abstention (needs proof of absence
    across all 268 docs) or true multi-hop Qs.
  - **Fully LLM-generated, unverified:** fast but circular and unsafe for ground truth.
- **Recommend:** Hand-author a small seed set first (all abstention + multi-hop —
  the eval's whole point, untrustworthy to generate), then LLM-assist single-hop
  expansion with human verification of each gold span.
- **Revisit when:** hand-authored n is too small to distinguish two configs (CIs
  overlap) → grow single-hop via verified LLM expansion.

### F6 — Answer-prompt contract (NEW — the other half of the measurement instrument)
Whether the system *can* abstain is a property of the generation prompt, not just
the model. Default (l) scores abstention as "system emits a refusal," but a bare
prompt makes that measure Haiku's default refusal temperament, not your system.
- **Options / tradeoffs:**
  - **(i) Bare "answer the question from this context":** the model tends to answer
    even when the context lacks the answer → abstention is unmeasurable / accidental.
  - **(ii) "Answer only from the provided context; if the answer isn't there, say
    so":** the minimum contract that makes abstention a real, controllable behavior.
  - **(iii) (ii) + mandatory inline citations:** stronger attribution, but citations
    are a Deferred upgrade — don't pull them into the baseline.
- **Recommend:** (ii). It's the least that makes abstention measurable and keeps
  citations deferred where they belong. **The prompt text is versioned and pinned in
  the config snapshot exactly like the judge rubric** — it is half the measurement
  instrument.
- **Revisit when:** groundedness/abstention errors trace to prompt phrasing rather
  than retrieval → tighten the contract (and consider pulling (iii)).

---

## 2. MEASUREMENT RISK — candidate D-013: real-entity contamination

Not a fork with clean options; a measurement cost to acknowledge and pre-register a
trigger for. These are web-researched dossiers on **real** people/companies, and the
generator's pretraining covers the same public web. So:
- **Correctness can pass without retrieval** — the model "knows" Aaron Silva / Serve
  Robotics from pretraining — and a judge can score *truth-in-the-world* rather than
  *support-in-the-provided-context*. Your eval can look great while retrieval is
  broken.
- **Mitigations that touch nothing now:** (1) rubric wording "supported by the
  provided context **alone**" (F3); (2) **groundedness primary, correctness
  secondary** (F3) — for a faithfulness bot, support-in-context is what you actually
  care about; (3) pre-register the signature below.
- **Pre-registered trigger (revisit-when):** *correctness high while hit-rate flat,*
  OR *answers cite facts absent from the retrieved chunks* → **un-park a names-only
  pseudonymizer subset.** D-001..D-007 mean it's half-built (`scripts/pseudonymize.py`
  has sound deterministic subject-mapping); a names-only swap is far cheaper than the
  full pass that was parked, and it removes the parametric shortcut.

---

## 3. DEFERRED FORKS (pre-registration — build only when its signature appears)

- **Reranking (cross-encoder):** hit-rate@k healthy but top-1 is the wrong
  person/passage → precision@1 gap.
- **Hybrid search (BM25 + dense):** misses on exact tokens (company names, "GCC",
  "Blue Book", tickers); abstention Qs get false answers on rare-term lookups.
- **Query rewriting / expansion:** multi-hop hit-rate << single-hop hit-rate.
- **Metadata filters (subject / company / section):** errors dominated by
  wrong-person retrievals.
- **Contextual headers / contextual retrieval:** section chunks retrieve but the
  groundedness judge can't tell who "he/his" is → subject-identity loss.
- **Answer-level citation/attribution (map claims → `[cite: N]`):** groundedness
  judge flags unsupported claims that *were* in retrieved context (also = F6 option iii).
- **Multi-turn memory / conversation state:** eval adds follow-ups; stateless
  baseline fails referential Qs.
- **Agentic / iterative retrieval:** multi-hop needs >1 retrieval round.

---

## 4. DEFAULTS I'M TAKING (vetoed by exception)

- **a. Parsing:** read `.md` as UTF-8, keep markdown markup in chunk text; drop the
  trailing `**Sources:**` URL list (opaque `vertexaisearch` redirects = noise); keep
  inline `[cite: N]` markers.
- **b. Retrieval scope:** single global flat pool, no per-subject filtering (dumbest
  baseline; subject filtering is a Deferred metadata-filter upgrade).
- **c. Normalization:** Unicode NFC; preserve original text verbatim (span fidelity);
  no lowercasing/stemming for dense retrieval. Same normalization applied to gold
  quotes and chunks (see F4).
- **d. Distance metric:** cosine over L2-normalized embeddings.
- **e. Index:** brute-force exact cosine over an in-memory numpy matrix (268 vectors
  at whole-doc baseline, ≤~4,000 if section-aware later → sub-ms; ANN premature).
- **f. top-k:** `k=3` for the whole-doc baseline. `# TUNABLE(3 whole docs ≈ 8.5k tok
  context, fits any generator; revisit when multi-hop needs facts from >3 subjects,
  or bump to k≈5–8 if we move to section chunks)`. Symptom wrong: a multi-hop gold
  span sits in the 4th-ranked doc.
- **g. Seeds:** all RNG seeds = 42, recorded in config snapshot. Generator & judge
  temp = 0.
- **h. Generator model (system-under-test, per-run variable, config-snapshotted):**
  default `claude-haiku-4-5`; not locked (varying it is the point). Note F3: if
  raised to Sonnet-class, the judge must move so it never grades its own tier.
- **i. Caching:** cache embeddings keyed by (model, normalized-text hash); do NOT
  cache generations or judgments (they are the measured output).
- **j. Dedup:** none — 268 distinct subjects, one doc each. Two `Solaru` subjects
  (Ademola/Adenuga) are different people; keep both.
- **k. File layout:** `runs/<run_id>/{config.json, results.jsonl, summary.json}`;
  question set + gold in `eval/questions.jsonl`. `run_id` = UTC timestamp + short git
  SHA; run dirs write-once; config captures git SHA + dirty flag. Config snapshot
  MUST include: embedder id, generator id, judge id, prompt-contract version (F6),
  and judge-rubric version (F3).
- **l. Abstention scoring (two-sided):** on `type=abstention` Qs, a pass = the system
  refuses / says "not in corpus" → **abstention-accuracy = correctly-abstained /
  abstention-Qs**. Report the *opposite direction too:* **false-refusal rate =
  wrongly-refused / answerable-Qs** (a system that always refuses scores 100% on the
  first and is caught only by the second). Always report both. Retrieval hit-rate is
  reported separately.
- **m. Summary metrics:** report **hit@1 and hit@3 separately, broken out per
  question type** (single-hop / multi-hop / abstention). hit@3 over 268 docs will
  saturate for single-hop; hit@1 is the discriminating number.

---

## Rows for you to fill (bring back with grounded reasons; then step-3 interrogation)

1. **F1+F2 (joint)** — chunk unit + embedder, resolved as one row.
2. **F6** — answer-prompt contract (new).
3. **F3** — judge, with the pin/calibration/per-lane-inputs resolution.
4. **F4** — span-matching.
5. **F5** — question-set method.
6. **D-013** — real-entity contamination cost + the pre-registered un-park trigger.

Approved without change: F4 & F5 recommendations, the Deferred list and its
signatures, Defaults a–k (l amended two-sided, m added).
