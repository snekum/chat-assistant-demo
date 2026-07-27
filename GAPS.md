# Gaps — questions I couldn't fully answer under interrogation

## G-001: Entity-resolution error is not covered by the planned eval
- Date: 2026-07-18
- Component: 1 (retrieval core) / D-016 person resolution
- Gap: The eval harness (D-009 abstention, D-011 hit-rate) has no notion of resolution
  error. When a user names a person, something must resolve name -> person_id; that step
  can fail two ways — resolve to the WRONG person confidently (worse), or return nothing.
  I confirmed this is a blind spot introduced after the design was set.
- What's missing / to build:
  1. A THIRD outcome state — "ask for confirmation / disambiguate" — distinct from D-009's
     two-sided refuse-vs-answer. Current scoring (Default l) cannot represent it.
  2. New question types: ambiguous-name queries where the correct behavior is ASK (the two
     Solaru subjects — Ademola vs Adenuga — are a ready-made in-corpus test), and
     not-in-corpus-person queries where correct behavior is REFUSE.
  3. Desired policy: on >1 match -> confirm/abstain, never auto-resolve to one confidently.
- Why it matters: real, semi-public people; a confidently wrong resolution is the exact
  reputational-risk failure D-009 was written to prevent (owner's own D-009 reason).
- Status: OPEN. Resolution LOGIC stays deferred (D-016), but the eval to catch its failure
  should be designed alongside Component 2 so the gap is measured, not assumed away.

## G-002: Judge calibration / drift-alarm protocol is undefined
- Date: 2026-07-21
- Component: 2 (LLM-as-judge, D-010)
- Gap: D-010 pre-registers "~20 human-checked judgments per rubric version" as a drift alarm,
  but the operational protocol is unspecified: (1) which verdicts to hand-check (random?
  stratified by lane × question type? disagreement-prone ones?), (2) what human/judge agreement
  threshold trips the alarm, (3) what happens to runs ALREADY written under the current rubric
  version when it trips (re-score / invalidate / annotate?).
- Why it matters: the judge is the ruler (D-010); an uncalibrated ruler silently mis-scores
  groundedness, which is the PRIMARY metric. Without a threshold + a defined response,
  "spot-check ~20" is theater, not a control.
- Owner could not answer under interrogation (2026-07-21): "gap."
- Shape of a fix (not a decision — for the owner to make later): stratify the ~20 across both
  lanes and all three types, INCLUDING the abstention/refusal calls; author human labels blind
  to the judge's verdict; report simple agreement now (Cohen's kappa once N is bigger); set the
  alarm threshold as a TUNABLE with a stated symptom; on trip, cut a NEW rubric version and
  re-score ALL runs under it (never mix versions, per D-010) rather than patching in place.
- Status: RESOLVED 2026-07-27 (Phase 1e; D-022 + calibration/20260726T091259Z-2158c98/report.json).
  All three unknowns are now answered and EXECUTED: (1) which verdicts → hybrid stratified design
  (natural real verdicts for the false-alarm direction + 8 injected adversarial plants for the
  dangerous direction the 33/0-skewed run can't produce); (2) trip threshold → asymmetric
  zero-tolerance (one judge-passed plant trips), natural set report-only; (3) on-trip response →
  bump RUBRIC_VERSION + re-score all runs (D-010). Result at d010-v1: no trip (6/6 plants caught),
  100% natural agreement, both gray-zone plants owner==judge==ungrounded, flip-rate 0%. kappa still
  DEFERRED until ≥50 cumulative labels (degenerate at n=33 all-true). The protocol is now a
  re-runnable instrument (eval/calibrate.py) armed for every future rubric bump — so judge numbers
  are trustworthy as config COMPARISONS at d010-v1, not just directional.

## G-003: D-019 refusal instrument — three items deferred to smoke-run 1b
- Date: 2026-07-24
- Component: 2 (eval harness) / D-019 deterministic refusal cross-check
- Gap: String-authoritative refusal labeling ships with three unresolved items, all correctly
  deferred until the FIRST online run (1b) produces real divergence data — named here so they get
  measured, not forgotten. Owner could not resolve #2/#3 under interrogation (2026-07-24) and chose
  to defer; the direction-of-bias warning below was delivered and accepted.
  1. Direction-blind divergence. `judge_divergence_n` fuses two OPPOSITE signals: (A)
     string=answer / judge=refusal = the bot refused OFF-SCRIPT (the "promote the judge" signal),
     vs (B) string=refusal / judge=answer = the JUDGE misread a clean refusal (a "judge is
     unreliable, do NOT promote it" signal). The count alone can't separate them. Recoverable:
     both labels are stored per-question in results.jsonl, so a directional split is a ~3-line
     summary add at 1b — nothing lost by deferring.
  2. abstention_accuracy is a FLOOR, not the truth. String-authoritative counts a refusal only if
     the bot emits the exact sentence, so a correct-but-off-script decline reads as "didn't
     refuse" → the reported number understates true semantic abstention ability by (at most) the
     hedge-direction divergence rate. Reporting caveat, not code: the README number is
     "exact-contract-compliance abstention"; the divergence rate is the gap to semantic.
  3. Revisit trigger is an unresolved fork. D-019 says high divergence → "tighten the contract
     (f6-v2) OR promote the judge" with no decider. Proposed ordering (decide at 1b, not now): try
     the CHEAP fix first — tighten the prompt to force exact-string compliance, keeping
     determinism — and promote the judge ONLY if compliance stays low after that, or if refusal
     phrasing proves inherently too varied to pin to one string.
- Why it matters: the bias is DIRECTIONAL. If Haiku refuses off-script often, false_refusal_rate
  AND groundedness both look BETTER than reality (optimistic / overclaim direction — a hedged
  refusal enters the groundedness denominator and the D-010 rubric scores refusals grounded=true
  by construction, so it lands as a grounded SUCCESS and pulls the rate toward 1.0), while
  abstention_accuracy looks worse (conservative). Two of three lean optimistic. This is NOT silent:
  the inflation set == the divergent set == the off-script refusals, so the 1b divergence rate is
  the GATE — read it before trusting groundedness or false_refusal_rate.
- Status: DEFERRED to 1b by design (no cutoff pre-committed before data, per D-013 posture). No
  work now; the divergence log + D-019 revisit-when already encode the gate. Cheap fixes (direction
  split; exclude/annotate divergent cases in the groundedness denominator) are 1b-sized.

## G-004: Aggregate/set queries — declared in scope, zero eval coverage
- Date: 2026-07-25
- Component: retrieval + eval instrument; lands with Phase 3 routing
- Gap: implicit-group queries ("Who can I talk to about expanding my business in China?") have NO
  eval coverage anywhere, and nothing on the roadmap explicitly owned them until this row. The
  ex-mh-1 removal drew the right line for the HIT@K GOLD SET — multi-hop must NAME its people so
  exactly-N docs are provably the answer — but the side effect is that the aggregate class was
  excluded entirely rather than measured differently. Owner has now declared aggregates IN SCOPE
  (2026-07-25): "CEOs routinely ask these kinds of questions" — for this product (CEOs finding
  CEOs for expansion/GTM/pivot outreach) the aggregate is arguably the PRIMARY query style.
- Why the current instrument can't grade them: hit@k needs a COMPLETE answer key. With an
  incomplete key, retrieval that correctly fetches a relevant doc missing from the key is scored a
  MISS — the instrument punishes correct behavior (the ex-mh-1 failure: real group ~45 docs, key
  listed 2). The gold-set rule STANDS (eval-goldset-review keeps rejecting aggregation questions
  from the hit@k set); aggregates need a DIFFERENT instrument, not different questions in the same
  one.
- Candidate instruments (build-time decision, NOT made here):
  1. Complete-key corpus sweep: for a handful of chosen aggregate questions, sweep all 268 docs
     (grep + LLM pass for phrasings grep misses), hand-verify, declare the full relevant set →
     exact recall measurable. The presence-proof mirror of the abstention absence-proof, similar
     cost (~20–30 min/question). Feasible ONLY at 268 docs; decays at 10x — state this honestly.
  2. Must-include partial key: 3–4 hand-verified certainly-relevant people; a recall floor that
     never punishes correct extras, but blind to junk.
  3. Judged per-retrieved-doc relevance: LLM judge scores each returned doc for relevance —
     precision with zero authoring cost, but blind to misses and puts the judge inside a retrieval
     metric (inherits the G-002 calibration burden).
  A serious design composes 1-or-2 (the missed-people side) with 3 (the junk side).
- System-side blocker, why this waits for Phase 3: retrieval fetches top-3; with ~9 truly relevant
  people the system CANNOT answer an aggregate well regardless of what the eval says. Aggregates
  likely need their own router lane (bigger k / filtered retrieval / different generation
  contract). Eval design follows that behavior decision — same sequencing logic as G-001's
  clarify state.
- Status: OPEN, deliberately deferred. Trigger: Phase 3 requirements memo / routing design names
  aggregates a lane → build the aggregate instrument alongside it. Related: negation-constrained
  variants ("China experience but NOT manufacturing") share instrument 1's complete-key machinery
  and are a known dense-retrieval weakness — evaluate together when this fires.
