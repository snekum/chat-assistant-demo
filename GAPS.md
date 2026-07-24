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
- Status: OPEN. Tied to D-010; resolve before judge numbers are trusted as config COMPARISONS
  (they're fine as directional signals now).
