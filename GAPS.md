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
