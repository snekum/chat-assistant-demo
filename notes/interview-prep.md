# Interview prep — self-quiz → talk-track

This is a **study aid**, not a working doc. DECISIONS.md / ROADMAP.md are the exhaustive
record (dense, by D-number); this is the distilled, thematic, scannable version you actually
review before an interview.

## How to use it (the loop that builds recall)
1. Pick a question. **Answer it cold, out loud or written, WITHOUT opening the source.**
2. Write your one-line answer in your own words under the question.
3. *Then* open the `(check: …)` pointer and diff. Where your cold answer was vague/wrong →
   that's a real gap → log it to GAPS.md.
4. Re-attempt (don't re-read). Retrieval practice > re-reading.

Filling this in *is* the studying; the filled-in version *is* your pre-interview talk-track.

## When to do it
- **At each phase boundary, while fresh** (mirrors the "3 hard questions" drill in CLAUDE.md).
- **A cumulative sweep every few phases** — re-attempt OLD sections cold (early decisions decay first).
- **A full pass 1–2 days before an interview.**
- Optionally: feed your COLD written answer (not the source) to your adversarial LLM critic and
  have it attack it — harsher than self-grading.

> Grows with the project. Sections below cover work completed through Phase 1c. Add a section per
> phase as you go (Phase 2 chunking A/B, Phase 3 multi-agent, etc.).

---

## 1. The measurement discipline (why the eval is built the way it is)
- Why **instrument before baseline** — what's the irreversible cost of baselining first?
  *(follow-up: your runs are write-once — what diagnostic is permanently lost if you don't record it up front?)* — my answer: ______  `(check: ROADMAP §1, D-018)`
- Runs are **write-once with full config snapshots**. Two runs disagree by 5 points — from the
  config alone, enumerate everything that could have moved. — ______  `(check: D-021)`
- What four things did the **repro-holes** work pin, and which one is only *auditable* not *fixed*?
  *(follow-up: why can't the embed-stack version be trusted for a cache-HIT run?)* — ______  `(check: D-021)`
- Gold is anchored to **doc_id + verbatim quote**. Why does that survive re-chunking, and what
  would break if you'd anchored to chunk_ids instead? — ______  `(check: D-011, D-011 amendment)`

## 2. Retrieval & storage
- **Whole-doc vs section chunking** — why did you deliberately build whole-doc *first*?
  *(follow-up: whole-doc's failure is QUIET — which metric saturates and hides it, and which two
  metrics do you actually watch?)* — ______  `(check: D-008)`
- **pgvector over numpy brute-force and over FAISS** — what capability justified the DB at 268
  rows, and what was your real-world FAISS scar? *(follow-up: steelman numpy — what's the strongest
  interview story you gave up?)* — ______  `(check: D-014)`
- **hit@k is doc_id-anchored.** What false-hit did the un-anchored version allow once quotes got
  generic ("Santa Clara University" in 15 docs)? — ______  `(check: D-011 amendment)`
- **k=3.** What's the symptom that would prove k=3 wrong for multi-hop? — ______  `(check: retrieve.py TUNABLE)`

## 3. The judge (the ruler)
- Your groundedness judge **never sees the gold answer** — why? *(follow-up: then what catches a
  fluent answer grounded in a mis-retrieved WRONG document?)* — ______  `(check: D-010)`
- Why is **groundedness primary and correctness secondary**? — ______  `(check: D-010, D-013)`
- You **can't set temperature** on the Sonnet judge — how do you know the ruler reads the same
  twice? — ______  `(check: D-010 temp-0 amendment; ROADMAP Tier-2 item 10 flip-rate)`
- **Single-provider (Claude) for gen AND judge** — what's the risk, and what single piece of
  evidence flips you to a cross-provider judge? — ______  `(check: D-017)`
- **G-002 (open gap):** what makes "spot-check ~20 judgments" theater rather than a control?
  What are the three unspecified pieces? — ______  `(check: GAPS G-002)`

## 4. Abstention & refusal (two-sided behavior)
- "Abstention accuracy is 100% for a bot that always refuses" — where's the counter-metric?
  *(follow-up: who labels is_refusal, and which OTHER metric silently breaks if that label is
  wrong?)* — ______  `(check: D-019; the groundedness denominator)`
- Why did you pick a **deterministic string** refusal label over the judge's semantic one, then
  **loosen equality→prefix**? What did the smoke run show that forced the change? — ______  `(check: D-019, 1b finding)`
- **Prove a private fact is absent from all 268 docs.** *(follow-up: what's the false-negative risk
  of your grep procedure?)* — the JNTU story is your live example. — ______  `(check: ab-010 notes, D-012 abstention)`
- **Superlative trap** ("who's the youngest founder?") — why is the correct behavior to refuse, and
  what's the failure mode you're testing for? — ______  `(check: ab-008/009, D-012)`

## 5. Citations
- The prompt mandates `[doc_id]` tags. What does your parser assert — claim truth, claim support,
  or citation *validity*? *(follow-up: what's a "fabricated citation" and why is it a contamination
  tell?)* — ______  `(check: D-020, D-009)`
- Why validity-only and not per-sentence coverage (option B)? What reopens that decision? — ______  `(check: D-020)`

## 6. Contamination (the subjects are real people)
- The generator may know these semi-public people from pretraining. Which of your metrics is
  contamination-PROOF, which is ROBUST, and which is VULNERABLE? — ______  `(check: D-013 metric note)`
- What's the exact cell (which metrics high, which low) that is the contamination signature in the
  wild, and why can neither groundedness nor hit-rate establish "retrieval was load-bearing"? — ______  `(check: D-013)`
- The **closed-book control** — what does open−closed correctness measure? — ______  `(check: D-013, ROADMAP item 14)`
- Why does **anonymizing the corpus NOT fix production contamination** — and what does? *(follow-up:
  Q1-measurement vs Q2-production; why did closed-book make anonymization unnecessary even for Q1?)* — ______  `(check: D-013 "what anonymization is FOR")`

## 7. Statistics & sizing (where most candidates die)
- **Wilson vs Wald** intervals — what artifact did Wald produce at p∈{0,1}? — ______  `(check: ROADMAP Tier-1 item 1)`
- **"n=45 looks tiny — why is your comparison legitimate?"** *(follow-up: what exactly makes it
  paired, and what were your discordant counts?)* — the McNemar argument. — ______  `(check: D-012 sizing)`
- **Multi-hop is only 6** — why is that principled and not lazy? *(follow-up: why is it
  coverage-limited, not power-limited?)* — ______  `(check: D-012 multi-hop sizing)`
- **Abstention is 2-per-mode** — what's the "mode" axis, and why did absence-proof COST order the
  work? — ______  `(check: D-012 abstention sizing)`
- Your growth trigger is "A/B delta inside the CIs," NOT "bands look wide." Why is the second one
  p-hacking? — ______  `(check: D-012)`

## 8. Multi-hop & why Phase 3 (not chunking) fixes it
- Multi-hop scores ~0 on all-spans. Why is that failure NOT fixed by better chunking?
  *(follow-up: a gold doc sits at rank 34 — chunking finer helps or hurts, and why?)* — ______  `(check: ROADMAP Phase 2 prediction; the offline before-picture)`
- What does Phase 3 do that fixes it — name the mechanism and the D-016 capability it activates. — ______  `(check: D-016, ROADMAP Phase 3)`
- **G-001:** "Compare Ross A and Ross B" and your own log shows the wrong Ross outranking the right
  one — walk through what happens and what a good system does instead of guessing. — ______  `(check: GAPS G-001; the clarify third state)`

## 9. Your own known gaps (naming them is strong signal)
- Name your four logged gaps and the trigger that would make each one urgent. — ______  `(check: GAPS G-001..G-004)`
- **G-004 (aggregates):** why can't hit@k grade "who can I talk to about expanding in China?" and
  what different instrument would? — ______  `(check: GAPS G-004)`

---

### Meta-questions an interviewer loves (no source — these test synthesis)
- "Walk me through your eval methodology in 2 minutes." (the whole instrument, top-down)
- "What's the weakest part of this system and how would you know?"
- "You had limited time — what did you deliberately NOT build, and how do you defend each cut?"
  *(the anti-bloat table + future-bucket triggers)*
- "Which single decision are you least sure about?" (honesty; point at an UNEXAMINED/gap)
