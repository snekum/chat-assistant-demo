---
name: phase-close
description: Use when closing out a ROADMAP phase — the owner says a phase is done ("wrap up Phase N", "let's close this phase", "/phase-close"). Runs the end-of-phase checklist: verify the phase is actually closeable, re-run the infra smoke check, refresh the README (numbers from the latest CLEAN-TREE baseline, architecture sketch, decision-log index, status), confirm every fork decided this phase has a DECISIONS row with the owner's reason + revisit-when, confirm GAPS opened this phase are resolved-with-evidence or deferred-with-a-trigger, and capture interviewer-voice drill questions into notes/interview-prep.md. Surfaces anything not ready rather than forcing closure; never invents a decision.
---

# Close a roadmap phase

An owner-triggered checklist run at the boundary between ROADMAP phases. Its job is to keep the
public-facing docs (README) and the private judgment ledger (DECISIONS / GAPS / interview-prep)
honest and in sync with what actually got built — so a recruiter reading the README and the
owner walking into an interview both see the current truth.

**This skill NEVER runs automatically.** There is no "phase done" event; only the owner knows a
phase closed. It fires when the owner triggers it. Do not schedule it or wire it to a hook.

**Prime directive — refresh and verify, never fabricate.** This skill edits documentation and
surfaces gaps. It does NOT make decisions, invent a reason, resolve a GAP that isn't resolved,
or write interview answers. Decisions are the owner's (CLAUDE.md protocol). If a fork was made
this phase without a DECISIONS row, STOP and ask the owner for their reason — do not draft one
for them.

## Step 0 — establish scope

Confirm with the owner (or infer from context, then confirm) **which phase is closing** and
read that phase's section in `ROADMAP.md`: its "what gets built", its interview questions, and
its design forks. That list is the closeability checklist for Steps 1–4. Also note the
baseline-of-record referenced by the current `runs/` — you will need it in Step 3.

## Step 1 — verify the phase is actually closeable (the gate)

Do this BEFORE editing any docs. If it fails, report what's missing and stop — a phase isn't
closed just because the code runs.

- **DECISIONS.md:** every fork this phase decided has a row with (a) the options, (b) the
  OWNER's reason grounded in this project (not a replay of the tradeoffs), (c) a revisit-when,
  and (d) `# TUNABLE(...)` for every threshold/number introduced ("no silent numbers",
  CLAUDE.md). Any fork the owner said "just build it" on must be logged `UNEXAMINED`. Missing a
  row → ask the owner, don't author it.
- **GAPS.md:** every gap opened during this phase is either RESOLVED with a dated evidence
  pointer (a run id / calibration report / commit), or explicitly deferred with a PRE-REGISTERED
  trigger signature. Update any stale status rows to match reality. A gap that's silently still
  open is the failure this step exists to catch.
- **Pre-registered predictions:** if the phase pre-registered a prediction (e.g. Phase 2's
  "multi-hop all-spans stays ~0 under any global chunking"), confirm the run either bore it out
  or the miss is recorded. A prediction made and then not checked is a hole.

## Step 2 — re-run the infra smoke check

```
./.venv/Scripts/python.exe scripts/smoke.py
```

Must exit 0 (deps, pgvector, embedder, key). If the phase added a component the smoke check
should now cover (e.g. Phase 5's observability backend), EXTEND `scripts/smoke.py` with a check
for it — the smoke script grows with the system. Report the result.

## Step 3 — refresh the README (four targeted parts, not a rewrite)

Only these drift per phase; leave the stable spine (design-philosophy framing, "why the corpus
is private") alone unless something structural changed.

1. **Numbers table** — refresh from the latest **CLEAN-TREE** baseline-of-record's
   `summary.json` (`git.dirty == false` in its `config.json`; a dirty run is warn-flagged per
   D-021 and must NOT become a headline number). Keep the "honest reading" column honest — carry
   forward caveats like the multi-hop structural zero and the false-refusal decomposition.
2. **Architecture sketch** — update only if the pipeline SHAPE changed this phase: chunk unit
   (Phase 2), + coordinator/router (Phase 3), + runtime gate (Phase 4), + serving/observability
   (Phase 5).
3. **Decision-log index** — add one-liners for any new D-rows (D-023+) under the right group.
4. **Status section** — state what's now done and what the next phase is, per ROADMAP.

If the phase produced a finalize-last artifact that was previously blocked (dashboard
screenshot, incident postmortem — Phase 5/6), wire it in now.

## Step 4 — capture interview drill questions (notes/interview-prep.md)

Per the interview-prep cadence: at phase end do a **consolidation sweep** for the components
built this phase. Constraints (do not violate):
- **Interviewer-VOICE scenario questions ONLY** — the hard "walk me through what happens
  when..." kind an actual staff interviewer asks. NEVER doc-enumeration ("list your decisions").
- **Do NOT write the answers.** The recall practice is the owner's; writing answers defeats it.
- Draw from the phase's ROADMAP "interview questions" as seeds, but sharpen them against what was
  ACTUALLY built and measured (real run numbers, real failure modes surfaced), not the generic
  roadmap phrasing.

## Step 5 — tee up the commit (owner confirms)

Stage ONLY the docs this skill touched (README, DECISIONS, GAPS, interview-prep, smoke script if
extended). Do NOT sweep in unrelated work-in-progress — ask if unsure what's related. Propose a
commit message; let the owner approve the commit + push (they push daily). **No AI co-author /
attribution trailer** in the message. Keep any sensitive-data rationale out of the public commit
text.

## What this skill deliberately does NOT do

- Cut a new baseline run (that's the phase's own work, not the close).
- Decide a fork, or write the owner's reason / interview answers.
- Force a GAP closed to make the phase "look done".
- Touch the stable README spine or unrelated files.
