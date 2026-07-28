# Eval question set — schema & authoring guide (D-012)

The question set is the eval's ground truth. Per D-012 you HAND-AUTHOR the abstention and
multi-hop questions (LLMs may brainstorm alongside you, never author unattended); single-hop
may be LLM-assisted but every gold span is human-verified. Run the validator after every
edit — it proves each gold quote actually matches the parsed corpus (D-011).

    ./.venv/Scripts/python.exe eval/validate_questions.py     # exit 0 = all valid

## Row schema (`eval/questions.jsonl`, one JSON object per line)

| field | meaning |
|---|---|
| `id` | stable unique id (e.g. `sh-001`, `mh-001`, `ab-001`) |
| `type` | `single-hop` \| `multi-hop` \| `abstention` |
| `question` | the user-style question text |
| `evidence` | list of `{doc_id, quote}` — the gold source spans (see rules) |
| `gold_answer` | short reference answer; for abstention, the expected refusal |
| `author` | `hand` \| `llm-assisted` \| `example-seed` (provenance, D-012) |
| `notes` | provenance / proof-of-absence / caveats |

`doc_id` is the document's canonical name (= filename stem, e.g. `Jane Doe`). Gold anchors
to **doc_id + verbatim quote**, never chunk ids (a fixed design constraint) — so re-chunking
never invalidates the set.

### Evidence rules (enforced by the validator)
- **single-hop**: exactly 1 evidence entry.
- **multi-hop**: >=2 evidence entries spanning >=2 distinct docs.
- **abstention**: evidence = `[]` (no doc contains the answer — that's the point).
- Every `quote` must be **copied verbatim from the doc's parsed text** and must survive
  D-011 normalization (lowercase, strip punctuation + `[cite:...]`, collapse whitespace).
  When in doubt, copy a short, distinctive phrase and let the validator confirm it.

### Authoring an abstention question — the hard part
You must PROVE the fact is absent from all 268 docs, not just from the obvious one. Grep the
raw corpus for every phrasing of the fact before trusting a refusal question. Record the
check in `notes`. Worked example: `ex-ab-1` (a company's revenue), verified via grep.

## Seed candidates (pointers — turn these into questions; verify every gold span)

**single-hop (LLM-assist ok, then verify):** any specific, checkable fact — a company
founding year, a person's degree/university, a company HQ city, a named product. Include some
**buried facts** (deep in a long dossier) like `ex-sh-2` (a fact far down a long document) —
those are where whole-doc is expected to fail, so they turn the dilution finding into a
measured number.

**multi-hop (hand-author):** facts that must be joined across >=2 docs.

CRITICAL distinction — multi-hop is NOT aggregation:
- **multi-hop** = the answer is a few SPECIFIC docs joined by reasoning. Scoreable.
- **aggregation / filter** ("who's in fintech?", "Harvard alums", "who can help in China")
  = the answer is the SET of every matching person. These are a COMPLETENESS TRAP: gold
  must enumerate ALL matches (42 docs mention fintech; 19 mention Harvard but "mention" !=
  "alum"), and missing one punishes a correct retrieval. They're also metadata-filter
  queries (a DEFERRED fork) that depend on `persons.meta` not yet extracted. Keep them OUT
  of this hit@k set -> they belong in a future set-scored recommendation track.

Because these dossiers do NOT cross-reference each other (D-004), true chain-multi-hop is
scarce here. The realistic form is a COMPARISON OF TWO NAMED PEOPLE (crisp gold, forces
retrieval to fetch both):
- "Which earned their MBA outside the US - Person A or Person B?" (one abroad, one domestic)
- "Which of X and Y bootstrapped without venture capital?"
Also avoid FUZZY gold: "who can help me expand in China" is a judgment, not a checkable
fact. If you can't point to an exact quote that settles it, it's not a hit@k question.

**abstention (hand-author + prove absence):** facts a dossier plausibly *could* hold but
doesn't — private financials, home address, exact salary, personal contact info, a fact
about a person NOT in the corpus at all.

**ambiguity / resolution (scoring deferred, but seed now):** two subjects who share a surname
are a ready-made test: a bare-surname query ("tell me about <surname>") should trigger a
clarify/confirm, not a confident wrong pick. These need a THIRD outcome state the current
scorer lacks — capture the questions now, wire scoring when name-resolution handling is
added. (Open question: how to score a clarify/confirm outcome.)

## Sizing (see eval/METRICS.md)
CI half-width ~ 1.96*sqrt(p(1-p)/N): N=40 -> +-0.14, N=100 -> +-0.09. Start with a hand seed
(all abstention + multi-hop by hand, a few single-hop) big enough to see direction; grow
single-hop via verified LLM-assist once you know which comparison you need to resolve.
