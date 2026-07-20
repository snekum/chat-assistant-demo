# Eval metrics reference — hit@k and comparing configs

Learning note for the eval harness (Component 2). Complements FORKS.md Default m
("report hit@1 and hit@3 separately, per question type") and F5 (CIs / question-set size).

## hit@k

For each question, the gold label names the **relevant doc** (the one containing the
answer). The retriever returns docs ranked by cosine similarity. **hit@k = is a relevant
doc within the top k?** Binary per question (1/0), then averaged over all N questions.

- `hit@1` = (# questions where the gold doc is ranked **#1**) / N
- `hit@3` = (# questions where the gold doc is in the **top 3**) / N
- Always `hit@3 >= hit@1` (top-3 contains top-1 — strictly easier bar).

Worked example (real probes, whole-doc baseline):

| question | gold doc | rank | hit@1 | hit@3 |
|---|---|---|---|---|
| banking-contract negotiation | Aaron Silva | 1 | 1 | 1 |
| Madagascar offshore center | Ross Fernandes | 67 | 0 | 0 |

## Why hit@1 discriminates between configs; why still report hit@3

- **Saturation:** with 268 docs and mostly single-hop lookups, the gold doc lands in the
  top 3 for almost any config -> `hit@3` sits near 1.0 with no headroom, so it can't
  separate whole-doc from section-chunked. `hit@1` is a harder bar, keeps spread, and is
  therefore the number to **compare configs on**.
- **But hit@3 still matters:** the generator receives all top-k=3 docs, so a gold doc at
  rank 2 still lets the answer be correct. `hit@3` = "did retrieval give the generator a
  chance"; `hit@1` = "how precise is the ranking." Report both; compare on hit@1.

## Significance — what to report NEXT TO hit@1

hit@1 is a proportion (X of N) and carries sampling noise. Report a 95% confidence
interval:

    CI half-width ~= 1.96 * sqrt( p*(1-p) / N )

| N | hit@1 = 0.70, 95% CI half-width |
|---|---|
| 40 | +-0.14 |
| 100 | +-0.09 |
| 400 | +-0.045 |

Consequences:
1. **Comparing A vs B:** if their CIs overlap, the difference is within noise — you cannot
   claim one is better (F5 "CIs overlap -> grow the question set").
2. **Sizing the set:** CI shrinks as 1/sqrt(N) -> halving it needs 4x the questions. A
   hand-authored N=40 set (D-012) can't resolve differences smaller than ~0.14. Know this
   before authoring.
3. **Sharper test (Component 2 decision):** since both configs run on the SAME questions,
   a PAIRED test (McNemar's, using only the questions where they disagree) is more powerful
   than comparing two independent CIs. Method choice (Wilson CI / bootstrap / McNemar) and
   target N are open harness decisions, folded into Default m.

## Distinct from other eval numbers (don't conflate)
- **hit@k** compares the *retrieved doc text* vs a *gold source span* (D-011, retrieval).
- **groundedness** judges the *generated answer* vs the *retrieved context* (D-010, primary).
- **abstention accuracy / false-refusal rate** score refuse-vs-answer behavior (Default l).
