"""Coverage ranking for several-needs questions (experiment arm 3; NOT wired into find_members).

One search blends a question's needs, so a person strong on ONE need ("healthcare") can outrank
people who cover all of them ("healthcare AND technology"). Here each need is searched on its own
with the product's hybrid ranking, and the per-need rankings are combined so that covering every
need beats excelling at one:

  rrf      reciprocal rank fusion across the needs: a person's points = sum of 1 / (60 + rank)
  min      a person's WORST need rank decides; rrf points break ties
  rrf_ctx  rrf plus the asker's own situation as one more list at CONTEXT_WEIGHT (core vs context)

One need: the question is searched as asked, so single-need questions change only via exclusions.
Exclusions: a person whose rank on an excluded topic beats their rank on every need moves below
everyone who doesn't -- they mainly fit the area the member said they don't want.

ONE implementation: eval/recommend_eval.py's needs_* arms call rank_by_needs, and find_members
would call the same function if an arm ships.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import hybrid

# TUNABLE(the asker's situation counts half a need: a fit matters, but it is context, not the ask.
#   Symptom wrong: people matching only the asker's industry outrank people matching the ask ->
#   lower; context-blind lists the reviewer marks down -> raise.)
CONTEXT_WEIGHT = 0.5
MODES = ("rrf", "min", "rrf_ctx")


def _order(conn, emb, text: str, scheme: str, keep) -> list[str]:
    return hybrid.rank_people(conn, emb.embed_query(text), text, scheme, keep=keep, top=0)[0]


def rank_by_needs(conn, emb, question: str, needs: list[str], exclude: list[str],
                  context: str, scheme: str, mode: str = "rrf",
                  keep: Callable[[str], bool] = lambda p: True) -> list[str]:
    """-> every kept person, best first."""
    needs = [n for n in needs if n.strip()]
    orders = ([_order(conn, emb, n, scheme, keep) for n in needs] if len(needs) >= 2
              else [_order(conn, emb, question, scheme, keep)])
    weights = [1.0] * len(orders)
    lists = list(orders)
    if mode == "rrf_ctx" and context.strip():
        lists.append(_order(conn, emb, context, scheme, keep))
        weights.append(CONTEXT_WEIGHT)
    points: dict[str, float] = defaultdict(float)
    for order, w in zip(lists, weights):
        for i, p in enumerate(order):
            points[p] += w / (hybrid.RRF_K + i + 1)

    need_rank = [{p: i for i, p in enumerate(o)} for o in orders]
    if mode == "min" and len(orders) > 1:
        ranked = sorted(points, key=lambda p: (max(r[p] for r in need_rank), -points[p]))
    else:
        ranked = sorted(points, key=lambda p: -points[p])

    exclude = [x for x in exclude if x.strip()]
    if exclude:
        excl_rank = [{p: i for i, p in enumerate(_order(conn, emb, x, scheme, keep))}
                     for x in exclude]
        mainly = {p for p in ranked
                  if min(r[p] for r in excl_rank) < min(r[p] for r in need_rank)}
        ranked = [p for p in ranked if p not in mainly] + [p for p in ranked if p in mainly]
    return ranked
