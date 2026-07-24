"""Interval statistics for eval proportions (Phase 1a, Tier-1 item 1 / D-018).

Wilson score interval replaces the Wald interval the harness shipped with. Wald
(p +- z*sqrt(p(1-p)/n)) has two failure modes that both appeared in run
20260721: it collapses to ZERO width at p in {0, 1} (the multi-hop `ci95_halfwidth: 0.0`
artifact -- claiming certainty from 3 questions) and it can run outside [0, 1]
(single-hop 0.60 +- 0.43 -> upper bound 1.03). Wilson fixes both by construction:
it asks "what range of true rates would plausibly produce what I observed", which
is asymmetric near the edges and always stays inside [0, 1].

Wilson is for a BINOMIAL proportion (each trial hits or misses): hit@k, and later
groundedness / correctness / abstention. It is NOT applied to span-recall's macro
average (a mean of per-question fractions, not a binomial) -- that stays a point
estimate for now (a bootstrap/t-interval is deferred; not worth it at n~12).
"""
from __future__ import annotations

import math

# 95% two-sided. z=1.96 matches eval/METRICS.md; the only tunable here and it is a
# convention, not a fit -- change it and every band in every future run moves.
Z_95 = 1.96


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float | None, float | None]:
    """Wilson score interval (lo, hi) for `successes` of `n`. (None, None) if n == 0.

    Asymmetric and clamped to [0, 1]. At successes=0, n=3 -> ~(0.00, 0.56); at
    successes=3, n=5 -> ~(0.23, 0.88) -- both honest where Wald degenerates.
    """
    if n == 0:
        return (None, None)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))
