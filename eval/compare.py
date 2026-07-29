"""Paired comparison of two chunk schemes on one binary metric -- McNemar's exact test (D-023,
ROADMAP Tier-2 item 11). Consumes an eval/sweep.py artifact and pairs questions on `id`.

WHY McNemar and not two independent confidence intervals (the n=45-is-tiny answer):
  The two arms answer the SAME questions, so the comparison is PAIRED. A question both arms hit
  (or both miss) tells you nothing about which is better -- it's the DISAGREEMENTS that carry the
  signal. McNemar looks only at the discordant pairs:
      b = A hit, B miss      c = A miss, B hit
  Under the null "the arms are equally good," a disagreement is a coin flip: b and c should be
  ~equal. So the test is an exact binomial test on b out of (b+c) with p=0.5. This reaches
  significance at ~8-10 discordant pairs when the effect is one-directional -- which is why ~40
  paired questions decide a delta that would need HUNDREDS of independent-CI samples (independent
  CIs must separate two noisy rates; McNemar only has to show the disagreements lean one way).

Concordant pairs (both hit / both miss) are reported but NOT in the test -- that is the whole
point, and the thing to say out loud in an interview.

Usage:
  ./.venv/Scripts/python.exe eval/compare.py analysis/sweep_<id>.json section whole_doc
  ./.venv/Scripts/python.exe eval/compare.py analysis/sweep_<id>.json section fixed268 --metric hit@5 --type single-hop
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p-value on the discordant pairs (b, c) under p=0.5. Exact (not the
    chi-square approximation) because the discordant count is small here. p=1.0 when b+c==0."""
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    # symmetric under p=0.5: two-sided = 2 * lower tail, capped at 1
    tail = sum(comb(n, i) for i in range(x + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def outcomes(per_q: list[dict], scheme: str, metric: str, qtype: str) -> dict[str, bool]:
    """{id: bool} for one scheme/metric/type."""
    return {
        r["id"]: bool(r[metric])
        for r in per_q
        if r["scheme"] == scheme and (qtype == "all" or r["type"] == qtype)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("scheme_a")
    ap.add_argument("scheme_b")
    ap.add_argument("--metric", default="hit@3")
    ap.add_argument("--type", default="single-hop",
                    help="single-hop | multi-hop | all")
    args = ap.parse_args()

    data = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
    per_q = data["per_question"]
    A = outcomes(per_q, args.scheme_a, args.metric, args.type)
    B = outcomes(per_q, args.scheme_b, args.metric, args.type)
    ids = sorted(set(A) & set(B))
    if not ids:
        raise SystemExit(f"no shared question ids for schemes {args.scheme_a!r}/{args.scheme_b!r} "
                         f"type={args.type!r} -- check the sweep artifact")

    both_hit = [i for i in ids if A[i] and B[i]]
    both_miss = [i for i in ids if not A[i] and not B[i]]
    a_only = [i for i in ids if A[i] and not B[i]]      # A hit, B miss  (b)
    b_only = [i for i in ids if not A[i] and B[i]]      # A miss, B hit  (c)
    b, c = len(a_only), len(b_only)
    p = mcnemar_exact_two_sided(b, c)

    ra = sum(1 for i in ids if A[i]) / len(ids)
    rb = sum(1 for i in ids if B[i]) / len(ids)

    print(f"\n=== McNemar: {args.scheme_a} vs {args.scheme_b}  ({args.metric}, {args.type}, n={len(ids)} paired) ===")
    print(f"{args.scheme_a} rate {ra:.2f}   {args.scheme_b} rate {rb:.2f}   (delta {ra - rb:+.2f})\n")
    print("2x2 paired table (rows = A, cols = B):")
    print(f"                     {args.scheme_b}=hit   {args.scheme_b}=miss")
    print(f"  {args.scheme_a}=hit        {len(both_hit):>5}        {b:>5}   <- b (A wins these)")
    print(f"  {args.scheme_a}=miss       {c:>5}        {len(both_miss):>5}   <- c is top-left of this row")
    print(f"\nconcordant (ignored by the test): both-hit={len(both_hit)}  both-miss={len(both_miss)}")
    print(f"DISCORDANT (the whole test): b={b} ({args.scheme_a} rescued) | c={c} ({args.scheme_b} rescued)")
    print(f"exact two-sided p-value = {p:.4f}   ({'significant' if p < 0.05 else 'NOT significant'} at 0.05)")
    if a_only:
        print(f"  {args.scheme_a}-only wins (b): {a_only}")
    if b_only:
        print(f"  {args.scheme_b}-only wins (c): {b_only}")
    if b + c < 8:
        print(f"NOTE: only {b + c} discordant pairs -- one-directional McNemar needs ~8-10 to reach "
              f"p<0.05, so a small delta here is UNDERPOWERED, not proven absent (D-012 sizing).")


if __name__ == "__main__":
    main()
