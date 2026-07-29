"""Phase 2 free retrieval k-sweep (DECISIONS D-023) -- the CORE chunking A/B, offline + free.

For every chunk_scheme present in the DB (whole_doc / section / fixed<N>) and every ANSWERABLE
question, retrieve the top-MAXK chunks WITH text (scheme-filtered, D-023) and compute hit@k /
span_recall@k for k in K_EVAL, plus each gold doc's rank in the full per-chunk ranking + MRR.
No generation, no judge -> ~$0 (D-023 cost re-scope: the retrieval comparison is the free part).

Reading (why this is the whole point): sweeping k at MATCHED size across schemes separates
"structure-aware boundaries helped" (section vs fixed<N> at the same k) from "we just fetched
more chunks" (same scheme across k) -- the fork-0 confound. hit@k is the metric that moves for
single-hop (recall@k == precision@1 == hit@k when there's one gold doc); span_recall diverges
only on multi-hop (partial credit).

CAVEAT on gold_rank: under section/fixed the ranking is over CHUNKS (~4k rows), and a gold doc
appears once per chunk -> gold_rank = the doc's BEST-placed chunk (D-018 MRR note). So "rank 40"
means 39 chunks outrank the gold's best chunk, NOT 39 docs -- a different denominator than the
268-doc whole_doc ranking. Compare ranks WITHIN a scheme, not across.

Writes analysis/sweep_<utc>.json: per-question rows (id, type, scheme, hit@k, gold_rank) for
eval/compare.py (McNemar pairs on id), plus per-scheme/per-type aggregates with Wilson bands.

Usage:  ./.venv/Scripts/python.exe eval/sweep.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

import hitrate  # noqa: E402
import store  # noqa: E402
from embedder import NomicLocal  # noqa: E402
from stats import wilson_interval  # noqa: E402

QUESTIONS = Path("eval/questions.jsonl")
OUT_DIR = Path("analysis")
ANSWERABLE = {"single-hop", "multi-hop"}
K_EVAL = [1, 3, 5, 8]   # eval cutoffs (fork 2); generation-run k picked from this curve
MAXK = max(K_EVAL)


def gold_rank(doc_id: str, ranking: list[dict]) -> int | None:
    for i, row in enumerate(ranking, 1):
        if row["doc_id"] == doc_id:
            return i
    return None


def schemes_in_db(conn) -> list[str]:
    rows = conn.execute(
        "SELECT chunk_scheme, count(*) FROM chunks GROUP BY chunk_scheme ORDER BY chunk_scheme"
    ).fetchall()
    print("db chunk_scheme rows:", dict(rows))
    return [s for s, _ in rows]


def wilson(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"n": n, "successes": k, "rate": (k / n if n else None), "ci_lo": lo, "ci_hi": hi}


def main() -> None:
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = [r for r in rows if r["type"] in ANSWERABLE]
    emb = NomicLocal()
    conn = store.connect()
    schemes = schemes_in_db(conn)

    qvecs = {r["id"]: emb.embed_query(r["question"]) for r in answerable}  # embed once, reuse per scheme

    per_q: list[dict] = []
    for scheme in schemes:
        for r in answerable:
            qv = qvecs[r["id"]]
            retrieved = store.search(conn, qv, k=MAXK, scheme=scheme)
            ranking = store.rank_all(conn, qv, scheme=scheme)
            ev = r["evidence"]
            total = len(ev)
            ranks = [gold_rank(e["doc_id"], ranking) for e in ev]
            valid = [x for x in ranks if x is not None]
            rec = {
                "id": r["id"], "type": r["type"], "scheme": scheme,
                "gold_ranks": ranks,
                "mrr": (1.0 / min(valid) if valid else None),
                "top_docs": [h["doc_id"] for h in retrieved[:MAXK]],
            }
            for k in K_EVAL:
                rec[f"hit@{k}"] = hitrate.hit_at_k(ev, retrieved, k)
                rec[f"span_recall@{k}"] = hitrate.spans_found_at_k(ev, retrieved, k) / total
            per_q.append(rec)

    summary = aggregate(per_q, schemes)
    print_table(summary, schemes)

    OUT_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"sweep_{run_id}.json"
    out.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "k_eval": K_EVAL, "schemes": schemes,
        "embedder": emb.model_id,
        "note": "offline retrieval sweep (D-023); gold_rank is a per-CHUNK rank for multi-chunk schemes",
        "per_question": per_q,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(per_q)} question x scheme rows) -- feed pairs to eval/compare.py")


def aggregate(per_q: list[dict], schemes: list[str]) -> dict:
    out: dict = {}
    for scheme in schemes:
        out[scheme] = {}
        for t in ("single-hop", "multi-hop", "answerable_overall"):
            sub = [x for x in per_q if x["scheme"] == scheme
                   and (t == "answerable_overall" or x["type"] == t)]
            if not sub:
                continue
            block: dict = {"n": len(sub)}
            for k in K_EVAL:
                hits = sum(1 for x in sub if x[f"hit@{k}"])
                block[f"hit@{k}"] = wilson(hits, len(sub))
                block[f"span_recall@{k}"] = round(
                    sum(x[f"span_recall@{k}"] for x in sub) / len(sub), 4)
            mrrs = [x["mrr"] for x in sub if x["mrr"] is not None]
            block["mrr"] = round(sum(mrrs) / len(mrrs), 4) if mrrs else None
            out[scheme][t] = block
    return out


def print_table(summary: dict, schemes: list[str]) -> None:
    print("\n=== Phase 2 retrieval sweep (single-hop hit@k, Wilson 95%) ===")
    hdr = "scheme".ljust(12) + "".join(f"hit@{k}".ljust(20) for k in K_EVAL) + "MRR"
    print(hdr)
    for scheme in schemes:
        b = summary.get(scheme, {}).get("single-hop")
        if not b:
            continue
        line = scheme.ljust(12)
        for k in K_EVAL:
            m = b[f"hit@{k}"]
            line += f"{m['rate']:.2f}[{m['ci_lo']:.2f},{m['ci_hi']:.2f}]".ljust(20)
        line += f"{b['mrr']:.3f}" if b["mrr"] is not None else "n/a"
        print(line)
    print("\n(single-hop n =", summary.get(schemes[0], {}).get("single-hop", {}).get("n"), ")")
    print("multi-hop span_recall@3 by scheme (all-spans hit stays ~0 -> Phase 3, per D-023):")
    for scheme in schemes:
        mb = summary.get(scheme, {}).get("multi-hop")
        if mb:
            print(f"  {scheme.ljust(12)} span_recall@3={mb['span_recall@3']:.2f}  hit@3={mb['hit@3']['rate']:.2f}")


if __name__ == "__main__":
    main()
