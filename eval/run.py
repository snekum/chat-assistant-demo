"""Eval run harness: retrieve -> rank diagnostics + hit-rate -> generate -> judge -> write-once artifacts.

Wires the scoring half together and emits an immutable run per FORKS Default k:
  runs/<run_id>/{config.json, results.jsonl, summary.json, rankings.jsonl}
run_id = UTC timestamp + short git SHA.

OFFLINE metrics (no API):
  - hit@1 / hit@3 per type -- all-spans, the headline (D-011). Reported with a WILSON 95% band
    (stats.wilson_interval, item 1) -- replaces the Wald `ci95_halfwidth` that degenerated to 0.0
    at p in {0,1}.
  - span-recall@k -- partial credit (found/total gold spans), secondary column (item 2). macro
    (per-question mean, headline) + micro (per-span pool, backstop). Nearly equal today (every
    multi-hop has 2 spans); diverges only if span counts do.
  - gold rank + MRR -- where each gold doc actually lands in the FULL 268-row ranking (D-018).
    hit@k is a red light; the rank is the diagnosis: rank 2-5 => ranking problem (rerank/bump-k),
    rank 40+ => representation problem (re-chunk). Cannot be backfilled into write-once runs, so
    recorded now. Full ranking goes to rankings.jsonl; results.jsonl keeps gold_ranks + mrr.

ONLINE metrics (generate + judge) run only if ANTHROPIC_API_KEY is set; otherwise SKIPPED and
their fields are null -- the run still produces a valid retrieval artifact.

Usage:  ./.venv/Scripts/python.exe eval/run.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # read .env into os.environ (does NOT override real shell exports) -> keys visible

import cost  # noqa: E402
import generate  # noqa: E402
import hitrate  # noqa: E402
import normalize  # noqa: E402
import store  # noqa: E402
from retrieve import DEFAULT_K, Retriever  # noqa: E402
from stats import wilson_interval  # noqa: E402

QUESTIONS = Path("eval/questions.jsonl")
RUNS_DIR = Path("runs")
SEED = 42  # Default g; recorded, not load-bearing for this deterministic pipeline
ANSWERABLE = {"single-hop", "multi-hop"}


def git_info() -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True)
            return out.stdout.strip() if out.returncode == 0 else None
        except FileNotFoundError:
            return None

    sha = _run(["rev-parse", "--short", "HEAD"]) or "nocommit"
    porcelain = _run(["status", "--porcelain"])
    dirty = bool(porcelain) if porcelain is not None else True
    return {"sha": sha, "dirty": dirty}


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def gold_rank(doc_id: str, full_ranking: list[dict]) -> int | None:
    """1-based rank of the FIRST chunk belonging to doc_id (D-018). 'First' = best rank among a
    doc's chunks, so the metric survives section-chunking (a doc's rank = its best-placed chunk)."""
    for i, row in enumerate(full_ranking, 1):
        if row["doc_id"] == doc_id:
            return i
    return None  # gold doc absent from the corpus -- a data error the validator should have caught


def is_exact_refusal(answer: str) -> bool:
    """Deterministic refusal detector (D-019, Tier-1 item 5): did the answer match the EXACT
    refusal sentence the contract mandates (generate.REFUSAL_STRING)? This is now the OFFICIAL
    refusal label -- it drives abstention scoring and filters the groundedness denominator; the
    judge's semantic is_refusal is kept only as a recorded cross-check (see summarize()).

    Normalized EQUALITY under the D-011 normalizer (not containment): a substantive answer can't
    normalize to EXACTLY the refusal sentence, so this can never wrongly pull a real answer out
    of the groundedness denominator (the reputational-risk direction, D-009). An off-script
    refusal (the bot's own words) reads False here and surfaces in the judge-divergence log --
    which is the point: that log is how we learn, at smoke-run 1b, whether the bot obeys "reply
    exactly" before we'd ever trust a smarter (judge) label instead.
    # TUNABLE(equality not containment; revisit at 1b. Symptom wrong: divergence log fills with
    #   genuine refusals that merely appended a citation/token -> loosen to normalized containment.)
    """
    return normalize.normalize_for_match(answer) == normalize.normalize_for_match(generate.REFUSAL_STRING)


def main() -> None:
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1

    api = generate.has_credentials()
    retriever = Retriever()
    generator = generate.Generator() if api else None
    judge = None
    if api:
        from judge import Judge  # noqa: E402

        judge = Judge()

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{git_info()['sha']}"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=False)  # write-once (Default k): never overwrite a run

    results: list[dict] = []
    rankings: list[dict] = []  # sidecar: full 268-row ranking per question (D-018 option c)
    for r in rows:
        qvec = retriever.emb.embed_query(r["question"])
        retrieved = store.search(retriever.conn, qvec, k=DEFAULT_K)  # top-k WITH text (hit@k + gen)
        full_ranking = store.rank_all(retriever.conn, qvec)  # all rows, doc_id+score, no text

        rec: dict = {
            "id": r["id"],
            "type": r["type"],
            "question": r["question"],
            "evidence": r["evidence"],
            "retrieved": [
                {"rank": i + 1, "doc_id": h["doc_id"], "score": round(h["score"], 4)}
                for i, h in enumerate(retrieved)
            ],
            "gold_ranks": None,   # rank of each gold doc in the full ranking (D-018)
            "mrr": None,          # reciprocal of the best-placed gold rank
            "hit@1": None,
            "hit@3": None,
            "span_recall@1": None,
            "span_recall@3": None,
            "spans_found@1": None,
            "spans_found@3": None,
            "spans_total": None,
            "answer": None,
            "is_refusal": None,      # judge's semantic call -- CROSS-CHECK only under D-019
            "refusal_exact": None,   # deterministic string match -- OFFICIAL label (D-019); null offline
            "grounded": None,
            "correct": None,
            "judge_reason": {"groundedness": None, "correctness": None},
            "cost": None,  # per-stage tokens/latency/$ (item 4); null on offline runs
        }

        if r["type"] in ANSWERABLE:  # abstention has no gold span -> retrieval metrics N/A
            ev = r["evidence"]
            rec["hit@1"] = hitrate.hit_at_k(ev, retrieved, 1)
            rec["hit@3"] = hitrate.hit_at_k(ev, retrieved, 3)
            f1 = hitrate.spans_found_at_k(ev, retrieved, 1)
            f3 = hitrate.spans_found_at_k(ev, retrieved, 3)
            total = len(ev)
            rec["spans_found@1"], rec["spans_found@3"], rec["spans_total"] = f1, f3, total
            rec["span_recall@1"], rec["span_recall@3"] = f1 / total, f3 / total
            ranks = [gold_rank(e["doc_id"], full_ranking) for e in ev]
            rec["gold_ranks"] = ranks
            valid = [x for x in ranks if x is not None]
            rec["mrr"] = 1.0 / min(valid) if valid else None  # best-placed gold (D-018 MRR note)

        if api:
            answer, gen_meta = generator.answer(r["question"], retrieved)
            context = generate.format_context(retrieved)
            g, g_meta = judge.groundedness(r["question"], context, answer)
            c, c_meta = judge.correctness(r["question"], r["gold_answer"], answer)
            rec.update(
                answer=answer,
                is_refusal=g["is_refusal"],
                refusal_exact=is_exact_refusal(answer),  # deterministic OFFICIAL label (D-019)
                grounded=g["grounded"],
                correct=c["correct"],
                judge_reason={"groundedness": g["reason"], "correctness": c["reason"]},
            )
            stages = {"generation": gen_meta, "groundedness": g_meta, "correctness": c_meta}
            for m in stages.values():
                m["cost_usd"] = cost.cost_usd(m)  # price the raw usage (item 4)
            rec["cost"] = {
                "stages": stages,
                "total_usd": round(sum(m["cost_usd"] or 0 for m in stages.values()), 6),
                "total_latency_s": round(sum(m["latency_s"] for m in stages.values()), 3),
            }

        results.append(rec)
        rankings.append({
            "id": r["id"],
            "ranking": [[row["doc_id"], round(row["score"], 4)] for row in full_ranking],
        })

        if rec["span_recall@3"] is not None:
            line = (f"  [{r['id']:<8}] {r['type']:<11} gold={rec['gold_ranks']} "
                    f"hit@1={rec['hit@1']} hit@3={rec['hit@3']} span@3={rec['span_recall@3']:.2f}")
        else:
            line = f"  [{r['id']:<8}] {r['type']:<11} (abstention -- no gold span)"
        if api:
            line += f" refuse={rec['is_refusal']} grounded={rec['grounded']}"
        print(line)

    summary = summarize(rows, results, api)
    config = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "embedder": {"id": retriever.emb.model_id, "dim": retriever.emb.dim},
        "generator": {"id": generate.GENERATOR_MODEL,
                      "prompt_contract_version": generate.PROMPT_CONTRACT_VERSION},
        "judge": ({"id": __import__("judge").JUDGE_MODEL,
                   "rubric_version": __import__("judge").RUBRIC_VERSION} if api else None),
        "retrieval": {"k": DEFAULT_K, "distance": "cosine",
                      "index": "pgvector-exact (D-014)"},
        "seed": SEED,
        "normalizer": "eval/normalize.py:normalize_for_match (D-011)",
        "ci_method": "wilson-score-95 (item 1; replaces Wald)",
        "question_set": {"path": str(QUESTIONS), "n": len(rows), "by_type": by_type},
        "stages": ["retrieval", "rank_diagnostics", "hit_rate", "span_recall"]
                  + (["generation", "judge"] if api else []),
        "artifacts": {"rankings": "rankings.jsonl -- full 268-row ranking/question (D-018 c)"},
        "api_available": api,
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "rankings.jsonl").open("w", encoding="utf-8") as f:
        for rk in rankings:
            f.write(json.dumps(rk, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report(run_id, out_dir, summary, api)


def summarize(rows: list[dict], results: list[dict], api: bool) -> dict:
    def binom(pred, subset) -> dict:
        """Binomial rate + Wilson 95% band (item 1). For hit@k and the API boolean lanes."""
        vals = [pred(x) for x in subset]
        vals = [v for v in vals if v is not None]
        n = len(vals)
        k = sum(1 for v in vals if v)
        p = k / n if n else None
        lo, hi = wilson_interval(k, n)
        return {"n": n, "successes": k, "rate": p, "ci_lo": lo, "ci_hi": hi}

    def macro(pred, subset) -> float | None:  # mean of per-question fractions (item 2)
        return mean([pred(x) for x in subset])

    def micro(found_key, subset) -> float | None:  # pooled spans (item 2 backstop)
        found = sum(x[found_key] for x in subset if x[found_key] is not None)
        total = sum(x["spans_total"] for x in subset if x["spans_total"] is not None)
        return found / total if total else None

    answerable = [x for x in results if x["type"] in ANSWERABLE]
    types = ("single-hop", "multi-hop")

    hit: dict = {}
    span: dict = {}
    mrr: dict = {}
    for t in types:
        sub = [x for x in results if x["type"] == t]
        hit[t] = {"hit@1": binom(lambda x: x["hit@1"], sub),
                  "hit@3": binom(lambda x: x["hit@3"], sub)}
        span[t] = {
            "span_recall@1": {"macro": macro(lambda x: x["span_recall@1"], sub),
                              "micro": micro("spans_found@1", sub)},
            "span_recall@3": {"macro": macro(lambda x: x["span_recall@3"], sub),
                              "micro": micro("spans_found@3", sub)},
        }
        mrr[t] = macro(lambda x: x["mrr"], sub)
    hit["answerable_overall"] = {"hit@1": binom(lambda x: x["hit@1"], answerable),
                                 "hit@3": binom(lambda x: x["hit@3"], answerable)}
    span["answerable_overall"] = {
        "span_recall@1": {"macro": macro(lambda x: x["span_recall@1"], answerable),
                          "micro": micro("spans_found@1", answerable)},
        "span_recall@3": {"macro": macro(lambda x: x["span_recall@3"], answerable),
                          "micro": micro("spans_found@3", answerable)},
    }
    mrr["answerable_overall"] = macro(lambda x: x["mrr"], answerable)

    summary: dict = {"hit_rate": hit, "span_recall": span, "mrr": mrr,
                     "groundedness": None, "correctness": None, "abstention": None,
                     "cost": None}
    if api:
        # OFFICIAL refusal label = deterministic refusal_exact (D-019); judge is_refusal is the
        # recorded cross-check. The label both scores abstention AND filters what gets grounded.
        answered = [x for x in answerable if not x["refusal_exact"]]  # substantive answers only
        non_refusal = [x for x in results if not x["refusal_exact"]]
        summary["groundedness"] = {  # PRIMARY: of substantive answers, fraction fully supported
            "primary_over_non_refusal": binom(lambda x: x["grounded"], non_refusal),
            "answerable_answered": binom(lambda x: x["grounded"], answered),
        }
        summary["correctness"] = binom(lambda x: x["correct"], results)  # secondary; per-type = item 7
        abst = [x for x in results if x["type"] == "abstention"]
        labeled = [x for x in results if x["refusal_exact"] is not None]  # both labels exist (api)
        divergent = [x for x in labeled if x["refusal_exact"] != x["is_refusal"]]
        summary["abstention"] = {  # two-sided (Default l); label = deterministic refusal_exact (D-019)
            "abstention_accuracy": binom(lambda x: x["refusal_exact"], abst),
            "false_refusal_rate": binom(lambda x: x["refusal_exact"], answerable),
            "refusal_label": {  # judge kept as cross-check; divergence = the bot went off-script
                "official": "refusal_exact",
                "judge_divergence_n": len(divergent),
                "judge_divergence_rate": round(len(divergent) / len(labeled), 3) if labeled else None,
                "divergent_ids": [x["id"] for x in divergent],
            },
        }
        priced = [x["cost"] for x in results if x.get("cost")]  # per-stage tokens/$/latency (item 4)
        if priced:
            n = len(priced)
            summary["cost"] = {
                "prices_dated": cost.PRICES_DATED,
                "total_usd": round(sum(c["total_usd"] for c in priced), 4),
                "mean_usd_per_question": round(sum(c["total_usd"] for c in priced) / n, 4),
                "by_stage": {
                    st: {
                        "input_tokens": sum(c["stages"][st]["input_tokens"] for c in priced),
                        "output_tokens": sum(c["stages"][st]["output_tokens"] for c in priced),
                        "total_usd": round(sum(c["stages"][st]["cost_usd"] or 0 for c in priced), 4),
                        "mean_latency_s": round(sum(c["stages"][st]["latency_s"] for c in priced) / n, 3),
                    }
                    for st in ("generation", "groundedness", "correctness")
                },
            }
    return summary


def report(run_id: str, out_dir: Path, summary: dict, api: bool) -> None:
    def fmt(m: dict) -> str:
        if m["rate"] is None:
            return "n/a"
        return f"{m['rate']:.2f} [{m['ci_lo']:.2f}, {m['ci_hi']:.2f}] (n={m['n']})"

    def sr(m: dict) -> str:
        return "n/a" if m["macro"] is None else f"macro={m['macro']:.2f} micro={m['micro']:.2f}"

    def mf(v) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    h, s, mr = summary["hit_rate"], summary["span_recall"], summary["mrr"]
    print(f"\n=== run {run_id} -> {out_dir} ===")
    print("hit@1 (all-spans headline; Wilson 95% band):")
    print(f"  single-hop: {fmt(h['single-hop']['hit@1'])}")
    print(f"  multi-hop : {fmt(h['multi-hop']['hit@1'])}   (2-doc Qs cannot hit@1 -- structural 0)")
    print(f"  overall   : {fmt(h['answerable_overall']['hit@1'])}")
    print(f"hit@3 overall: {fmt(h['answerable_overall']['hit@3'])}")
    print("span-recall@3 (partial credit; secondary):")
    print(f"  single-hop: {sr(s['single-hop']['span_recall@3'])}")
    print(f"  multi-hop : {sr(s['multi-hop']['span_recall@3'])}")
    print(f"MRR (best-placed gold): single={mf(mr['single-hop'])}  "
          f"multi={mf(mr['multi-hop'])}  overall={mf(mr['answerable_overall'])}")
    if api:
        g = summary["groundedness"]["primary_over_non_refusal"]
        print(f"groundedness (PRIMARY, non-refusal answers): {fmt(g)}")
        print(f"correctness  (secondary): {fmt(summary['correctness'])}")
        a = summary["abstention"]
        print(f"abstention-accuracy: {fmt(a['abstention_accuracy'])}  |  "
              f"false-refusal rate: {fmt(a['false_refusal_rate'])}")
        rl = a["refusal_label"]
        dr = "n/a" if rl["judge_divergence_rate"] is None else f"{rl['judge_divergence_rate']:.0%}"
        line = f"refusal label: OFFICIAL=refusal_exact (deterministic) | judge divergence: {rl['judge_divergence_n']} ({dr})"
        if rl["divergent_ids"]:
            line += f" ids={rl['divergent_ids']}"
        print(line)
        if summary.get("cost"):
            cst = summary["cost"]
            by = cst["by_stage"]
            print(f"cost: ${cst['total_usd']:.4f} total, ${cst['mean_usd_per_question']:.4f}/question "
                  f"(gen ${by['generation']['total_usd']:.4f} / ground ${by['groundedness']['total_usd']:.4f} "
                  f"/ correct ${by['correctness']['total_usd']:.4f}; prices {cst['prices_dated']})")
    else:
        print("groundedness / correctness / abstention: SKIPPED (no ANTHROPIC_API_KEY).")


if __name__ == "__main__":
    main()
