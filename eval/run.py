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

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
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

# Override with EVAL_QUESTIONS=<path> to score a subset (e.g. a smoke-run slice) without
# touching the gold file. config.question_set.{path,sha256} record whatever was actually used.
QUESTIONS = Path(os.environ.get("EVAL_QUESTIONS", "eval/questions.jsonl"))
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


def sha256_file(path: Path) -> str:
    """Content fingerprint of the question set (repro item 8). `n` in the config is a weak
    fingerprint -- a quote can be edited (changing which gold is scored) without changing the
    row count, so two runs silently score different gold under a same-looking config. Hashing
    the raw bytes pins the exact set: any edit -> different hash -> visible in a config diff."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embed_stack_versions() -> dict:
    """Versions of the packages that determine the EMBEDDINGS, snapshotted because they are
    INVISIBLE to the cache key (model_id + role + sha256(text)) -- the torch caveat. A torch or
    sentence-transformers upgrade can change what a cache-MISS produces while old cached vectors
    persist, and nothing in the cache key would flag the mix. Recording the versions here makes
    such a mismatch auditable across two runs (it does NOT fix the cache-key blindness).

    Read via importlib.metadata (package METADATA, not the DLLs), NOT `import torch`, so a
    fully-cached offline run still never loads torch -- preserving the D-015 cache payoff.
    HONESTY CAVEAT: this is the version INSTALLED NOW. For a cache-HIT run it is not necessarily
    the version that produced the cached vector (which was made whenever the cache was populated).
    """
    def _v(pkg: str) -> str | None:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    return {pkg: _v(pkg) for pkg in ("torch", "sentence-transformers", "numpy")}


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
    """Deterministic refusal detector (D-019, Tier-1 item 5): does the answer BEGIN with the
    exact refusal sentence the contract mandates (generate.REFUSAL_STRING)? This is the OFFICIAL
    refusal label -- it drives abstention scoring and filters the groundedness denominator; the
    judge's semantic is_refusal is kept only as a recorded cross-check (see summarize()).

    PREFIX match under the D-011 normalizer -- loosened from strict equality at smoke-run 1b. The
    run 20260725T142745Z (5 Qs) fired the pre-registered symptom: Haiku refuses ON-script but
    APPENDS a helpful explanation ("I don't know based on the provided reports.\\n\\nThe report
    does not contain..."), so strict equality missed 3/3 genuine refusals -- crashing
    abstention_accuracy to 0.00 and leaking those refusals into the groundedness denominator.
    All divergence was one-directional (judge=refusal, string=no-match = genuine off-script
    refusal; zero judge misreads), which is exactly the "loosen the string" signal D-019 named.

    Prefix keeps determinism AND nearly keeps equality's key guarantee: a substantive answer
    almost never OPENS with the exact refusal sentence, so a real answer still can't be wrongly
    pulled out of the groundedness denominator (the reputational-risk direction, D-009). The
    trailing-space guard (nr + " ") keeps the match token-aligned (won't fire on a longer word
    that merely starts with the sentence's last token).
    # TUNABLE(prefix not equality; loosened at 1b on the append-explanation evidence above.
    #   Symptom wrong: a substantive answer that OPENS with the refusal sentence then keeps
    #   answering gets mislabeled a refusal -> tighten back toward equality or add a length cap.)
    """
    na = normalize.normalize_for_match(answer)
    nr = normalize.normalize_for_match(generate.REFUSAL_STRING)
    return bool(nr) and (na == nr or na.startswith(nr + " "))


_CITE_TAG = re.compile(r"\[([^\]]+)\]")  # any [....] bracket in the answer text
_CITE_PREFIX = re.compile(r"^\s*cite\s*:\s*", re.I)  # corpus-style "cite:" the generator copies
_SECTION_REF = re.compile(r"^section\s+\d+$", re.I)  # a within-doc pointer, not a report citation


def _citation_names(answer: str) -> list[str]:
    """Doc-citation candidates from the answer's [..] tags (D-020). The contract form is
    [Full Name], but at the 1d baseline the generator also emitted corpus-style variants --
    [cite: Ed Fahey], [cite: 1], [Ed Fahey, Section 5] -- so normalize before matching: strip a
    leading 'cite:' prefix (baseline sh-008/mh-005 mis-flagged VALID names fabricated because
    'cite' leaked into the normalized token), and drop non-doc brackets (bare numerics = corpus
    [cite: N] markers; 'Section N' = a within-doc pointer). This lowers only FALSE fabrications;
    a real fabricated citation (a name not in the retrieved set) still surfaces.
    # TUNABLE(strip cite:-prefix + drop numeric/Section-N brackets; revisit if a NEW non-name
    #   bracket form appears in a run, or a real doc named like one of these gets dropped.)
    """
    out: list[str] = []
    for tag in _CITE_TAG.findall(answer):
        for raw in re.split(r"[,;]", tag):
            name = _CITE_PREFIX.sub("", raw).strip()
            if name and not name.isdigit() and not _SECTION_REF.match(name):
                out.append(name)
    return out


def parse_citations(answer: str, retrieved: list[dict]) -> dict:
    """Deterministic citation validity check (D-020, Tier-1 item 6). The contract (f6-v1) makes
    the generator tag every claim with a document-level [doc_id]; doc_id == the person's full
    name. Here we read those tags back and ask, for each, whether it names one of the reports
    actually RETRIEVED for this question -- NOT the whole corpus. A tag that names a doc the
    generator was never shown is a FABRICATED citation: the exact "unverified citation launders
    hallucinations" failure D-009 warns of, and a contamination tell (the name came from memory,
    not the provided reports -- D-013).

    Match = full-name normalized-exact (reuse the D-011 normalizer) against the retrieved doc_ids;
    plural brackets are split on comma/semicolon because the contract sanctions "report(s)". Tag
    extraction + corpus-format normalization live in _citation_names (see its note for the 1d fix).

    Validity only (option A): counts + a fabricated flag, no per-sentence coverage (that needs a
    fuzzy "is this a factual sentence" splitter -- deferred, D-020). Computed for every answer but
    summarized over NON-REFUSAL answers (a refusal correctly carries no citation).
    """
    retrieved_ids = {normalize.normalize_for_match(h["doc_id"]) for h in retrieved}
    cited = _citation_names(answer)
    fabricated = [n for n in cited if normalize.normalize_for_match(n) not in retrieved_ids]
    return {
        "n_citations": len(cited),
        "n_valid": len(cited) - len(fabricated),
        "n_fabricated": len(fabricated),
        "has_any_citation": bool(cited),
        "has_fabricated": bool(fabricated),
        "fabricated": sorted(set(fabricated)),  # the offending names, for eyeballing
    }


def warn_if_dirty() -> None:
    """Reproducibility, dirty-tree rule: WARN, don't block. A hard block
    would tax the constant throwaway offline runs that are the harness's tight-feedback point; the
    loud warning is the nudge and config.git.dirty is the permanent audit trail. A comparison-grade
    baseline-of-record must be cut from a clean tree -- this only reminds, it does not enforce."""
    if git_info()["dirty"]:
        print("!" * 72)
        print("!! DIRTY TREE: uncommitted changes present. This run is NOT comparison-grade --")
        print("!! the recorded SHA pins nothing. Commit before cutting a baseline-of-record.")
        print("!" * 72)


def main() -> None:
    warn_if_dirty()
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
            "citations": None,  # [doc_id]-tag validity vs retrieved set (item 6, D-020); null offline
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
            rec["citations"] = parse_citations(answer, retrieved)  # deterministic [doc_id] check (D-020)
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
        "normalizer": {"impl": "eval/normalize.py:normalize_for_match (D-011)",
                       "version": normalize.NORMALIZER_VERSION},  # snapshot like the rubric (item 8)
        "embed_stack": embed_stack_versions(),  # torch/st versions -- invisible to cache key (item 8)
        "ci_method": "wilson-score-95 (item 1; replaces Wald)",
        "question_set": {"path": str(QUESTIONS), "n": len(rows), "by_type": by_type,
                         "sha256": sha256_file(QUESTIONS)},  # pins exact gold bytes (item 8)
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
                     "citations": None, "cost": None}
    if api:
        # OFFICIAL refusal label = deterministic refusal_exact (D-019); judge is_refusal is the
        # recorded cross-check. The label both scores abstention AND filters what gets grounded.
        answered = [x for x in answerable if not x["refusal_exact"]]  # substantive answers only
        non_refusal = [x for x in results if not x["refusal_exact"]]
        summary["groundedness"] = {  # PRIMARY: of substantive answers, fraction fully supported
            "primary_over_non_refusal": binom(lambda x: x["grounded"], non_refusal),
            "answerable_answered": binom(lambda x: x["grounded"], answered),
        }
        # Correctness (secondary), split by type (item 7). A blended number hides opposite
        # failures -- strong single-hop can mask weak abstention (confidently answering questions
        # it should refuse). answerable_overall vs abstention is the split the contamination story
        # (D-013) + closed-book control (item 14) care about. Types derived from data so a future
        # type (e.g. resolution, G-001) is included automatically.
        correctness: dict = {}
        for t in sorted({x["type"] for x in results}):
            correctness[t] = binom(lambda x: x["correct"], [x for x in results if x["type"] == t])
        correctness["answerable_overall"] = binom(lambda x: x["correct"], answerable)
        correctness["overall"] = binom(lambda x: x["correct"], results)  # old blended number, kept
        summary["correctness"] = correctness
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
        # Citation validity (item 6, D-020), over NON-REFUSAL answers (a refusal has no citations).
        cited_answers = [x for x in non_refusal if x["citations"] is not None]
        if cited_answers:
            summary["citations"] = {
                "n_answers": len(cited_answers),
                "fabricated_citation_rate": binom(lambda x: x["citations"]["has_fabricated"], cited_answers),
                "zero_citation_rate": binom(lambda x: not x["citations"]["has_any_citation"], cited_answers),
                "mean_citations_per_answer": macro(lambda x: x["citations"]["n_citations"], cited_answers),
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
        print("correctness (secondary, per type):")
        for t, m in summary["correctness"].items():
            print(f"  {t:<18}: {fmt(m)}")
        a = summary["abstention"]
        print(f"abstention-accuracy: {fmt(a['abstention_accuracy'])}  |  "
              f"false-refusal rate: {fmt(a['false_refusal_rate'])}")
        rl = a["refusal_label"]
        dr = "n/a" if rl["judge_divergence_rate"] is None else f"{rl['judge_divergence_rate']:.0%}"
        line = f"refusal label: OFFICIAL=refusal_exact (deterministic) | judge divergence: {rl['judge_divergence_n']} ({dr})"
        if rl["divergent_ids"]:
            line += f" ids={rl['divergent_ids']}"
        print(line)
        if summary.get("citations"):
            ct = summary["citations"]
            print(f"citations (over {ct['n_answers']} non-refusal answers): "
                  f"fabricated-rate {fmt(ct['fabricated_citation_rate'])} | "
                  f"zero-citation {fmt(ct['zero_citation_rate'])} | "
                  f"mean {mf(ct['mean_citations_per_answer'])}/answer")
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
