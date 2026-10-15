"""Recommendation-quality eval (D-048..D-050): score a member-ranking method against the answer key
(analysis/answer_key_v3_haiku55.jsonl: every member x every question, graded 0/1/2 by Haiku 5.5 on
rubric v3). Local and free for embedding-only methods; every method is scored the same way.

Metrics, per question, then averaged:
  ndcg@5      are the 5 people shown the best ones, best first? (gain 3 for a 2, 1 for a 1;
              1.0 = the ideal 5 in the ideal order). The headline: the product shows 5 people.
  useful@5    share of the 5 shown that the key rates useful (1 or 2)
  best@5      share of the possible "clearly useful" (2) people that made the 5
              (denominator min(5, number of 2s))
  recall@k    share of ALL clearly useful (2) people ranked within the top k (10/20/50): did the
              search at least FIND them, even if not on screen? Low recall@50 = a retrieval
              problem (re-represent, hybrid, decompose); good recall@50 but low ndcg@5 = a
              ranking problem (rerank).

Questions dropped (D-050 follow-up, assistant rule): no member graded 2 (cannot test "found the
best"), or more than half the network graded 2 (every method looks perfect).

Search input (D-052): every method searches with the chat model's own wording of the question --
the find_members criterion it writes, recorded once per question in LIVE_QUERIES and frozen, so
runs stay comparable -- because that is what the product sends. Runs before D-052 searched with
the raw question; `--raw` reproduces them. Every run records its input in config.json.

Usage:  python eval/recommend_eval.py <method> [<method> ...] [--raw]   (see METHODS)
        python eval/recommend_eval.py compare <run_a> <run_b> [metric]
Writes runs/<ts>-<sha>-rec-<method>/ : config.json + summary.json (metrics only, committable) and
results.jsonl (ranked person_ids per question -- gitignored by the runs/** rule).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

KEY = ROOT / "analysis" / "answer_key_v3_haiku55.jsonl"
QUESTIONS = ROOT / "eval" / "recommend_questions.jsonl"
SHOWN = 5                    # the product's member cap (tools.TOP_N_MEMBERS)
RECALL_KS = (10, 20, 50)
GAIN = {0: 0, 1: 1, 2: 3}
# TUNABLE(drop a question when more than half the network is "clearly useful": every method
#   scores ~perfect on it. Symptom wrong: a kept question shows ndcg@5 ~1.0 for every method.)
MAX_SHARE_OF_2S = 0.5
# The recorded chat-model wording per question (eval/split_needs.py), PINNED by name: a new
# recording never changes the baseline silently. Re-record (~$0.04, Batch API) and re-pin when the
# agent prompt or the find_members description changes -- _splits() warns when they have.
LIVE_QUERIES = ROOT / "analysis" / "needs_haiku45_20261014T204144Z.jsonl"
QUERY_INPUT = "live"         # "raw" = the question as typed (every run before D-052)


def load_key() -> tuple[dict[str, dict[str, int]], list[dict]]:
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for line in KEY.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        grades[r["qid"]][r["person_id"]] = r["grade"]
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l]
    n = len(next(iter(grades.values())))
    kept = [q for q in qs if 0 < sum(g == 2 for g in grades[q["id"]].values()) <= MAX_SHARE_OF_2S * n]
    return grades, kept


def score(ranked: list[str], g: dict[str, int]) -> dict:
    twos = {p for p, v in g.items() if v == 2}
    dcg = sum(GAIN[g.get(p, 0)] / math.log2(i + 2) for i, p in enumerate(ranked[:SHOWN]))
    ideal = sorted((GAIN[v] for v in g.values()), reverse=True)[:SHOWN]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    out = {"ndcg@5": dcg / idcg,
           "useful@5": sum(g.get(p, 0) > 0 for p in ranked[:SHOWN]) / SHOWN,
           "best@5": len(twos & set(ranked[:SHOWN])) / min(SHOWN, len(twos))}
    for k in RECALL_KS:
        out[f"recall@{k}"] = len(twos & set(ranked[:k])) / len(twos)
    return out


# --- methods: question text -> every member ranked, best first ---------------------------------

# How a person's section similarities become one score. "max" is today's find_members (a person
# = their single best-matching section); the others reward people who match in several sections
# (the multi-need hypothesis). Rows arrive best-first, so s[:n] are the person's top n.
AGG = {"max": lambda s: s[0], "sum2": lambda s: sum(s[:2]), "sum3": lambda s: sum(s[:3]),
       "mean3": lambda s: sum(s[:3]) / len(s[:3])}
# BM25, tokenizer and fusion are the PRODUCT's (src/hybrid.py): one implementation, so the
# eval measures exactly what members get.
from hybrid import BM25, keyword_index, rank_people, rrf as _rrf  # noqa: E402,F401


def _dense_people(conn, emb, question: str) -> dict[str, list[float]]:
    import store
    from tools import SCHEME
    per: dict[str, list[float]] = defaultdict(list)
    for row in store.search(conn, emb.embed_query(question), k=100_000, scheme=SCHEME):
        per[row["person_id"]].append(row["score"])
    return per


def _rank(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda p: -scores[p])


def method_dense(conn, emb, agg: str = "max"):
    """Meaning-based search; agg "max" is today's member search (find_members' ordering,
    extended past its 5-person cap so recall@k can look deeper)."""
    def rank(question: str) -> list[str]:
        return _rank({p: AGG[agg](s)
                      for p, s in _dense_people(conn, emb, query_of(question)).items()})
    return rank


def method_bm25(conn, emb):
    """Keyword search alone (diagnostic): a person = their best-matching section by BM25."""
    from tools import SCHEME
    index, _, section_people = keyword_index(conn, SCHEME)
    everyone = sorted(set(section_people))

    def rank(question: str) -> list[str]:
        best: dict[str, float] = {p: 0.0 for p in everyone}
        for p, s in zip(section_people, index.scores(query_of(question))):
            best[p] = max(best[p], s)
        return _rank(best)
    return rank


def method_hybrid(conn, emb, agg: str = "max"):
    """Keyword + meaning, merged by reciprocal rank fusion. agg "sum3" IS the shipped product
    ranking (src/hybrid.rank_people, D-051); other aggs compose the same primitives."""
    from tools import SCHEME
    if agg == "sum3":
        return lambda q: rank_people(conn, emb.embed_query(query_of(q)), query_of(q), SCHEME)[0]
    dense, keyword = method_dense(conn, emb, agg), method_bm25(conn, emb)
    return lambda question: _rrf(dense(question), keyword(question))


def method_rerank(conn, emb, model_name: str, agg: str, unit: str = "sections"):
    """Cross-encoder re-orders hybrid_sum3's top rerank.POOL people (src/rerank.py); people past
    the pool keep hybrid order, so recall@50 equals the pool's by construction. The pool and the
    reranker both get query_of(question) -- what find_members would receive."""
    from importlib.metadata import version

    import rerank
    from tools import SCHEME
    def rank(question: str) -> list[str]:
        text = query_of(question)
        return rerank.rerank_people(conn, emb.embed_query(text), text, SCHEME,
                                    model_name, agg, unit=unit)[0]
    rank.stats = lambda: rerank.STATS
    rank.config = {"rerank": {"model": rerank.MODELS[model_name], "agg": agg, "unit": unit,
                              "pool": rerank.POOL, "per_person": rerank.PER_PERSON,
                              "max_tokens": rerank.MAX_TOKENS[model_name],
                              "device": rerank.device(), "dtype": rerank.dtype(model_name)},
                   "rerank_stack": {p: version(p) for p in
                                    ("sentence-transformers", "transformers", "torch")}}
    return rank


def _splits() -> tuple[str, dict[str, dict]]:
    """LIVE_QUERIES keyed by question: what the chat model put in its find_members call (today's
    tool: "live"; the needs/exclusions variant: "split"). Warns, never blocks, when the agent
    prompt or tool description changed since the recording."""
    import hashlib

    import split_needs
    kept = {q["question"] for q in load_key()[1]}
    rows = [json.loads(l) for l in LIVE_QUERIES.read_text(encoding="utf-8").splitlines() if l]
    by_q = {r["question"]: r for r in rows if "live" in r and "split" in r}
    if not kept <= set(by_q):
        raise FileNotFoundError(f"{LIVE_QUERIES.name} misses scored questions -- re-record")
    now = split_needs._stamp()
    stale = [k for k in ("system_sha256", "split_tool_sha256") if rows[0].get(k) != now[k]]
    if not split_needs.live_tool_matches_agent():
        stale.append("find_members description")
    if stale:
        print(f"!! {LIVE_QUERIES.name} is STALE ({', '.join(stale)} changed since recording): the "
              "chat model would now word its searches differently -- re-record and re-pin.")
    _splits.sha256 = hashlib.sha256(LIVE_QUERIES.read_bytes()).hexdigest()
    return LIVE_QUERIES.name, by_q


def query_of(question: str) -> str:
    """The text a method searches with: the recorded chat-model wording (QUERY_INPUT "live") or
    the question as typed ("raw"). Unrecorded text (the warm-up) passes through unchanged."""
    if QUERY_INPUT == "raw":
        return question
    if not hasattr(query_of, "by_q"):
        query_of.by_q = _splits()[1]
    row = query_of.by_q.get(question)
    return row["live"]["criterion"] if row else question


def method_live_criterion(conn, emb):
    """Hybrid on the chat model's recorded wording whatever QUERY_INPUT says -- the diagnostic
    that led to D-052 (under the live input it equals hybrid_sum3; kept so the old run reproduces)."""
    from tools import SCHEME
    name, by_q = _splits()

    def rank(question: str) -> list[str]:
        text = by_q[question]["live"]["criterion"] if question in by_q else question
        return rank_people(conn, emb.embed_query(text), text, SCHEME)[0]
    rank.config = {"needs_file": name, "split_model": next(iter(by_q.values()))["model"]}
    return rank


def method_needs(conn, emb, mode: str):
    """Arm 3 (src/needs.py): the chat model's split of the question into needs, exclusions and
    the asker's context; per-need hybrid rankings combined by coverage."""
    import needs
    from tools import SCHEME
    name, by_q = _splits()

    def rank(question: str) -> list[str]:
        s = by_q.get(question, {}).get("split", {})
        return needs.rank_by_needs(conn, emb, query_of(question), s.get("needs", []),
                                   s.get("exclude_topics", []), s.get("asker_context", ""),
                                   SCHEME, mode)
    rank.config = {"needs_file": name, "split_model": next(iter(by_q.values()))["model"],
                   "needs": {"mode": mode, "context_weight": needs.CONTEXT_WEIGHT}}
    return rank


def method_llm(conn, emb, variant: str):
    """LLM reranker (eval/llm_rerank.py): Haiku 4.5's ordering of hybrid's top 15 on the live
    input, read from the newest complete analysis/llm_rerank_<variant>_*.jsonl (the rankings are
    produced by a batch job, so this arm only scores them)."""
    import hashlib
    kept = {q["question"] for q in load_key()[1]}
    for path in sorted((ROOT / "analysis").glob(f"llm_rerank_{variant}_*.jsonl"), reverse=True):
        rows = {r["question"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        if kept <= set(rows):
            break
    else:
        raise FileNotFoundError(f"no complete llm_rerank_{variant} file -- run eval/llm_rerank.py")
    if QUERY_INPUT != "live":
        raise ValueError("the LLM rankings were made on the live input; --raw would mislabel them")

    def rank(question: str) -> list[str]:
        return rows[question]["ranked"] if question in rows else []
    first = next(iter(rows.values()))
    rank.config = {"llm_rerank": {"variant": variant, "file": path.name,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "model": first["model"], "pool": first["pool"],
                                  "batch_id": first["batch_id"],
                                  "letters_repaired": sum(r["repaired"] for r in rows.values())}}
    return rank


METHODS = {"dense": method_dense, "bm25": method_bm25,
           "llm_rules": lambda c, e: method_llm(c, e, "rules"),
           "llm_plain": lambda c, e: method_llm(c, e, "plain"),
           "live_criterion": method_live_criterion,
           **{f"needs_{m}": (lambda m: lambda c, e: method_needs(c, e, m))(m)
              for m in ("rrf", "min", "rrf_ctx")},
           **{f"dense_{a}": (lambda a: lambda c, e: method_dense(c, e, a))(a)
              for a in ("sum2", "sum3", "mean3")},
           "hybrid": method_hybrid,
           **{f"hybrid_{a}": (lambda a: lambda c, e: method_hybrid(c, e, a))(a)
              for a in ("sum2", "sum3", "mean3")},
           **{f"ce_{m}_{a}": (lambda m, a: lambda c, e: method_rerank(c, e, m, a))(m, a)
              for m in ("minilm", "bge") for a in ("max", "sum3")},
           "ce_m3_profile": lambda c, e: method_rerank(c, e, "m3", "max", "profile")}


def compare(run_a: str, run_b: str, metric: str = "ndcg@5") -> None:
    """B vs A question by question: wins/losses and a paired bootstrap 95% CI on the mean
    difference. 34 questions is small: a mean gain whose CI crosses 0 is not yet a gain."""
    import random
    pa = json.loads((ROOT / "runs" / run_a / "summary.json").read_text())["per_question"]
    pb = json.loads((ROOT / "runs" / run_b / "summary.json").read_text())["per_question"]
    author = {q["id"]: q.get("author") for q in
              (json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l)}
    rng = random.Random(0)
    inputs = [json.loads((ROOT / "runs" / r / "config.json").read_text()).get(
        "query_input", {"mode": "raw"})["mode"] for r in (run_a, run_b)]   # pre-D-052 = raw
    print(f"{run_b.split('-rec-')[-1]} vs {run_a.split('-rec-')[-1]} on {metric}:")
    if inputs[0] != inputs[1]:
        print(f"  !! different search inputs ({run_a.split('-rec-')[-1]}: {inputs[0]}, "
              f"{run_b.split('-rec-')[-1]}: {inputs[1]}) -- not a like-for-like comparison")
    # Real users' questions and assistant-drafted ones are reported apart: drafted questions
    # skewed keyword-friendly once already, so they never decide a comparison alone.
    for label, keep in (("all", lambda q: True), ("real", lambda q: author.get(q) != "assistant"),
                        ("drafted", lambda q: author.get(q) == "assistant")):
        diffs = [pb[q][metric] - pa[q][metric] for q in pa if q in pb and keep(q)]
        if not diffs:
            continue
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10_000))
        print(f"  {label:8}(n={len(diffs):>2}) mean {sum(diffs) / len(diffs):+.3f} "
              f"[95% CI {boots[250]:+.3f}, {boots[9750]:+.3f}] | wins {sum(d > 0.005 for d in diffs)}"
              f", losses {sum(d < -0.005 for d in diffs)}, ties {sum(abs(d) <= 0.005 for d in diffs)}")


def main(method: str) -> None:
    import store
    from embedder import NomicLocal

    grades, kept = load_key()
    conn, emb = store.connect(), NomicLocal()
    rank = METHODS[method](conn, emb)
    rank("warm-up")                         # model + index load, kept out of the timing
    stats = getattr(rank, "stats", None)    # reranker arms: fresh vs memo-reused pairs
    before = dict(stats()) if stats else None
    per_q, results, times = {}, [], []
    for q in kept:
        t0 = time.perf_counter()
        ranked = rank(q["question"])
        times.append(time.perf_counter() - t0)
        per_q[q["id"]] = score(ranked, grades[q["id"]])
        results.append({"qid": q["id"], "ranked": ranked[:max(RECALL_KS)]})
    pairs = ({k: stats()[k] - before[k] for k in before} if stats else {})
    metrics = list(next(iter(per_q.values())))
    overall = {m: round(sum(s[m] for s in per_q.values()) / len(per_q), 3) for m in metrics}
    by_tag = defaultdict(list)
    for q in kept:
        for t in q["tags"]:
            by_tag[t].append(per_q[q["id"]]["ndcg@5"])

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, cwd=ROOT).stdout.strip())
    tag = f"live-{method}" if QUERY_INPUT == "live" else method   # raw keeps the pre-D-052 name
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{sha}-rec-{tag}"
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "runner": "eval/recommend_eval.py", "method": method,
        "git": {"sha": sha, "dirty": dirty},
        "answer_key": KEY.name, "rubric": "v3", "grader": "claude-haiku-5-5",
        "questions_kept": [q["id"] for q in kept], "shown": SHOWN, "gain": GAIN,
        "max_share_of_2s": MAX_SHARE_OF_2S,
        "query_input": ({"mode": "raw"} if QUERY_INPUT == "raw" else
                        {"mode": "live", "file": LIVE_QUERIES.name,
                         "sha256": getattr(_splits, "sha256", None)}),
        **getattr(rank, "config", {})},
        indent=1), encoding="utf-8")
    p95 = sorted(times)[math.ceil(0.95 * len(times)) - 1]
    (out / "summary.json").write_text(json.dumps({
        "n_questions": len(kept), "overall": overall,
        "ms_per_question": round(1000 * sum(times) / len(kept)), "ms_p95": round(1000 * p95),
        **({"rerank_pairs": pairs} if pairs else {}),
        "ndcg@5_by_tag": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_tag.items())},
        "per_question": {q: {m: round(v, 3) for m, v in s.items()} for q, s in per_q.items()}},
        indent=1), encoding="utf-8")
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n",
                                       encoding="utf-8")
    print(f"{run_id}: {len(kept)} questions, {1000 * sum(times) / len(kept):.0f} ms per question "
          f"(p95 {1000 * p95:.0f} ms)")
    if pairs.get("reused"):
        print(f"  NOTE: {pairs['reused']} of {pairs['reused'] + pairs['scored']} reranker pairs were "
              "scored by an earlier arm in this process -- the time above excludes that scoring")
    for m, v in overall.items():
        print(f"  {m:10} {v:.3f}")
    print("  ndcg@5 by question type: " + ", ".join(
        f"{t} {sum(v) / len(v):.2f} (n={len(v)})" for t, v in sorted(by_tag.items())))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(*sys.argv[2:])
    else:
        if "--raw" in sys.argv:
            QUERY_INPUT = "raw"
        for m in [a for a in sys.argv[1:] if a != "--raw"]:   # one process shares the memo
            main(m)
